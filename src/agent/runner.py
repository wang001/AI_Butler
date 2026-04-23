# -*- coding: utf-8 -*-
"""
agent/runner.py — LLM ↔ Tool 执行引擎（AgentRunner）

参考 NanoBot 的 AgentRunner 设计，将纯粹的 LLM 调用 + Tool Call 循环
从 Butler 上帝类中抽离。AgentRunner 不感知记忆系统、上下文组装等上层逻辑，
只关心：给定 messages → 调用 LLM → 如有 tool_calls 则执行工具 → 循环。

职责：
  1. LLM 调用（带限流指数退避重试）
  2. Tool Call 循环 — 非流式版本（_tool_call_loop）
  3. Tool Call 循环 — 流式版本（_tool_call_loop_streaming）
  4. 工具批处理（按并发安全性分组并行/串行）
  5. 工具标记过滤（部分模型输出 tool_calls_section 标记）
  6. 通过 AgentHook 通知生命周期事件

不关心：
  - 对话历史管理（由 Butler 维护）
  - 上下文组装（由 ContextBuilder 负责）
  - 记忆系统（由 MemoryManager 负责）
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, AsyncGenerator, TYPE_CHECKING

from openai import AsyncOpenAI
from event import AgentEvent, make_agent_event, new_event_id

if TYPE_CHECKING:
    from agent.hooks import AgentHook
    from tools import ToolDispatcher

# ── 常量 ─────────────────────────────────────────────────────────────────────
MAX_TOOL_ROUNDS = 6  # tool call 最大轮次（防止死循环）

# ── 工具标记过滤 ─────────────────────────────────────────────────────────────
# 部分模型（如 Kimi / GLM 系）会把工具调用原始标记段输出到 content 字段。
# 以下工具负责将其剥离，保证对用户输出干净的纯文本。

_TOOL_SECTION_RE = re.compile(
    r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>",
    re.DOTALL,
)
_MINIMAX_TOOL_CALL_RE = re.compile(
    r"<minimax:tool_call>.*?</minimax:tool_call>",
    re.DOTALL,
)
_TOOL_SECTION_OPEN = "<|tool_calls_section_begin|>"
_TOOL_SECTION_CLOSE = "<|tool_calls_section_end|>"
_MINIMAX_TOOL_CALL_OPEN = "<minimax:tool_call>"
_MINIMAX_TOOL_CALL_CLOSE = "</minimax:tool_call>"
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _strip_tool_markup(text: str) -> str:
    """
    从非流式文本中剥离 <|tool_calls_section_begin|>...<|tool_calls_section_end|> 段落。
    同时去掉段落前后多余的空白行。
    """
    if not text or _TOOL_SECTION_OPEN not in text:
        cleaned = text
    else:
        cleaned = _TOOL_SECTION_RE.sub("", text)
    cleaned = _MINIMAX_TOOL_CALL_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


class _ToolMarkupFilter:
    """
    逐 chunk 过滤模型输出中的 tool_calls_section 标记。

    这样我们既能在同一条流里处理 content delta，又能保留 reasoning delta 的原始顺序。
    """

    def __init__(self) -> None:
        self._keep = max(len(_TOOL_SECTION_OPEN), len(_TOOL_SECTION_CLOSE)) - 1
        self._buf = ""
        self._suppressing = False

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        out: list[str] = []

        while True:
            if self._suppressing:
                idx = self._buf.find(_TOOL_SECTION_CLOSE)
                if idx == -1:
                    self._buf = self._buf[-self._keep:] if len(self._buf) > self._keep else self._buf
                    break
                self._buf = self._buf[idx + len(_TOOL_SECTION_CLOSE):]
                self._suppressing = False
                continue

            idx = self._buf.find(_TOOL_SECTION_OPEN)
            if idx == -1:
                if len(self._buf) > self._keep:
                    out.append(self._buf[:-self._keep])
                    self._buf = self._buf[-self._keep:]
                break

            prefix = self._buf[:idx].rstrip()
            if prefix:
                out.append(prefix)
            self._buf = self._buf[idx + len(_TOOL_SECTION_OPEN):]
            self._suppressing = True

        return out

    def flush(self) -> list[str]:
        if self._buf and not self._suppressing:
            out = [self._buf]
        else:
            out = []
        self._buf = ""
        self._suppressing = False
        return out


class _ContentBlockFilter:
    """
    逐 chunk 拆分 content：
      - tool_calls_section 直接抑制
      - <think>...</think> 输出为 reasoning 片段
      - 其余内容输出为 text 片段

    这样即便 provider 只会在 content 里塞 think 标签，我们也能转成结构化事件。
    """

    def __init__(self) -> None:
        self._keep = max(
            len(_TOOL_SECTION_OPEN),
            len(_TOOL_SECTION_CLOSE),
            len(_MINIMAX_TOOL_CALL_OPEN),
            len(_MINIMAX_TOOL_CALL_CLOSE),
            len(_THINK_OPEN),
            len(_THINK_CLOSE),
        ) - 1
        self._buf = ""
        self._mode = "text"

    def _emit_partial(self, kind: str) -> list[tuple[str, str]]:
        if len(self._buf) <= self._keep:
            return []
        piece = self._buf[:-self._keep]
        self._buf = self._buf[-self._keep:]
        return [(kind, piece)] if piece else []

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        self._buf += chunk
        out: list[tuple[str, str]] = []

        while True:
            if self._mode == "tool":
                close_candidates = [
                    idx for idx in (
                        self._buf.find(_TOOL_SECTION_CLOSE),
                        self._buf.find(_MINIMAX_TOOL_CALL_CLOSE),
                    )
                    if idx != -1
                ]
                if not close_candidates:
                    if len(self._buf) > self._keep:
                        self._buf = self._buf[-self._keep:]
                    break
                idx = min(close_candidates)
                close_marker = (
                    _TOOL_SECTION_CLOSE
                    if idx == self._buf.find(_TOOL_SECTION_CLOSE)
                    else _MINIMAX_TOOL_CALL_CLOSE
                )
                self._buf = self._buf[idx + len(close_marker):]
                self._mode = "text"
                continue

            if self._mode == "reasoning":
                idx = self._buf.find(_THINK_CLOSE)
                if idx == -1:
                    out.extend(self._emit_partial("reasoning"))
                    break
                piece = self._buf[:idx]
                if piece:
                    out.append(("reasoning", piece))
                self._buf = self._buf[idx + len(_THINK_CLOSE):]
                self._mode = "text"
                continue

            candidates = [
                (self._buf.find(_THINK_OPEN), "reasoning", len(_THINK_OPEN)),
                (self._buf.find(_TOOL_SECTION_OPEN), "tool", len(_TOOL_SECTION_OPEN)),
                (
                    self._buf.find(_MINIMAX_TOOL_CALL_OPEN),
                    "tool",
                    len(_MINIMAX_TOOL_CALL_OPEN),
                ),
            ]
            candidates = [item for item in candidates if item[0] != -1]
            if not candidates:
                out.extend(self._emit_partial("text"))
                break

            idx, next_mode, marker_len = min(candidates, key=lambda item: item[0])
            piece = self._buf[:idx]
            if piece:
                out.append(("text", piece))
            self._buf = self._buf[idx + marker_len:]
            self._mode = next_mode

        return out

    def flush(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self._buf and self._mode != "tool":
            kind = "reasoning" if self._mode == "reasoning" else "text"
            out.append((kind, self._buf))
        self._buf = ""
        self._mode = "text"
        return out


def _textify(value: Any) -> str:
    """将 provider 特有的 delta 结构尽量规整成字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_textify(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value and value[key] is not None:
                return _textify(value[key])
        return ""

    for key in ("text", "content", "value"):
        attr = getattr(value, key, None)
        if attr is not None:
            return _textify(attr)
    return ""


def _extract_reasoning_text(delta: Any) -> str:
    """兼容 OpenAI 兼容接口里常见的 reasoning delta 字段。"""
    for key in ("reasoning_content", "reasoning"):
        text = _textify(getattr(delta, key, None))
        if text:
            return text

    extra = getattr(delta, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        text = _textify(extra.get(key))
        if text:
            return text

    return ""


def _extract_content_text(delta: Any) -> str:
    return _textify(getattr(delta, "content", None))


def _new_event_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _split_reasoning_and_reply(text: str) -> tuple[str, str]:
    """
    将普通文本拆成 reasoning 与最终展示文本。

    兼容只会输出 <think>...</think> 的 provider，同时剥离工具标记段。
    """
    cleaned = _strip_tool_markup(text or "")
    if not cleaned:
        return "", ""

    reasoning_parts: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        block = (match.group(1) or "").strip()
        if block:
            reasoning_parts.append(block)
        return ""

    reply = _THINK_BLOCK_RE.sub(_collect, cleaned)
    reply = re.sub(r"\n{3,}", "\n\n", reply).strip()
    reasoning = "\n\n".join(reasoning_parts).strip()
    return reasoning, reply


def _tool_result_preview(result: str, limit: int = 1200) -> tuple[str, bool]:
    if len(result) <= limit:
        return result, False
    return result[:limit].rstrip() + "\n...", True


def _tool_output_payload(result: str) -> dict[str, Any]:
    preview, truncated = _tool_result_preview(result)
    return {
        "preview": preview,
        "truncated": truncated,
    }


def _tool_input_text(args: Any) -> str:
    if not args:
        return "{}"
    return json.dumps(args, ensure_ascii=False, indent=2)


def _provider_rejects_system_role(base_url: str, model: str) -> bool:
    """
    某些 OpenAI 兼容接口不接受 system role，例如当前接入的 MiniMax 兼容层。
    """
    base = (base_url or "").lower()
    mdl = (model or "").lower()
    return (
        "minimax" in base
        or "minnimax" in base
        or mdl.startswith("minimax")
    )


def _rewrite_messages_without_system(messages: list[dict]) -> list[dict]:
    """
    将 system 消息折叠进第一条 user 消息，兼容不支持 system role 的 provider。
    """
    system_chunks: list[str] = []
    rewritten: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            content = (msg.get("content") or "").strip()
            if content:
                system_chunks.append(content)
            continue
        rewritten.append(dict(msg))

    if not system_chunks:
        return rewritten

    preamble = (
        "[以下内容是系统指令、长期记忆和检索上下文，请严格遵守，不要把它们当作用户原话复述]\n\n"
        + "\n\n---\n\n".join(system_chunks)
    )

    for msg in rewritten:
        if msg.get("role") == "user":
            user_content = msg.get("content") or ""
            msg["content"] = f"{preamble}\n\n---\n\n{user_content}".strip()
            return rewritten

    return [{"role": "user", "content": preamble}] + rewritten


def _prepare_messages_for_llm(
    llm: AsyncOpenAI,
    model: str,
    messages: list[dict],
) -> list[dict]:
    """
    在发给 provider 前做兼容性适配，避免不同兼容层的协议差异直接炸请求。
    """
    base_url = str(getattr(llm, "base_url", "") or "")
    if _provider_rejects_system_role(base_url, model):
        return _rewrite_messages_without_system(messages)
    return messages


# ── LLM 调用（带限流重试）────────────────────────────────────────────────────

async def _llm_call(llm: AsyncOpenAI, **kwargs):
    """指数退避重试，处理 429 限流。最多 3 次，间隔 2→4→8s。"""
    from openai import RateLimitError
    if "messages" in kwargs and "model" in kwargs:
        kwargs["messages"] = _prepare_messages_for_llm(
            llm=llm,
            model=kwargs["model"],
            messages=kwargs["messages"],
        )
    wait = 2
    for attempt in range(3):
        try:
            return await llm.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == 2:
                raise
            await asyncio.sleep(wait)
            wait = min(wait * 2, 32)


# ── AgentRunner ──────────────────────────────────────────────────────────────

class AgentRunner:
    """
    LLM ↔ Tool 执行引擎。

    给定 messages + tools → 调用 LLM → 执行工具 → 循环，
    直到模型不再发出 tool_calls 或达到最大轮次。

    通过 AgentHook 通知外部（Channel 层）执行进度。
    """

    def __init__(
        self,
        llm: AsyncOpenAI,
        model: str,
        dispatcher: "ToolDispatcher",
        hook: "AgentHook | None" = None,
    ):
        self._llm = llm
        self._model = model
        self._dispatcher = dispatcher
        self._hook = hook

    @property
    def dispatcher(self) -> "ToolDispatcher":
        return self._dispatcher

    async def _execute_tool_call(self, tc) -> tuple[str, str, str]:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {}

        if self._hook:
            await self._hook.on_tool_start(name, args)

        try:
            result = await self._dispatcher.run(name, tc.function.arguments)
        except Exception as exc:
            result = f"[工具执行失败] {name}: {exc}"

        if self._hook:
            await self._hook.on_tool_end(name, result)

        return tc.id, name, result

    async def _run_tool_batch(self, tool_calls: list) -> list[tuple[str, str, str]]:
        """
        执行一批工具调用（批内按并发安全性分组并行），返回 (tool_call_id, name, result)。
        """
        # 按并发安全性分批：相邻安全工具合并并行，不安全工具独占串行
        batches: list[list] = []
        for tc in tool_calls:
            safe = self._dispatcher.concurrent_safe(tc.function.name)
            if safe and batches and all(
                self._dispatcher.concurrent_safe(t.function.name) for t in batches[-1]
            ):
                batches[-1].append(tc)
            else:
                batches.append([tc])

        completed: list[tuple[str, str, str]] = []
        for batch in batches:
            if len(batch) == 1:
                completed.append(await self._execute_tool_call(batch[0]))
            else:
                completed.extend(await asyncio.gather(*[
                    self._execute_tool_call(tc) for tc in batch
                ]))

        return completed

    async def run(
        self,
        messages: list[dict],
    ) -> tuple[str, list[dict]]:
        """
        Tool Call 循环（非流式）。

        返回 (最终回复文本, 本轮新增消息列表)。
        批内并行，批间串行；超过 MAX_TOOL_ROUNDS 强制不带工具再请求一次。
        """
        new_messages: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            if self._hook:
                await self._hook.on_llm_start(messages + new_messages)

            response = await _llm_call(
                self._llm,
                model=self._model,
                messages=messages + new_messages,
                tools=self._dispatcher.tools,
                tool_choice="auto",
                stream=False,
            )

            if self._hook:
                await self._hook.on_llm_end(response)

            msg = response.choices[0].message

            if not msg.tool_calls:
                _reasoning, reply = _split_reasoning_and_reply(msg.content or "")
                new_messages.append({"role": "assistant", "content": reply})
                return reply, new_messages

            # 记录 tool_call 消息
            _reasoning, content = _split_reasoning_and_reply(msg.content or "")
            new_messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tid, _name, result in await self._run_tool_batch(msg.tool_calls):
                new_messages.append({"role": "tool", "tool_call_id": tid, "content": result})

        # 超过最大轮次，强制不带工具再请求
        response = await _llm_call(
            self._llm,
            model=self._model,
            messages=messages + new_messages,
            stream=False,
        )
        _reasoning, reply = _split_reasoning_and_reply(
            response.choices[0].message.content or ""
        )
        new_messages.append({"role": "assistant", "content": reply})
        return reply, new_messages

    async def run_stream(
        self,
        messages: list[dict],
        conversation_id: str = "",
        message_id: str = "",
    ) -> AsyncGenerator[AgentEvent | list[dict], None]:
        """
        Tool Call 循环（流式版本）。

        yield 三类内容：
          - AgentEvent                         内部 canonical 事件
          - list[dict]（最后一次）            new_messages，供 Butler 做持久化

        工具调用阶段：非流式等待工具完成。
        最终回复阶段：stream=True，逐 delta yield。
        超过 MAX_TOOL_ROUNDS 后强制以流式输出最终回复（不带 tools）。
        """
        new_messages: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            step_id = new_event_id("step")
            yield make_agent_event(
                kind="step.started",
                source="agent",
                conversation_id=conversation_id,
                message_id=message_id,
                step_id=step_id,
            )

            # 先非流式探测是否有 tool_calls
            if self._hook:
                await self._hook.on_llm_start(messages + new_messages)

            response = await _llm_call(
                self._llm,
                model=self._model,
                messages=messages + new_messages,
                tools=self._dispatcher.tools,
                tool_choice="auto",
                stream=False,
            )

            if self._hook:
                await self._hook.on_llm_end(response)

            msg = response.choices[0].message

            if not msg.tool_calls:
                # 没有工具调用 → 以流式重新请求并 yield 结构化事件
                reply_parts: list[str] = []
                async for event in self._stream_reply(
                    messages + new_messages,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                ):
                    if event["kind"] == "text.delta":
                        reply_parts.append(str(event["payload"].get("delta") or ""))
                        if self._hook:
                            await self._hook.on_stream_token(str(event["payload"].get("delta") or ""))
                    yield event
                yield make_agent_event(
                    kind="step.finished",
                    source="agent",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={"finishReason": "stop"},
                )
                new_messages.append({"role": "assistant", "content": "".join(reply_parts)})
                yield new_messages
                return

            # 有 tool_calls → 记录 & 通知 & 执行
            reasoning, content = _split_reasoning_and_reply(msg.content or "")
            new_messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            display_reasoning = "\n\n".join(part for part in (reasoning, content or "") if part).strip()
            if display_reasoning:
                reasoning_id = new_event_id("reasoning")
                yield make_agent_event(
                    kind="reasoning.started",
                    source="agent",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={"partId": reasoning_id},
                )
                yield make_agent_event(
                    kind="reasoning.delta",
                    source="agent",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={"partId": reasoning_id, "delta": display_reasoning},
                )
                yield make_agent_event(
                    kind="reasoning.finished",
                    source="agent",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={"partId": reasoning_id},
                )

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {"_raw": tc.function.arguments}
                yield make_agent_event(
                    kind="tool.call.started",
                    source="tool",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={"toolCallId": tc.id, "toolName": tc.function.name},
                )
                yield make_agent_event(
                    kind="tool.call.arguments",
                    source="tool",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={
                        "toolCallId": tc.id,
                        "toolName": tc.function.name,
                        "inputTextDelta": _tool_input_text(args),
                        "input": args,
                    },
                )

            # 执行工具
            for tid, _name, result in await self._run_tool_batch(msg.tool_calls):
                new_messages.append({"role": "tool", "tool_call_id": tid, "content": result})
                yield make_agent_event(
                    kind="tool.call.finished",
                    source="tool",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={
                        "toolCallId": tid,
                        "toolName": _name,
                        "output": _tool_output_payload(result),
                    },
                )
            yield make_agent_event(
                kind="step.finished",
                source="agent",
                conversation_id=conversation_id,
                message_id=message_id,
                step_id=step_id,
                payload={"finishReason": "tool-calls"},
            )

        # 超过最大轮次，强制流式输出（不带 tools）
        step_id = new_event_id("step")
        yield make_agent_event(
            kind="step.started",
            source="agent",
            conversation_id=conversation_id,
            message_id=message_id,
            step_id=step_id,
        )
        reply_parts: list[str] = []
        async for event in self._stream_reply(
            messages + new_messages,
            use_tools=False,
            conversation_id=conversation_id,
            message_id=message_id,
            step_id=step_id,
        ):
            if event["kind"] == "text.delta":
                reply_parts.append(str(event["payload"].get("delta") or ""))
                if self._hook:
                    await self._hook.on_stream_token(str(event["payload"].get("delta") or ""))
            yield event
        yield make_agent_event(
            kind="step.finished",
            source="agent",
            conversation_id=conversation_id,
            message_id=message_id,
            step_id=step_id,
            payload={"finishReason": "stop"},
        )
        new_messages.append({"role": "assistant", "content": "".join(reply_parts)})
        yield new_messages

    async def _stream_reply(
        self,
        messages: list[dict],
        use_tools: bool = False,
        conversation_id: str = "",
        message_id: str = "",
        step_id: str = "",
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        以 stream=True 请求 LLM，逐 delta yield 结构化事件。
        """
        from openai import RateLimitError
        kwargs = dict(
            model=self._model,
            messages=_prepare_messages_for_llm(
                llm=self._llm,
                model=self._model,
                messages=messages,
            ),
            stream=True,
        )
        if use_tools:
            kwargs["tools"] = self._dispatcher.tools
            kwargs["tool_choice"] = "auto"

        wait = 2
        for attempt in range(3):
            try:
                stream = await self._llm.chat.completions.create(**kwargs)
                break
            except RateLimitError:
                if attempt == 2:
                    raise
                await asyncio.sleep(wait)
                wait = min(wait * 2, 32)

        content_filter = _ContentBlockFilter()
        active_reasoning_id: str | None = None
        active_text_id: str | None = None

        async def _close_text_if_needed() -> AsyncGenerator[AgentEvent, None]:
            nonlocal active_text_id
            if active_text_id is not None:
                yield make_agent_event(
                    kind="text.finished",
                    source="agent",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={"partId": active_text_id},
                )
                active_text_id = None

        async def _close_reasoning_if_needed() -> AsyncGenerator[AgentEvent, None]:
            nonlocal active_reasoning_id
            if active_reasoning_id is not None:
                yield make_agent_event(
                    kind="reasoning.finished",
                    source="agent",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={"partId": active_reasoning_id},
                )
                active_reasoning_id = None

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            reasoning_text = _extract_reasoning_text(delta)
            if reasoning_text:
                async for event in _close_text_if_needed():
                    yield event
                if active_reasoning_id is None:
                    active_reasoning_id = new_event_id("reasoning")
                    yield make_agent_event(
                        kind="reasoning.started",
                        source="agent",
                        conversation_id=conversation_id,
                        message_id=message_id,
                        step_id=step_id,
                        payload={"partId": active_reasoning_id},
                    )
                yield make_agent_event(
                    kind="reasoning.delta",
                    source="agent",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={"partId": active_reasoning_id, "delta": reasoning_text},
                )

            content_text = _extract_content_text(delta)
            if not content_text:
                continue

            for kind, piece in content_filter.feed(content_text):
                if not piece:
                    continue
                if kind == "reasoning":
                    async for event in _close_text_if_needed():
                        yield event
                    if active_reasoning_id is None:
                        active_reasoning_id = new_event_id("reasoning")
                        yield make_agent_event(
                            kind="reasoning.started",
                            source="agent",
                            conversation_id=conversation_id,
                            message_id=message_id,
                            step_id=step_id,
                            payload={"partId": active_reasoning_id},
                        )
                    yield make_agent_event(
                        kind="reasoning.delta",
                        source="agent",
                        conversation_id=conversation_id,
                        message_id=message_id,
                        step_id=step_id,
                        payload={"partId": active_reasoning_id, "delta": piece},
                    )
                    continue

                async for event in _close_reasoning_if_needed():
                    yield event
                if active_text_id is None:
                    active_text_id = new_event_id("text")
                    yield make_agent_event(
                        kind="text.started",
                        source="agent",
                        conversation_id=conversation_id,
                        message_id=message_id,
                        step_id=step_id,
                        payload={"partId": active_text_id},
                    )
                yield make_agent_event(
                    kind="text.delta",
                    source="agent",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={"partId": active_text_id, "delta": piece},
                )

        for kind, piece in content_filter.flush():
            if not piece:
                continue
            if kind == "reasoning":
                async for event in _close_text_if_needed():
                    yield event
                if active_reasoning_id is None:
                    active_reasoning_id = new_event_id("reasoning")
                    yield make_agent_event(
                        kind="reasoning.started",
                        source="agent",
                        conversation_id=conversation_id,
                        message_id=message_id,
                        step_id=step_id,
                        payload={"partId": active_reasoning_id},
                    )
                yield make_agent_event(
                    kind="reasoning.delta",
                    source="agent",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={"partId": active_reasoning_id, "delta": piece},
                )
                continue

            async for event in _close_reasoning_if_needed():
                yield event
            if active_text_id is None:
                active_text_id = new_event_id("text")
                yield make_agent_event(
                    kind="text.started",
                    source="agent",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={"partId": active_text_id},
                )
            yield make_agent_event(
                kind="text.delta",
                source="agent",
                conversation_id=conversation_id,
                message_id=message_id,
                step_id=step_id,
                payload={"partId": active_text_id, "delta": piece},
            )

        async for event in _close_reasoning_if_needed():
            yield event
        async for event in _close_text_if_needed():
            yield event
