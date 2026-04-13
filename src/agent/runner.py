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
from typing import AsyncGenerator, TYPE_CHECKING

from openai import AsyncOpenAI

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
_TOOL_SECTION_OPEN  = "<|tool_calls_section_begin|>"
_TOOL_SECTION_CLOSE = "<|tool_calls_section_end|>"


def _strip_tool_markup(text: str) -> str:
    """
    从非流式文本中剥离 <|tool_calls_section_begin|>...<|tool_calls_section_end|> 段落。
    同时去掉段落前后多余的空白行。
    """
    if not text or _TOOL_SECTION_OPEN not in text:
        return text
    cleaned = _TOOL_SECTION_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


async def _strip_tool_sections_stream(
    gen: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """
    从流式 token 序列中过滤 <|tool_calls_section_begin|>...<|tool_calls_section_end|>。

    采用滑动缓冲区方案：
    - NORMAL 模式：安全部分立即 yield，末尾保留足够字符以检测开始标记。
    - SUPPRESS 模式：丢弃所有内容直到找到结束标记。
    """
    OPEN  = _TOOL_SECTION_OPEN
    CLOSE = _TOOL_SECTION_CLOSE
    KEEP  = max(len(OPEN), len(CLOSE)) - 1

    buf         = ""
    suppressing = False

    async for token in gen:
        buf += token

        while True:
            if suppressing:
                idx = buf.find(CLOSE)
                if idx == -1:
                    buf = buf[-KEEP:] if len(buf) > KEEP else buf
                    break
                buf        = buf[idx + len(CLOSE):]
                suppressing = False
            else:
                idx = buf.find(OPEN)
                if idx == -1:
                    if len(buf) > KEEP:
                        yield buf[:-KEEP]
                        buf = buf[-KEEP:]
                    break
                prefix = buf[:idx].rstrip()
                if prefix:
                    yield prefix
                buf        = buf[idx + len(OPEN):]
                suppressing = True

    if buf and not suppressing:
        yield buf


# ── LLM 调用（带限流重试）────────────────────────────────────────────────────

async def _llm_call(llm: AsyncOpenAI, **kwargs):
    """指数退避重试，处理 429 限流。最多 3 次，间隔 2→4→8s。"""
    from openai import RateLimitError
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

    async def _run_tool_batch(self, tool_calls: list) -> list[dict]:
        """
        执行一批工具调用（批内按并发安全性分组并行），返回 tool role 消息列表。
        """
        from tools import TOOL_CONCURRENT_SAFE

        # 按并发安全性分批：相邻安全工具合并并行，不安全工具独占串行
        batches: list[list] = []
        for tc in tool_calls:
            safe = TOOL_CONCURRENT_SAFE.get(tc.function.name, False)
            if safe and batches and all(
                TOOL_CONCURRENT_SAFE.get(t.function.name, False) for t in batches[-1]
            ):
                batches[-1].append(tc)
            else:
                batches.append([tc])

        async def _exec(tc):
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}

            if self._hook:
                await self._hook.on_tool_start(name, args)

            result = await self._dispatcher.run(name, tc.function.arguments)

            if self._hook:
                await self._hook.on_tool_end(name, result)

            return tc.id, result

        tool_messages: list[dict] = []
        for batch in batches:
            if len(batch) == 1:
                tid, result = await _exec(batch[0])
                tool_messages.append({"role": "tool", "tool_call_id": tid, "content": result})
            else:
                for tid, result in await asyncio.gather(*[_exec(tc) for tc in batch]):
                    tool_messages.append({"role": "tool", "tool_call_id": tid, "content": result})

        return tool_messages

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
                reply = _strip_tool_markup(msg.content or "")
                new_messages.append({"role": "assistant", "content": reply})
                return reply, new_messages

            # 记录 tool_call 消息
            new_messages.append({
                "role": "assistant",
                "content": _strip_tool_markup(msg.content or "") or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            new_messages.extend(await self._run_tool_batch(msg.tool_calls))

        # 超过最大轮次，强制不带工具再请求
        response = await _llm_call(
            self._llm,
            model=self._model,
            messages=messages + new_messages,
            stream=False,
        )
        reply = _strip_tool_markup(response.choices[0].message.content or "")
        new_messages.append({"role": "assistant", "content": reply})
        return reply, new_messages

    async def run_stream(
        self,
        messages: list[dict],
    ) -> AsyncGenerator[str | list[dict], None]:
        """
        Tool Call 循环（流式版本）。

        yield 三类内容：
          - str（以 \\x00TOOL_CALL: 开头）  工具开始调用事件
          - str（以 \\x00TOOL_RESULT: 开头）工具执行完毕事件
          - str（普通字符串）               最终回复的 token
          - list[dict]（最后一次）          new_messages，供 Butler 做持久化

        工具调用阶段：非流式等待工具完成。
        最终回复阶段：stream=True，逐 token yield。
        超过 MAX_TOOL_ROUNDS 后强制以流式输出最终回复（不带 tools）。
        """
        new_messages: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
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
                # 没有工具调用 → 以流式重新请求并 yield tokens
                reply_tokens: list[str] = []
                async for token in _strip_tool_sections_stream(
                    self._stream_reply(messages + new_messages)
                ):
                    reply_tokens.append(token)
                    if self._hook:
                        await self._hook.on_stream_token(token)
                    yield token
                new_messages.append({"role": "assistant", "content": "".join(reply_tokens)})
                yield new_messages
                return

            # 有 tool_calls → 记录 & 通知 & 执行
            new_messages.append({
                "role": "assistant",
                "content": _strip_tool_markup(msg.content or "") or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {"_raw": tc.function.arguments}
                payload = json.dumps({"name": name, "args": args}, ensure_ascii=False)
                yield f"\x00TOOL_CALL:{payload}"

            # 执行工具
            tool_messages = await self._run_tool_batch(msg.tool_calls)
            new_messages.extend(tool_messages)

            tool_names = [tc.function.name for tc in msg.tool_calls]
            for name in tool_names:
                payload = json.dumps({"name": name}, ensure_ascii=False)
                yield f"\x00TOOL_RESULT:{payload}"

        # 超过最大轮次，强制流式输出（不带 tools）
        reply_tokens: list[str] = []
        async for token in _strip_tool_sections_stream(
            self._stream_reply(messages + new_messages, use_tools=False)
        ):
            reply_tokens.append(token)
            if self._hook:
                await self._hook.on_stream_token(token)
            yield token
        new_messages.append({"role": "assistant", "content": "".join(reply_tokens)})
        yield new_messages

    async def _stream_reply(
        self,
        messages: list[dict],
        use_tools: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        以 stream=True 请求 LLM，逐 token yield 文本片段。
        """
        from openai import RateLimitError
        kwargs = dict(
            model=self._model,
            messages=messages,
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

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
