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
from typing import Any, AsyncGenerator, TYPE_CHECKING

from event import AgentEvent, make_agent_event, new_event_id
from provider import ChatProvider, ProviderToolCall

if TYPE_CHECKING:
    from agent.hooks import AgentHook
    from tools import ToolDispatcher

# ── 常量 ─────────────────────────────────────────────────────────────────────
MAX_TOOL_ROUNDS = 6  # tool call 最大轮次（防止死循环）

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


def _tool_call_id(tc: Any) -> str:
    return str(getattr(tc, "id", "") or "")


def _tool_call_name(tc: Any) -> str:
    if isinstance(tc, ProviderToolCall):
        return tc.name
    fn = getattr(tc, "function", None)
    return str(getattr(fn, "name", "") or "")


def _tool_call_arguments(tc: Any) -> str:
    if isinstance(tc, ProviderToolCall):
        return tc.arguments or "{}"
    fn = getattr(tc, "function", None)
    return str(getattr(fn, "arguments", "") or "{}")


def _parse_tool_args(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {"_raw": arguments}


def _serialize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": _tool_call_id(tc),
            "type": "function",
            "function": {
                "name": _tool_call_name(tc),
                "arguments": _tool_call_arguments(tc),
            },
        }
        for tc in tool_calls
    ]


def _tool_calls_from_stream_buffers(
    buffers: dict[int, dict[str, str]],
) -> list[ProviderToolCall]:
    result: list[ProviderToolCall] = []
    for index in sorted(buffers):
        item = buffers[index]
        name = item.get("name") or ""
        if not name:
            continue
        result.append(
            ProviderToolCall(
                id=item.get("id") or new_event_id("call"),
                name=name,
                arguments=item.get("arguments") or "{}",
            )
        )
    return result


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
        provider: ChatProvider,
        dispatcher: "ToolDispatcher",
        hook: "AgentHook | None" = None,
    ):
        self._provider = provider
        self._dispatcher = dispatcher
        self._hook = hook

    @property
    def dispatcher(self) -> "ToolDispatcher":
        return self._dispatcher

    @property
    def provider(self) -> ChatProvider:
        return self._provider

    async def _execute_tool_call(self, tc) -> tuple[str, str, str]:
        tid = _tool_call_id(tc)
        name = _tool_call_name(tc)
        arguments = _tool_call_arguments(tc)
        args = _parse_tool_args(arguments)

        if self._hook:
            await self._hook.on_tool_start(name, args)

        try:
            result = await self._dispatcher.run(name, arguments)
        except Exception as exc:
            result = f"[工具执行失败] {name}: {exc}"

        if self._hook:
            await self._hook.on_tool_end(name, result)

        return tid, name, result

    async def _run_tool_batch(self, tool_calls: list) -> list[tuple[str, str, str]]:
        """
        执行一批工具调用（批内按并发安全性分组并行），返回 (tool_call_id, name, result)。
        """
        # 按并发安全性分批：相邻安全工具合并并行，不安全工具独占串行
        batches: list[list] = []
        for tc in tool_calls:
            safe = self._dispatcher.concurrent_safe(_tool_call_name(tc))
            if safe and batches and all(
                self._dispatcher.concurrent_safe(_tool_call_name(t)) for t in batches[-1]
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

            response = await self._provider.create_completion(
                messages=messages + new_messages,
                tools=self._dispatcher.tools,
                tool_choice="auto",
                stream=False,
            )

            if self._hook:
                await self._hook.on_llm_end(response)

            msg = response.choices[0].message

            if not msg.tool_calls:
                _reasoning, reply = self._provider.split_reasoning_and_reply(msg.content or "")
                new_messages.append({"role": "assistant", "content": reply})
                return reply, new_messages

            # 记录 tool_call 消息
            _reasoning, content = self._provider.split_reasoning_and_reply(msg.content or "")
            new_messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": _serialize_tool_calls(msg.tool_calls),
            })

            for tid, _name, result in await self._run_tool_batch(msg.tool_calls):
                new_messages.append({"role": "tool", "tool_call_id": tid, "content": result})

        # 超过最大轮次，强制不带工具再请求
        response = await self._provider.create_completion(
            messages=messages + new_messages,
            stream=False,
        )
        _reasoning, reply = self._provider.split_reasoning_and_reply(
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

        每个 step 使用一次 stream=True 请求，同时收集正文、reasoning 与 tool call delta。
        如果该 step 产生工具调用，则流结束后执行工具并进入下一 step。
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

            # 单次流式请求内同时收集 text/reasoning 与 tool call delta。
            if self._hook:
                await self._hook.on_llm_start(messages + new_messages)

            reply_parts: list[str] = []
            tool_call_buffers: dict[int, dict[str, str]] = {}
            async for event in self._stream_reply(
                messages + new_messages,
                use_tools=True,
                conversation_id=conversation_id,
                message_id=message_id,
                step_id=step_id,
                tool_call_buffers=tool_call_buffers,
            ):
                if event["kind"] == "text.delta":
                    reply_parts.append(str(event["payload"].get("delta") or ""))
                    if self._hook:
                        await self._hook.on_stream_token(str(event["payload"].get("delta") or ""))
                yield event

            if self._hook:
                await self._hook.on_llm_end({"stream": True})

            tool_calls = _tool_calls_from_stream_buffers(tool_call_buffers)
            if not tool_calls:
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
            new_messages.append({
                "role": "assistant",
                "content": "".join(reply_parts).strip() or None,
                "tool_calls": _serialize_tool_calls(tool_calls),
            })

            for tc in tool_calls:
                args = _parse_tool_args(_tool_call_arguments(tc))
                yield make_agent_event(
                    kind="tool.call.started",
                    source="tool",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={"toolCallId": _tool_call_id(tc), "toolName": _tool_call_name(tc)},
                )
                yield make_agent_event(
                    kind="tool.call.arguments",
                    source="tool",
                    conversation_id=conversation_id,
                    message_id=message_id,
                    step_id=step_id,
                    payload={
                        "toolCallId": _tool_call_id(tc),
                        "toolName": _tool_call_name(tc),
                        "inputTextDelta": _tool_input_text(args),
                        "input": args,
                    },
                )

            # 执行工具
            for tid, _name, result in await self._run_tool_batch(tool_calls):
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
        tool_call_buffers: dict[int, dict[str, str]] | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        以 stream=True 请求 LLM，逐 delta yield 结构化事件。
        """
        stream = await self._provider.create_completion(
            messages=messages,
            tools=self._dispatcher.tools if use_tools else None,
            tool_choice="auto" if use_tools else None,
            stream=True,
        )

        content_filter = self._provider.create_content_block_filter()
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

            if tool_call_buffers is not None:
                for tc_delta in self._provider.extract_delta_tool_calls(delta):
                    buf = tool_call_buffers.setdefault(
                        tc_delta.index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    if tc_delta.id:
                        buf["id"] = tc_delta.id
                    if tc_delta.name:
                        buf["name"] = tc_delta.name
                    if tc_delta.arguments_delta:
                        buf["arguments"] += tc_delta.arguments_delta

            reasoning_text = self._provider.extract_delta_reasoning(delta)
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

            content_text = self._provider.extract_delta_content(delta)
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
