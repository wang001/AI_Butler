"""
provider/openai_responses_provider.py — OpenAI Responses API Provider

Responses API 的协议和 Chat Completions 不同：
  - 请求使用 input / instructions
  - function tools 使用扁平 schema
  - tool result 使用 function_call_output
  - streaming 使用 response.* 语义事件

本 provider 将这些差异适配回 AgentRunner 已经消费的统一 Provider delta。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from openai import RateLimitError

from provider.openai_compatible_provider import (
    OpenAICompatibleProvider,
    _get_value,
    _textify,
)


class OpenAIResponsesProvider(OpenAICompatibleProvider):
    @property
    def name(self) -> str:
        return "openai_responses"

    async def create_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        stream: bool = False,
    ) -> Any:
        request = self._build_request(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
        )

        wait = 2
        for attempt in range(self._MAX_RETRIES):
            try:
                response = await self.client.responses.create(**request)
                self._ensure_responses_api_object(response, stream=stream)
                if stream:
                    return self._stream_to_chat_chunks(response)
                self._capture_response_id(response)
                return self._response_to_chat_completion(response)
            except RateLimitError:
                if attempt == self._MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(wait)
                wait = min(wait * 2, 32)

    def _ensure_responses_api_object(self, response: Any, *, stream: bool) -> None:
        if stream:
            if hasattr(response, "__aiter__"):
                return
            raise RuntimeError(
                "OpenAI Responses API returned a non-stream response. "
                "Please verify LLM_BASE_URL supports /responses."
            )

        if isinstance(response, str):
            preview = response.strip().replace("\n", " ")[:160]
            raise RuntimeError(
                "OpenAI Responses API returned plain text/HTML instead of a Response object. "
                f"Please verify LLM_BASE_URL supports /responses. Preview: {preview}"
            )

    def _build_request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        tool_outputs = self._latest_tool_outputs(messages)
        if tool_outputs and getattr(self, "_previous_response_id", ""):
            input_items = tool_outputs
            instructions = ""
            previous_response_id = self._previous_response_id
        else:
            instructions, input_items = self._messages_to_responses_input(messages)
            previous_response_id = None

        request: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "stream": stream,
        }
        if instructions:
            request["instructions"] = instructions
        if previous_response_id:
            request["previous_response_id"] = previous_response_id
        if tools:
            request["tools"] = self._tools_to_responses_tools(tools)
        if tool_choice is not None and tools:
            request["tool_choice"] = tool_choice
        return request

    def _messages_to_responses_input(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        instructions: list[str] = []
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""
            if role == "system":
                if content.strip():
                    instructions.append(content.strip())
                continue
            if role in ("user", "assistant"):
                if msg.get("tool_calls"):
                    for tool_call in msg.get("tool_calls") or []:
                        fn = tool_call.get("function") or {}
                        input_items.append({
                            "type": "function_call",
                            "call_id": tool_call.get("id") or "",
                            "name": fn.get("name") or "",
                            "arguments": fn.get("arguments") or "{}",
                            "status": "completed",
                        })
                    continue
                if content:
                    input_items.append({"role": role, "content": content})
                continue
            if role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or "",
                    "output": content,
                })

        return "\n\n".join(instructions).strip(), input_items

    def _latest_tool_outputs(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for msg in reversed(messages):
            role = msg.get("role", "")
            if role == "tool":
                outputs.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or "",
                    "output": msg.get("content") or "",
                })
                continue
            if outputs:
                break
        outputs.reverse()
        return outputs

    def _tools_to_responses_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function") or {}
            result.append({
                "type": "function",
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters") or {"type": "object"},
            })
        return result

    def _response_to_chat_completion(self, response: Any) -> Any:
        tool_calls = []
        for item in _get_value(response, "output") or []:
            if _get_value(item, "type") != "function_call":
                continue
            tool_calls.append(SimpleNamespace(
                id=str(_get_value(item, "call_id") or _get_value(item, "id") or ""),
                function=SimpleNamespace(
                    name=str(_get_value(item, "name") or ""),
                    arguments=str(_get_value(item, "arguments") or "{}"),
                ),
            ))

        message = SimpleNamespace(
            content=self._response_output_text(response),
            tool_calls=tool_calls,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def _response_output_text(self, response: Any) -> str:
        output_text = _textify(_get_value(response, "output_text"))
        if output_text:
            return output_text

        parts: list[str] = []
        for item in _get_value(response, "output") or []:
            if _get_value(item, "type") != "message":
                continue
            for part in _get_value(item, "content") or []:
                if _get_value(part, "type") in ("output_text", "text"):
                    text = _textify(_get_value(part, "text"))
                    if text:
                        parts.append(text)
        return "".join(parts)

    async def _stream_to_chat_chunks(self, stream: Any):
        call_meta: dict[int, dict[str, str]] = {}
        argument_seen: dict[int, bool] = {}

        async for event in stream:
            event_type = _get_value(event, "type")

            if event_type in ("response.created", "response.completed"):
                response = _get_value(event, "response")
                if response is not None:
                    self._capture_response_id(response)

            if event_type == "response.output_text.delta":
                yield self._chunk(content=str(_get_value(event, "delta") or ""))
                continue

            if event_type in (
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            ):
                yield self._chunk(reasoning=str(_get_value(event, "delta") or ""))
                continue

            if event_type == "response.output_item.added":
                output_index = int(_get_value(event, "output_index") or 0)
                item = _get_value(event, "item") or {}
                if _get_value(item, "type") != "function_call":
                    continue
                meta = call_meta.setdefault(output_index, {"id": "", "name": ""})
                meta["id"] = str(_get_value(item, "call_id") or _get_value(item, "id") or "")
                meta["name"] = str(_get_value(item, "name") or "")
                yield self._chunk(tool_calls=[self._tool_delta(
                    output_index=output_index,
                    call_id=meta["id"],
                    name=meta["name"],
                )])
                continue

            if event_type == "response.function_call_arguments.delta":
                output_index = int(_get_value(event, "output_index") or 0)
                argument_seen[output_index] = True
                meta = call_meta.setdefault(output_index, {"id": "", "name": ""})
                yield self._chunk(tool_calls=[self._tool_delta(
                    output_index=output_index,
                    call_id=meta.get("id", ""),
                    name=meta.get("name", ""),
                    arguments_delta=str(_get_value(event, "delta") or ""),
                )])
                continue

            if event_type == "response.function_call_arguments.done":
                output_index = int(_get_value(event, "output_index") or 0)
                meta = call_meta.setdefault(output_index, {"id": "", "name": ""})
                if _get_value(event, "name"):
                    meta["name"] = str(_get_value(event, "name") or "")
                if _get_value(event, "item_id") and not meta.get("id"):
                    meta["id"] = str(_get_value(event, "item_id") or "")
                if argument_seen.get(output_index):
                    continue
                yield self._chunk(tool_calls=[self._tool_delta(
                    output_index=output_index,
                    call_id=meta.get("id", ""),
                    name=meta.get("name", ""),
                    arguments_delta=str(_get_value(event, "arguments") or "{}"),
                )])

    def _capture_response_id(self, response: Any) -> None:
        response_id = _get_value(response, "id")
        if response_id:
            self._previous_response_id = str(response_id)

    def _chunk(
        self,
        *,
        content: str = "",
        reasoning: str = "",
        tool_calls: list[Any] | None = None,
    ) -> Any:
        delta = SimpleNamespace(
            content=content or None,
            reasoning_content=reasoning or None,
            reasoning=reasoning or None,
            tool_calls=tool_calls,
        )
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    def _tool_delta(
        self,
        *,
        output_index: int,
        call_id: str = "",
        name: str = "",
        arguments_delta: str = "",
    ) -> Any:
        return SimpleNamespace(
            index=output_index,
            id=call_id or None,
            function=SimpleNamespace(
                name=name or None,
                arguments=arguments_delta or None,
            ),
        )
