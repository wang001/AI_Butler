"""
provider/openai_compatible_provider.py — OpenAI 兼容 Provider

当前 Butler 绝大多数模型都走 OpenAI 兼容接口，但不同兼容层仍然存在：
  - 是否支持 system role
  - reasoning delta 字段命名
  - 是否把 <think>...</think> 或工具标记塞进 content

这些差异统一封装在这里，避免 Runner 继续累积 provider 特判。
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from openai import AsyncOpenAI, RateLimitError

from provider.base import (
    ChatProvider,
    ContentBlockFilter,
    ProviderModelConfig,
    ProviderToolCallDelta,
)

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


def _textify(value: Any) -> str:
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


def _get_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _strip_tool_markup(text: str) -> str:
    if not text or _TOOL_SECTION_OPEN not in text:
        cleaned = text
    else:
        cleaned = _TOOL_SECTION_RE.sub("", text)
    cleaned = _MINIMAX_TOOL_CALL_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


class _OpenAIContentBlockFilter:
    """
    逐 chunk 拆分 content：
      - tool_calls_section 直接抑制
      - <think>...</think> 输出为 reasoning 片段
      - 其余内容输出为 text 片段
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


class OpenAICompatibleProvider(ChatProvider):
    _MAX_RETRIES = 3

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "",
        api_key: str = "",
    ):
        super().__init__(model=model)
        self._base_url = base_url
        self._api_key = api_key
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @property
    def name(self) -> str:
        return "openai_compatible"

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def _rejects_system_role(self) -> bool:
        base = (self._base_url or "").lower()
        mdl = (self.model or "").lower()
        return (
            "minimax" in base
            or "minnimax" in base
            or mdl.startswith("minimax")
        )

    def _rewrite_messages_without_system(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        system_chunks: list[str] = []
        rewritten: list[dict[str, Any]] = []

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

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prepared = (
            self._rewrite_messages_without_system(messages)
            if self._rejects_system_role()
            else [dict(msg) for msg in messages]
        )
        # Some OpenAI-compatible gateways reject assistant messages that mix
        # non-empty content with tool_calls. Tool results already carry the
        # actionable context, so clearing content is the most portable form.
        for msg in prepared:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["content"] = None
        return prepared

    async def create_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        stream: bool = False,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.prepare_messages(messages),
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None and tools:
            kwargs["tool_choice"] = tool_choice

        wait = 2
        for attempt in range(self._MAX_RETRIES):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except RateLimitError:
                if attempt == self._MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(wait)
                wait = min(wait * 2, 32)

    def split_reasoning_and_reply(self, text: str) -> tuple[str, str]:
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

    def extract_delta_reasoning(self, delta: Any) -> str:
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

    def extract_delta_content(self, delta: Any) -> str:
        return _textify(getattr(delta, "content", None))

    def create_content_block_filter(self) -> ContentBlockFilter:
        return _OpenAIContentBlockFilter()

    def extract_delta_tool_calls(self, delta: Any) -> list[ProviderToolCallDelta]:
        raw_calls = _get_value(delta, "tool_calls")
        if not raw_calls:
            extra = getattr(delta, "model_extra", None) or {}
            raw_calls = extra.get("tool_calls")
        if not raw_calls:
            return []

        result: list[ProviderToolCallDelta] = []
        for idx, raw in enumerate(raw_calls):
            index = _get_value(raw, "index")
            fn = _get_value(raw, "function")
            result.append(
                ProviderToolCallDelta(
                    index=int(index) if index is not None else idx,
                    id=str(_get_value(raw, "id") or ""),
                    name=str(_get_value(fn, "name") or "") if fn is not None else "",
                    arguments_delta=(
                        str(_get_value(fn, "arguments") or "")
                        if fn is not None
                        else ""
                    ),
                )
            )
        return result

    def export_model_config(self) -> ProviderModelConfig:
        return ProviderModelConfig(
            model=self.model,
            api_key=self._api_key,
            base_url=self._base_url,
            litellm_provider="openai",
        )
