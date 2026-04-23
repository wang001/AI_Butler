# -*- coding: utf-8 -*-
"""
event/mapper.py — 内部事件到外部流式协议的映射

当前主要提供：
  AgentEvent -> StreamEvent

这样 Agent Runtime 可以只关心内部 canonical 事件，
而 Web/CLI 继续维持现有协议与 UI 兼容性。
"""
from __future__ import annotations

from event.stream import StreamEvent
from event.types import AgentEvent


def to_stream_events(event: AgentEvent) -> list[StreamEvent]:
    kind = event["kind"]
    payload = event["payload"]

    if kind == "message.started":
        return [{"type": "start", "messageId": event["messageId"]}]

    if kind == "message.finished":
        return [{
            "type": "finish",
            "finishReason": str(payload.get("finishReason") or "stop"),
        }]

    if kind == "message.error":
        return [{
            "type": "error",
            "errorText": str(payload.get("errorText") or "未知错误"),
        }]

    if kind == "step.started":
        return [{"type": "start-step"}]

    if kind == "step.finished":
        return [{
            "type": "finish-step",
            "finishReason": str(payload.get("finishReason") or "stop"),
        }]

    if kind == "reasoning.started":
        return [{"type": "reasoning-start", "id": str(payload.get("partId") or "")}]

    if kind == "reasoning.delta":
        return [{
            "type": "reasoning-delta",
            "id": str(payload.get("partId") or ""),
            "delta": str(payload.get("delta") or ""),
        }]

    if kind == "reasoning.finished":
        return [{"type": "reasoning-end", "id": str(payload.get("partId") or "")}]

    if kind == "tool.call.started":
        return [{
            "type": "tool-input-start",
            "toolCallId": str(payload.get("toolCallId") or ""),
            "toolName": str(payload.get("toolName") or ""),
        }]

    if kind == "tool.call.arguments":
        events: list[StreamEvent] = []
        input_text_delta = str(payload.get("inputTextDelta") or "")
        if input_text_delta:
            events.append({
                "type": "tool-input-delta",
                "toolCallId": str(payload.get("toolCallId") or ""),
                "inputTextDelta": input_text_delta,
            })
        events.append({
            "type": "tool-input-available",
            "toolCallId": str(payload.get("toolCallId") or ""),
            "toolName": str(payload.get("toolName") or ""),
            "input": payload.get("input"),
        })
        return events

    if kind == "tool.call.finished":
        return [{
            "type": "tool-output-available",
            "toolCallId": str(payload.get("toolCallId") or ""),
            "output": payload.get("output"),
        }]

    if kind == "text.started":
        return [{"type": "text-start", "id": str(payload.get("partId") or "")}]

    if kind == "text.delta":
        return [{
            "type": "text-delta",
            "id": str(payload.get("partId") or ""),
            "delta": str(payload.get("delta") or ""),
        }]

    if kind == "text.finished":
        return [{"type": "text-end", "id": str(payload.get("partId") or "")}]

    return []
