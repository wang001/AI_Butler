# -*- coding: utf-8 -*-
"""
event/stream.py — 对外 Web 流式事件定义

这一层保持当前前端已经消费的 chunk 协议稳定：
  - start / finish
  - start-step / finish-step
  - text-start/delta/end
  - reasoning-start/delta/end
  - tool-input-* / tool-output-available
  - error

它是 channel-facing 协议，不是系统内部的 canonical 事件模型。
"""
from __future__ import annotations

from typing import Any, Literal, TypeAlias, TypedDict


class StartEvent(TypedDict):
    type: Literal["start"]
    messageId: str


class StartStepEvent(TypedDict):
    type: Literal["start-step"]


class FinishStepEvent(TypedDict):
    type: Literal["finish-step"]
    finishReason: str


class FinishEvent(TypedDict):
    type: Literal["finish"]
    finishReason: str


class ErrorEvent(TypedDict):
    type: Literal["error"]
    errorText: str


class TextStartEvent(TypedDict):
    type: Literal["text-start"]
    id: str


class TextDeltaEvent(TypedDict):
    type: Literal["text-delta"]
    id: str
    delta: str


class TextEndEvent(TypedDict):
    type: Literal["text-end"]
    id: str


class ReasoningStartEvent(TypedDict):
    type: Literal["reasoning-start"]
    id: str


class ReasoningDeltaEvent(TypedDict):
    type: Literal["reasoning-delta"]
    id: str
    delta: str


class ReasoningEndEvent(TypedDict):
    type: Literal["reasoning-end"]
    id: str


class ToolInputStartEvent(TypedDict):
    type: Literal["tool-input-start"]
    toolCallId: str
    toolName: str


class ToolInputDeltaEvent(TypedDict):
    type: Literal["tool-input-delta"]
    toolCallId: str
    inputTextDelta: str


class ToolInputAvailableEvent(TypedDict):
    type: Literal["tool-input-available"]
    toolCallId: str
    toolName: str
    input: Any


class ToolOutputAvailableEvent(TypedDict):
    type: Literal["tool-output-available"]
    toolCallId: str
    output: Any


StreamEvent: TypeAlias = (
    StartEvent
    | StartStepEvent
    | FinishStepEvent
    | FinishEvent
    | ErrorEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ReasoningStartEvent
    | ReasoningDeltaEvent
    | ReasoningEndEvent
    | ToolInputStartEvent
    | ToolInputDeltaEvent
    | ToolInputAvailableEvent
    | ToolOutputAvailableEvent
)
