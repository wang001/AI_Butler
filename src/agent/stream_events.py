# -*- coding: utf-8 -*-
"""
agent/stream_events.py — 流式事件定义

Web 事件协议尽量贴近 AI SDK UI message stream：
  - start / finish                消息级生命周期
  - start-step / finish-step      单次 LLM step 生命周期
  - text-start/delta/end          最终展示文本
  - reasoning-start/delta/end     可折叠思考内容
  - tool-input-*                  工具输入生成/可用
  - tool-output-available         工具输出可用
  - error                         流式错误

这样前端既可以做自定义渲染，也更容易向现成开源实现靠拢。
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
