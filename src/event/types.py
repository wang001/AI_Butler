# -*- coding: utf-8 -*-
"""
event/types.py — 内部 canonical 事件定义

这一层不直接面向 Web 协议，而是面向系统内部的状态变化：
  - message / step 生命周期
  - reasoning / text 内容块
  - tool call 生命周期
  - session / memory 等副作用通知

外部 channel 可以在此基础上做映射，而不是直接依赖 AgentRunner 的产物。
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal, TypedDict


EventSource = Literal["agent", "tool", "session", "memory"]

AgentEventKind = Literal[
    "message.started",
    "message.finished",
    "message.error",
    "step.started",
    "step.finished",
    "reasoning.started",
    "reasoning.delta",
    "reasoning.finished",
    "tool.call.started",
    "tool.call.arguments",
    "tool.call.finished",
    "text.started",
    "text.delta",
    "text.finished",
    "session.snapshot.updated",
    "memory.update.requested",
]


class AgentEvent(TypedDict):
    eventId: str
    ts: float
    conversationId: str
    messageId: str
    stepId: str
    source: EventSource
    kind: AgentEventKind
    payload: dict[str, Any]


def new_event_id(prefix: str = "evt") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def make_agent_event(
    *,
    kind: AgentEventKind,
    source: EventSource,
    conversation_id: str = "",
    message_id: str = "",
    step_id: str = "",
    payload: dict[str, Any] | None = None,
) -> AgentEvent:
    return {
        "eventId": new_event_id("evt"),
        "ts": time.time(),
        "conversationId": conversation_id,
        "messageId": message_id,
        "stepId": step_id,
        "source": source,
        "kind": kind,
        "payload": payload or {},
    }
