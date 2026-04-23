# -*- coding: utf-8 -*-
"""
event — 事件定义与协议适配层

分两类事件：

1. AgentEvent
   系统内部的 canonical 事件，用于描述一轮 agent 执行中的状态变化。

2. StreamEvent
   对外 Web/CLI 暴露的流式事件，当前保持与 AI SDK 风格协议兼容。

长期目标是：
  Agent Runtime / Tool Executor / Background Services 先产出 AgentEvent，
  再由不同 channel 适配为各自的外部协议。
"""

from event.mapper import to_stream_events
from event.stream import StreamEvent
from event.types import AgentEvent, AgentEventKind, EventSource, make_agent_event, new_event_id

__all__ = [
    "AgentEvent",
    "AgentEventKind",
    "EventSource",
    "StreamEvent",
    "make_agent_event",
    "new_event_id",
    "to_stream_events",
]
