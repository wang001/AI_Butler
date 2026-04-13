# -*- coding: utf-8 -*-
"""
agent/hooks.py — 生命周期钩子抽象

参考 NanoBot 的 AgentHook / CompositeHook 设计，提供可插拔的
Agent 事件回调机制。Channel 层（CLI spinner、Web 推流、日志审计等）
通过实现 AgentHook 子类接入，而非直接修改 Agent 核心代码。

用法：
    class MyHook(AgentHook):
        async def on_tool_start(self, name, args):
            print(f"[工具调用] {name}")

    hooks = CompositeHook([MyHook(), AnotherHook()])
    butler = await Butler.create(cfg, hooks=hooks)
"""
from __future__ import annotations

from abc import ABC


class AgentHook(ABC):
    """
    Agent 生命周期钩子基类。

    所有方法均为空实现（opt-in 模式），子类按需覆写。
    """

    async def on_llm_start(self, messages: list[dict]) -> None:
        """LLM 请求即将发出。"""

    async def on_llm_end(self, response) -> None:
        """LLM 响应已收到（非流式场景）。"""

    async def on_tool_start(self, name: str, args: dict) -> None:
        """工具即将被调用。"""

    async def on_tool_end(self, name: str, result: str) -> None:
        """工具执行完毕。"""

    async def on_stream_token(self, token: str) -> None:
        """流式模式下收到一个 token。"""

    async def on_error(self, error: Exception) -> None:
        """执行过程中出现异常。"""


class CompositeHook(AgentHook):
    """
    组合钩子：将多个 AgentHook 按注册顺序依次执行。

    任何单个 hook 抛异常不会中断其他 hook 的执行（静默吞掉异常）。
    """

    def __init__(self, hooks: list[AgentHook] | None = None):
        self._hooks: list[AgentHook] = list(hooks) if hooks else []

    def add(self, hook: AgentHook) -> None:
        self._hooks.append(hook)

    async def on_llm_start(self, messages: list[dict]) -> None:
        for h in self._hooks:
            try:
                await h.on_llm_start(messages)
            except Exception:
                pass

    async def on_llm_end(self, response) -> None:
        for h in self._hooks:
            try:
                await h.on_llm_end(response)
            except Exception:
                pass

    async def on_tool_start(self, name: str, args: dict) -> None:
        for h in self._hooks:
            try:
                await h.on_tool_start(name, args)
            except Exception:
                pass

    async def on_tool_end(self, name: str, result: str) -> None:
        for h in self._hooks:
            try:
                await h.on_tool_end(name, result)
            except Exception:
                pass

    async def on_stream_token(self, token: str) -> None:
        for h in self._hooks:
            try:
                await h.on_stream_token(token)
            except Exception:
                pass

    async def on_error(self, error: Exception) -> None:
        for h in self._hooks:
            try:
                await h.on_error(error)
            except Exception:
                pass
