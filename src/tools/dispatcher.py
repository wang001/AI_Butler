"""
ToolDispatcher — 兼容 facade

当前工具系统已拆为三层：

  - Tool          : 工具基类（name / description / parameters / execute）
  - ToolRegistry  : 工具注册与发现
  - ToolExecutor  : 参数校验、执行、结果 spill

保留 ToolDispatcher 这个入口，是为了不打断现有上层调用：
  - dispatcher.tools
  - await dispatcher.run(name, arguments)
  - dispatcher.concurrent_safe(name)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.executor import ToolExecutor
from tools.registry import ToolRegistry, build_default_tool_registry


class ToolDispatcher:
    """
    Tool Call 调度 facade。

    对外继续暴露旧接口，但内部真正职责已下放到：
      - ToolRegistry
      - ToolExecutor
    """

    def __init__(
        self,
        reme: Any,
        history: Any = None,
        command_executor: Any = None,
        browser_agent: Any = None,
        tool_call_dir: str | None = None,
        memory_update_service: Any = None,
        memory_tools: list[dict] | None = None,
    ):
        self.command_executor = command_executor
        self.browser_agent = browser_agent
        self._tool_call_dir = Path(tool_call_dir) if tool_call_dir else Path.cwd()

        self._registry, self.memory = build_default_tool_registry(
            reme=reme,
            history=history,
            command_executor=command_executor,
            browser_agent=browser_agent,
            memory_update_service=memory_update_service,
            memory_tools=memory_tools,
        )
        self._executor = ToolExecutor(
            registry=self._registry,
            tool_call_dir=str(self._tool_call_dir),
        )

    @property
    def tools(self) -> list[dict]:
        return self._registry.tools

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def concurrent_safe(self, name: str) -> bool:
        return self._registry.is_concurrent_safe(name)

    async def run(self, name: str, arguments: str) -> str:
        return await self._executor.run(name, arguments)
