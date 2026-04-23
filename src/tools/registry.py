"""
tools/registry.py — 工具注册表

负责：
  - 按名称注册 Tool
  - 动态暴露当前可用 schema
  - 提供并发安全元信息

不负责：
  - 参数校验
  - 实际执行
  - 结果 spill
"""
from __future__ import annotations

from typing import Any, Iterable

from tools.base import Tool
from tools.memory_tools import MEMORY_TOOLS, MemoryTools


def _schema_name(schema: dict[str, Any]) -> str:
    return str(((schema.get("function") or {}).get("name")) or "")


def _schema_by_name(
    schemas: Iterable[dict[str, Any]],
    name: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    for schema in schemas:
        if _schema_name(schema) == name:
            return schema
    return fallback

class ToolRegistry:
    """按名称持有 Tool 实例，并暴露当前启用的工具集合。"""

    def __init__(self, tools: Iterable[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def require(self, name: str) -> Tool:
        tool = self.get(name)
        if tool is None:
            raise KeyError(name)
        return tool

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    @property
    def concurrent_safe_map(self) -> dict[str, bool]:
        return {
            name: tool.concurrency_safe for name, tool in self._tools.items()
        }

    def is_concurrent_safe(self, name: str) -> bool:
        tool = self.get(name)
        return bool(tool.concurrency_safe) if tool else False


def build_default_tool_registry(
    *,
    reme: Any,
    history: Any = None,
    command_executor: Any = None,
    browser_agent: Any = None,
    memory_update_service: Any = None,
    memory_tools: list[dict[str, Any]] | None = None,
) -> tuple[ToolRegistry, MemoryTools]:
    """
    根据当前 runtime 可用能力动态构建默认 ToolRegistry。

    返回 (registry, memory_tools_impl)：
      - registry          供 ToolDispatcher / AgentRunner 使用
      - memory_tools_impl 供上层保留 memory/search_history/update_memory 能力实例
    """
    memory_impl = MemoryTools(
        reme=reme,
        history=history,
        memory_update_service=memory_update_service,
    )
    effective_memory_schemas = memory_tools if memory_tools is not None else MEMORY_TOOLS

    tools: list[Tool] = [
    ]

    from tools.browser_use_tool import BrowserUseTool
    from tools.current_time_tool import CurrentTimeTool
    from tools.memory_tools import (
        SearchHistoryTool,
        SearchMemoryTool,
        UpdateMemoryTool,
    )
    from tools.read_file_tool import ReadFileTool
    from tools.run_command_tool import RunCommandTool
    from tools.web_fetcher_tool import WebFetcherTool
    from tools.web_search_tool import WebSearchTool

    visible_memory_names = {
        _schema_name(schema) for schema in effective_memory_schemas
    } if memory_tools is not None else {"search_memory", "search_history", "update_memory"}

    if "search_memory" in visible_memory_names:
        tools.append(SearchMemoryTool(memory_impl))
    if "search_history" in visible_memory_names:
        tools.append(SearchHistoryTool(memory_impl))
    if "update_memory" in visible_memory_names:
        tools.append(UpdateMemoryTool(memory_impl))

    tools.extend([
        CurrentTimeTool(),
        WebSearchTool(),
        WebFetcherTool(),
        ReadFileTool(),
    ])

    if command_executor is not None:
        tools.append(RunCommandTool(command_executor))

    if browser_agent is not None:
        tools.append(BrowserUseTool(browser_agent))

    return ToolRegistry(tools), memory_impl
