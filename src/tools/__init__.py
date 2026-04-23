"""
tools — AI Butler 工具层

各模块职责：
  - base.py        : Tool 基类
  - registry.py    : ToolRegistry（工具注册与发现）
  - executor.py    : ToolExecutor（参数校验 / 执行 / spill）
  - dispatcher.py  : 兼容 facade（保留旧入口）
  - memory_tools/  : 记忆工具目录（search_memory / search_history / update_memory）
  - web_search_tool/: 网络搜索工具目录（web_search）
  - web_fetcher_tool.py : 网页正文抓取（web_fetcher）
  - read_file_tool.py   : 纯文本文件读取（read_file）
  - run_command_tool.py : shell 命令执行（run_command）
  - browser_use_tool.py : 浏览器自动化（browser_use）
  - current_time_tool.py: 当前时间工具

可用工具列表（dispatcher.tools）仍在运行时动态构建：
  - command_executor 为 None 时不注册 run_command
  - browser_agent    为 None 时不注册 browser_use
"""
from tools.base import Tool
from tools.dispatcher import ToolDispatcher
from tools.registry import ToolRegistry

__all__ = ["Tool", "ToolDispatcher", "ToolRegistry"]
