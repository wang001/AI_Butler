"""
tools — AI Butler 工具层

各模块职责：
  - dispatcher.py  : Tool Call 调度器（按名称分发 + 结果溢出写文件）
  - memory.py      : 记忆工具（search_memory / search_history）
  - search.py      : 网络搜索（web_search）
  - file_reader.py : 纯文本文件读取（read_file）
  - command.py     : Docker 容器内 shell 命令执行（run_command）
  - browser.py     : 浏览器自动化（browser_use）

可用工具列表（dispatcher.tools）在运行时根据 command_executor / browser_agent
是否为 None 动态构建，COMMAND_ENABLED=false 或 BROWSER_ENABLED=false 时
对应工具不会出现在发给模型的 schema 中。
"""
from tools.dispatcher import TOOL_CONCURRENT_SAFE, ToolDispatcher

__all__ = ["TOOL_CONCURRENT_SAFE", "ToolDispatcher"]
