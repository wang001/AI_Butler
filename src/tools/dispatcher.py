"""
Tool Call 调度器

负责两件事：
  1. 根据运行时配置动态拼装当前可用工具的 schema 列表（dispatcher.tools）
  2. 接收模型发出的 tool call，按名称分发给对应的工具实现

工具实现分布：
  - memory.py      : search_memory, search_history（记忆 / 历史检索）
  - search.py      : web_search（搜索结果页检索）
  - web_fetcher.py : web_fetcher（已知 URL 的正文抓取）
  - file_reader.py : read_file（纯文本文件读取）
  - command.py     : run_command（Docker 容器内 shell 命令执行，COMMAND_ENABLED 控制）
  - browser.py     : browser_use（AI 驱动的浏览器自动化，BROWSER_ENABLED 控制）
  - 本文件内联     : get_current_time（纯计算，无需独立模块）

并发安全策略见 TOOL_CONCURRENT_SAFE：
  True  = 只读/无副作用，可与其他安全工具并行执行
  False = 有写操作或共享有状态资源，必须独占串行执行

结果溢出策略（_spill_to_file）：
  工具返回结果超过字符上限时，将完整内容写入 tool_call_dir/<工具名>_<时间戳>.txt，
  并返回文件路径提示，让模型通过 read_file 工具按需读取，避免截断导致信息丢失。
"""
import inspect
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable

from tools.memory import MEMORY_TOOLS, MemoryTools
from tools.search import SEARCH_TOOLS, web_search
from tools.file_reader import FILE_READER_TOOLS, read_file as _read_file
from tools.web_fetcher import WEB_FETCHER_TOOLS, web_fetcher
from tools.command import COMMAND_TOOLS
from tools.browser import BROWSER_TOOLS

# ── get_current_time schema（纯计算内联，保留在本文件）────────────────────────

_TIME_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "返回当前的日期和时间。用户询问时间、日期、星期时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "tz_offset": {
                        "type": "number",
                        "description": (
                            "UTC 偏移小时数，默认 8（北京时间）。"
                            "例如：东京 9、伦敦 0、纽约 -5、洛杉矶 -8。"
                            "支持半小时时区，如印度 5.5、尼泊尔 5.75。"
                        ),
                        "default": 8,
                    },
                },
                "required": [],
            },
        },
    },
]

# ── 并发安全标记 ───────────────────────────────────────────────────────────────
# True  = 只读/无副作用，可与其他并发安全工具并行执行
# False = 有写操作或共享有状态资源，必须独占串行执行
TOOL_CONCURRENT_SAFE: dict[str, bool] = {
    "search_memory":    True,   # 只读：向量检索
    "search_history":   True,   # 只读：SQLite FTS 查询
    "update_memory":    False,  # 写 MEMORY.md，必须串行独占
    "get_current_time": True,   # 纯计算，无 IO
    "web_search":       True,   # 只读：外部搜索引擎
    "web_fetcher":      True,   # 只读：网页正文抓取
    "read_file":        True,   # 只读：本地文件读取
    "run_command":      False,  # 可能写文件、修改系统状态
    "browser_use":      False,  # 共享浏览器实例，有状态操作
}

# ── 调度器 ─────────────────────────────────────────────────────────────────────

class ToolDispatcher:
    """
    Tool Call 调度器。

    持有各工具实例，接收模型发出的 tool call 名称 + 参数，
    分发给对应的工具实现并返回字符串结果。

    tools 属性根据实际启用的工具动态构建：
      - command_executor 为 None 时不注册 run_command
      - browser_agent    为 None 时不注册 browser_use
    这样模型只会看到当前真正可用的工具。

    结果溢出策略：
      结果字符数超过 _RESULT_CHAR_LIMIT 时，完整内容写入
      <tool_call_dir>/<tool>_<timestamp>.txt，
      返回给模型的是文件路径提示，模型可用 read_file 按需读取。
    """

    # 工具结果直接内联的字符上限（约 800 tokens，保守估算 3 字符/token）
    _RESULT_CHAR_LIMIT = 800 * 3

    def __init__(
        self,
        reme: Any,
        history: Any = None,
        command_executor: Any = None,  # tools.command.CommandExecutor 实例，None 表示禁用
        browser_agent: Any = None,     # tools.browser.BrowserAgent 实例，None 表示禁用
        tool_call_dir: str | None = None,
        memory_update_service: Any = None,
        memory_tools: list[dict] | None = None,
    ):
        self.memory = MemoryTools(
            reme=reme,
            history=history,
            memory_update_service=memory_update_service,
        )
        self.command_executor = command_executor
        self.browser_agent = browser_agent
        self._tool_call_dir = Path(tool_call_dir) if tool_call_dir else Path.cwd()

        # 根据实际启用的工具动态构建 schema 列表
        self.tools: list[dict] = (
            (memory_tools if memory_tools is not None else MEMORY_TOOLS)
            + SEARCH_TOOLS
            + WEB_FETCHER_TOOLS
            + FILE_READER_TOOLS
            + _TIME_TOOLS
            + (COMMAND_TOOLS if command_executor is not None else [])
            + (BROWSER_TOOLS if browser_agent is not None else [])
        )

    def _spill_to_file(self, tool_name: str, content: str) -> str:
        """
        将过长的工具结果写入临时文件，返回提示字符串。

        文件保存在 <tool_call_dir>/<tool>_<timestamp>.txt，
        模型可通过 run_command 读取（如 cat、head、grep 等）。
        """
        out_dir = self._tool_call_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{tool_name}_{ts}.txt"
        file_path = out_dir / filename
        file_path.write_text(content, encoding="utf-8")

        return (
            f"[结果过长，已写入文件]\n"
            f"路径：{file_path}\n"
            f"共 {len(content)} 字符。\n"
            f"请使用 read_file 工具读取所需内容，例如：\n"
            f"  read_file(path='{file_path}')                        # 读取全文\n"
            f"  read_file(path='{file_path}', start_line=1, end_line=50)  # 读取前 50 行\n"
            f"  read_file(path='{file_path}', pattern='关键词')      # 按关键词过滤"
        )

    def _handle_result(self, tool_name: str, result: str) -> str:
        """结果在限制内直接返回，超限则溢出到文件。"""
        if len(result) <= self._RESULT_CHAR_LIMIT:
            return result
        return self._spill_to_file(tool_name, result)

    def _describe_tool_signature(
        self,
        handler: Callable[..., Any],
    ) -> tuple[list[str], list[str], bool]:
        """
        返回 (allowed_names, required_names, accepts_var_keyword)。

        用于在真正调用工具前给出可恢复的参数错误，而不是直接抛 TypeError 中断整轮 agent。
        """
        signature = inspect.signature(handler)
        allowed_names: list[str] = []
        required_names: list[str] = []
        accepts_var_keyword = False

        for name, param in signature.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_var_keyword = True
                continue
            if param.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            allowed_names.append(name)
            if param.default is inspect.Signature.empty:
                required_names.append(name)

        return allowed_names, required_names, accepts_var_keyword

    def _format_tool_arg_error(
        self,
        tool_name: str,
        message: str,
        allowed_names: list[str],
        required_names: list[str],
    ) -> str:
        parts = [f"[{tool_name} 参数错误] {message}"]
        if allowed_names:
            parts.append("允许参数：" + ", ".join(allowed_names))
        if required_names:
            parts.append("必填参数：" + ", ".join(required_names))
        parts.append("请根据工具 schema 修正参数后重试。")
        return "\n".join(parts)

    async def _call_tool(
        self,
        tool_name: str,
        handler: Callable[..., Any],
        args: dict[str, Any],
        allow_spill: bool = True,
    ) -> str:
        """
        通用工具调用包装：
          - 预校验参数名 / 必填参数
          - 捕获 TypeError 等可恢复错误并转成 tool_result 文本
          - 避免单个工具调用直接炸掉整轮推理
        """
        allowed_names, required_names, accepts_var_keyword = (
            self._describe_tool_signature(handler)
        )

        if not accepts_var_keyword:
            unexpected = sorted(set(args) - set(allowed_names))
            if unexpected:
                return self._format_tool_arg_error(
                    tool_name=tool_name,
                    message="不支持参数: " + ", ".join(unexpected),
                    allowed_names=allowed_names,
                    required_names=required_names,
                )

        missing = [name for name in required_names if name not in args]
        if missing:
            return self._format_tool_arg_error(
                tool_name=tool_name,
                message="缺少必填参数: " + ", ".join(missing),
                allowed_names=allowed_names,
                required_names=required_names,
            )

        try:
            result = handler(**args)
            if inspect.isawaitable(result):
                result = await result
        except TypeError as exc:
            return self._format_tool_arg_error(
                tool_name=tool_name,
                message=str(exc),
                allowed_names=allowed_names,
                required_names=required_names,
            )
        except Exception as exc:
            return f"[{tool_name} 失败] {exc}"

        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, indent=2)

        if not allow_spill:
            return result
        return self._handle_result(tool_name, result)

    async def run(self, name: str, arguments: str) -> str:
        """
        按工具名称分发执行，返回字符串结果（回传给 LLM 作为 tool_result）。
        arguments 是 JSON 字符串。
        """
        try:
            args: dict = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}

        if name == "search_memory":
            return await self._call_tool(name, self.memory.search_memory, args)
        elif name == "search_history":
            return await self._call_tool(name, self.memory.search_history, args)
        elif name == "update_memory":
            return await self._call_tool(name, self.memory.update_memory, args)
        elif name == "get_current_time":
            return await self._call_tool(name, self._get_current_time, args)
        elif name == "web_search":
            return await self._call_tool(name, web_search, args)
        elif name == "web_fetcher":
            return await self._call_tool(name, web_fetcher, args)
        elif name == "read_file":
            # 同步调用；自身有 _MAX_LINES 行数兜底，且结果不能再写入文件（会死循环），
            # 必须直接 return，绕过 _handle_result / _spill_to_file。
            return await self._call_tool(name, _read_file, args, allow_spill=False)
        elif name == "run_command":
            return await self._call_tool(name, self._run_command, args)
        elif name == "browser_use":
            return await self._call_tool(name, self._browser_use, args)
        else:
            return f"[未知工具: {name}]"

    # ── 内联工具实现（仅 get_current_time，其余见各自模块） ────────────────────

    def _get_current_time(self, tz_offset: float = 8) -> str:
        """返回指定时区的当前时间。tz_offset 为 UTC 偏移小时数，默认 8（北京时间）。"""
        tz = timezone(timedelta(hours=tz_offset))
        now = datetime.now(tz)
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        wd = weekdays[now.weekday()]
        # 生成时区标签，如 UTC+8、UTC-5、UTC+5:30
        offset_mins = int(tz_offset * 60)
        sign = "+" if offset_mins >= 0 else "-"
        h, m = divmod(abs(offset_mins), 60)
        tz_label = f"UTC{sign}{h}" if m == 0 else f"UTC{sign}{h}:{m:02d}"
        return now.strftime(f"%Y年%m月%d日 {wd} %H:%M:%S（{tz_label}）")

    async def _run_command(
        self,
        command: str,
        timeout: int = 30,
        workdir: str | None = None,
    ) -> str:
        """委托给 tools.command.CommandExecutor 执行。"""
        if self.command_executor is None:
            return "[run_command 不可用] 命令执行器未初始化。"

        result = await self.command_executor.run(
            command=command,
            timeout=timeout,
            workdir=workdir,
        )

        if result.get("error"):
            return f"[命令执行失败] {result['error']}"

        parts = []
        if result.get("timed_out"):
            parts.append(f"[超时] 命令执行超过 {timeout} 秒限制")

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        if stdout:
            parts.append(f"[stdout]\n{stdout}")
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        if not stdout and not stderr:
            parts.append("（命令无输出）")

        parts.append(f"[退出码] {result.get('exit_code', -1)}")
        return "\n".join(parts)

    async def _browser_use(self, task: str) -> str:
        """委托给 tools.browser.BrowserAgent 执行。"""
        if self.browser_agent is None:
            return "[browser_use 不可用] 浏览器 Agent 未初始化。"

        result = await self.browser_agent.run_task(task=task)

        if not result.get("success"):
            return f"[浏览器操作失败] {result.get('error', '未知错误')}"

        parts = []
        if result.get("result"):
            parts.append(result["result"])
        if result.get("steps"):
            parts.append(f"\n（共执行 {result['steps']} 个操作步骤）")

        return "\n".join(parts) if parts else "浏览器任务已完成。"
