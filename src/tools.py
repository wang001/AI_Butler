"""
Tool Call 定义 & 执行器

工具列表：
  - search_memory    : 主动检索长期记忆（模型显式发起）
  - search_history   : 全文检索原始对话日志（SQLite FTS5）
  - get_current_time : 返回当前时间
  - web_search       : 网络搜索（union-search-skill，no_api_key 分组）
  - run_command      : 在容器环境中执行命令（subprocess）
  - browser_use      : AI 驱动的浏览器操作（基于 browser-use）
"""
import asyncio
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# 将 union-search-skill 的 scripts 目录加入 sys.path，以便 import 其内部模块
_UNION_SEARCH_SCRIPTS = Path(__file__).parent.parent / "vendor" / "union-search-skill" / "scripts"
if str(_UNION_SEARCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_UNION_SEARCH_SCRIPTS))

# ── OpenAI function-calling schema ──────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "在长期记忆库中检索与查询相关的历史信息。"
                "当用户询问过去发生的事、之前的偏好、或你觉得有历史背景需要确认时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于检索记忆的查询语句，用自然语言描述想找的内容",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最多返回条数，默认 5",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": (
                "在完整的原始对话历史日志中按关键词检索。"
                "当用户询问过去某次具体对话的细节、某件具体的事情、"
                "或记忆系统可能遗漏的内容时使用。"
                "按关键词命中数打分排序：命中越多排越前，不要求全部命中。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "搜索关键词，用空格分隔多个词。"
                            "必须提取核心名词/实体，不要传整句话。"
                            "示例：'五一 普吉岛' 而非 '用户五一想去哪里玩'"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最多返回条数，默认 8，最大 20",
                        "default": 8,
                    },
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant"],
                        "description": "可选，只搜索指定角色的消息",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "返回当前的日期和时间（北京时间）。用户询问时间、日期、星期时调用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "搜索互联网获取实时信息。"
                "用于查询新闻、天气、价格、最新资讯等训练数据截止后的内容。"
                "可通过 platforms 参数指定搜索平台，不同场景应选择合适的平台组合。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题",
                    },
                    "platforms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "指定搜索平台列表，留空则使用默认通用组合。"
                            "请根据查询意图选择合适的平台：\n"
                            "【通用搜索引擎】baidu_direct(百度), bing_int_direct(必应国际), brave_direct(Brave)\n"
                            "【AI 驱动搜索】startpage_direct(隐私友好,Google代理), "
                            "ecosia_direct(环保搜索引擎), mojeek(独立索引,无追踪)\n"
                            "【问答/知识】duckduckgo_instant(即时答案,适合事实性问题), "
                            "wolfram_direct(数学/科学/数据计算)\n"
                            "【中文资讯】toutiao_direct(今日头条,新闻/热点), "
                            "sogou_direct(搜狗,微信公众号内容), so360_direct(360搜索)\n"
                            "【示例】旅游攻略→baidu_direct,sogou_direct; "
                            "国际新闻→bing_int_direct,brave_direct; "
                            "科学计算→wolfram_direct; "
                            "热点事件→toutiao_direct,baidu_direct"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "在容器环境中执行 shell 命令。"
                "用于文件操作、数据处理、代码运行、系统信息查询等。"
                "工作目录为 /workspace，可读写。"
                "容器本身提供安全隔离，命令不会影响宿主机。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 shell 命令，如 'ls -la' 或 'python3 script.py'",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "超时秒数，默认 30，最大 120",
                        "default": 30,
                    },
                    "workdir": {
                        "type": "string",
                        "description": "执行命令的工作目录，默认 /workspace",
                        "default": "/workspace",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_use",
            "description": (
                "使用 AI 驱动的浏览器自动化执行网页操作任务。"
                "只需用自然语言描述要做的事，浏览器 Agent 会自主完成：\n"
                "  - 打开网页、点击按钮、填写表单\n"
                "  - 提取页面内容、截图\n"
                "  - 多步骤复杂操作（如搜索比价、填写申请等）\n"
                "适合需要真实浏览器交互的场景，比 web_search 更强大但更慢。\n"
                "典型用法：'打开 example.com 并提取主要内容'、'在淘宝搜索 XX 并比较价格'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "自然语言描述的浏览器操作任务",
                    },
                },
                "required": ["task"],
            },
        },
    },
]

# ── 工具并发安全标记 ──────────────────────────────────────────────────────────
# True  = 只读/无副作用，可与其他并发安全工具并行执行
# False = 有写操作或共享有状态资源，必须独占执行
TOOL_CONCURRENT_SAFE: dict[str, bool] = {
    "search_memory":   True,   # 只读：向量检索
    "search_history":  True,   # 只读：SQLite FTS 查询
    "get_current_time": True,  # 纯计算，无 IO
    "web_search":      True,   # 只读：外部搜索引擎
    "run_command":     False,  # 可能写文件、修改系统状态
    "browser_use":     False,  # 共享浏览器实例，有状态操作
}

# ── 执行器 ────────────────────────────────────────────────────────────────────

class ToolExecutor:
    """持有 reme 实例及外部工具实例，负责实际执行每个工具。"""

    def __init__(
        self,
        reme: Any,
        history: Any = None,
        sandbox: Any = None,       # SandboxExecutor 实例
        browser_agent: Any = None, # BrowserAgent 实例
    ):
        self.reme = reme
        self.history = history   # ChatHistory 实例，可为 None（向后兼容）
        self.sandbox = sandbox   # CommandExecutor 实例
        self.browser_agent = browser_agent  # browser-use Agent

    # 工具结果最大 token 数，超出截断（参考 Claude Code contentReplacementState）
    _RESULT_TOKEN_LIMIT = 800

    @staticmethod
    def _truncate_result(result: str, limit: int = _RESULT_TOKEN_LIMIT) -> str:
        """
        超过 limit tokens 时截断结果，附说明。
        粗估：中文约 1 字/token，英文约 4 字符/token，取保守的 3 字符/token。
        """
        char_limit = limit * 3
        if len(result) <= char_limit:
            return result
        truncated = result[:char_limit]
        return (
            truncated
            + f"\n\n…（结果过长已截断，原始长度 {len(result)} 字符，"
            f"仅保留约 {limit} tokens。如需完整内容请缩小搜索范围或分步查询。）"
        )

    async def run(self, name: str, arguments: str) -> str:
        """
        执行工具，返回字符串结果（将作为 tool_result 回传给 LLM）。
        arguments 是 JSON 字符串。结果超过 token 限制时自动截断。
        """
        try:
            args: dict = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}

        if name == "search_memory":
            result = await self._search_memory(**args)
        elif name == "search_history":
            result = self._search_history(**args)
        elif name == "get_current_time":
            result = self._get_current_time()
        elif name == "web_search":
            result = await self._web_search(**args)
        elif name == "run_command":
            result = await self._run_command(**args)
        elif name == "browser_use":
            result = await self._browser_use(**args)
        else:
            result = f"[未知工具: {name}]"

        return self._truncate_result(result)

    # ── 工具实现 ──────────────────────────────────────────────────────────────

    async def _search_memory(self, query: str, max_results: int = 5) -> str:
        try:
            result = await self.reme.memory_search(query, max_results=max_results)
            if result is None:
                return "未找到相关记忆。"
            # ReMe v0.3+ 返回 ToolResponse，content 是 list[dict|TextBlock]
            content = getattr(result, "content", None)
            if not content:
                return "未找到相关记忆。"
            texts = []
            for block in content:
                text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                if text:
                    texts.append(text)
            if not texts:
                return "未找到相关记忆。"
            return "检索到以下历史记忆：\n" + "\n\n".join(texts)
        except Exception as e:
            return f"[记忆检索失败: {e}]"

    def _search_history(
        self,
        query: str,
        limit: int = 8,
        role: str | None = None,
    ) -> str:
        """全文检索原始对话日志（SQLite FTS5）。"""
        if self.history is None:
            return "[search_history 不可用] 历史日志系统未初始化。"
        limit = min(limit, 20)
        try:
            results = self.history.search(query, limit=limit, role=role)
        except Exception as e:
            return f"[search_history 失败] {e}"

        if not results:
            return f"在对话历史中未找到与「{query}」相关的内容。"

        from datetime import datetime, timezone, timedelta
        cst = timezone(timedelta(hours=8))
        lines = [f"在对话历史中找到 {len(results)} 条与「{query}」相关的记录：\n"]
        for i, r in enumerate(results, 1):
            dt = datetime.fromtimestamp(r["ts"], tz=cst).strftime("%Y-%m-%d %H:%M")
            role_label = {"user": "你", "assistant": "Butler", "tool": "工具"}.get(r["role"], r["role"])
            content = r["content"]
            # 内容过长时截取前 300 字
            if len(content) > 300:
                content = content[:300] + "…"
            score_tag = f" (匹配度:{r['score']:.0%})" if "score" in r else ""
            lines.append(f"{i}. [{dt}]{score_tag} {role_label}：{content}")
        return "\n".join(lines)

    def _get_current_time(self) -> str:
        cst = timezone(timedelta(hours=8))
        now = datetime.now(cst)
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        wd = weekdays[now.weekday()]
        return now.strftime(f"%Y年%m月%d日 {wd} %H:%M:%S（北京时间）")

    async def _web_search(self, query: str, platforms: list[str] | None = None) -> str:
        """
        调用 union-search-skill 执行多平台聚合搜索。
        默认使用 no_api_key 分组（无需任何 API Key）。
        union_search 是同步函数，用 asyncio.to_thread 包一层避免阻塞事件循环。
        """
        try:
            from union_search import union_search, PLATFORM_GROUPS
        except ImportError as e:
            return f"[web_search 不可用] union-search-skill 未正确安装: {e}"

        # 默认通用搜索引擎（3 个足够覆盖中英文），模型可通过 platforms 指定垂类平台
        _DEFAULT_PLATFORMS = ["baidu_direct", "bing_int_direct", "brave_direct"]
        search_platforms = platforms or _DEFAULT_PLATFORMS

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    union_search,
                    keyword=query,
                    platforms=search_platforms,
                    limit=5,          # 每个平台最多 5 条
                    max_workers=6,    # 并发数
                    timeout=15,       # 单平台超时
                    deduplicate=True,
                ),
                timeout=20,           # 整体最多等 20 秒
            )
        except asyncio.TimeoutError:
            return f"[web_search 超时] 搜索「{query}」超过 20 秒，请稍后重试。"
        except Exception as e:
            return f"[web_search 失败] {e}"

        # 格式化结果给 LLM
        items: list[dict] = result.get("final_items", [])
        if not items:
            # final_items 为空时从各平台 results 聚合
            for platform_result in result.get("results", {}).values():
                items.extend(platform_result.get("items", []))

        if not items:
            return f"搜索「{query}」未找到结果。"

        lines = [f"搜索「{query}」的结果："]
        for i, item in enumerate(items[:15], 1):
            title = item.get("title") or item.get("name") or ""
            url   = item.get("url") or item.get("link") or item.get("href") or ""
            snippet = item.get("snippet") or item.get("description") or item.get("content", "")[:200]
            source = item.get("source") or item.get("platform") or ""
            line = f"{i}. [{source}] {title}"
            if url:
                line += f"\n   {url}"
            if snippet:
                line += f"\n   {snippet.strip()[:150]}"
            lines.append(line)

        summary = result.get("summary", {})
        lines.append(
            f"\n（共检索 {summary.get('total_platforms', 0)} 个平台，"
            f"获得 {summary.get('total_items', 0)} 条原始结果，"
            f"去重后 {summary.get('deduplicated_total_items', len(items))} 条）"
        )
        return "\n".join(lines)

    async def _run_command(
        self,
        command: str,
        timeout: int = 30,
        workdir: str | None = None,
    ) -> str:
        """在容器环境中执行命令。"""
        if self.sandbox is None:
            return "[run_command 不可用] 命令执行器未初始化。"

        result = await self.sandbox.run(
            command=command,
            timeout=timeout,
            workdir=workdir,
        )

        if result.get("error"):
            return f"[命令执行失败] {result['error']}"

        parts = []
        if result.get("timed_out"):
            parts.append(f"[超时] 命令执行超过 {timeout} 秒限制")

        exit_code = result.get("exit_code", -1)
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        if stdout:
            parts.append(f"[stdout]\n{stdout}")
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        if not stdout and not stderr:
            parts.append("（命令无输出）")

        parts.append(f"[退出码] {exit_code}")
        return "\n".join(parts)

    async def _browser_use(self, task: str) -> str:
        """使用 browser-use 执行浏览器操作任务。"""
        if self.browser_agent is None:
            return "[browser_use 不可用] 浏览器 Agent 未初始化。"

        result = await self.browser_agent.run_task(task=task)

        if not result.get("success"):
            error = result.get("error", "未知错误")
            return f"[浏览器操作失败] {error}"

        parts = []
        text_result = result.get("result", "")
        if text_result:
            parts.append(text_result)

        steps = result.get("steps", 0)
        if steps:
            parts.append(f"\n（共执行 {steps} 个操作步骤）")

        return "\n".join(parts) if parts else "浏览器任务已完成。"
