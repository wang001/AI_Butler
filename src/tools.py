"""
Tool Call 定义 & 执行器

工具列表：
  - search_memory    : 主动检索长期记忆（模型显式发起）
  - get_current_time : 返回当前时间
  - web_search       : 网络搜索（union-search-skill，no_api_key 分组）
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
                "支持 30+ 平台，默认使用无需 API Key 的搜索引擎。"
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
                            "可选，指定搜索平台列表。"
                            "留空默认使用 no_api_key 分组（百度、Bing、DuckDuckGo 等）。"
                            "可选平台示例: baidu_direct, duckduckgo_html, brave_direct, "
                            "github, zhihu, bilibili, youtube"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ── 执行器 ────────────────────────────────────────────────────────────────────

class ToolExecutor:
    """持有 reme 实例，负责实际执行每个工具。"""

    def __init__(self, reme: Any):
        self.reme = reme

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
        elif name == "get_current_time":
            result = self._get_current_time()
        elif name == "web_search":
            result = await self._web_search(**args)
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

        # 默认用 no_api_key 分组，用户可通过参数覆盖
        search_platforms = platforms or PLATFORM_GROUPS.get("no_api_key", [])

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
