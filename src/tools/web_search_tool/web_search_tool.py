"""
网络搜索工具（Web Search）

直接使用本地的百度 / Bing 抓取引擎，不再依赖 union-search-skill。
"""
from __future__ import annotations

import asyncio

from tools.base import Tool
from tools.web_search_tool.search_engine import choose_engine, run_search

# ── OpenAI function-calling schema ────────────────────────────────────────────

SEARCH_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "搜索互联网获取实时信息。"
                "用于查询新闻、天气、价格、最新资讯等训练数据截止后的内容。"
                "支持百度和 Bing 国际版。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题",
                    },
                    "engine": {
                        "type": "string",
                        "enum": ["auto", "baidu", "bing"],
                        "default": "auto",
                        "description": (
                            "搜索引擎。auto 会按查询语言自动选择：中文偏向 baidu，"
                            "英文/国际内容偏向 bing。"
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                        "description": "返回结果条数，默认 5，最多 10。",
                    },
                    "proxy": {
                        "type": "string",
                        "description": "可选代理地址，例如 http://127.0.0.1:7890",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

async def web_search(
    query: str,
    engine: str = "auto",
    max_results: int = 5,
    proxy: str | None = None,
) -> str:
    """执行网页搜索，返回格式化结果字符串。"""
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(
                run_search,
                engine,
                query,
                max_results,
                proxy,
            ),
            timeout=20,
        )
    except asyncio.TimeoutError:
        return f"[web_search 超时] 搜索「{query}」超过 20 秒，请稍后重试。"
    except Exception as exc:
        return f"[web_search 失败] {exc}"

    if not results:
        return f"搜索「{query}」未找到结果。"

    actual_engine = choose_engine(query) if engine == "auto" else engine
    lines = [f"搜索「{query}」的结果（引擎：{actual_engine}）："]
    for idx, item in enumerate(results, 1):
        title = item.get("title") or ""
        url = item.get("href") or ""
        snippet = item.get("body") or ""
        source = item.get("source") or actual_engine
        line = f"{idx}. [{source}] {title}"
        if url:
            line += f"\n   {url}"
        if snippet:
            line += f"\n   {snippet.strip()[:150]}"
        lines.append(line)

    lines.append(f"\n（共返回 {len(results)} 条结果）")
    return "\n".join(lines)


class WebSearchTool(Tool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "搜索互联网获取实时信息。"
            "用于查询新闻、天气、价格、最新资讯等训练数据截止后的内容。"
            "支持百度和 Bing 国际版。"
        )

    @property
    def parameters(self) -> dict:
        return SEARCH_TOOLS[0]["function"]["parameters"]

    @property
    def read_only(self) -> bool:
        return True

    @property
    def concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        query: str,
        engine: str = "auto",
        max_results: int = 5,
        proxy: str | None = None,
    ) -> str:
        return await web_search(
            query=query,
            engine=engine,
            max_results=max_results,
            proxy=proxy,
        )
