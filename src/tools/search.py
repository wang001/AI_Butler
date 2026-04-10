"""
网络搜索工具（Web Search）

封装 web_search 工具：
  - 调用 union-search-skill 执行多平台聚合搜索
  - 支持通过 platforms 参数指定搜索平台（默认百度 + 必应 + Brave）
  - union_search 是同步函数，用 asyncio.to_thread 包装避免阻塞事件循环

SEARCH_TOOLS 是供 OpenAI function-calling 使用的 schema，
由 dispatcher.py 合并进全局 TOOLS。
"""
import asyncio
import sys
from pathlib import Path

# 将 union-search-skill 的 scripts 目录加入 sys.path
_UNION_SEARCH_SCRIPTS = Path(__file__).parent.parent.parent / "vendor" / "union-search-skill" / "scripts"
if str(_UNION_SEARCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_UNION_SEARCH_SCRIPTS))

# ── OpenAI function-calling schema ────────────────────────────────────────────

SEARCH_TOOLS: list[dict] = [
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
]

# ── 工具实现 ───────────────────────────────────────────────────────────────────

_DEFAULT_PLATFORMS = ["baidu_direct", "bing_int_direct", "brave_direct"]


async def web_search(query: str, platforms: list[str] | None = None) -> str:
    """调用 union-search-skill 执行多平台聚合搜索，返回格式化结果字符串。"""
    try:
        from union_search import union_search
    except ImportError as e:
        return f"[web_search 不可用] union-search-skill 未正确安装: {e}"

    search_platforms = platforms or _DEFAULT_PLATFORMS

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                union_search,
                keyword=query,
                platforms=search_platforms,
                limit=5,        # 每个平台最多 5 条
                max_workers=6,  # 并发数
                timeout=15,     # 单平台超时（秒）
                deduplicate=True,
            ),
            timeout=20,         # 整体超时（秒）
        )
    except asyncio.TimeoutError:
        return f"[web_search 超时] 搜索「{query}」超过 20 秒，请稍后重试。"
    except Exception as e:
        return f"[web_search 失败] {e}"

    items: list[dict] = result.get("final_items", [])
    if not items:
        for platform_result in result.get("results", {}).values():
            items.extend(platform_result.get("items", []))

    if not items:
        return f"搜索「{query}」未找到结果。"

    lines = [f"搜索「{query}」的结果："]
    for i, item in enumerate(items[:15], 1):
        title   = item.get("title") or item.get("name") or ""
        url     = item.get("url") or item.get("link") or item.get("href") or ""
        snippet = item.get("snippet") or item.get("description") or item.get("content", "")[:200]
        source  = item.get("source") or item.get("platform") or ""
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
