"""
记忆工具（Memory Tools）

封装三个面向长期记忆 / 对话历史的工具：
  - search_memory  : 向量检索长期记忆（ReMe）
  - search_history : 全文检索原始对话日志（SQLite FTS5）
  - update_memory  : 触发独立上下文更新 MEMORY.md

MEMORY_TOOLS 是供 OpenAI function-calling 使用的 schema 列表，
由 dispatcher.py 合并进全局 TOOLS。

MemoryTools 类持有 reme / history / memory_update_service 三个依赖，
对外暴露同名异步/同步方法供 ToolDispatcher 调用。
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any


MEMORY_SEARCH_TOOL: dict = {
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
}

MEMORY_HISTORY_TOOL: dict = {
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
}

MEMORY_UPDATE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "update_memory",
        "description": (
            "当你明确判断有必要更新长期记忆 MEMORY.md 时调用。"
            "适用于用户明确要求“记住这件事”、稳定偏好发生变化、"
            "或对未来多轮对话持续有价值的重要背景。"
            "这个工具会使用独立模型上下文执行，不会直接把当前对话原样写入记忆。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "string",
                    "description": (
                        "你建议写入长期记忆的关键信息摘要。"
                        "应只包含稳定、长期有价值的信息，不要写临时任务。"
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "为什么值得更新长期记忆，例如：用户明确要求记住、长期偏好更新。",
                },
                "detail_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选。若细节很多，可给出相关详细文档路径，记忆里保留简述 + 路径。",
                },
            },
            "required": ["observations"],
        },
    },
}

MEMORY_TOOLS: list[dict] = [
    MEMORY_SEARCH_TOOL,
    MEMORY_HISTORY_TOOL,
    MEMORY_UPDATE_TOOL,
]

MEMORY_READONLY_TOOLS: list[dict] = [
    MEMORY_SEARCH_TOOL,
    MEMORY_HISTORY_TOOL,
]


class MemoryTools:
    """
    记忆工具实现。

    持有 ReMe 实例（长期记忆）、ChatHistory 实例（对话日志）和
    MemoryUpdateService（MEMORY.md 更新服务）。
    """

    def __init__(self, reme: Any, history: Any = None, memory_update_service: Any = None):
        self.reme = reme
        self.history = history
        self.memory_update_service = memory_update_service

    async def search_memory(self, query: str, max_results: int = 5) -> str:
        """向量检索长期记忆（ReMe）。"""
        try:
            result = await self.reme.memory_search(query, max_results=max_results)
            if result is None:
                return "未找到相关记忆。"
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

    def search_history(
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

        cst = timezone(timedelta(hours=8))
        lines = [f"在对话历史中找到 {len(results)} 条与「{query}」相关的记录：\n"]
        for i, r in enumerate(results, 1):
            dt = datetime.fromtimestamp(r["ts"], tz=cst).strftime("%Y-%m-%d %H:%M")
            role_label = {"user": "你", "assistant": "Butler", "tool": "工具"}.get(r["role"], r["role"])
            content = r["content"]
            if len(content) > 300:
                content = content[:300] + "…"
            score_tag = f" (匹配度:{r['score']:.0%})" if "score" in r else ""
            lines.append(f"{i}. [{dt}]{score_tag} {role_label}：{content}")
        return "\n".join(lines)

    async def update_memory(
        self,
        observations: str,
        reason: str = "",
        detail_paths: list[str] | None = None,
    ) -> str:
        """触发独立上下文的 MEMORY.md 更新。"""
        if self.memory_update_service is None:
            return "[update_memory 不可用] 记忆更新服务未初始化。"

        note_lines = [observations.strip()]
        if reason.strip():
            note_lines.append(f"更新原因：{reason.strip()}")
        if detail_paths:
            clean_paths = [p.strip() for p in detail_paths if p and p.strip()]
            if clean_paths:
                note_lines.append("详细资料路径：" + "；".join(clean_paths))

        notes = "\n".join(line for line in note_lines if line)
        return await self.memory_update_service.request_update(
            trigger="tool",
            proposed_notes=notes,
        )
