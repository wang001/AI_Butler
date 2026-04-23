from tools.base import Tool
from tools.memory_tools.memory_runtime import MEMORY_HISTORY_TOOL, MemoryTools


class SearchHistoryTool(Tool):
    def __init__(self, memory: MemoryTools):
        self._memory = memory

    @property
    def name(self) -> str:
        return "search_history"

    @property
    def description(self) -> str:
        return (
            "在完整的原始对话历史日志中按关键词检索。"
            "当用户询问过去某次具体对话的细节、某件具体的事情、"
            "或记忆系统可能遗漏的内容时使用。"
            "按关键词命中数打分排序：命中越多排越前，不要求全部命中。"
        )

    @property
    def parameters(self) -> dict:
        return MEMORY_HISTORY_TOOL["function"]["parameters"]

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        query: str,
        limit: int = 8,
        role: str | None = None,
    ) -> str:
        return self._memory.search_history(query=query, limit=limit, role=role)
