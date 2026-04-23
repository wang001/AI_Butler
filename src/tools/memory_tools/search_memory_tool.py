from tools.base import Tool
from tools.memory_tools.memory_runtime import MEMORY_SEARCH_TOOL, MemoryTools


class SearchMemoryTool(Tool):
    def __init__(self, memory: MemoryTools):
        self._memory = memory

    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return (
            "在长期记忆库中检索与查询相关的历史信息。"
            "当用户询问过去发生的事、之前的偏好、或你觉得有历史背景需要确认时调用。"
        )

    @property
    def parameters(self) -> dict:
        return MEMORY_SEARCH_TOOL["function"]["parameters"]

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, query: str, max_results: int = 5) -> str:
        return await self._memory.search_memory(query=query, max_results=max_results)
