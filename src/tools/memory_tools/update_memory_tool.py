from tools.base import Tool
from tools.memory_tools.memory_runtime import MEMORY_UPDATE_TOOL, MemoryTools


class UpdateMemoryTool(Tool):
    def __init__(self, memory: MemoryTools):
        self._memory = memory

    @property
    def name(self) -> str:
        return "update_memory"

    @property
    def description(self) -> str:
        return (
            "当你明确判断有必要更新长期记忆 MEMORY.md 时调用。"
            "适用于用户明确要求“记住这件事”、稳定偏好发生变化、"
            "或对未来多轮对话持续有价值的重要背景。"
            "这个工具会使用独立模型上下文执行，不会直接把当前对话原样写入记忆。"
        )

    @property
    def parameters(self) -> dict:
        return MEMORY_UPDATE_TOOL["function"]["parameters"]

    async def execute(
        self,
        observations: str,
        reason: str = "",
        detail_paths: list[str] | None = None,
    ) -> str:
        return await self._memory.update_memory(
            observations=observations,
            reason=reason,
            detail_paths=detail_paths,
        )
