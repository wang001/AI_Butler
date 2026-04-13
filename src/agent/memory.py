# -*- coding: utf-8 -*-
"""
agent/memory.py — 记忆系统封装

将 ReMe 交互、MEMORY.md 读取、被动检索、对话沉淀等记忆相关职责
从原 Butler 上帝类中抽离，形成独立的 MemoryManager。

职责：
  1. 管理 ReMe 实例的生命周期（start / close）
  2. pre_reasoning_hook — 短期记忆压缩 + 异步沉淀
  3. passive_recall     — 静默向量检索（相似度过滤）
  4. read_memory_md     — 读取 MEMORY.md 用户画像
  5. settle             — 退出时将对话主动沉淀到长期记忆

ReMe Msg ↔ dict 转换也封装在此模块内部，对外仅暴露 dict 接口。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# ── 常量 ─────────────────────────────────────────────────────────────────────
PASSIVE_RECALL_K     = 8     # 静默检索候选条数
RECENT_RESERVE_CHARS = 2000  # 退出时保留最近 N 字符不压缩


# ── ReMe Msg ↔ dict 转换（内部使用）─────────────────────────────────────────

def _dicts_to_msgs(messages: list[dict]) -> list:
    """OpenAI dict 列表 → agentscope Msg 列表（过滤 tool call 中间过程）。"""
    from agentscope.message import Msg as _Msg
    result = []
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content") or ""
        if role == "tool":
            continue
        if role == "assistant" and "tool_calls" in m:
            if not content:
                continue
        result.append(_Msg(role=role, content=content, name=role))
    return result


def _msgs_to_dicts(msgs: list) -> list[dict]:
    """agentscope Msg 列表 → OpenAI dict 列表。"""
    return [{"role": m.role, "content": m.content} for m in msgs]


# ── MemoryManager ────────────────────────────────────────────────────────────

class MemoryManager:
    """
    记忆系统管理器。

    封装 ReMe（短期压缩 + 长期沉淀 + 向量检索）和 MEMORY.md（用户画像）
    的全部交互逻辑，对外仅暴露 dict 格式接口。
    """

    def __init__(self, reme: Any, working_dir: str, similarity_threshold: float = 0.5):
        self._reme = reme
        self._working_dir = Path(working_dir)
        self._similarity_threshold = similarity_threshold

    @property
    def reme(self) -> Any:
        """暴露 reme 实例，供 ToolDispatcher 使用。"""
        return self._reme

    @classmethod
    async def create(
        cls,
        working_dir: str,
        llm_api_key: str,
        llm_base_url: str,
        llm_model: str,
        emb_api_key: str,
        emb_base_url: str,
        emb_model: str,
        similarity_threshold: float = 0.5,
    ) -> "MemoryManager":
        """
        工厂方法：初始化 ReMe 并返回可用的 MemoryManager 实例。
        """
        from reme.reme_light import ReMeLight

        reme = ReMeLight(
            working_dir=working_dir,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_api_key=emb_api_key,
            embedding_base_url=emb_base_url,
            default_as_llm_config={"model_name": llm_model},
            default_embedding_model_config={"model_name": emb_model},
            default_file_store_config={"fts_enabled": True, "vector_enabled": True},
            enable_load_env=False,
        )
        await reme.start()

        return cls(
            reme=reme,
            working_dir=working_dir,
            similarity_threshold=similarity_threshold,
        )

    async def pre_reasoning_hook(
        self,
        messages: list[dict],
        system_prompt: str,
        compressed_summary: str,
        max_input_length: int,
        compact_ratio: float,
        memory_compact_reserve: int,
    ) -> tuple[list[dict], str]:
        """
        ReMe pre_reasoning_hook：短期记忆压缩 + 异步沉淀。

        Args:
            messages:             OpenAI dict 格式的对话历史
            system_prompt:        系统提示词
            compressed_summary:   上一轮的压缩摘要
            max_input_length:     最大输入长度
            compact_ratio:        压缩比
            memory_compact_reserve: 压缩保留字符数

        Returns:
            (压缩后的 messages, 新的 compressed_summary)
        """
        msgs_for_reme = _dicts_to_msgs(messages)
        msgs_for_reme, new_summary = await self._reme.pre_reasoning_hook(
            messages=msgs_for_reme,
            system_prompt=system_prompt,
            compressed_summary=compressed_summary,
            max_input_length=max_input_length,
            compact_ratio=compact_ratio,
            memory_compact_reserve=memory_compact_reserve,
            enable_tool_result_compact=True,
        )
        return _msgs_to_dicts(msgs_for_reme), new_summary

    async def passive_recall(self, query: str) -> list[str]:
        """
        静默向量检索，不经模型决策，相似度低于阈值直接过滤。

        Returns:
            匹配到的记忆片段列表（可能为空）
        """
        try:
            result = await self._reme.memory_search(
                query,
                max_results=PASSIVE_RECALL_K,
                min_score=self._similarity_threshold,
            )
            if not result or not result.content:
                return []
            snippets = []
            for block in result.content:
                text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                if text:
                    snippets.append(text)
            return snippets
        except Exception:
            return []

    def read_memory_md(self) -> str:
        """读取 MEMORY.md 用户画像文件，不存在时返回空串。"""
        try:
            f = self._working_dir / "MEMORY.md"
            return f.read_text(encoding="utf-8") if f.exists() else ""
        except Exception:
            return ""

    async def settle(self, messages: list[dict]) -> None:
        """
        退出时调用：将当前对话主动沉淀到长期记忆。

        保留最近 RECENT_RESERVE_CHARS 字符不压缩（避免压缩正在讨论的内容），
        将更早的对话异步提交给 ReMe 做摘要沉淀。
        """
        clean_msgs = _dicts_to_msgs(messages)
        tail_chars, split_idx = 0, 0
        for i in range(len(clean_msgs) - 1, -1, -1):
            tail_chars += len(clean_msgs[i].content or "")
            if tail_chars >= RECENT_RESERVE_CHARS:
                split_idx = i
                break
        if msgs_to_summarize := clean_msgs[:split_idx]:
            self._reme.add_async_summary_task(messages=msgs_to_summarize)
        await self._reme.await_summary_tasks()

    async def close(self) -> None:
        """关闭 ReMe 资源。"""
        await self._reme.close()
