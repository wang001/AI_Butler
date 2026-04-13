# -*- coding: utf-8 -*-
"""
agent/context.py — 上下文组装器（ContextBuilder）

从原 assembler.py 升级而来，参考 NanoBot 的 ContextBuilder 设计。
负责将各层上下文（system prompt、MEMORY.md、对话历史、检索片段）
拼装成可直接送入 LLM 的 OpenAI messages 格式。

与原 assembler.py 相比的改进：
  - 从纯函数升级为类，可扩展 token 预算管理、Bootstrap 文件等能力
  - 支持通过 MemoryManager 自动获取 MEMORY.md 和检索结果
  - 保留独立的 build() 静态方法以兼容简单场景

用法：
    builder = ContextBuilder(system_prompt, memory_manager)
    messages = await builder.build_context(messages, user_input, cfg)
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.memory import MemoryManager
    from config import Config


class ContextBuilder:
    """
    上下文组装器。

    参考 NanoBot 的 ContextBuilder，将分散在 Butler 中的上下文准备逻辑
    （ReMe hook、被动检索、MEMORY.md 读取、messages 拼装）统一在此。
    """

    def __init__(self, system_prompt: str, memory: "MemoryManager"):
        self._system_prompt = system_prompt
        self._memory = memory

    async def build_context(
        self,
        messages: list[dict],
        user_input: str,
        compressed_summary: str,
        cfg: "Config",
    ) -> tuple[list[dict], list[dict], str]:
        """
        完整的上下文构建流程：
          1. 追加用户消息到 messages
          2. ReMe pre_reasoning_hook（短期记忆压缩 + 异步沉淀）
          3. 静默检索长期记忆
          4. 读取 MEMORY.md
          5. 拼装最终 messages

        Args:
            messages:           当前会话的对话历史
            user_input:         本轮用户输入
            compressed_summary: 上一轮的压缩摘要
            cfg:                运行时配置

        Returns:
            (final_messages, updated_messages, new_compressed_summary)
            - final_messages:    组装好的完整 messages（含 system + history + retrieval）
            - updated_messages:  经 ReMe 压缩后的对话历史（需回写 Butler._messages）
            - new_summary:       更新后的压缩摘要
        """
        # 追加用户消息
        messages = messages + [{"role": "user", "content": user_input}]

        # Step 1: ReMe hook（短期记忆压缩 + 异步沉淀）
        messages, new_summary = await self._memory.pre_reasoning_hook(
            messages=messages,
            system_prompt=self._system_prompt,
            compressed_summary=compressed_summary,
            max_input_length=cfg.max_input_length,
            compact_ratio=cfg.compact_ratio,
            memory_compact_reserve=cfg.memory_compact_reserve,
        )

        # Step 2: 静默检索长期记忆
        passive_snippets = await self._memory.passive_recall(user_input)

        # Step 3: 读取 MEMORY.md（用户画像）
        memory_md = self._memory.read_memory_md()

        # Step 4: 组装 context
        final_messages = self.build(
            system_prompt=self._system_prompt,
            history=messages,
            memory_md=memory_md,
            retrieval_snippets=passive_snippets or None,
        )

        return final_messages, messages, new_summary

    @staticmethod
    def build(
        system_prompt: str,
        history: list[dict],
        memory_md: str = "",
        retrieval_snippets: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        组装 messages 列表（纯函数版本，兼容原 assembler.build）。

        Args:
            system_prompt:      系统提示词（来自 prompts/system.txt）
            history:            对话历史 [{"role": "user"|"assistant", "content": "..."}]
            memory_md:          MEMORY.md 内容（用户画像/长期记忆摘要），可为空
            retrieval_snippets: 静默检索结果片段列表，放在所有历史之后
                                以保证 system+memory+history 前缀稳定、可被 LLM 缓存命中

        Returns:
            OpenAI messages 格式列表
        """
        # ── 组装 system message（仅含静态内容，保持前缀稳定）────────────────
        system_parts = [system_prompt.strip()]

        if memory_md and memory_md.strip():
            system_parts.append(
                "\n\n---\n## 用户长期记忆（MEMORY.md）\n\n" + memory_md.strip()
            )

        system_content = "".join(system_parts)

        messages: list[dict] = [{"role": "system", "content": system_content}]

        # ── 追加对话历史（含本轮用户消息）───────────────────────────────────
        for turn in history:
            role = turn.get("role", "")
            content = turn.get("content") or ""
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        # ── 追加检索片段（放在最末尾，不影响前缀缓存命中）──────────────────
        if retrieval_snippets:
            snippets_text = "\n\n---\n".join(s.strip() for s in retrieval_snippets if s.strip())
            if snippets_text:
                messages.append({
                    "role": "system",
                    "content": "## 从记忆库检索到的相关历史记录\n\n" + snippets_text,
                })

        return messages
