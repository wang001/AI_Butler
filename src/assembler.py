# -*- coding: utf-8 -*-
"""
assembler.py — 组装发给 LLM 的 messages 列表

build() 将各层上下文拼装成 OpenAI messages 格式：
  [system]  = system_prompt + MEMORY.md（用户画像）
  [history] = 原始对话历史（user/assistant 交替，含本轮用户消息）
  [user]    = retrieval_snippets（静默检索，追加在最末尾以最大化前缀缓存命中率）
"""

from typing import Optional


def build(
    system_prompt: str,
    history: list[dict],
    memory_md: str = "",
    retrieval_snippets: Optional[list[str]] = None,
) -> list[dict]:
    """
    组装 messages 列表。

    Args:
        system_prompt: 系统提示词（来自 prompts/system.txt）
        history: 对话历史，OpenAI 格式 [{"role": "user"|"assistant", "content": "..."}]，
                 最后一条通常是本轮用户消息
        memory_md: MEMORY.md 内容（用户画像/长期记忆摘要），可为空
        retrieval_snippets: 静默向量检索结果片段列表，None 或空列表表示无检索结果。
                            放在所有历史（含当前用户消息）之后，以保证
                            system+memory+history 前缀稳定、可被 LLM 缓存命中。

    Returns:
        OpenAI messages 格式列表，可直接传给 chat.completions.create(messages=...)
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
