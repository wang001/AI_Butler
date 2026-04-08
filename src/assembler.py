"""
assembler — 组装发给 LLM 的 messages 列表

设计目标：前缀缓存友好（Prompt Cache / KV Cache）。

messages 分层结构（从前到后）：
  Layer 0 — system prompt（人格指令）         ← 全局固定，可缓存
  Layer 1 — system prompt（MEMORY.md 画像）   ← session 内固定，可缓存
  Layer 2 — 历史对话 user/assistant           ← 增量追加，前缀可缓存
  Layer 3 — system prompt（检索结果注入）     ← 每轮变化，放末尾不影响前缀
  Layer 4 — 当前 user 消息                    ← 当前输入

关键：不变的内容在前面保持稳定，变化的内容（检索结果）放在末尾，
这样 LLM API 提供商的前缀缓存可以命中 Layer 0~2 的部分。
"""

import tiktoken

# token 预算
BUDGET = {
    "system":    800,
    "memory_md": 500,   # MEMORY.md（长期用户画像）
    "retrieval": 1500,  # 高相似度检索结果（可选注入）
    "reserve":   2000,  # LLM 回复预留
    "ctx_limit": 128000,
}

def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 3  # 中文粗估


def truncate_to_budget(messages: list[dict], token_budget: int) -> list[dict]:
    """从最旧消息开始截断，直到 token 总数 <= budget"""
    msgs = list(messages)
    while msgs:
        total = sum(count_tokens(m.get("content", "")) for m in msgs)
        if total <= token_budget:
            break
        msgs = msgs[1:]
    return msgs


def build(
    system_prompt: str,
    history: list[dict],
    memory_md: str = "",
    retrieval_snippets: list[str] | None = None,
) -> list[dict]:
    """
    组装最终发给 LLM 的 messages。

    分层策略（前缀缓存友好）：
      1. Layer 0: system prompt — 人格指令（固定）
      2. Layer 1: system prompt — MEMORY.md 用户画像（session 内固定）
      3. Layer 2: 历史对话（增量追加）
      4. Layer 3: system prompt — 检索结果注入（每轮变化，放末尾）

    注意：history 中的最后一条消息就是当前 user 输入（由 main.py 在
    调用 build 之前 append 进去的），检索结果插在它前面。
    """
    messages: list[dict] = []
    used_tokens = BUDGET["reserve"]

    # Layer 0: 人格指令（全局固定）
    messages.append({"role": "system", "content": system_prompt})
    used_tokens += count_tokens(system_prompt)

    # Layer 1: MEMORY.md 用户画像（session 内固定）
    if memory_md:
        profile_text = memory_md[: BUDGET["memory_md"] * 3]
        profile_content = f"## 关于用户的长期记忆\n{profile_text}"
        messages.append({"role": "system", "content": profile_content})
        used_tokens += count_tokens(profile_content)

    # Layer 2: 历史对话（增量追加，前缀稳定）
    # 先预留 retrieval 的预算，再分配给历史消息
    retrieval_budget = 0
    if retrieval_snippets:
        joined = "\n".join(f"- {s}" for s in retrieval_snippets)
        retrieval_content = f"## 可能相关的历史记忆（供参考，自行判断是否有用）\n{joined}"
        retrieval_budget = count_tokens(retrieval_content)

    remaining = BUDGET["ctx_limit"] - used_tokens - retrieval_budget
    recent = truncate_to_budget(history, remaining)

    # 如果有检索结果，插在历史消息的最后一条（当前 user 输入）之前
    if retrieval_snippets and recent:
        # 分离出当前 user 消息（最后一条）
        history_prefix = recent[:-1]
        current_msg = recent[-1]
        messages.extend(history_prefix)
        messages.append({"role": "system", "content": retrieval_content})
        messages.append(current_msg)
    else:
        messages.extend(recent)

    return messages
