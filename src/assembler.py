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

    策略：
    - system = 人格 prompt + MEMORY.md（始终带，上限 BUDGET['memory_md'] tokens）
    - 若 retrieval_snippets 非空（调用方已做相似度过滤），追加到 system 末尾
    - history 按剩余 token 预算从旧到新截断
    """
    # 1. 拼 system
    system_content = system_prompt

    if memory_md:
        # 粗截到预算字符数（中文1字≈1token，英文约0.25token，取3倍保守）
        profile_text = memory_md[: BUDGET["memory_md"] * 3]
        system_content += f"\n\n## 关于用户的长期记忆\n{profile_text}"

    if retrieval_snippets:
        joined = "\n".join(f"- {s}" for s in retrieval_snippets)
        system_content += f"\n\n## 可能相关的历史记忆（供参考，自行判断是否有用）\n{joined}"

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # 2. 历史消息按剩余预算截断
    used = count_tokens(system_content) + BUDGET["reserve"]
    remaining = BUDGET["ctx_limit"] - used
    recent = truncate_to_budget(history, remaining)
    messages.extend(recent)

    return messages
