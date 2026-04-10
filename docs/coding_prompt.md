# AI Butler — 编码任务提示词

## 背景

这是一个个人 AI 管家项目，目标是实现跨会话的长期记忆能力。
**必读**：`docs/tech_spec.md`（完整技术方案）。

## 核心原则

ReMe (`reme-ai[light]`) 已经内置了记忆检索、BM25、向量存储、短期记忆压缩、异步沉淀，**不要重复造轮子**。

我们只需要实现：
1. CLI 事件循环（`src/main.py`）
2. 把 ReMe 接进去
3. 查询路由（三档，`src/router.py`）
4. Context 组装（`src/assembler.py`）

---

## 第一步：安装依赖

更新 `requirements.txt`，内容替换为：

```
reme-ai[light]>=0.3.0
openai>=1.30.0
python-dotenv>=1.0.0
tiktoken>=0.7.0
```

安装：

```bash
pip install -r requirements.txt
```

---

## 第二步：目录结构

在仓库根目录下创建以下文件（仅这些，不要多建）：

```
src/
├── main.py
├── config.py
├── router.py
├── assembler.py
└── prompts/
    └── system.txt
.env.example
```

---

## 第三步：各文件实现

### `.env.example`

```env
LLM_BASE_URL=https://qianfan.baidubce.com/v2/coding
LLM_API_KEY=your_key_here
LLM_MODEL=glm-5

# Embedding（OpenRouter 免费模型，2048维）
EMB_BASE_URL=https://openrouter.ai/api/v1
EMB_API_KEY=your_key_here
EMB_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
```

---

### `src/config.py`

从 `.env` 读取配置，暴露一个 `Config` dataclass：

```python
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "glm-5"

    emb_base_url: str = ""
    emb_api_key: str = ""
    emb_model: str = ""

    working_dir: str = "data/.reme"
    max_input_length: int = 128000
    compact_ratio: float = 0.7
    memory_compact_reserve: int = 10000

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", ""),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "glm-5"),
            emb_base_url=os.getenv("EMB_BASE_URL", ""),
            emb_api_key=os.getenv("EMB_API_KEY", ""),
            emb_model=os.getenv("EMB_MODEL", ""),
        )
```

---

### `src/prompts/system.txt`

写一段中文管家人格，风格参考：

```
你是 AI Butler，用户的个人 AI 管家。

性格：
- 温暖但不谄媚，直接回答问题，不说"好的好的"之类的废话
- 有自己的判断，可以提出不同看法，但尊重用户决定
- 记得用户说过的事，自然引用，不刻意强调"我记得你说过"

记忆使用：
- 对话历史是当前上下文，优先依赖
- 长期记忆（MEMORY.md 内容）是"老朋友的印象"，自然融入，不生硬引用
- 如果记忆有冲突，以最新的为准

边界：
- 不替用户做决定，只提供信息和建议
- 不确定时直接说"我不太确定"，不编造
```

---

### `src/router.py`

三档路由，规则判断：

```python
DEEP_KEYWORDS = [
    "上次", "之前", "以前", "记得吗", "你还记得",
    "我说过", "我提过", "上回", "那时候", "之前说的",
]

DIRECT_KEYWORDS = [
    "几点", "多少", "帮我算", "你好", "现在", "今天",
    "明天", "天气", "时间", "计算", "翻译",
]

def classify(user_input: str) -> str:
    """返回 'DIRECT' / 'SHALLOW' / 'DEEP'"""
    for kw in DEEP_KEYWORDS:
        if kw in user_input:
            return "DEEP"
    for kw in DIRECT_KEYWORDS:
        if kw in user_input:
            return "DIRECT"
    return "SHALLOW"
```

---

### `src/assembler.py`

按 token 预算组装最终发给 LLM 的 messages：

```python
import tiktoken

# token 预算（参照 tech_spec 第九节）
BUDGET = {
    "system":    800,
    "profile":   500,   # MEMORY.md 内容，仅 SHALLOW/DEEP
    "retrieval": 1500,  # 检索结果，仅 DEEP
    "input":     500,   # 当前用户输入
    "reserve":   2000,  # LLM 回复预留
}

def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 3   # 中文粗估

def truncate_to_budget(messages: list[dict], token_budget: int) -> list[dict]:
    """从最旧消息开始截断，直到 token 总数 <= budget"""
    while messages:
        total = sum(count_tokens(m.get("content", "")) for m in messages)
        if total <= token_budget:
            break
        messages = messages[1:]  # 丢最旧的
    return messages

def build(
    system_prompt: str,
    history: list[dict],
    retrieval_context: str,
    user_input: str,
    route: str,
    memory_md: str = "",
) -> list[dict]:
    """
    组装最终 messages。
    - DIRECT:  system + 最近3轮 + user_input
    - SHALLOW: system + memory_md + history + user_input
    - DEEP:    system + memory_md + retrieval + history + user_input
    """
    messages = []

    # system prompt（固定，程序记忆）
    system_content = system_prompt
    if route in ("SHALLOW", "DEEP") and memory_md:
        profile_text = memory_md[:BUDGET["profile"] * 3]  # 粗截，后面精确算
        system_content += f"\n\n## 关于用户的长期记忆\n{profile_text}"
    if route == "DEEP" and retrieval_context:
        system_content += f"\n\n## 相关历史记忆\n{retrieval_context}"

    messages.append({"role": "system", "content": system_content})

    # 历史消息（工作记忆）
    if route == "DIRECT":
        recent = history[-6:] if len(history) > 6 else history  # 最近3轮
    else:
        # 计算剩余预算
        used = count_tokens(system_content) + BUDGET["input"] + BUDGET["reserve"]
        remaining = 128000 - used
        recent = truncate_to_budget(list(history), remaining)

    messages.extend(recent)

    return messages
```

---

### `src/main.py`

CLI 事件循环，核心入口：

```python
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from reme.reme_light import ReMeLight
from openai import AsyncOpenAI

from config import Config
from router import classify
from assembler import build

async def main():
    cfg = Config.from_env()
    system_prompt = Path("src/prompts/system.txt").read_text(encoding="utf-8")

    # 初始化 ReMe（记忆底座）
    reme = ReMeLight(
        working_dir=cfg.working_dir,
        llm_api_key=cfg.llm_api_key,
        llm_base_url=cfg.llm_base_url + "/chat/completions",  # ReMe 需要完整路径
        embedding_api_key=cfg.emb_api_key,
        embedding_base_url=cfg.emb_base_url,
        default_as_llm_config={"model_name": cfg.llm_model},
        default_embedding_model_config={"model_name": cfg.emb_model},
        default_file_store_config={"fts_enabled": True, "vector_enabled": True},
        enable_load_env=False,
    )
    await reme.start()

    # 初始化 OpenAI 客户端（用于主对话，流式输出）
    llm = AsyncOpenAI(
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
    )

    messages = []
    compressed_summary = ""

    print("=" * 40)
    print("AI Butler 已启动，输入 quit 退出")
    print("=" * 40)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q", "退出"):
            break

        messages.append({"role": "user", "content": user_input})

        try:
            # Step 1: ReMe hook（工具压缩 + token检查 + 短期记忆压缩 + 异步沉淀）
            messages, compressed_summary = await reme.pre_reasoning_hook(
                messages=messages,
                system_prompt=system_prompt,
                compressed_summary=compressed_summary,
                max_input_length=cfg.max_input_length,
                compact_ratio=cfg.compact_ratio,
                memory_compact_reserve=cfg.memory_compact_reserve,
                enable_tool_result_compact=True,
            )

            # Step 2: 路由判断
            route = classify(user_input)

            # Step 3: 按需检索长期记忆（仅 DEEP 档）
            retrieval_context = ""
            if route == "DEEP":
                try:
                    results = await reme.memory_search(user_input, max_results=10)
                    if results:
                        retrieval_context = "\n".join(
                            f"- {r.content}" for r in results
                        )
                except Exception as e:
                    print(f"[检索失败: {e}]", file=sys.stderr)

            # Step 4: 读取长期记忆（MEMORY.md）
            memory_md = ""
            if route in ("SHALLOW", "DEEP"):
                try:
                    memory_file = Path(cfg.working_dir) / "MEMORY.md"
                    if memory_file.exists():
                        memory_md = memory_file.read_text(encoding="utf-8")
                except Exception:
                    pass

            # Step 5: 组装 context
            final_messages = build(
                system_prompt=system_prompt,
                history=messages,
                retrieval_context=retrieval_context,
                user_input=user_input,
                route=route,
                memory_md=memory_md,
            )

            # Step 6: 调 LLM，流式输出
            print("\nButler: ", end="", flush=True)
            reply = ""
            stream = await llm.chat.completions.create(
                model=cfg.llm_model,
                messages=final_messages,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                reply += delta
            print()

            # Step 7: 更新工作记忆
            messages.append({"role": "assistant", "content": reply})

        except KeyboardInterrupt:
            print("\n[中断]")
            break
        except Exception as e:
            print(f"\n[错误: {e}]", file=sys.stderr)
            # 出错时保留用户消息，移除未完成的 assistant 消息
            if messages and messages[-1]["role"] == "user":
                pass  # 保留，下轮重试
            continue

    # 等待异步沉淀任务完成
    print("\n正在保存记忆...")
    await reme.await_summary_tasks()
    await reme.close()
    print("再见！")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 运行方式

```bash
# 1. 复制并填写环境变量
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY、EMB_API_KEY 等

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行
cd AI_Butler
python src/main.py
```

---

## .gitignore 追加

确保以下内容在 `.gitignore` 里：

```
data/
.env
__pycache__/
*.pyc
```

---

## 注意事项

1. **ReMe 的 `working_dir`** 设为 `data/.reme`，它自己管 MEMORY.md、daily notes、BM25、向量索引，不要另起一套存储
2. **`LLM_BASE_URL`** 填到 `/v2/coding` 这层（不含 `/chat/completions`），ReMe 初始化时会自己拼
3. **Embedding 维度** ReMe 会自动探测，不需要手动指定
4. **BM25 + 向量检索** 全由 ReMe 内部处理，`memory_search` 直接调用即可
5. **M6 梦境模块不在本次范围**，ReMe 的 `summary_memory` 已经覆盖了轻量版的异步沉淀需求
6. 如果 `reme.memory_search` 接口签名与示例不符，以 ReMe 实际 API 为准（可看 `reme/reme_light.py`）
