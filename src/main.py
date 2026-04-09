"""
AI Butler — 主入口

流程：
  1. ReMe pre_reasoning_hook（短期记忆压缩 + 异步沉淀）
  2. 静默检索长期记忆：相似度 >= 阈值才注入 system（被动记忆提示）
  3. 读取 MEMORY.md（用户画像）
  4. 组装 messages（assembler.build）
  5. Tool Call 循环：模型可主动调用 search_memory / get_current_time / web_search
  6. 流式输出最终回复
  7. 更新工作记忆
"""
# 只 import 轻量标准库，重型依赖在后台预热
import asyncio
import json
import sys
import time
import threading
import warnings
import concurrent.futures
from pathlib import Path

# 屏蔽 chromadb 在 Python 3.14 上触发的 DeprecationWarning
warnings.filterwarnings(
    "ignore",
    message="'asyncio.iscoroutinefunction' is deprecated",
    category=DeprecationWarning,
    module="chromadb",
)

from dotenv import load_dotenv
load_dotenv()

# 轻量依赖，先 import
from openai import AsyncOpenAI
from config import Config
from assembler import build
from history import ChatHistory

# prompt_toolkit：输入/输出区域隔离
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import HTML

# ── 后台预热：在独立线程里提前 import 重型库 ──────────────────────────
# transformers / chromadb / agentscope / reme 加起来约 10s+
# 用 Future 让主流程在第一次需要时等它完成即可
_heavy_future: concurrent.futures.Future = concurrent.futures.Future()

def _load_heavy():
    """在后台线程里 import 所有重型模块，完成后把结果放入 Future。"""
    try:
        from reme.reme_light import ReMeLight        # noqa: F401
        from agentscope.message import Msg           # noqa: F401
        _heavy_future.set_result(True)
    except Exception as e:
        _heavy_future.set_exception(e)

threading.Thread(target=_load_heavy, daemon=True, name="heavy-import").start()


def _ensure_heavy_loaded():
    """阻塞直到重型模块加载完毕（首条消息发送前调用）。"""
    if not _heavy_future.done():
        safe_print("（正在初始化记忆系统，请稍候…）")
    _heavy_future.result()   # 有异常会在此 re-raise
    # 加载完成后才能 import
    from reme.reme_light import ReMeLight   # noqa: F401, F811
    from agentscope.message import Msg     # noqa: F401, F811
    from tools import TOOLS, ToolExecutor  # noqa: F401, F811


# 延迟 import 的全局占位（实际在 main() 中赋值）
ReMeLight = None  # type: ignore
Msg = None        # type: ignore
TOOLS = None      # type: ignore
ToolExecutor = None  # type: ignore

# ── safe_print：在 patch_stdout 上下文中安全输出，不干扰输入行 ──────────
def safe_print(*args, sep=" ", end="\n", file=None):
    """替代 print()，在 prompt_toolkit patch_stdout 保护下安全输出。"""
    text = sep.join(str(a) for a in args) + end
    # patch_stdout 会把这里的 sys.stdout 包装为线程安全版本
    sys.stdout.write(text)
    sys.stdout.flush()


# ---------- ReMe Msg ↔ dict 转换工具 ----------

def dicts_to_msgs(messages: list[dict]) -> list:
    """将 OpenAI 格式 dict 列表转换为 agentscope Msg 列表（供 ReMe 使用）。

    过滤规则（只让纯对话内容进入长期记忆，tool call 中间过程不沉淀）：
      - role="tool" → 跳过（工具返回值对长期记忆无意义）
      - role="assistant" 且有 tool_calls 且 content 为空 → 跳过
      - role="assistant" 且有 tool_calls 但 content 有值 → 只保留 content，丢弃 tool_calls
      - 其余正常保留
    """
    from agentscope.message import Msg as _Msg
    result = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""

        # 跳过 tool result 消息
        if role == "tool":
            continue
        # assistant 的 tool_call 调度消息：只在有 content 时保留文字部分
        if role == "assistant" and "tool_calls" in m:
            if not content:
                continue
            # 有 content → 只保留文字，不带 tool_calls metadata

        result.append(_Msg(role=role, content=content, name=role))
    return result


def msgs_to_dicts(msgs: list) -> list[dict]:
    """将 agentscope Msg 列表转回 OpenAI 格式 dict 列表。

    注意：经 ReMe 压缩后的消息只包含纯对话内容（tool call 已在 dicts_to_msgs 中过滤），
    所以这里只还原 role + content，不再恢复 tool_calls / tool_call_id。
    """
    result = []
    for m in msgs:
        result.append({"role": m.role, "content": m.content})
    return result

# 相似度高于此阈值才把静默检索结果注入 system（0~1，越高越严格）
# ReMe hybrid search 分数通常在 0.3~0.8，0.5 可覆盖大多数有效匹配
SIMILARITY_INJECT_THRESHOLD = 0.5
# 静默检索候选条数
PASSIVE_RECALL_K = 8
# tool call 最大轮次（防止死循环）
MAX_TOOL_ROUNDS = 6


async def passive_recall(
    reme: ReMeLight,
    query: str,
    threshold: float,
    k: int,
) -> list[str]:
    """
    静默检索：不经模型决策，直接向量搜索。
    ReMe v0.3+ memory_search 返回 ToolResponse 对象（content 是 list[TextBlock]）。
    将所有文本内容拼接后作为单个片段返回（已由 ReMe 内部按 min_score 过滤）。
    """
    try:
        result = await reme.memory_search(query, max_results=k, min_score=threshold)
        if result is None:
            return []
        # ToolResponse: result.content 是 list[dict] 或 list[TextBlock]
        content = result.content
        if not content:
            return []
        snippets = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
            else:
                text = getattr(block, "text", "") or ""
            if text:
                snippets.append(text)
        return snippets
    except Exception as e:
        safe_print(f"[静默检索失败: {e}]")
        return []


async def llm_create_with_retry(llm: AsyncOpenAI, **kwargs) -> object:
    """
    带指数退避的 LLM 调用，处理 429 限流。
    最多重试 5 次，等待间隔：2s → 4s → 8s → 16s → 32s。
    """
    from openai import RateLimitError
    max_retries = 3
    wait = 2
    for attempt in range(max_retries):
        try:
            return await llm.chat.completions.create(**kwargs)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            safe_print(f"\n[限流，{wait}秒后重试 ({attempt+1}/{max_retries})...]")
            await asyncio.sleep(wait)
            wait = min(wait * 2, 32)


async def run_tool_call_loop(
    llm: AsyncOpenAI,
    model: str,
    messages: list[dict],
    executor: object,
    tools: list,
    history: object = None,
) -> tuple[str, list[dict]]:
    """
    执行 Tool Call 循环，返回 (最终回复文本, 追加到 messages 的新消息列表)。
    新消息列表包含所有 assistant tool_call 消息 + tool result 消息 + 最终 assistant 回复。
    history: ChatHistory 实例，用于记录工具调用（只记工具名，不记参数/结果）。
    """
    new_messages: list[dict] = []

    for _round in range(MAX_TOOL_ROUNDS):
        response = await llm_create_with_retry(
            llm,
            model=model,
            messages=messages + new_messages,
            tools=tools,
            tool_choice="auto",
            stream=False,  # tool call 阶段不流式，避免处理复杂性
        )

        msg = response.choices[0].message

        # 没有 tool call → 最终回复
        if not msg.tool_calls:
            reply = msg.content or ""
            new_messages.append({"role": "assistant", "content": reply})
            return reply, new_messages

        # 有 tool call → 执行并追加结果
        # assistant 的 tool_call 消息
        tool_call_msg: dict = {
            "role": "assistant",
            "content": msg.content,  # 可能为 None
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        }
        new_messages.append(tool_call_msg)

        # 按并发安全性分批执行 tool calls
        # 规则：按模型输出顺序扫描，相邻的并发安全工具合并为一批并行执行，
        #       遇到非并发安全工具则独立为一批串行执行。
        from tools import TOOL_CONCURRENT_SAFE

        tool_names = [tc.function.name for tc in msg.tool_calls]
        for name in tool_names:
            safe_print(f"\n[工具调用] {name}")
        # 记录本轮调用了哪些工具（只记名称）
        if history is not None:
            history.append("tool_call", "、".join(tool_names))

        # 分批：相邻可并行的合为一组，不可并行的各自独占一组
        batches: list[list] = []
        for tc in msg.tool_calls:
            is_safe = TOOL_CONCURRENT_SAFE.get(tc.function.name, False)
            if is_safe and batches and all(
                TOOL_CONCURRENT_SAFE.get(t.function.name, False) for t in batches[-1]
            ):
                # 当前工具可并行，且上一批也全是可并行的 → 合入
                batches[-1].append(tc)
            else:
                # 新开一批（不可并行工具独占，或首个工具）
                batches.append([tc])

        async def _exec_one(tc):
            result = await executor.run(tc.function.name, tc.function.arguments)
            safe_print(f"[工具结果:{tc.function.name}] {result[:200]}{'...' if len(result) > 200 else ''}")
            return tc.id, result

        # 逐批执行：批内并行，批间串行
        for batch in batches:
            if len(batch) == 1:
                tid, result = await _exec_one(batch[0])
                new_messages.append({
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": result,
                })
            else:
                results = await asyncio.gather(*[_exec_one(tc) for tc in batch])
                for tid, result in results:
                    new_messages.append({
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": result,
                    })

    # 超过最大轮次，强制不带工具再请求一次
    response = await llm_create_with_retry(
        llm,
        model=model,
        messages=messages + new_messages,
        stream=False,
    )
    reply = response.choices[0].message.content or ""
    new_messages.append({"role": "assistant", "content": reply})
    return reply, new_messages


async def stream_final_reply(
    llm: AsyncOpenAI,
    model: str,
    messages: list[dict],
) -> str:
    """对已经执行完 tool call 的 messages 做最终流式输出。"""
    safe_print("\nButler: ", end="")
    reply = ""
    stream = await llm.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        sys.stdout.write(delta)
        sys.stdout.flush()
        reply += delta
    safe_print()
    return reply


async def main():
    cfg = Config.from_env()
    # system.txt 位于源码目录中，通过 __file__ 定位（兼容容器内 /app 和本地开发）
    _src_dir = Path(__file__).parent
    system_prompt = (_src_dir / "prompts" / "system.txt").read_text(encoding="utf-8")

    session = PromptSession()

    # patch_stdout 包住整个交互循环：
    # 所有 print/sys.stdout.write 都会在输出后自动重绘输入行，互不干扰
    with patch_stdout():
        safe_print("=" * 40)
        safe_print("AI Butler 已启动，输入 quit 退出")
        safe_print("=" * 40)

        llm = AsyncOpenAI(
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
        )

        messages: list[dict] = []
        compressed_summary = ""
        reme = None
        executor = None

        # 历史日志系统：轻量级，直接在主流程初始化（无重型依赖）
        history = ChatHistory(data_dir=cfg.working_dir)

        while True:
            # prompt_toolkit 异步输入：Ctrl+C → KeyboardInterrupt，Ctrl+D → EOFError
            try:
                user_input = await session.prompt_async("\n你: ")
                user_input = user_input.strip()
            except KeyboardInterrupt:
                safe_print("（输入已取消，继续对话；输入 quit 退出）")
                continue
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q", "退出"):
                break

            # 首条消息发出前，确保重型模块已加载完毕，并完成 ReMe 初始化
            if reme is None:
                _ensure_heavy_loaded()
                # 重新 import（此时后台线程已把它们装进 sys.modules）
                from reme.reme_light import ReMeLight as _ReMeLight
                from tools import TOOLS as _TOOLS, ToolExecutor as _ToolExecutor
                reme = _ReMeLight(
                    working_dir=cfg.working_dir,
                    llm_api_key=cfg.llm_api_key,
                    llm_base_url=cfg.llm_base_url,
                    embedding_api_key=cfg.emb_api_key,
                    embedding_base_url=cfg.emb_base_url,
                    default_as_llm_config={"model_name": cfg.llm_model},
                    default_embedding_model_config={"model_name": cfg.emb_model},
                    default_file_store_config={"fts_enabled": True, "vector_enabled": True},
                    enable_load_env=False,
                )
                await reme.start()

                # 初始化命令执行器（容器内直接通过 subprocess 执行）
                sandbox = None
                if cfg.command_enabled:
                    try:
                        from sandbox import CommandExecutor, CommandConfig
                        sandbox = CommandExecutor(CommandConfig(
                            workdir=cfg.command_workdir,
                            default_timeout=cfg.command_default_timeout,
                        ))
                    except Exception as e:
                        safe_print(f"[命令执行器初始化跳过] {e}")

                # 初始化 browser-use Agent
                browser_agent = None
                if cfg.browser_enabled:
                    try:
                        from browser import BrowserAgent, BrowserUseConfig
                        browser_agent = BrowserAgent(BrowserUseConfig(
                            headless=cfg.browser_headless,
                            max_steps=cfg.browser_max_steps,
                            llm_model=cfg.llm_model,
                            llm_base_url=cfg.llm_base_url,
                            llm_api_key=cfg.llm_api_key,
                        ))
                    except Exception as e:
                        safe_print(f"[浏览器初始化跳过] {e}")

                executor = _ToolExecutor(
                    reme,
                    history=history,
                    sandbox=sandbox,
                    browser_agent=browser_agent,
                )
                TOOLS_LIST = _TOOLS
            else:
                from tools import TOOLS as TOOLS_LIST  # noqa: F811

            messages.append({"role": "user", "content": user_input})

            try:
                # Step 1: ReMe hook（短期记忆压缩 + 异步沉淀）
                msgs_for_reme = dicts_to_msgs(messages)
                msgs_for_reme, compressed_summary = await reme.pre_reasoning_hook(
                    messages=msgs_for_reme,
                    system_prompt=system_prompt,
                    compressed_summary=compressed_summary,
                    max_input_length=cfg.max_input_length,
                    compact_ratio=cfg.compact_ratio,
                    memory_compact_reserve=cfg.memory_compact_reserve,
                    enable_tool_result_compact=True,
                )
                messages = msgs_to_dicts(msgs_for_reme)

                # Step 2: 静默检索长期记忆
                passive_snippets = await passive_recall(
                    reme=reme,
                    query=user_input,
                    threshold=cfg.memory_similarity_threshold,
                    k=PASSIVE_RECALL_K,
                )

                # Step 3: 读取 MEMORY.md（用户画像）
                memory_md = ""
                try:
                    memory_file = Path(cfg.working_dir) / "MEMORY.md"
                    if memory_file.exists():
                        memory_md = memory_file.read_text(encoding="utf-8")
                except Exception:
                    pass

                # Step 4: 组装 context
                final_messages = build(
                    system_prompt=system_prompt,
                    history=messages,
                    memory_md=memory_md,
                    retrieval_snippets=passive_snippets if passive_snippets else None,
                )

                # Step 5: Tool Call 循环
                reply, new_msgs = await run_tool_call_loop(
                    llm=llm,
                    model=cfg.llm_model,
                    messages=final_messages,
                    executor=executor,
                    tools=TOOLS_LIST,
                    history=history,
                )

                # Step 6: 输出回复
                if reply:
                    safe_print(f"\nButler: {reply}")
                else:
                    reply = await stream_final_reply(llm, cfg.llm_model, final_messages + new_msgs)

                # Step 7: 追加到工作记忆
                messages.extend(new_msgs)

                # Step 8: 持久化原始对话日志（JSONL + SQLite FTS5）
                # tool_call 条目已在 run_tool_call_loop 内部实时写入
                history.append("user", user_input)
                if reply:
                    history.append("assistant", reply)

            except KeyboardInterrupt:
                safe_print("\n（当前回复已中断，继续对话；输入 quit 退出）")
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
                continue
            except Exception as e:
                safe_print(f"\n[错误: {e}]")
                import traceback
                traceback.print_exc()
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
                continue

        # 清理浏览器资源
        if executor and hasattr(executor, 'browser_agent') and executor.browser_agent:
            try:
                await executor.browser_agent.close()
            except Exception:
                pass

        safe_print("\n正在保存记忆...")
        try:
            if reme is not None:
                # 主动沉淀当前对话到长期记忆
                # pre_reasoning_hook 只在 context 溢出时才触发沉淀，
                # 短对话从不溢出，所以需要在退出时显式提交。
                #
                # 重要：保留最近 RECENT_RESERVE_CHARS 字符的原始对话不压缩，
                # 只把更早的部分送去做 summary。避免最近的对话细节被摘要丢失。
                RECENT_RESERVE_CHARS = 2000
                clean_msgs = dicts_to_msgs(messages)
                # 从末尾往前累计字符数，找到切分点：
                # 保留最近 >= RECENT_RESERVE_CHARS 的原始消息不压缩，
                # 只把更早的部分送去 summary。
                # 若整段对话不足 RECENT_RESERVE_CHARS，则全部保留、不压缩。
                tail_chars = 0
                split_idx = 0  # 默认不压缩任何消息
                for i in range(len(clean_msgs) - 1, -1, -1):
                    tail_chars += len(clean_msgs[i].content or "")
                    if tail_chars >= RECENT_RESERVE_CHARS:
                        split_idx = i
                        break
                msgs_to_summarize = clean_msgs[:split_idx]
                if msgs_to_summarize:
                    reme.add_async_summary_task(messages=msgs_to_summarize)
                await reme.await_summary_tasks()
                await reme.close()
        except Exception as e:
            safe_print(f"[保存记忆失败] {e}")
        history.close()
        safe_print("再见！")


if __name__ == "__main__":
    asyncio.run(main())
