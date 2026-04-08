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
import concurrent.futures
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# 轻量依赖，先 import
from openai import AsyncOpenAI
from config import Config
from assembler import build

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
        print("（正在初始化记忆系统，请稍候…）", flush=True)
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


# ---------- ReMe Msg ↔ dict 转换工具 ----------

def dicts_to_msgs(messages: list[dict]) -> list:
    """将 OpenAI 格式 dict 列表转换为 agentscope Msg 列表（供 ReMe 使用）。"""
    from agentscope.message import Msg as _Msg
    result = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        metadata = {}
        if "tool_calls" in m:
            metadata["tool_calls"] = m["tool_calls"]
        if "tool_call_id" in m:
            metadata["tool_call_id"] = m["tool_call_id"]
        result.append(_Msg(role=role, content=content, name=role, metadata=metadata))
    return result


def msgs_to_dicts(msgs: list) -> list[dict]:
    """将 agentscope Msg 列表转回 OpenAI 格式 dict 列表。"""
    result = []
    for m in msgs:
        d: dict = {"role": m.role, "content": m.content}
        if m.metadata.get("tool_calls"):
            d["tool_calls"] = m.metadata["tool_calls"]
        if m.metadata.get("tool_call_id"):
            d["tool_call_id"] = m.metadata["tool_call_id"]
        result.append(d)
    return result

# 相似度高于此阈值才把静默检索结果注入 system（0~1，越高越严格）
SIMILARITY_INJECT_THRESHOLD = 0.75
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
        print(f"[静默检索失败: {e}]", file=sys.stderr)
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
            print(f"\n[限流，{wait}秒后重试 ({attempt+1}/{max_retries})...]", flush=True)
            await asyncio.sleep(wait)
            wait = min(wait * 2, 32)


async def run_tool_call_loop(
    llm: AsyncOpenAI,
    model: str,
    messages: list[dict],
    executor: object,
    tools: list,
) -> tuple[str, list[dict]]:
    """
    执行 Tool Call 循环，返回 (最终回复文本, 追加到 messages 的新消息列表)。
    新消息列表包含所有 assistant tool_call 消息 + tool result 消息 + 最终 assistant 回复。
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

        # 并发执行所有 tool call（与 Claude Code StreamingToolExecutor 思路一致）
        for tc in msg.tool_calls:
            print(f"\n[工具调用] {tc.function.name}({tc.function.arguments})", flush=True)

        async def _exec_one(tc):
            result = await executor.run(tc.function.name, tc.function.arguments)
            print(f"[工具结果:{tc.function.name}] {result[:200]}{'...' if len(result) > 200 else ''}", flush=True)
            return tc.id, result

        results = await asyncio.gather(*[_exec_one(tc) for tc in msg.tool_calls])

        for tool_call_id, result in results:
            new_messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
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
    """对已经执行完 tool call 的 messages 做最终流式输出（可选：直接打印）。"""
    print("\nButler: ", end="", flush=True)
    reply = ""
    stream = await llm.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        reply += delta
    print()
    return reply


async def main():
    cfg = Config.from_env()
    system_prompt = Path("src/prompts/system.txt").read_text(encoding="utf-8")

    # CLI 立刻出现，记忆系统在后台加载
    print("=" * 40)
    print("AI Butler 已启动，输入 quit 退出")
    print("=" * 40)

    llm = AsyncOpenAI(
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
    )

    messages: list[dict] = []
    compressed_summary = ""
    reme = None
    executor = None

    while True:
        # 等待用户输入：Ctrl+C 取消本次输入并重新提示，不退出；Ctrl+D(EOFError) 才退出
        try:
            user_input = input("\n你: ").strip()
        except KeyboardInterrupt:
            print("\n（输入已取消，继续对话；输入 quit 退出）")
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
                llm_base_url=cfg.llm_base_url + "/chat/completions",
                embedding_api_key=cfg.emb_api_key,
                embedding_base_url=cfg.emb_base_url,
                default_as_llm_config={"model_name": cfg.llm_model},
                default_embedding_model_config={"model_name": cfg.emb_model},
                default_file_store_config={"fts_enabled": True, "vector_enabled": True},
                enable_load_env=False,
            )
            await reme.start()
            executor = _ToolExecutor(reme)
            TOOLS_LIST = _TOOLS
        else:
            from tools import TOOLS as TOOLS_LIST  # noqa: F811

        messages.append({"role": "user", "content": user_input})

        try:
            # Step 1: ReMe hook（短期记忆压缩 + 异步沉淀）
            # ReMe 需要 agentscope Msg 对象，转换后再调用，结果转回 dict
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

            # Step 2: 静默检索长期记忆（向量相似度过滤）
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

            # Step 5: Tool Call 循环（模型主动调工具）
            reply, new_msgs = await run_tool_call_loop(
                llm=llm,
                model=cfg.llm_model,
                messages=final_messages,
                executor=executor,
                tools=TOOLS_LIST,
            )

            # Step 6: 如果 tool call 循环已经给出了 reply，直接打印
            if reply:
                print(f"\nButler: {reply}")
            else:
                reply = await stream_final_reply(llm, cfg.llm_model, final_messages + new_msgs)

            # Step 7: 把本轮所有消息（含 tool call 往返）追加到工作记忆
            messages.extend(new_msgs)

        except KeyboardInterrupt:
            # LLM 处理中途按 Ctrl+C：取消本轮，移除未完成的 user 消息，继续等待
            print("\n（当前回复已中断，继续对话；输入 quit 退出）")
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            continue
        except Exception as e:
            print(f"\n[错误: {e}]", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # 出错时移除最后追加的 user 消息，避免历史污染
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            continue

    print("\n正在保存记忆...")
    try:
        if reme is not None:
            await reme.await_summary_tasks()
            await reme.close()
    except Exception:
        pass  # 退出时忽略清理错误
    print("再见！")


if __name__ == "__main__":
    asyncio.run(main())
