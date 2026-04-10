"""
agent.py — 核心 Agent（Butler）

负责一次完整的"用户输入 → 回复"推理流程，与输入来源无关：
  1. ReMe pre_reasoning_hook（短期记忆压缩 + 异步沉淀）
  2. 静默检索长期记忆，相似度 >= 阈值才注入 system
  3. 读取 MEMORY.md（用户画像）
  4. 组装 messages（skills.assembler.build）
  5. Tool Call 循环（并发批处理）
  6. 返回最终回复文本

调用方（channels.cli / channels.feishu / gateway 等）只需：
  butler = await Butler.create(cfg)
  reply  = await butler.chat(user_input)

会话状态（messages、compressed_summary）由 Butler 内部维护，
Channel 层完全不感知记忆、工具、LLM 细节。
"""
import asyncio
import threading
import concurrent.futures
import uuid
import warnings
from pathlib import Path
from typing import Callable

from openai import AsyncOpenAI

from config import Config
from history import ChatHistory
from assembler import build
from tools import TOOL_CONCURRENT_SAFE, ToolDispatcher

# ── 常量 ───────────────────────────────────────────────────────────────────────
MAX_TOOL_ROUNDS      = 6     # tool call 最大轮次（防止死循环）
PASSIVE_RECALL_K     = 8     # 静默检索候选条数
RECENT_RESERVE_CHARS = 2000  # 退出时保留最近 N 字符不压缩

# ── 后台预热：重型依赖（chromadb / agentscope / reme）在独立线程预加载 ─────────
warnings.filterwarnings(
    "ignore",
    message="'asyncio.iscoroutinefunction' is deprecated",
    category=DeprecationWarning,
    module="chromadb",
)

_heavy_future: concurrent.futures.Future = concurrent.futures.Future()


def _load_heavy():
    try:
        from reme.reme_light import ReMeLight   # noqa: F401
        from agentscope.message import Msg      # noqa: F401
        _heavy_future.set_result(True)
    except Exception as e:
        _heavy_future.set_exception(e)


threading.Thread(target=_load_heavy, daemon=True, name="heavy-import").start()


def wait_heavy_loaded(on_waiting: Callable[[], None] | None = None) -> None:
    """阻塞直到重型模块加载完毕。on_waiting 在等待期间被调用一次（用于显示提示）。"""
    if not _heavy_future.done() and on_waiting:
        on_waiting()
    _heavy_future.result()   # 有异常会在此 re-raise


# ── ReMe Msg ↔ dict 转换 ───────────────────────────────────────────────────────

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


# ── LLM 调用（带限流重试） ─────────────────────────────────────────────────────

async def _llm_call(llm: AsyncOpenAI, **kwargs):
    """指数退避重试，处理 429 限流。最多 3 次，间隔 2→4→8s。"""
    from openai import RateLimitError
    wait = 2
    for attempt in range(3):
        try:
            return await llm.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == 2:
                raise
            await asyncio.sleep(wait)
            wait = min(wait * 2, 32)


# ── 核心 Agent ─────────────────────────────────────────────────────────────────

class Butler:
    """
    AI Butler 核心推理引擎。

    与输入来源（CLI / 企微 / 飞书 / HTTP）完全解耦，
    所有 Channel 只需调用 `await butler.chat(text)` 获取回复。

    on_tool_call / on_tool_result 是可选回调，由各 Channel 在拿到实例后自行注入：
        butler.on_tool_call   = lambda name: ...
        butler.on_tool_result = lambda name, result: ...
    """

    def __init__(
        self,
        cfg: Config,
        system_prompt: str,
        llm: AsyncOpenAI,
        reme,
        history: ChatHistory,
        dispatcher: ToolDispatcher,
        session_id: str = "",
        channel: str = "unknown",
        on_tool_call: Callable[[str], None] | None = None,
        on_tool_result: Callable[[str, str], None] | None = None,
    ):
        self._cfg            = cfg
        self._system_prompt  = system_prompt
        self._llm            = llm
        self._reme           = reme
        self._history        = history
        self._dispatcher     = dispatcher
        self.session_id      = session_id   # 供 Channel 层展示或追踪
        self.channel         = channel
        self.on_tool_call    = on_tool_call  # channel 注入的进度回调
        self.on_tool_result  = on_tool_result

        # 会话状态（跨轮次保留）
        self._messages: list[dict] = []
        self._compressed_summary: str = ""

    @classmethod
    async def create(cls, cfg: Config, channel: str = "unknown") -> "Butler":
        """
        工厂方法：初始化所有重型依赖，返回可用的 Butler 实例。

        Args:
            cfg     : 运行时配置
            channel : 渠道标识，如 "cli" / "feishu" / "wecom" / "api"，
                      用于对话日志的渠道标记，默认 "unknown"

        回调（on_tool_call / on_tool_result）由各 Channel 在拿到实例后自行注入：
            butler.on_tool_call   = lambda name: ...
            butler.on_tool_result = lambda name, result: ...
        """
        session_id = str(uuid.uuid4())
        from reme.reme_light import ReMeLight

        src_dir = Path(__file__).parent
        system_prompt = (src_dir / "prompts" / "system.txt").read_text(encoding="utf-8")

        llm = AsyncOpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)

        reme = ReMeLight(
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

        history = ChatHistory(
            data_dir=cfg.working_dir,
            session_id=session_id,
            channel=channel,
        )

        # 可选工具初始化
        command_executor = None
        if cfg.command_enabled:
            try:
                from tools.command import CommandExecutor, CommandConfig
                command_executor = CommandExecutor(CommandConfig(
                    workdir=cfg.command_workdir,
                    default_timeout=cfg.command_default_timeout,
                ))
            except Exception:
                pass

        browser_agent = None
        if cfg.browser_enabled:
            try:
                from tools.browser import BrowserAgent, BrowserUseConfig
                browser_agent = BrowserAgent(BrowserUseConfig(
                    headless=cfg.browser_headless,
                    max_steps=cfg.browser_max_steps,
                    llm_model=cfg.llm_model,
                    llm_base_url=cfg.llm_base_url,
                    llm_api_key=cfg.llm_api_key,
                ))
            except Exception:
                pass

        dispatcher = ToolDispatcher(
            reme,
            history=history,
            command_executor=command_executor,
            browser_agent=browser_agent,
            working_dir=cfg.working_dir,
        )

        return cls(
            cfg=cfg,
            system_prompt=system_prompt,
            llm=llm,
            reme=reme,
            history=history,
            dispatcher=dispatcher,
            session_id=session_id,
            channel=channel,
        )

    async def chat(self, user_input: str) -> str:
        """
        处理一条用户消息，返回 Butler 的回复文本。
        会话状态（messages / summary）自动跨轮次保留。
        """
        self._messages.append({"role": "user", "content": user_input})

        # Step 1: ReMe hook（短期记忆压缩 + 异步沉淀）
        msgs_for_reme = _dicts_to_msgs(self._messages)
        msgs_for_reme, self._compressed_summary = await self._reme.pre_reasoning_hook(
            messages=msgs_for_reme,
            system_prompt=self._system_prompt,
            compressed_summary=self._compressed_summary,
            max_input_length=self._cfg.max_input_length,
            compact_ratio=self._cfg.compact_ratio,
            memory_compact_reserve=self._cfg.memory_compact_reserve,
            enable_tool_result_compact=True,
        )
        self._messages = _msgs_to_dicts(msgs_for_reme)

        # Step 2: 静默检索长期记忆
        passive_snippets = await self._passive_recall(user_input)

        # Step 3: 读取 MEMORY.md（用户画像）
        memory_md = self._read_memory_md()

        # Step 4: 组装 context
        final_messages = build(
            system_prompt=self._system_prompt,
            history=self._messages,
            memory_md=memory_md,
            retrieval_snippets=passive_snippets or None,
        )

        # Step 5: Tool Call 循环
        reply, new_msgs = await self._tool_call_loop(final_messages)

        # Step 6: 追加到工作记忆
        self._messages.extend(new_msgs)

        # Step 7: 持久化对话日志
        self._history.append("user", user_input)
        if reply:
            self._history.append("assistant", reply)

        return reply

    async def close(self):
        """退出时调用：沉淀记忆、关闭资源。"""
        # 关闭浏览器
        if self._dispatcher.browser_agent:
            try:
                await self._dispatcher.browser_agent.close()
            except Exception:
                pass

        # 主动沉淀当前对话到长期记忆
        clean_msgs = _dicts_to_msgs(self._messages)
        tail_chars, split_idx = 0, 0
        for i in range(len(clean_msgs) - 1, -1, -1):
            tail_chars += len(clean_msgs[i].content or "")
            if tail_chars >= RECENT_RESERVE_CHARS:
                split_idx = i
                break
        if msgs_to_summarize := clean_msgs[:split_idx]:
            self._reme.add_async_summary_task(messages=msgs_to_summarize)
        await self._reme.await_summary_tasks()
        await self._reme.close()

        self._history.close()

    # ── 内部方法 ───────────────────────────────────────────────────────────────

    async def _passive_recall(self, query: str) -> list[str]:
        """静默向量检索，不经模型决策，相似度低于阈值直接过滤。"""
        try:
            result = await self._reme.memory_search(
                query,
                max_results=PASSIVE_RECALL_K,
                min_score=self._cfg.memory_similarity_threshold,
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

    def _read_memory_md(self) -> str:
        """读取 MEMORY.md 用户画像文件，不存在时返回空串。"""
        try:
            f = Path(self._cfg.working_dir) / "MEMORY.md"
            return f.read_text(encoding="utf-8") if f.exists() else ""
        except Exception:
            return ""

    async def _tool_call_loop(
        self,
        messages: list[dict],
    ) -> tuple[str, list[dict]]:
        """
        Tool Call 循环。
        返回 (最终回复文本, 本轮新增消息列表)。
        批内并行，批间串行；超过 MAX_TOOL_ROUNDS 强制不带工具再请求一次。
        """
        new_messages: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            response = await _llm_call(
                self._llm,
                model=self._cfg.llm_model,
                messages=messages + new_messages,
                tools=self._dispatcher.tools,
                tool_choice="auto",
                stream=False,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                reply = msg.content or ""
                new_messages.append({"role": "assistant", "content": reply})
                return reply, new_messages

            # 记录 tool_call 消息
            new_messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            # 通知 channel 层（打印进度 / 推送中间状态）
            tool_names = [tc.function.name for tc in msg.tool_calls]
            for name in tool_names:
                if self.on_tool_call:
                    self.on_tool_call(name)
            self._history.append("tool_call", "、".join(tool_names))

            # 按并发安全性分批：相邻安全工具合并并行，不安全工具独占串行
            batches: list[list] = []
            for tc in msg.tool_calls:
                safe = TOOL_CONCURRENT_SAFE.get(tc.function.name, False)
                if safe and batches and all(
                    TOOL_CONCURRENT_SAFE.get(t.function.name, False) for t in batches[-1]
                ):
                    batches[-1].append(tc)
                else:
                    batches.append([tc])

            async def _exec(tc):
                result = await self._dispatcher.run(tc.function.name, tc.function.arguments)
                if self.on_tool_result:
                    self.on_tool_result(tc.function.name, result)
                return tc.id, result

            for batch in batches:
                if len(batch) == 1:
                    tid, result = await _exec(batch[0])
                    new_messages.append({"role": "tool", "tool_call_id": tid, "content": result})
                else:
                    for tid, result in await asyncio.gather(*[_exec(tc) for tc in batch]):
                        new_messages.append({"role": "tool", "tool_call_id": tid, "content": result})

        # 超过最大轮次，强制不带工具再请求
        response = await _llm_call(
            self._llm,
            model=self._cfg.llm_model,
            messages=messages + new_messages,
            stream=False,
        )
        reply = response.choices[0].message.content or ""
        new_messages.append({"role": "assistant", "content": reply})
        return reply, new_messages
