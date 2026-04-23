# -*- coding: utf-8 -*-
"""
agent — AI Butler 核心 Agent 包

原 agent.py（669 行上帝类）已拆分为：
  - agent/runner.py   : AgentRunner — LLM ↔ Tool 执行循环
  - agent/context.py  : ContextBuilder — 上下文组装
  - agent/memory.py   : MemoryManager — 记忆系统封装
  - agent/hooks.py    : AgentHook / CompositeHook — 生命周期钩子

Butler 类现在是一个薄协调层，组合上述模块完成推理流程。

用法：
    hook = MyHook()
    butler = await Butler.create(cfg, channel="cli", hook=hook)
    reply  = await butler.chat(user_input)
    async for token in butler.chat_stream(user_input):
        ...
    await butler.close()
"""
from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, TYPE_CHECKING

from openai import AsyncOpenAI

from agent.hooks import AgentHook, CompositeHook
from agent.memory import MemoryManager
from agent.context import ContextBuilder
from agent.runner import AgentRunner
from event import StreamEvent, make_agent_event, to_stream_events

if TYPE_CHECKING:
    from config import Config
    from history import ChatHistory
    from tools import ToolDispatcher

__all__ = [
    "Butler",
    "wait_heavy_loaded",
    "AgentHook",
    "CompositeHook",
    "MemoryManager",
    "ContextBuilder",
    "AgentRunner",
    "StreamEvent",
]


# ── 后台预热：重型依赖在独立线程预加载 ───────────────────────────────────────
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
    _heavy_future.result()


# ── Butler：薄协调层 ──────────────────────────────────────────────────────────

class Butler:
    """
    AI Butler 核心推理引擎。

    组合 MemoryManager、ContextBuilder、AgentRunner 完成推理流程。
    Channel 层通过 AgentHook 接入生命周期事件（工具调用进度、流式 token 等）。
    """

    def __init__(
        self,
        cfg: "Config",
        memory: MemoryManager,
        history: "ChatHistory",
        dispatcher: "ToolDispatcher",
        runner: AgentRunner,
        context_builder: ContextBuilder,
        session_id: str = "",
        channel: str = "unknown",
    ):
        self._cfg = cfg
        self._memory = memory
        self._history = history
        self._dispatcher = dispatcher
        self._runner = runner
        self._context_builder = context_builder
        self.session_id = session_id
        self.channel = channel

        # 会话状态（跨轮次保留）
        self._messages: list[dict] = []
        self._compressed_summary: str = ""

    @classmethod
    async def create(
        cls,
        cfg: "Config",
        channel: str = "unknown",
        hook: AgentHook | None = None,
        session_id: str | None = None,
        initial_messages: list[dict] | None = None,
        initial_compressed_summary: str = "",
        session_title: str = "",
        memory_update_service: Any = None,
    ) -> "Butler":
        """
        工厂方法：初始化所有依赖组件，返回可用的 Butler 实例。

        Args:
            cfg     : 运行时配置
            channel : 渠道标识（"cli" / "feishu" / "wecom" / "api"）
            hook    : AgentHook 实例，用于接收工具调用/流式 token 等事件
        """
        session_id = session_id or str(uuid.uuid4())

        src_dir = Path(__file__).parent.parent
        system_prompt = (src_dir / "prompts" / "system.txt").read_text(encoding="utf-8")

        llm = AsyncOpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)

        memory = await MemoryManager.create(
            memory_dir=cfg.memory_dir,
            llm_api_key=cfg.llm_api_key,
            llm_base_url=cfg.llm_base_url,
            llm_model=cfg.llm_model,
            emb_api_key=cfg.emb_api_key,
            emb_base_url=cfg.emb_base_url,
            emb_model=cfg.emb_model,
            similarity_threshold=cfg.memory_similarity_threshold,
        )

        from history import ChatHistory
        history = ChatHistory(
            data_dir=cfg.memory_dir,
            session_id=session_id,
            channel=channel,
        )
        history.create_session(
            session_id=session_id,
            channel=channel,
            title=session_title,
        )

        command_executor = None
        if cfg.command_enabled:
            try:
                from tools.run_command_tool import CommandExecutor, CommandConfig
                command_executor = CommandExecutor(CommandConfig(
                    workdir=cfg.workspace_dir,
                    default_timeout=cfg.command_default_timeout,
                ))
            except Exception:
                pass

        browser_agent = None
        if cfg.browser_enabled:
            try:
                from tools.browser_use_tool import BrowserAgent, BrowserUseConfig
                browser_agent = BrowserAgent(BrowserUseConfig(
                    headless=cfg.browser_headless,
                    max_steps=cfg.browser_max_steps,
                    llm_model=cfg.llm_model,
                    llm_base_url=cfg.llm_base_url,
                    llm_api_key=cfg.llm_api_key,
                ))
            except Exception:
                pass

        from tools import ToolDispatcher
        dispatcher = ToolDispatcher(
            memory.reme,
            history=history,
            command_executor=command_executor,
            browser_agent=browser_agent,
            tool_call_dir=cfg.tool_call_dir,
            memory_update_service=memory_update_service,
        )

        runner = AgentRunner(
            llm=llm,
            model=cfg.llm_model,
            dispatcher=dispatcher,
            hook=hook,
        )

        context_builder = ContextBuilder(
            system_prompt=system_prompt,
            memory=memory,
        )

        inst = cls(
            cfg=cfg,
            memory=memory,
            history=history,
            dispatcher=dispatcher,
            runner=runner,
            context_builder=context_builder,
            session_id=session_id,
            channel=channel,
        )
        inst._memory_update_service = memory_update_service
        inst._messages = list(initial_messages or [])
        inst._compressed_summary = initial_compressed_summary or ""
        return inst

    async def chat(self, user_input: str) -> str:
        """
        处理一条用户消息，返回完整回复文本（非流式）。
        会话状态（messages / summary）自动跨轮次保留。
        """
        final_messages, self._messages, self._compressed_summary = (
            await self._context_builder.build_context(
                messages=self._messages,
                user_input=user_input,
                compressed_summary=self._compressed_summary,
                cfg=self._cfg,
            )
        )

        reply, new_msgs = await self._runner.run(final_messages)

        self._messages.extend(new_msgs)
        self._persist_turn(user_input=user_input, new_msgs=new_msgs, reply=reply)

        return reply

    async def chat_stream(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
        """
        处理一条用户消息，以 AsyncGenerator 逐 token yield 回复（流式）。

        yield 内容：
          - start / finish               消息生命周期
          - start-step / finish-step     单次 LLM step 生命周期
          - reasoning-*                  过程性思考文本
          - tool-input-* / tool-output-* 工具调用与结果
          - text-*                       最终回复正文
        """
        final_messages, self._messages, self._compressed_summary = (
            await self._context_builder.build_context(
                messages=self._messages,
                user_input=user_input,
                compressed_summary=self._compressed_summary,
                cfg=self._cfg,
            )
        )

        reply_parts: list[str] = []
        new_msgs: list[dict] = []
        message_id = f"msg_{uuid.uuid4().hex}"

        started = make_agent_event(
            kind="message.started",
            source="agent",
            conversation_id=self.session_id,
            message_id=message_id,
        )
        for stream_event in to_stream_events(started):
            yield stream_event

        async for item in self._runner.run_stream(
            final_messages,
            conversation_id=self.session_id,
            message_id=message_id,
        ):
            if isinstance(item, dict):
                for stream_event in to_stream_events(item):
                    yield stream_event
                if item["kind"] == "text.delta":
                    reply_parts.append(str(item["payload"].get("delta") or ""))
            else:
                new_msgs = item

        reply = "".join(reply_parts)
        finished = make_agent_event(
            kind="message.finished",
            source="agent",
            conversation_id=self.session_id,
            message_id=message_id,
            payload={"finishReason": "stop"},
        )
        for stream_event in to_stream_events(finished):
            yield stream_event

        self._messages.extend(new_msgs)
        self._persist_turn(user_input=user_input, new_msgs=new_msgs, reply=reply)

    def touch_session(self) -> None:
        """刷新会话活跃时间。"""
        self._history.touch_session(
            status="active",
            last_active_at=time.time(),
        )

    def snapshot_state(self) -> dict:
        """导出会话运行态，用于热态回收后的恢复。"""
        tail_history_rows = self._count_history_rows_in_runtime_messages(self._messages)
        return {
            "session_id": self.session_id,
            "compressed_summary": self._compressed_summary,
            "tail_messages": self._messages,
            "tail_history_rows": tail_history_rows,
            "summary_history_id": self._history.get_summary_boundary_id(tail_history_rows),
        }

    def _persist_turn(self, user_input: str, new_msgs: list[dict], reply: str) -> None:
        """
        将本轮 user / assistant / tool_call 持久化到 history，
        并更新 sessions 表里的恢复快照。
        """
        self._history.append("user", user_input)

        for msg in new_msgs:
            role = msg.get("role")
            content = msg.get("content") or ""

            if role == "assistant" and content:
                self._history.append("assistant", content)

            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    payload = {
                        "id": tc.get("id"),
                        "name": ((tc.get("function") or {}).get("name")),
                        "arguments": ((tc.get("function") or {}).get("arguments")),
                    }
                    self._history.append(
                        "tool_call",
                        json.dumps(payload, ensure_ascii=False),
                    )

        self._persist_session_state(preview=reply or user_input)
        if getattr(self, "_memory_update_service", None):
            self._memory_update_service.notify_new_messages()

    def _persist_session_state(self, preview: str = "") -> None:
        tail_history_rows = self._count_history_rows_in_runtime_messages(self._messages)
        title = self._suggest_session_title()
        self._history.touch_session(
            title=title,
            status="active",
            preview=(preview or "").strip()[:240],
            compressed_summary=self._compressed_summary,
            summary_history_id=self._history.get_summary_boundary_id(tail_history_rows),
            tail_messages_json=json.dumps(self._messages, ensure_ascii=False),
            last_active_at=time.time(),
            last_message_at=time.time(),
        )

    def _suggest_session_title(self) -> str:
        current = self._history.get_session() or {}
        existing = (current.get("title") or "").strip()
        if existing and existing != "新会话":
            return existing

        for msg in self._messages:
            if msg.get("role") == "user" and (msg.get("content") or "").strip():
                content = (msg.get("content") or "").strip().splitlines()[0]
                return (content[:30] + "…") if len(content) > 30 else content
        return existing or "新会话"

    @staticmethod
    def _count_history_rows_in_runtime_messages(messages: list[dict]) -> int:
        """
        估算当前 tail 在 history 表里对应多少条“可重建”的历史行。

        规则：
          - user / assistant(content) 各记 1 条
          - assistant.tool_calls 中的每个 tool_call 记 1 条 role=tool_call
          - tool 结果不入 history 恢复轨迹
        """
        rows = 0
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role in ("user", "assistant") and content:
                rows += 1
            if role == "assistant" and msg.get("tool_calls"):
                rows += len(msg["tool_calls"])
        return rows

    async def close(self):
        """退出时调用：沉淀记忆、关闭资源。"""
        self._history.touch_session(
            status="idle",
            compressed_summary=self._compressed_summary,
            summary_history_id=self._history.get_summary_boundary_id(
                self._count_history_rows_in_runtime_messages(self._messages)
            ),
            tail_messages_json=json.dumps(self._messages, ensure_ascii=False),
            last_active_at=time.time(),
        )
        if self._dispatcher.browser_agent:
            try:
                await self._dispatcher.browser_agent.close()
            except Exception:
                pass

        await self._memory.settle(self._messages)
        await self._memory.close()
        self._history.close()
