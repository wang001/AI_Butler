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
import threading
import uuid
import warnings
from pathlib import Path
from typing import AsyncGenerator, Callable, TYPE_CHECKING

from openai import AsyncOpenAI

from agent.hooks import AgentHook, CompositeHook
from agent.memory import MemoryManager
from agent.context import ContextBuilder
from agent.runner import AgentRunner

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
    ) -> "Butler":
        """
        工厂方法：初始化所有依赖组件，返回可用的 Butler 实例。

        Args:
            cfg     : 运行时配置
            channel : 渠道标识（"cli" / "feishu" / "wecom" / "api"）
            hook    : AgentHook 实例，用于接收工具调用/流式 token 等事件
        """
        session_id = str(uuid.uuid4())

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

        from cron import MemoryUpdateService
        memory_update_service = MemoryUpdateService(
            cfg=cfg,
            history=history,
            reme=memory.reme,
            llm_model=cfg.llm_model,
        )

        command_executor = None
        if cfg.command_enabled:
            try:
                from tools.command import CommandExecutor, CommandConfig
                command_executor = CommandExecutor(CommandConfig(
                    workdir=cfg.workspace_dir,
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
        self._history.append("user", user_input)
        if reply:
            self._history.append("assistant", reply)

        return reply

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        处理一条用户消息，以 AsyncGenerator 逐 token yield 回复（流式）。

        yield 内容：
          - 普通字符串：回复文本 token
          - \\x00TOOL_CALL:{json}    工具开始调用事件
          - \\x00TOOL_RESULT:{json}  工具执行完毕事件
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

        async for item in self._runner.run_stream(final_messages):
            if isinstance(item, str):
                yield item
                if not item.startswith("\x00"):
                    reply_parts.append(item)
            else:
                new_msgs = item

        reply = "".join(reply_parts)

        self._messages.extend(new_msgs)
        self._history.append("user", user_input)
        if reply:
            self._history.append("assistant", reply)

    async def close(self):
        """退出时调用：沉淀记忆、关闭资源。"""
        if self._dispatcher.browser_agent:
            try:
                await self._dispatcher.browser_agent.close()
            except Exception:
                pass

        if getattr(self, "_memory_update_service", None):
            await self._memory_update_service.stop()

        await self._memory.settle(self._messages)
        await self._memory.close()
        self._history.close()
