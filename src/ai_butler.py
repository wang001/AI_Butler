"""
ai_butler.py — 统一启动入口 & 应用核心

架构概览
--------
                ┌──────────────────────────────────┐
                │         AIButlerApp              │
                │                                  │
  CLI Channel ──┤──► inbox (asyncio.Queue)         │
  HTTP Channel ─┤           │                      │
  飞书 Channel ──┤       Agent 主循环               │
                │       (串行消费)                  │
                │           │                      │
                │       Butler.chat()              │
                └──────────────────────────────────┘

设计要点：
- 所有 Channel 通过 app.send(text) 向 inbox 压消息，等待 Future 返回回复
- Agent 主循环串行消费 inbox，保证 Butler 内部状态（对话历史）线程安全
- CLI Channel 通过 CliHook（AgentHook 子类）接收工具进度事件驱动 ThinkingSpinner
- Gateway 模式使用 uvicorn 的异步启动，与 Agent 主循环共享同一事件循环

启动方式
--------
  python ai_butler.py                          # CLI 对话（默认）
  python ai_butler.py --mode cli               # CLI 对话（显式）
  python ai_butler.py --mode gateway           # Gateway 常驻服务
  python ai_butler.py --mode gateway --port 9000 --host 127.0.0.1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

# ── sys.path：兼容直接运行和容器内 PYTHONPATH=/app/src ────────────────────────
_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

warnings.filterwarnings(
    "ignore",
    message="'asyncio.iscoroutinefunction' is deprecated",
    category=DeprecationWarning,
    module="chromadb",
)

from dotenv import load_dotenv
load_dotenv()

from config import Config

if TYPE_CHECKING:
    from agent import Butler
    from agent.stream_events import StreamEvent


# ── AIButlerApp ────────────────────────────────────────────────────────────────
_DEFAULT_SESSION_KEY = "__default__"
_WEB_SESSION_IDLE_TTL = 3600
_WEB_SESSION_SWEEP_INTERVAL = 60


@dataclass
class _Runtime:
    session_id: str
    channel: str
    butler: "Butler"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_active_at: float = field(default_factory=time.time)


class AIButlerApp:
    """
    AI Butler 应用主体。

    按会话维护多个 runtime，每个 runtime 用一把 asyncio.Lock 保证串行访问。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._runtimes: dict[str, _Runtime] = {}
        self._runtimes_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._memory_update_service = None
        self._default_channel = "unknown"
        self._running = False

    # ── 对外接口（Channel 调用） ────────────────────────────────────────────────

    async def send(
        self,
        text: str,
        *,
        session_id: str | None = None,
        channel: str | None = None,
    ) -> str:
        """非流式：阻塞至收到完整回复后返回。"""
        runtime = await self._get_or_create_runtime(
            session_id=session_id,
            channel=channel or self._default_channel,
        )
        async with runtime.lock:
            runtime.last_active_at = time.time()
            runtime.butler.touch_session()
            return await runtime.butler.chat(text)

    async def send_stream(
        self,
        text: str,
        *,
        session_id: str | None = None,
        channel: str | None = None,
    ):
        """
        纯文本流：仅逐段产出最终回复正文。

        CLI 使用这个接口，忽略 reasoning/tool 事件，只关心最终回复文本。
        """
        runtime = await self._get_or_create_runtime(
            session_id=session_id,
            channel=channel or self._default_channel,
        )
        async with runtime.lock:
            runtime.last_active_at = time.time()
            runtime.butler.touch_session()
            async for event in runtime.butler.chat_stream(text):
                if event["type"] == "text-delta":
                    yield event["delta"]

    async def send_event_stream(
        self,
        text: str,
        *,
        session_id: str | None = None,
        channel: str | None = None,
    ):
        """
        结构化事件流：逐条产出 reasoning/tool/text 事件。

        Web 渠道使用这个接口，以便精细渲染过程与最终回复。
        """
        runtime = await self._get_or_create_runtime(
            session_id=session_id,
            channel=channel or "web",
        )
        async with runtime.lock:
            runtime.last_active_at = time.time()
            runtime.butler.touch_session()
            async for event in runtime.butler.chat_stream(text):
                yield event

    def list_sessions(self, *, channel: str = "web", limit: int = 50) -> list[dict]:
        """列出指定 channel 的会话。"""
        store = self._open_session_store()
        try:
            return store.list_sessions(limit=limit, channel=channel)
        finally:
            store.close()

    def get_session(self, session_id: str) -> dict | None:
        """读取单个会话元数据。"""
        store = self._open_session_store()
        try:
            return store.get_session(session_id)
        finally:
            store.close()

    def get_session_messages(self, session_id: str, limit: int = 200) -> list[dict]:
        """读取会话历史消息。"""
        store = self._open_session_store()
        try:
            return store.get_session_messages(session_id=session_id, limit=limit)
        finally:
            store.close()

    async def create_session(
        self,
        *,
        channel: str = "web",
        title: str = "新会话",
    ) -> dict:
        """创建一个新的持久化会话元记录。"""
        sid = str(uuid.uuid4())
        store = self._open_session_store()
        try:
            store.create_session(
                session_id=sid,
                channel=channel,
                title=title,
            )
            return store.get_session(sid) or {
                "id": sid,
                "title": title,
                "channel": channel,
                "status": "active",
            }
        finally:
            store.close()

    async def _get_or_create_runtime(
        self,
        *,
        session_id: str | None,
        channel: str,
    ) -> _Runtime:
        from agent import Butler

        sid = session_id or _DEFAULT_SESSION_KEY
        async with self._runtimes_lock:
            existing = self._runtimes.get(sid)
            if existing is not None:
                existing.last_active_at = time.time()
                return existing

            store = self._open_session_store()
            try:
                session_meta = store.get_session(sid)
            finally:
                store.close()
            initial_messages: list[dict] | None = None
            initial_compressed_summary = ""
            session_title = ""

            if session_meta is not None:
                session_title = session_meta.get("title") or ""
                initial_compressed_summary = session_meta.get("compressed_summary") or ""
                tail_messages_json = session_meta.get("tail_messages_json") or "[]"
                try:
                    decoded = json.loads(tail_messages_json)
                    if isinstance(decoded, list):
                        initial_messages = decoded
                except Exception:
                    initial_messages = None

            butler = await Butler.create(
                self.cfg,
                channel=channel,
                session_id=sid,
                initial_messages=initial_messages,
                initial_compressed_summary=initial_compressed_summary,
                session_title=session_title or ("新会话" if channel == "web" else ""),
                memory_update_service=self._memory_update_service,
            )

            runtime = _Runtime(
                session_id=sid,
                channel=channel,
                butler=butler,
            )
            self._runtimes[sid] = runtime
            return runtime

    def _open_session_store(self):
        from history import ChatHistory

        return ChatHistory(
            data_dir=self.cfg.memory_dir,
            session_id="__app__",
            channel="meta",
        )

    def _create_memory_update_service(self):
        from cron import MemoryUpdateService
        from history import ChatHistory

        history = ChatHistory(
            data_dir=self.cfg.memory_dir,
            session_id="__memory_updater__",
            channel="meta",
        )
        return MemoryUpdateService(
            cfg=self.cfg,
            history=history,
            reme=None,
            llm_model=self.cfg.llm_model,
        )

    async def _cleanup_idle_runtimes(self) -> None:
        while self._running:
            await asyncio.sleep(_WEB_SESSION_SWEEP_INTERVAL)
            now = time.time()
            stale_ids: list[str] = []

            async with self._runtimes_lock:
                for sid, runtime in self._runtimes.items():
                    if sid == _DEFAULT_SESSION_KEY:
                        continue
                    if runtime.channel != "web":
                        continue
                    if runtime.lock.locked():
                        continue
                    if now - runtime.last_active_at >= _WEB_SESSION_IDLE_TTL:
                        stale_ids.append(sid)

            for sid in stale_ids:
                await self._evict_runtime(sid)

    async def _evict_runtime(self, session_id: str) -> None:
        async with self._runtimes_lock:
            runtime = self._runtimes.pop(session_id, None)
        if runtime is None:
            return
        await runtime.butler.close()

    # ── 主入口 ─────────────────────────────────────────────────────────────────

    async def run(self, mode: str, host: str = "0.0.0.0", port: int = 8080) -> None:
        """
        启动 App：初始化会话存储，并按模式挂起对应 Channel。
        """
        from agent import Butler, wait_heavy_loaded

        self._memory_update_service = self._create_memory_update_service()
        self._memory_update_service.start()

        if mode == "cli":
            from channels.cli import CliHook
            from cli.stream import safe_print
            wait_heavy_loaded(on_waiting=lambda: safe_print("（正在初始化记忆系统，请稍候…）"))
            cli_hook = CliHook()
            self._default_channel = "cli"
            butler = await Butler.create(
                self.cfg,
                channel="cli",
                hook=cli_hook,
                session_id=_DEFAULT_SESSION_KEY,
                memory_update_service=self._memory_update_service,
            )
            self._runtimes[_DEFAULT_SESSION_KEY] = _Runtime(
                session_id=_DEFAULT_SESSION_KEY,
                channel="cli",
                butler=butler,
            )
        else:
            wait_heavy_loaded()
            self._default_channel = mode
            cli_hook = None

        self._running = True
        if mode == "gateway":
            self._cleanup_task = asyncio.create_task(
                self._cleanup_idle_runtimes(),
                name="web-session-cleanup",
            )

        try:
            if mode == "cli":
                await self._start_cli(cli_hook)
            elif mode == "gateway":
                await self._start_gateway(host, port)
        finally:
            self._running = False
            if self._cleanup_task is not None:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            async with self._runtimes_lock:
                runtimes = list(self._runtimes.values())
                self._runtimes.clear()

            for runtime in runtimes:
                await runtime.butler.close()

            if self._memory_update_service is not None:
                await self._memory_update_service.stop()
                self._memory_update_service = None

    # ── Channel 启动器 ─────────────────────────────────────────────────────────

    async def _start_cli(self, cli_hook) -> None:
        """启动 CLI Channel。"""
        from channels.cli import run as cli_run
        await cli_run(
            hook=cli_hook,
            send_fn=self.send,
            send_stream_fn=self.send_stream,
        )

    async def _start_gateway(self, host: str, port: int) -> None:
        """启动 Gateway Channel：常驻 HTTP / WebSocket 服务。"""
        try:
            import uvicorn
        except ImportError:
            print(
                "[错误] gateway 模式需要 uvicorn，请运行：\n"
                "  pip install 'uvicorn[standard]'",
                file=sys.stderr,
            )
            raise SystemExit(1)

        from gateway import server as gw_server
        gw_server.set_app(self)

        print(f"[AI Butler Gateway] http://{host}:{port}")
        print(f"  GET  /          网页对话界面")
        print(f"  GET  /health    健康检查")
        print(f"  POST /api/chat  REST 问答")
        print(f"  WS   /api/ws    WebSocket 流式对话")
        print(f"  POST /feishu    飞书 Webhook")

        config = uvicorn.Config(
            "gateway.server:app",
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()


# ── CLI 参数解析 ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai_butler",
        description="AI Butler — 个人 AI 管家",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python ai_butler.py                               # CLI 对话（默认）
  python ai_butler.py --mode cli                    # CLI 对话（显式）
  python ai_butler.py --mode gateway                # Gateway 服务（0.0.0.0:8080）
  python ai_butler.py --mode gateway --port 9000    # 自定义端口
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "gateway"],
        default="cli",
        help="运行模式：cli（交互终端）或 gateway（常驻 HTTP 服务），默认 cli",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="[gateway 专用] 监听地址，默认 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="[gateway 专用] 监听端口，默认 8080",
    )
    return parser.parse_args()


# ── 主入口 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    cfg  = Config.from_env()
    app  = AIButlerApp(cfg)
    asyncio.run(app.run(mode=args.mode, host=args.host, port=args.port))


if __name__ == "__main__":
    main()
