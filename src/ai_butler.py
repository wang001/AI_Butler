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
import sys
import warnings
from pathlib import Path

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


# ── AIButlerApp ────────────────────────────────────────────────────────────────

class AIButlerApp:
    """
    AI Butler 应用主体。

    持有唯一的 Butler 实例和一个消息队列（inbox）。
    所有 Channel 通过 send() / send_stream() 发送消息，Agent 主循环串行处理。
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.butler = None
        self._inbox: asyncio.Queue[tuple[str, asyncio.Future[str], asyncio.Queue | None]] = asyncio.Queue()
        self._running = False

    # ── 对外接口（Channel 调用） ────────────────────────────────────────────────

    async def send(self, text: str) -> str:
        """非流式：阻塞至收到完整回复后返回。"""
        loop   = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        await self._inbox.put((text, future, None))
        return await future

    async def send_stream(self, text: str):
        """
        流式：以 AsyncGenerator 逐 token yield 回复。

        普通 token 直接 yield；\\x00 开头的工具事件 marker 也透传，
        由各 Channel 自行决定是否消费（CLI 通过 CliHook 已处理，可忽略）。
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        await self._inbox.put((text, future, token_queue))

        while True:
            token = await token_queue.get()
            if token is None:
                break
            yield token

        await future

    # ── Agent 主循环 ────────────────────────────────────────────────────────────

    async def _agent_loop(self) -> None:
        """
        后台协程：串行消费 inbox，保证 Butler 内部对话历史的一致性。
        """
        while self._running:
            try:
                item = await asyncio.wait_for(self._inbox.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            text, future, token_queue = item

            try:
                if token_queue is None:
                    reply = await self.butler.chat(text)
                    if not future.done():
                        future.set_result(reply)
                else:
                    reply_parts: list[str] = []
                    async for token in self.butler.chat_stream(text):
                        if not token.startswith("\x00"):
                            reply_parts.append(token)
                        await token_queue.put(token)
                    await token_queue.put(None)  # sentinel
                    if not future.done():
                        future.set_result("".join(reply_parts))
            except Exception as exc:
                if token_queue is not None:
                    await token_queue.put(None)
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._inbox.task_done()

    # ── 主入口 ─────────────────────────────────────────────────────────────────

    async def run(self, mode: str, host: str = "0.0.0.0", port: int = 8080) -> None:
        """
        启动 App：初始化 Butler，启动 Agent 主循环，然后按模式挂起对应 Channel。
        """
        from agent import Butler, wait_heavy_loaded

        if mode == "cli":
            from channels.cli import CliHook
            from cli.stream import safe_print
            wait_heavy_loaded(on_waiting=lambda: safe_print("（正在初始化记忆系统，请稍候…）"))
            cli_hook = CliHook()
            self.butler = await Butler.create(self.cfg, channel="cli", hook=cli_hook)
        else:
            wait_heavy_loaded()
            self.butler = await Butler.create(self.cfg, channel=mode)
            cli_hook = None

        self._running = True
        loop_task = asyncio.create_task(self._agent_loop(), name="agent-loop")

        memory_cron = getattr(self.butler, "_memory_update_service", None)
        if memory_cron:
            memory_cron.start()

        try:
            if mode == "cli":
                await self._start_cli(cli_hook)
            elif mode == "gateway":
                await self._start_gateway(host, port)
        finally:
            self._running = False
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass
            await self.butler.close()

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
