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
- CLI Channel 额外持有 butler 引用，用于注册工具回调（ThinkingSpinner 进度显示）
- Gateway 模式使用 uvicorn 的异步启动，与 Agent 主循环共享同一事件循环
- 两种模式均以 asyncio.run(app.run(...)) 驱动，入口统一

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
from typing import Callable, Awaitable

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
    所有 Channel 通过 send() 发送消息，Agent 主循环串行处理并返回回复。

    使用方式（各 Channel）：
        reply = await app.send("用户消息")
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.butler = None          # 由 run() 初始化
        self._inbox: asyncio.Queue[tuple[str, asyncio.Future[str], asyncio.Queue | None]] = asyncio.Queue()
        self._running = False

    # ── 对外接口（Channel 调用） ────────────────────────────────────────────────

    async def send(self, text: str) -> str:
        """
        Channel 调用此方法向 Agent 发送消息，阻塞至收到完整回复后返回（非流式）。

        线程安全：多个 Channel 可同时调用，消息会按到达顺序串行处理。
        """
        loop    = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        await self._inbox.put((text, future, None))
        return await future

    async def send_stream(self, text: str):
        """
        Channel 调用此方法向 Agent 发送消息，以 AsyncGenerator 逐 token yield 回复（流式）。

        实现方式：
        - 向 inbox 放入消息时附带一个 asyncio.Queue 作为 token 通道。
        - Agent 主循环检测到 token_queue 不为 None 时，调用 butler.chat_stream()
          并将每个 token 放入 token_queue；结束时放入 sentinel（None）。
        - 本方法从 token_queue 逐个取出 token 并 yield 给调用方。
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        await self._inbox.put((text, future, token_queue))

        # 逐 token yield（直到 sentinel None）
        while True:
            token = await token_queue.get()
            if token is None:
                break
            yield token

        # 等待 future 确保 agent loop 完成（异常也会在此抛出）
        await future

    # ── Agent 主循环 ────────────────────────────────────────────────────────────

    async def _agent_loop(self) -> None:
        """
        后台协程：串行消费 inbox，保证 Butler 内部对话历史的一致性。

        inbox 中的每条消息格式为 (text, future, token_queue)：
        - token_queue 为 None  → 非流式，调用 butler.chat()，结果写入 future。
        - token_queue 不为 None → 流式，调用 butler.chat_stream()，
          逐 token 放入 token_queue，结束放 sentinel(None)，future 写入完整回复。
        """
        while self._running:
            try:
                item = await asyncio.wait_for(self._inbox.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            text, future, token_queue = item

            try:
                if token_queue is None:
                    # 非流式
                    reply = await self.butler.chat(text)
                    if not future.done():
                        future.set_result(reply)
                else:
                    # 流式：逐事件推送到 token_queue
                    # \x00 开头的是工具调用事件 marker，不计入最终回复文本
                    reply_parts: list[str] = []
                    async for token in self.butler.chat_stream(text):
                        if not token.startswith("\x00"):
                            reply_parts.append(token)
                        await token_queue.put(token)
                    await token_queue.put(None)  # sentinel
                    reply = "".join(reply_parts)
                    if not future.done():
                        future.set_result(reply)
            except Exception as exc:
                if token_queue is not None:
                    await token_queue.put(None)  # 异常时也要结束 sentinel
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._inbox.task_done()

    # ── 主入口 ─────────────────────────────────────────────────────────────────

    async def run(self, mode: str, host: str = "0.0.0.0", port: int = 8080) -> None:
        """
        启动 App：初始化 Butler，启动 Agent 主循环，然后按模式挂起对应 Channel。

        退出条件：
        - cli 模式：用户输入 quit / Ctrl-D / Ctrl-C
        - gateway 模式：uvicorn 收到 SIGTERM / SIGINT
        """
        from agent import Butler, wait_heavy_loaded

        if mode == "cli":
            from cli.stream import safe_print
            wait_heavy_loaded(on_waiting=lambda: safe_print("（正在初始化记忆系统，请稍候…）"))
        else:
            wait_heavy_loaded()
        self.butler   = await Butler.create(self.cfg, channel=mode)
        self._running = True

        loop_task = asyncio.create_task(self._agent_loop(), name="agent-loop")

        try:
            if mode == "cli":
                await self._start_cli()
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

    async def _start_cli(self) -> None:
        """
        启动 CLI Channel（当前终端）。

        CLI 既持有 butler 引用（注册 ThinkingSpinner 工具回调），
        又通过 self.send_stream() 走 inbox 队列流式发送消息，逐 token 打印。
        """
        from channels.cli import run as cli_run
        await cli_run(
            self.butler,
            send_fn=self.send,
            send_stream_fn=self.send_stream,
        )

    async def _start_gateway(self, host: str, port: int) -> None:
        """
        启动 Gateway Channel：常驻 HTTP / WebSocket 服务。

        uvicorn 以异步模式（Server.serve()）启动，
        与 Agent 主循环共享同一个 asyncio 事件循环。
        """
        try:
            import uvicorn
        except ImportError:
            print(
                "[错误] gateway 模式需要 uvicorn，请运行：\n"
                "  pip install 'uvicorn[standard]'",
                file=sys.stderr,
            )
            raise SystemExit(1)

        # 把本 App 注册到 gateway.server，供路由处理器调用 app.send()
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
        await server.serve()   # 异步阻塞，直到 SIGTERM/SIGINT


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
