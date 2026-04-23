"""
channels/cli.py — 命令行消息渠道

把当前终端作为一个 Channel 接入 AI Butler。
通过 CliHook（AgentHook 子类）接收工具调用进度事件，驱动 ThinkingSpinner。

调用流程：
    hook = CliHook()
    butler = await Butler.create(cfg, channel="cli", hook=hook)
    await cli_run(hook=hook, send_stream_fn=app.send_stream)
"""
from __future__ import annotations

import sys
import threading
from typing import Callable, AsyncGenerator, Awaitable

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from agent.hooks import AgentHook
from cli.stream import safe_print, ThinkingSpinner, _STDOUT_LOCK


class CliHook(AgentHook):
    """
    CLI 专用 AgentHook。

    仅维护 spinner 生命周期。
    CLI 不再展示工具过程，只显示最终回复文本。
    """

    def __init__(self):
        self._spinner: ThinkingSpinner | None = None
        self._spinner_lock = threading.Lock()

    def set_spinner(self, spinner: ThinkingSpinner | None) -> None:
        with self._spinner_lock:
            self._spinner = spinner

    async def on_tool_start(self, name: str, args: dict) -> None:
        return None

    async def on_tool_end(self, name: str, result: str) -> None:
        return None


async def run(
    hook: CliHook,
    send_fn: Callable[[str], Awaitable[str]] | None = None,
    send_stream_fn: Callable[[str], AsyncGenerator[str, None]] | None = None,
) -> None:
    """
    启动命令行交互 session（流式输出优先）。

    Args:
        hook          : CliHook 实例，与 Butler 创建时传入的同一个对象。
        send_fn       : 非流式发送函数（当 send_stream_fn 未提供时使用）。
        send_stream_fn: 流式发送函数（优先使用），来自 AIButlerApp.send_stream。
    """
    session = PromptSession()

    with patch_stdout():
        safe_print("=" * 40)
        safe_print("AI Butler 已启动，输入 quit 退出")
        safe_print("=" * 40)

        try:
            while True:
                # ── 读取输入 ──────────────────────────────────────────────────
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

                # ── 流式输出 ──────────────────────────────────────────────────
                if send_stream_fn is not None:
                    sp = None
                    try:
                        sp = ThinkingSpinner()
                        sp.__enter__()
                        hook.set_spinner(sp)

                        first_token = True
                        gen = send_stream_fn(user_input)
                        async for token in gen:
                            if first_token:
                                # 第一个 token 到来：关闭 spinner，开始打印
                                # 关键：必须在 hook.set_spinner(None) 之前停止 spinner
                                hook.set_spinner(None)
                                sp.stop(clear=True)
                                with _STDOUT_LOCK:
                                    sys.stdout.write("\nButler: ")
                                    sys.stdout.flush()
                                first_token = False
                            with _STDOUT_LOCK:
                                sys.stdout.write(token)
                                sys.stdout.flush()

                        # 流式结束：如果收到了 token，打印换行；否则打印 "Butler: " 前缀
                        if not first_token:
                            with _STDOUT_LOCK:
                                sys.stdout.write("\n")
                                sys.stdout.flush()
                        else:
                            # 没有收到任何 token，仍需清除 spinner 并打印前缀
                            hook.set_spinner(None)
                            sp.stop(clear=True)
                            with _STDOUT_LOCK:
                                sys.stdout.write("\nButler: （无回复）\n")
                                sys.stdout.flush()

                        sp.__exit__(None, None, None)
                    except KeyboardInterrupt:
                        hook.set_spinner(None)
                        if sp:
                            sp.stop(clear=True)
                            sp.__exit__(None, None, None)
                        with _STDOUT_LOCK:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        safe_print("（当前回复已中断，继续对话；输入 quit 退出）")
                    except Exception as e:
                        hook.set_spinner(None)
                        if sp:
                            sp.stop(clear=True)
                            sp.__exit__(None, None, None)
                        safe_print(f"\n[错误: {e}]")
                        import traceback
                        traceback.print_exc()

                # ── 非流式兜底 ────────────────────────────────────────────────
                elif send_fn is not None:
                    sp = None
                    try:
                        sp = ThinkingSpinner()
                        sp.__enter__()
                        hook.set_spinner(sp)
                        reply = await send_fn(user_input)
                        hook.set_spinner(None)
                        sp.stop(clear=True)
                        sp.__exit__(None, None, None)
                        safe_print(f"\nButler: {reply}")
                    except KeyboardInterrupt:
                        hook.set_spinner(None)
                        if sp:
                            sp.stop(clear=True)
                            sp.__exit__(None, None, None)
                        safe_print("\n（当前回复已中断，继续对话；输入 quit 退出）")
                    except Exception as e:
                        hook.set_spinner(None)
                        if sp:
                            sp.stop(clear=True)
                            sp.__exit__(None, None, None)
                        safe_print(f"\n[错误: {e}]")
                        import traceback
                        traceback.print_exc()

        finally:
            hook.set_spinner(None)
            safe_print("\n正在保存记忆...")
            safe_print("再见！")
