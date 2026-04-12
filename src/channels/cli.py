"""
channels/cli.py — 命令行消息渠道

把当前终端作为一个 Channel 接入 AI Butler。
与飞书、HTTP 等渠道统一的调用模型：

    async for token in send_stream_fn(user_input):  # 流式，逐 token 打印
        ...

CLI 特有行为：
  - 用 prompt_toolkit 读取用户输入（历史、Ctrl+C/D）
  - 工具调用时显示 ThinkingSpinner 与进度行（通过 butler 回调）
  - 流式模式：第一个 token 到来前显示 ThinkingSpinner，
    收到第一个 token 后关闭 Spinner，开始逐 token 打印
  - 支持 quit / exit / q / 退出 关键字退出当前 session

参数说明：
  butler        — Butler 实例，仅用于注册 on_tool_call / on_tool_result 回调，
                  不直接调用 butler.chat()。
  send_stream_fn — 异步生成器，签名 (str) -> AsyncGenerator[str, None]，
                   由 AIButlerApp 提供（app.send_stream）。
"""
from __future__ import annotations

import sys
from typing import Callable, AsyncGenerator, Awaitable

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from agent import Butler
from cli.stream import safe_print, ThinkingSpinner


async def run(
    butler: Butler,
    send_fn: Callable[[str], Awaitable[str]] | None = None,
    send_stream_fn: Callable[[str], AsyncGenerator[str, None]] | None = None,
) -> None:
    """
    启动命令行交互 session（流式输出优先）。

    Args:
        butler        : 已初始化的 Butler 实例（用于注册工具回调）。
        send_fn       : 非流式发送函数（向后兼容，当 send_stream_fn 未提供时使用）。
        send_stream_fn: 流式发送函数（优先使用），来自 AIButlerApp.send_stream。
    """
    session = PromptSession()

    # ── CLI 特有：工具调用进度回调 ────────────────────────────────────────────
    spinner: ThinkingSpinner | None = None

    def on_tool_call(name: str):
        nonlocal spinner
        if spinner:
            with spinner.pause():
                safe_print(f"\n[工具调用] {name}")

    def on_tool_result(name: str, result: str):
        if spinner:
            with spinner.pause():
                preview = result[:200] + ("..." if len(result) > 200 else "")
                safe_print(f"[工具结果:{name}] {preview}")

    butler.on_tool_call   = on_tool_call
    butler.on_tool_result = on_tool_result

    # ── 主循环 ────────────────────────────────────────────────────────────────
    with patch_stdout():
        safe_print("=" * 40)
        safe_print("AI Butler 已启动，输入 quit 退出")
        safe_print("=" * 40)

        try:
            while True:
                # 读取输入
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
                    try:
                        first_token = True
                        with ThinkingSpinner() as sp:
                            spinner = sp
                            gen = send_stream_fn(user_input)
                            async for token in gen:
                                    # 跳过工具事件 marker（CLI 通过 on_tool_call 回调展示）
                                    if token.startswith("\x00"):
                                        continue
                                    if first_token:
                                        # 第一个 token 到来：关闭 spinner，开始打印
                                        sp._stop.set()
                                        sp._thread.join(timeout=1)
                                        sys.stdout.write("\r\033[K")   # 清除 spinner 行
                                        sys.stdout.write("\nButler: ")
                                        sys.stdout.flush()
                                        first_token = False
                                        spinner = None
                                    sys.stdout.write(token)
                                    sys.stdout.flush()
                        if not first_token:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        spinner = None
                    except KeyboardInterrupt:
                        spinner = None
                        sys.stdout.write("\n")
                        safe_print("（当前回复已中断，继续对话；输入 quit 退出）")
                    except Exception as e:
                        spinner = None
                        safe_print(f"\n[错误: {e}]")
                        import traceback
                        traceback.print_exc()

                # ── 非流式兜底（无 send_stream_fn 时） ────────────────────────
                else:
                    _send = send_fn or butler.chat
                    try:
                        with ThinkingSpinner() as sp:
                            spinner = sp
                            reply = await _send(user_input)
                        spinner = None
                        safe_print(f"\nButler: {reply}")
                    except KeyboardInterrupt:
                        spinner = None
                        safe_print("\n（当前回复已中断，继续对话；输入 quit 退出）")
                    except Exception as e:
                        spinner = None
                        safe_print(f"\n[错误: {e}]")
                        import traceback
                        traceback.print_exc()

        finally:
            safe_print("\n正在保存记忆...")
            safe_print("再见！")
