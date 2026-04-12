"""
channels/cli.py — 命令行消息渠道

把当前终端作为一个 Channel 接入 AI Butler。
与飞书、HTTP 等渠道统一的调用模型：

    reply = await send_fn(user_input)   # 走 AIButlerApp.inbox 队列

CLI 特有行为：
  - 用 prompt_toolkit 读取用户输入（历史、Ctrl+C/D）
  - 工具调用时显示 ThinkingSpinner 与进度行（通过 butler 回调）
  - 支持 quit / exit / q / 退出 关键字退出当前 session

参数说明：
  butler  — Butler 实例，仅用于注册 on_tool_call / on_tool_result 回调，
            不直接调用 butler.chat()。
  send_fn — 异步可调用，签名 (str) -> str，由 AIButlerApp 提供，
            内部走 inbox 队列，Agent 主循环串行处理后返回回复。
            若未传入则退化为直接调用 butler.chat()（兼容单测场景）。
"""
from __future__ import annotations

import sys
from typing import Callable, Awaitable

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from agent import Butler
from cli.stream import safe_print, ThinkingSpinner


async def run(
    butler: Butler,
    send_fn: Callable[[str], Awaitable[str]] | None = None,
) -> None:
    """
    启动命令行交互 session。

    Args:
        butler : 已初始化的 Butler 实例（用于注册工具回调，不直接 chat）。
        send_fn: 异步消息发送函数，来自 AIButlerApp.send()；
                 默认 None 时退化为 butler.chat()（向后兼容）。
    """
    _send = send_fn or butler.chat

    session = PromptSession()

    # ── CLI 特有：工具调用进度回调（在 Agent 主循环调用 butler.chat() 期间触发）──
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

    # ── 主循环 ─────────────────────────────────────────────────────────────────
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

                # 通过 send_fn 走 AIButlerApp.inbox 队列（或直接 butler.chat）
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
            # butler.close() 由 AIButlerApp.run() 统一调用，此处无需重复
            safe_print("再见！")
