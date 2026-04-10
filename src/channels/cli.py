"""
channels/cli.py — 命令行消息渠道

将来自终端的用户输入/输出封装为 Channel，与飞书、企微等渠道保持统一的核心调用方式：
    channel_specific_setup()
    reply = await butler.chat(user_input)
    channel_specific_output(reply)

CLI 特有行为：
    - 用 prompt_toolkit 读取用户输入（历史、Ctrl+C/D）
    - 工具调用时显示 ThinkingSpinner 与进度行
    - 支持 quit / exit / q / 退出 关键字退出当前 session
"""
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from agent import Butler
from cli.stream import safe_print, ThinkingSpinner


async def run(butler: Butler) -> None:
    """
    启动命令行交互 session。

    接收一个已初始化的 Butler 实例，负责终端侧的全部 I/O，
    核心推理逻辑完全由 Butler.chat() 承接，与其他 Channel 保持一致。
    """
    session = PromptSession()

    # ── CLI 特有：工具调用进度回调 ──────────────────────────────────────────────
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

                # CLI 特有：quit 指令退出 session
                if user_input.lower() in ("quit", "exit", "q", "退出"):
                    break

                # 核心推理：与其他 Channel 完全相同的调用方式
                try:
                    with ThinkingSpinner() as sp:
                        spinner = sp
                        reply = await butler.chat(user_input)
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
            try:
                await butler.close()
            except Exception as e:
                safe_print(f"[保存记忆失败] {e}")
            safe_print("再见！")
