"""
cli/commands.py — 命令行入口

负责初始化 Butler 并将控制权交给 channels.cli 处理终端 I/O。
终端交互的所有细节（输入读取、Spinner、quit 退出等）均在 channels/cli.py 中实现。
"""
from config import Config
from agent import Butler, wait_heavy_loaded
from cli.stream import safe_print


async def run(cfg: Config) -> None:
    """启动命令行交互会话。"""
    from channels.cli import run as cli_channel_run

    # 重型模块预热（chromadb / agentscope / reme）
    wait_heavy_loaded(on_waiting=lambda: safe_print("（正在初始化记忆系统，请稍候…）"))

    butler = await Butler.create(cfg, channel="cli")
    await cli_channel_run(butler)
