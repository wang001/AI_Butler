"""
cli/commands.py — CLI 模式启动适配器

兼容旧调用方式（cli/main.py → commands.run()），
内部委托给 AIButlerApp 统一处理，确保消息走 inbox 队列。
"""
from config import Config


async def run(cfg: Config) -> None:
    """启动 CLI 模式（委托给 AIButlerApp）。"""
    from ai_butler import AIButlerApp
    app = AIButlerApp(cfg)
    await app.run(mode="cli")
