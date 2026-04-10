"""
AI Butler — CLI 启动入口

只做三件事：
  1. 把 src/ 加入 sys.path（兼容直接运行和容器 /app）
  2. 加载 .env
  3. 把控制权交给 cli.commands.run()

核心推理逻辑在 skills.agent.Butler，
终端 I/O 在 cli.commands，流式渲染在 cli.stream。
"""
import asyncio
import sys
import warnings
from pathlib import Path

# src/ 加入 sys.path，保证各子包可直接 import
_SRC_DIR = Path(__file__).parent.parent
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
from cli.commands import run


if __name__ == "__main__":
    asyncio.run(run(Config.from_env()))
