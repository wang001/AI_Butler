"""
cli/main.py — CLI 启动入口（兼容保留）

直接委托给 ai_butler.py 的统一入口，等价于：
  python src/ai_butler.py --mode cli

保留此文件是为了兼容现有的 docker CMD 和开发习惯。
"""
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

# 委托给统一入口，强制 CLI 模式
if __name__ == "__main__":
    import sys
    # 注入 --mode cli，避免被外部 argv 干扰
    sys.argv = [sys.argv[0], "--mode", "cli"]
    from ai_butler import main
    main()
