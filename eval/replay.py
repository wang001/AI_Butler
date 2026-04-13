# -*- coding: utf-8 -*-
"""
eval/replay.py — 评测历史对话重放器

将 user_c_daily/day*.json 的历史对话按天写入 ReMe 记忆文件（评测专用路径），
同时写入 ChatHistory（jsonl + SQLite FTS5），模拟生产环境的记忆积累过程。

关键设计：
  - 直接写文件：不调用 LLM 摘要，直接按天写入 memory/YYYY-MM-DD.md
  - 格式对齐：文件格式与 ReMe 生成的一致，确保向量索引正确建立
  - ChatHistory：所有对话顺序写入 JSONL + SQLite FTS5，供 search_history 检索

评测专用路径：eval/eval_data/memory/
生产路径：data/memory/（不会被触碰）

用法：
  python eval/replay.py              # 清空后重放所有历史对话
  python eval/replay.py --no-reset   # 不清空，直接追加（用于断点续跑）
"""

import asyncio
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from config import Config
from history import ChatHistory

EVAL_DATA_DIR = Path(__file__).parent / "eval_data" / "memory"
DAILY_DIR = Path(__file__).parent / "conversations" / "user_c_daily"

NO_RESET = "--no-reset" in sys.argv


def get_eval_config() -> Config:
    cfg = Config.from_env()
    cfg.memory_dir = str(EVAL_DATA_DIR)
    return cfg


def load_days() -> list[dict]:
    """
    按天加载对话文件，返回 list of {"date": "YYYY-MM-DD", "turns": [...]}。
    同一天的主文件 + continuation 合并到一起。
    """
    pattern = str(DAILY_DIR / "day*.json")
    files = sorted(glob.glob(pattern))

    # 过滤掉 test_TC-* 文件
    files = [f for f in files if not Path(f).name.startswith("test_")]

    # 按日期分组：day01_0103.json 和 day01_continuation.json 同属一天
    day_groups: dict[str, dict] = {}  # date -> {"date", "turns", "meta"}

    for fpath in files:
        fname = Path(fpath).name
        with open(fpath, encoding="utf-8") as fp:
            data = json.load(fp)

        if isinstance(data, dict):
            date = data.get("meta", {}).get("date", "")
            turns = data.get("turns", [])
            meta = data.get("meta", {})
        elif isinstance(data, list):
            # continuation 文件：list 格式，无 meta，从文件名推断日期
            turns = data
            meta = {}
            # 从同名主文件找 date
            m = re.match(r"day(\d+)_(\d{4})", fname)
            if m:
                mmdd = m.group(2)
                date = f"2025-{mmdd[:2]}-{mmdd[2:]}"
            else:
                date = ""
        else:
            continue

        if not date:
            continue

        if date not in day_groups:
            day_groups[date] = {"date": date, "turns": [], "meta": meta}

        for turn in turns:
            role = turn.get("role", "")
            if role in ("user", "assistant"):
                day_groups[date]["turns"].append({
                    "role": role,
                    "content": turn.get("content", ""),
                    "timestamp": turn.get("timestamp", ""),
                })

    # 按日期排序
    return sorted(day_groups.values(), key=lambda d: d["date"])


def format_conversation_to_markdown(turns: list[dict], date: str, meta: dict) -> str:
    """
    将一天的对话格式化为 markdown，格式与 ReMe 生成的一致。
    """
    lines = [
        f"# 记忆记录 - {date}",
        "",
        "## 持久化记忆",
        "",
        f"### 日期信息",
        f"- **日期**: {date}",
    ]

    # 从 meta 提取 memory_seeds 作为关键记忆点
    memory_seeds = meta.get("memory_seeds", [])
    day_desc = meta.get("day_desc", "")

    if day_desc:
        lines.extend([f"- **当日主题**: {day_desc}"])

    if memory_seeds:
        lines.extend(["", "### 关键记忆点"])
        for seed in memory_seeds:
            lines.append(f"- {seed}")

    # 对话摘要（前 10 轮）
    lines.extend(["", "### 对话摘要"])
    for i, turn in enumerate(turns[:20]):  # 只取前 20 轮避免太长
        role = "用户" if turn["role"] == "user" else "AI"
        content = turn["content"][:200]  # 截断
        if len(turn["content"]) > 200:
            content += "..."
        lines.append(f"- **{role}**: {content}")

    # 完整对话记录
    lines.extend(["", "## 完整对话记录", ""])
    for turn in turns:
        role = "👤 用户" if turn["role"] == "user" else "🤖 AI"
        lines.append(f"{role}: {turn['content']}")
        lines.append("")

    return "\n".join(lines)


async def replay():
    cfg = get_eval_config()

    if not NO_RESET:
        if EVAL_DATA_DIR.exists():
            print(f"清空评测记忆目录: {EVAL_DATA_DIR}")
            shutil.rmtree(EVAL_DATA_DIR)
    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    memory_dir = EVAL_DATA_DIR / "memory"
    memory_dir.mkdir(exist_ok=True)

    print(f"评测记忆目录: {EVAL_DATA_DIR}")

    days = load_days()
    total_turns = sum(len(d["turns"]) for d in days)
    print(f"共 {len(days)} 天对话，{total_turns} 轮消息")

    # 初始化 ChatHistory
    history = ChatHistory(data_dir=cfg.memory_dir)

    # 按天写入文件
    for day in days:
        date = day["date"]
        turns = day["turns"]
        meta = day.get("meta", {})

        if not turns:
            continue

        print(f"\n[{date}] {len(turns)} 轮对话")

        # 写入 ChatHistory
        for turn in turns:
            history.append(turn["role"], turn["content"])

        # 直接写 memory 文件
        md_content = format_conversation_to_markdown(turns, date, meta)
        md_file = memory_dir / f"{date}.md"
        md_file.write_text(md_content, encoding="utf-8")
        print(f"  ✅ 写入 {md_file.name} ({len(md_content)} 字符)")

    history.close()

    # 初始化 ReMe 让它扫描文件建立索引
    print("\n初始化 ReMe 建立向量索引...")
    from reme.reme_light import ReMeLight

    reme = ReMeLight(
        working_dir=cfg.memory_dir,
        llm_api_key=cfg.llm_api_key,
        llm_base_url=cfg.llm_base_url,
        embedding_api_key=cfg.emb_api_key,
        embedding_base_url=cfg.emb_base_url,
        default_as_llm_config={"model_name": cfg.llm_model},
        default_embedding_model_config={"model_name": cfg.emb_model},
        default_file_store_config={"fts_enabled": True, "vector_enabled": True},
        enable_load_env=False,
    )
    await reme.start()
    print("ReMe 已启动")

    # 强制扫描所有文件到向量库
    print("  扫描记忆文件到向量库...")
    try:
        for name, watcher in reme.service_context.file_watchers.items():
            print(f"    扫描 file_watcher[{name}]...")
            await watcher._scan_existing_files()
        print("  向量库同步完成")
    except Exception as e:
        print(f"  [同步失败] {e}")
        import traceback; traceback.print_exc()

    await asyncio.sleep(2)

    # 验证
    print("\n验证向量库写入...")
    memory_files = list(reme.memory_path.glob("*.md"))
    print(f"  memory/ 目录共 {len(memory_files)} 个文件")

    test_queries = [
        ("阅读 书 额尔古纳", "TC-C01: 读书推荐"),
        ("咖啡 睡眠 12点", "TC-C02: 睡眠建议"),
        ("智能家居 Home Assistant 小爱", "TC-C03: 智能家居"),
        ("泰国 普吉岛 免签", "TC-C04: 泰国旅行"),
        ("世博公园 春天 郁金香", "TC-C05: 春天出行"),
    ]

    for query, desc in test_queries:
        try:
            result = await reme.memory_search(query, max_results=3, min_score=0.2)
            content = getattr(result, "content", None)
            hits = 0
            if content:
                for block in content:
                    text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                    if text and text.strip() not in ("[]", ""):
                        hits += 1
            icon = "✅" if hits > 0 else "❌"
            print(f"  {icon} {desc}: {hits} 条")
        except Exception as e:
            print(f"  ❌ {desc}: 检索失败 {e}")

    try:
        await reme.close()
    except Exception:
        pass

    print(f"\n重放完成。评测记忆已写入: {EVAL_DATA_DIR}")
    print("现在可以运行: python eval/run_eval.py")


if __name__ == "__main__":
    asyncio.run(replay())
