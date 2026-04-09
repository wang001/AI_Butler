# -*- coding: utf-8 -*-
"""
eval/replay.py — 评测历史对话重放器

将 user_c_daily/day*.json 的历史对话真实写入 ReMe（评测专用路径），
同时写入 ChatHistory（jsonl + SQLite FTS5），模拟生产环境的记忆积累过程。

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
import shutil
import sys
from pathlib import Path

# 把 src/ 加入 sys.path，复用主程序的模块
REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from config import Config
from history import ChatHistory

# ── 评测专用路径（与生产完全隔离）──────────────────────────────────────────
EVAL_DATA_DIR = Path(__file__).parent / "eval_data" / "memory"
DAILY_DIR = Path(__file__).parent / "conversations" / "user_c_daily"

NO_RESET = "--no-reset" in sys.argv


def get_eval_config() -> Config:
    """复用生产 .env 配置，只替换 working_dir 为评测专用路径。"""
    cfg = Config.from_env()
    cfg.working_dir = str(EVAL_DATA_DIR)
    return cfg


def load_daily_turns() -> list[dict]:
    """
    按时间顺序加载所有 day*.json 和 day*_continuation.json，
    返回完整的 turns 列表（只含 user/assistant 角色）。
    """
    pattern = str(DAILY_DIR / "day*.json")
    files = sorted(glob.glob(pattern))  # 文件名字母序 = 时间序

    all_turns = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)

        if isinstance(data, dict):
            turns = data.get("turns", [])
        elif isinstance(data, list):
            turns = data
        else:
            continue

        for turn in turns:
            role = turn.get("role", "")
            if role in ("user", "assistant"):
                all_turns.append({
                    "role": role,
                    "content": turn.get("content", ""),
                    "timestamp": turn.get("timestamp", ""),
                })

    return all_turns


def pair_turns(turns: list[dict]) -> list[tuple[str, str]]:
    """
    将 turns 列表配对为 (user_msg, assistant_reply) 元组列表。
    跳过不完整的配对（连续两个 user 或连续两个 assistant）。
    """
    pairs = []
    i = 0
    while i < len(turns) - 1:
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            pairs.append((turns[i]["content"], turns[i + 1]["content"]))
            i += 2
        else:
            i += 1
    return pairs


async def replay():
    cfg = get_eval_config()

    # ── 准备评测目录 ────────────────────────────────────────────────────────
    if not NO_RESET:
        if EVAL_DATA_DIR.exists():
            print(f"清空评测记忆目录: {EVAL_DATA_DIR}")
            shutil.rmtree(EVAL_DATA_DIR)
    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"评测记忆目录: {EVAL_DATA_DIR}")

    # ── 加载历史对话 ────────────────────────────────────────────────────────
    turns = load_daily_turns()
    pairs = pair_turns(turns)
    print(f"历史对话: {len(turns)} 轮 → {len(pairs)} 对 user/assistant")

    # ── 初始化 ReMe（评测专用路径）──────────────────────────────────────────
    print("初始化 ReMe（评测专用）...")
    from reme.reme_light import ReMeLight
    from agentscope.message import Msg

    reme = ReMeLight(
        working_dir=cfg.working_dir,
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

    # ── 初始化 ChatHistory（评测专用路径）──────────────────────────────────
    history = ChatHistory(data_dir=cfg.working_dir)

    # ── 重放对话：按批次直接调用 summary_memory 写入 ReMe 向量库 ────────────
    # 不依赖 pre_reasoning_hook（它只在 context 溢出时才触发，通常不会触发）。
    # 直接按 BATCH_SIZE 对分批，每批生成摘要并写入向量库。
    #
    # 同时写入 ChatHistory（JSONL + FTS5），供 search_history 工具使用。
    BATCH_SIZE = 15  # 每 15 对（约 30 条消息）沉淀一次

    all_msgs = []   # 全量 Msg 列表，用于最终兜底沉淀
    batch_msgs = []  # 当前批次

    print(f"\n开始重放 {len(pairs)} 对对话...")
    for idx, (user_msg, assistant_reply) in enumerate(pairs):
        u = Msg(role="user", content=user_msg, name="user")
        a = Msg(role="assistant", content=assistant_reply, name="assistant")
        batch_msgs.append(u)
        batch_msgs.append(a)
        all_msgs.append(u)
        all_msgs.append(a)

        history.append("user", user_msg)
        history.append("assistant", assistant_reply)

        # 每 BATCH_SIZE 对触发一次沉淀（异步任务，最后统一 await）
        if (idx + 1) % BATCH_SIZE == 0:
            print(f"  [{idx + 1}/{len(pairs)}] 提交沉淀任务（{len(batch_msgs)} 条消息）...")
            try:
                reme.add_async_summary_task(messages=list(batch_msgs))
            except Exception as e:
                print(f"  [提交沉淀失败] {e}")
            batch_msgs = []

        if (idx + 1) % 30 == 0:
            print(f"  已重放 {idx + 1}/{len(pairs)} 对")

    # 最后一批（不足 BATCH_SIZE 的尾部）
    if batch_msgs:
        print(f"  [{len(pairs)}/{len(pairs)}] 提交最后一批沉淀任务（{len(batch_msgs)} 条消息）...")
        try:
            reme.add_async_summary_task(messages=list(batch_msgs))
        except Exception as e:
            print(f"  [提交最后一批沉淀失败] {e}")

    # 等待所有异步沉淀任务完成
    print("\n等待所有沉淀任务完成...")
    try:
        await reme.await_summary_tasks()
        print("  沉淀完成")
    except Exception as e:
        print(f"  [等待沉淀失败] {e}")

    await reme.close()
    history.close()

    print(f"\n重放完成。评测记忆已写入: {EVAL_DATA_DIR}")
    print("现在可以运行: python eval/run_eval.py")


if __name__ == "__main__":
    asyncio.run(replay())
