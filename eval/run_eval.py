# -*- coding: utf-8 -*-
"""
AI管家记忆系统评测脚本 v4
- 评测模式：内联复用 main.py 完整流程（含 tool call loop、passive_recall）
- 被测模型：从 .env 读取（llm_model）
- 评判模型：kimi-k2.5

前置步骤：
  python eval/replay.py       # 重放历史对话写入 ReMe（评测专用路径）
  python eval/run_eval.py     # 正式评测

评测流程（每个 TC）：
  1. pre_reasoning_hook（与生产一致，但 messages 为空，主要触发静默初始化）
  2. passive_recall：静默向量检索（与 main.py 完全一致）
  3. 读取评测专用 MEMORY.md
  4. assembler.build 组装 messages
  5. run_tool_call_loop：被测模型可主动调用 search_memory / search_history 等工具
  6. 评判模型打分

用法：
  python eval/run_eval.py
  python eval/run_eval.py --dry-run   # 不调 API，只检查流程
"""

import asyncio
import glob
import json
import os
import re
import sys
import yaml
import warnings
from pathlib import Path
from datetime import datetime

# 屏蔽 chromadb 在 Python 3.14 上的 DeprecationWarning
warnings.filterwarnings(
    "ignore",
    message="'asyncio.iscoroutinefunction' is deprecated",
    category=DeprecationWarning,
    module="chromadb",
)

# ── 路径设置 ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
EVAL_DIR = Path(__file__).parent
EVAL_DATA_DIR = EVAL_DIR / "eval_data" / "memory"
DAILY_DIR = EVAL_DIR / "conversations" / "user_c_daily"
GT_FILE = EVAL_DIR / "ground_truth" / "user_c_gt.yaml"
OUTPUT_FILE = EVAL_DIR / "eval-results-user-c.json"

sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from config import Config
from assembler import build
from history import ChatHistory

# 从 main.py 内联复用关键函数
import importlib.util
_main_spec = importlib.util.spec_from_file_location("main_module", SRC_DIR / "main.py")
_main_module = importlib.util.module_from_spec(_main_spec)
# 不执行 __main__ 块，只加载模块定义
_main_spec.loader.exec_module(_main_module)

passive_recall = _main_module.passive_recall
run_tool_call_loop = _main_module.run_tool_call_loop
dicts_to_msgs = _main_module.dicts_to_msgs
msgs_to_dicts = _main_module.msgs_to_dicts

# ── 配置 ───────────────────────────────────────────────────────────────────
DRY_RUN = "--dry-run" in sys.argv

JUDGE_MODEL = "kimi-k2.5"
PASSIVE_RECALL_K = 8
SIMILARITY_THRESHOLD = 0.5


def get_eval_config() -> Config:
    """复用生产 .env，只替换 working_dir 为评测专用路径。"""
    cfg = Config.from_env()
    cfg.working_dir = str(EVAL_DATA_DIR)
    return cfg


# ── LLM 同步调用（评判模型用，避免嵌套 asyncio）────────────────────────
import requests

def call_llm_sync(model: str, messages: list, api_key: str, base_url: str,
                  max_tokens: int = 800) -> str:
    if DRY_RUN:
        return json.dumps({
            "required_results": [],
            "forbidden_triggered": False,
            "forbidden_reason": "",
            "bonus_count": 0,
            "bonus_reasons": [],
            "overall_pass": True,
            "score": 1.0,
            "summary": "[DRY-RUN]",
        }, ensure_ascii=False)

    sess = requests.Session()
    sess.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    r = sess.post(
        base_url.rstrip("/") + "/chat/completions",
        json={"model": model, "messages": messages, "max_tokens": max_tokens},
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ── 数据加载 ───────────────────────────────────────────────────────────────
def load_test_files() -> list:
    pattern = str(DAILY_DIR / "test_TC-C*.json")
    files = sorted(glob.glob(pattern))
    result = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            result.append(json.load(fp))
    return result


def load_ground_truth() -> dict:
    with open(GT_FILE, encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    return {tc["id"]: tc for tc in data["test_cases"]}


def get_trigger_message(test_data: dict) -> str:
    for t in test_data.get("turns", []):
        if t["role"] == "user":
            return t["content"]
    return ""


# ── 自动关键词检查 ─────────────────────────────────────────────────────────
def check_keyword(text: str, keywords: list, mode: str = "any") -> bool:
    text_lower = text.lower()
    hits = [kw.lower() in text_lower for kw in keywords]
    return any(hits) if mode == "any" else not any(hits)


def auto_check(item: dict, answer: str):
    """返回 (result: bool|None, note: str)"""
    check = item.get("check", "")
    if check.startswith("response_mentions_any"):
        m = re.search(r'\[([^\]]+)\]', check)
        if m:
            kws = [k.strip().strip('"\'') for k in m.group(1).split(",")]
            r = check_keyword(answer, kws, "any")
            return r, f"关键词{kws}: {'命中' if r else '未命中'}"
    elif check.startswith("response_not_mentions_any"):
        m = re.search(r'\[([^\]]+)\]', check)
        if m:
            kws = [k.strip().strip('"\'') for k in m.group(1).split(",")]
            r = check_keyword(answer, kws, "none")
            return r, f"禁止词{kws}: {'通过' if r else '出现禁止词'}"
    return None, "需要评判模型"


# ── 评判 prompt ────────────────────────────────────────────────────────────
def build_judge_prompt(tc: dict, butler_answer: str, trigger_msg: str,
                       passive_snippets: list, tool_calls_made: list) -> list:
    required_text = "\n".join(
        f"  [{r['id']}] {r['desc']}"
        + (f"\n       标准: {r.get('rubric','').strip()}" if r.get("rubric") else "")
        for r in tc.get("required", [])
    )
    forbidden_text = "\n".join(f"  - {f['desc']}" for f in tc.get("forbidden", []))
    bonus_text = "\n".join(f"  - {b['desc']}" for b in tc.get("bonus", []))
    snippets_text = "\n---\n".join(passive_snippets) if passive_snippets else "（无静默检索结果）"
    tools_text = "、".join(tool_calls_made) if tool_calls_made else "（无工具调用）"

    system = f"""你是严格的 AI 记忆能力评测专家。

【用例】{tc['id']} — {tc['name']}
【描述】{tc.get('description','').strip()}

【必须满足（required）】
{required_text}

【绝对禁止（forbidden）】
{forbidden_text if forbidden_text else '无'}

【加分项（bonus）】
{bonus_text if bonus_text else '无'}

【ReMe 静默检索到的记忆片段】
{snippets_text}

【模型主动调用的工具】
{tools_text}

【触发问题】
{trigger_msg}

【AI 管家最终回答】
{butler_answer}

按以下 JSON 格式输出，不要有其他内容：
{{
  "required_results": [{{"id": "<id>", "pass": <true/false>, "reason": "<说明>"}}],
  "forbidden_triggered": <true/false>,
  "forbidden_reason": "<触发了哪条>",
  "bonus_count": <0-3>,
  "bonus_reasons": ["<说明>"],
  "overall_pass": <true/false>,
  "score": <0.0-1.0>,
  "summary": "<一句话总结>"
}}

规则：overall_pass = 所有 required 通过 AND forbidden_triggered=false；
forbidden 触发时 score 必须为 0。"""

    return [{"role": "system", "content": system}]


def parse_judge(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {
        "required_results": [], "forbidden_triggered": False,
        "forbidden_reason": "", "bonus_count": 0, "bonus_reasons": [],
        "overall_pass": False, "score": 0.0,
        "summary": f"解析失败: {text[:100]}",
    }


# ── 核心评测逻辑 ───────────────────────────────────────────────────────────
async def run_eval():
    # 检查重放是否已完成
    if not EVAL_DATA_DIR.exists():
        print(f"评测记忆目录不存在: {EVAL_DATA_DIR}")
        print("请先运行: python eval/replay.py")
        return

    cfg = get_eval_config()
    system_prompt = (SRC_DIR / "prompts" / "system.txt").read_text(encoding="utf-8")

    print("=" * 60)
    print("AI管家记忆评测 v4 — 用户C（林晨）")
    print(f"被测模型: {cfg.llm_model} | 评判模型: {JUDGE_MODEL}")
    print(f"评测记忆: {EVAL_DATA_DIR}")
    print(f"模式: {'DRY-RUN' if DRY_RUN else '正式评测'}")
    print("=" * 60)

    # 初始化评测专用 ReMe（与 main.py 完全一致的初始化参数）
    print("\n初始化评测专用 ReMe...")
    from reme.reme_light import ReMeLight
    from openai import AsyncOpenAI
    from tools import TOOLS, ToolExecutor

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

    # 初始化评测专用 ChatHistory（让模型可以调 search_history）
    history = ChatHistory(data_dir=cfg.working_dir)

    llm = AsyncOpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)

    # ToolExecutor：注入 history，让 search_history 工具可用
    executor = ToolExecutor(reme, history=history)

    test_files = load_test_files()
    ground_truth = load_ground_truth()

    results = []
    stats = {"total": 0, "pass": 0, "total_weight": 0.0, "weighted_score": 0.0}

    for test_data in test_files:
        meta = test_data.get("meta", {})
        tc_id = meta.get("test_case_id", "UNKNOWN")
        if tc_id not in ground_truth:
            print(f"  {tc_id} 在 ground_truth 中找不到，跳过")
            continue

        tc = ground_truth[tc_id]
        weight = tc.get("scoring", {}).get("weight", 1.0)
        trigger_msg = get_trigger_message(test_data)

        print(f"\n[{tc_id}] {tc['name']}")
        print(f"  记忆间隔: {tc.get('memory_gap_days','?')} 天 | 权重: {weight}")
        print(f"  触发问题: {trigger_msg[:80]}...")

        # ── Step 1: 静默向量检索（与 main.py passive_recall 完全一致）──
        # 注意：用文件顶部的 SIMILARITY_THRESHOLD（0.5），不用 cfg（0.75 太严格）
        passive_snippets = []
        if not DRY_RUN:
            passive_snippets = await passive_recall(
                reme=reme,
                query=trigger_msg,
                threshold=SIMILARITY_THRESHOLD,
                k=PASSIVE_RECALL_K,
            )
        print(f"  ReMe 静默检索: {len(passive_snippets)} 条片段")

        # ── Step 2: 读取评测专用 MEMORY.md ────────────────────────────
        memory_md = ""
        memory_file = EVAL_DATA_DIR / "MEMORY.md"
        if memory_file.exists():
            memory_md = memory_file.read_text(encoding="utf-8")

        # ── Step 3: 组装 messages（与 main.py assembler.build 一致）──
        # history 只有当前触发问题（评测每个 TC 都是独立的新对话）
        final_messages = build(
            system_prompt=system_prompt,
            history=[{"role": "user", "content": trigger_msg}],
            memory_md=memory_md,
            retrieval_snippets=passive_snippets if passive_snippets else None,
        )

        # ── Step 4: Tool Call 循环（与 main.py run_tool_call_loop 完全一致）
        butler_answer = "[DRY-RUN]"
        tool_calls_made: list[str] = []

        if not DRY_RUN:
            # 用一个代理 history 记录工具调用，不写入真实历史
            # 注意：Python 嵌套类不是闭包，无法直接引用外层变量。
            # 通过把 tool_calls_made 作为实例属性传入来绕过这个限制。
            class _TrackHistory:
                """只追踪工具调用名称，不真正写入 JSONL/DB。"""
                def __init__(self, calls_list: list):
                    self._calls = calls_list

                def append(self, role: str, content: str):
                    if role == "tool_call":
                        self._calls.extend(
                            [t.strip() for t in content.split("、") if t.strip()]
                        )

            try:
                reply, new_msgs = await run_tool_call_loop(
                    llm=llm,
                    model=cfg.llm_model,
                    messages=final_messages,
                    executor=executor,
                    tools=TOOLS,
                    history=_TrackHistory(tool_calls_made),
                )
                butler_answer = reply or ""
            except Exception as e:
                butler_answer = f"[ERROR: {e}]"
                print(f"  被测模型调用失败: {e}")
                import traceback; traceback.print_exc()

        if tool_calls_made:
            print(f"  模型调用工具: {', '.join(tool_calls_made)}")
        print(f"  管家回答: {butler_answer[:120]}...")

        # ── Step 5: 自动关键词检查 ─────────────────────────────────────
        auto_results = {}
        forbidden_auto = False

        for req in tc.get("required", []):
            r, note = auto_check(req, butler_answer)
            if r is not None:
                auto_results[req["id"]] = (r, note)
                icon = "✅" if r else "❌"
                print(f"  {icon} [{req['id']}] 自动: {note}")

        for forb in tc.get("forbidden", []):
            r, note = auto_check(forb, butler_answer)
            if r is not None and not r:
                forbidden_auto = True
                print(f"  Forbidden 触发: {forb['desc']}")

        # ── Step 6: 评判模型打分 ────────────────────────────────────────
        judge_messages = build_judge_prompt(
            tc, butler_answer, trigger_msg, passive_snippets, tool_calls_made
        )
        try:
            judge_raw = call_llm_sync(
                JUDGE_MODEL, judge_messages,
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url,
                max_tokens=800,
            )
            judge_result = parse_judge(judge_raw)
        except Exception as e:
            print(f"  评判调用失败: {e}")
            judge_result = {
                "required_results": [], "forbidden_triggered": False,
                "forbidden_reason": "", "bonus_count": 0, "bonus_reasons": [],
                "overall_pass": False, "score": 0.0,
                "summary": f"评判失败: {e}",
            }

        # 自动检查结果覆盖评判（自动更可靠）
        for req_id, (auto_pass, _) in auto_results.items():
            for jr in judge_result.get("required_results", []):
                if jr["id"] == req_id:
                    jr["pass"] = auto_pass
        if forbidden_auto:
            judge_result["forbidden_triggered"] = True
            judge_result["overall_pass"] = False
            judge_result["score"] = 0.0

        overall_pass = judge_result.get("overall_pass", False)
        score = judge_result.get("score", 0.0)
        bonus = judge_result.get("bonus_count", 0)
        summary = judge_result.get("summary", "")

        stats["total"] += 1
        stats["total_weight"] += weight
        stats["weighted_score"] += weight * score
        if overall_pass:
            stats["pass"] += 1

        icon = "✅ PASS" if overall_pass else "❌ FAIL"
        print(f"  {icon} | 得分: {score:.2f} | 加权: {weight * score:.2f} | {summary}")
        for jr in judge_result.get("required_results", []):
            ji = "✅" if jr.get("pass") else "❌"
            print(f"    {ji} [{jr['id']}] {jr.get('reason','')}")

        results.append({
            "id": tc_id,
            "name": tc["name"],
            "memory_gap_days": tc.get("memory_gap_days"),
            "weight": weight,
            "trigger_msg": trigger_msg,
            "passive_snippets": passive_snippets,
            "tool_calls_made": tool_calls_made,
            "butler_answer": butler_answer,
            "auto_checks": {k: {"pass": v[0], "note": v[1]} for k, v in auto_results.items()},
            "forbidden_auto": forbidden_auto,
            "judge_result": judge_result,
            "overall_pass": overall_pass,
            "score": score,
            "weighted_score": weight * score,
            "bonus": bonus,
            "summary": summary,
        })

    # 清理
    history.close()
    try:
        await reme.close()
    except Exception:
        pass

    # ── 汇总 ──────────────────────────────────────────────────────────────
    tw = stats["total_weight"]
    ws = stats["weighted_score"]
    final_pct = (ws / tw * 100) if tw > 0 else 0
    pass_rate = (stats["pass"] / stats["total"] * 100) if stats["total"] > 0 else 0

    print("\n" + "=" * 60)
    print("评测结果汇总")
    print("=" * 60)
    print(f"通过率:   {stats['pass']}/{stats['total']} ({pass_rate:.1f}%)")
    print(f"加权得分: {ws:.2f}/{tw:.2f} ({final_pct:.1f}%)")
    for r in results:
        i = "✅" if r["overall_pass"] else "❌"
        print(f"  {i} {r['id']}: {r['score']:.2f} × {r['weight']} = {r['weighted_score']:.2f}")

    output = {
        "run_at": datetime.now().isoformat(),
        "butler_model": cfg.llm_model,
        "judge_model": JUDGE_MODEL,
        "eval_data_dir": str(EVAL_DATA_DIR),
        "dry_run": DRY_RUN,
        "stats": {
            "total": stats["total"],
            "pass": stats["pass"],
            "pass_rate": round(pass_rate, 1),
            "total_weight": tw,
            "weighted_score": round(ws, 3),
            "final_score_pct": round(final_pct, 1),
        },
        "results": results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(run_eval())
