"""
cron — 定时任务模块。

当前包含：
  - MemoryUpdateService：周期性读取最新 history，独立上下文判断并更新 MEMORY.md
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI

from agent.runner import AgentRunner
from tools.dispatcher import ToolDispatcher
from tools.memory import MEMORY_READONLY_TOOLS


class MemoryUpdateService:
    """管理 MEMORY.md 的后台更新与独立上下文执行。"""

    _MAX_MEMORY_CHARS = 2000
    _DEFAULT_INTERVAL_SECONDS = 6 * 60 * 60

    def __init__(self, cfg, history, reme, llm_model: str):
        self._cfg = cfg
        self._history = history
        self._reme = reme
        self._llm_model = llm_model
        self._memory_dir = Path(cfg.memory_dir)
        self._memory_path = self._memory_dir / "MEMORY.md"
        self._meta_path = self._memory_dir / ".memory_update_meta.json"
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(), name="memory-update-cron")

    async def stop(self) -> None:
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def request_update(
        self,
        trigger: str,
        recent_messages: list[dict] | None = None,
        proposed_notes: str = "",
    ) -> str:
        """由模型工具或后台任务触发的独立记忆更新入口。"""
        async with self._lock:
            return await self._run_update(
                trigger=trigger,
                recent_messages=recent_messages,
                proposed_notes=proposed_notes,
            )

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._DEFAULT_INTERVAL_SECONDS)
            try:
                await self.request_update(trigger="scheduled")
            except Exception:
                # 定时任务不能影响主服务运行
                pass

    async def _run_update(
        self,
        trigger: str,
        recent_messages: list[dict] | None = None,
        proposed_notes: str = "",
    ) -> str:
        since_ts = self._load_last_update_ts()
        history_rows = recent_messages or self._history.get_since(since_ts=since_ts, limit=500)
        if not history_rows and not proposed_notes.strip():
            return "未发现新的对话记录，也没有新的记忆候选信息，无需更新 MEMORY.md。"

        payload = self._build_history_payload(history_rows)
        current_memory = self._read_memory()
        prompt = self._build_prompt(
            current_memory=current_memory,
            history_payload=payload,
            trigger=trigger,
            proposed_notes=proposed_notes,
        )
        updated_memory = await self._generate_memory(prompt)
        normalized = self._normalize_memory(updated_memory)

        latest_ts = max((row["ts"] for row in history_rows), default=since_ts)

        if normalized == current_memory.strip():
            self._save_last_update_ts(latest_ts)
            return "已检查最新对话，现有 MEMORY.md 无需更新。"

        self._write_memory(normalized)
        self._save_last_update_ts(latest_ts)
        return (
            f"MEMORY.md 已更新（触发方式: {trigger}）。"
            f" 当前长度 {len(normalized)} 字。"
        )

    def _build_history_payload(self, rows: list[dict]) -> str:
        parts: list[str] = []
        for row in rows:
            role = row.get("role", "unknown")
            if role not in {"user", "assistant"}:
                continue
            content = (row.get("content") or "").strip()
            if not content:
                continue
            parts.append(f"[{role}] {content}")
        return "\n".join(parts[-200:])

    def _build_prompt(
        self,
        current_memory: str,
        history_payload: str,
        trigger: str,
        proposed_notes: str = "",
    ) -> str:
        return (
            "你是用户长期记忆整理助手。你的唯一任务是维护 MEMORY.md。\n"
            "要求：\n"
            "1. 你必须基于新的对话内容，判断是否值得更新长期记忆。\n"
            "2. 输出必须是最终版 MEMORY.md 正文，不要解释过程。\n"
            "3. 最终内容必须不超过 2000 个汉字字符。\n"
            "4. 只保留对未来有长期价值的信息：稳定偏好、长期项目、关键约定、明确身份背景。\n"
            "5. 临时琐事、短期情绪、一次性任务不要写入。\n"
            "6. 如果内容很多，优先精简；必要时采用“简要描述 -> 详细文档路径”的形式。\n"
            "7. 如果无需更新，也请输出当前 MEMORY.md 的原文，不要额外解释。\n\n"
            f"触发方式：{trigger}\n\n"
            "【当前 MEMORY.md】\n"
            f"{current_memory or '（空）'}\n\n"
            "【模型建议补充的记忆候选】\n"
            f"{proposed_notes or '（无）'}\n\n"
            "【最新对话历史】\n"
            f"{history_payload or '（无）'}\n"
        )

    async def _generate_memory(self, prompt: str) -> str:
        llm = AsyncOpenAI(base_url=self._cfg.llm_base_url, api_key=self._cfg.llm_api_key)
        dispatcher = ToolDispatcher(
            reme=self._reme,
            history=self._history,
            command_executor=None,
            browser_agent=None,
            tool_call_dir=self._cfg.tool_call_dir,
            memory_update_service=self,
            memory_tools=MEMORY_READONLY_TOOLS,
        )
        runner = AgentRunner(
            llm=llm,
            model=self._llm_model,
            dispatcher=dispatcher,
            hook=None,
        )
        reply, _ = await runner.run([
            {"role": "system", "content": "你只负责生成 MEMORY.md 最终正文。"},
            {"role": "user", "content": prompt},
        ])
        return reply.strip()

    def _normalize_memory(self, text: str) -> str:
        text = (text or "").strip()
        if len(text) <= self._MAX_MEMORY_CHARS:
            return text
        simplified = text[: self._MAX_MEMORY_CHARS - 1].rstrip()
        return simplified + "…"

    def _read_memory(self) -> str:
        try:
            return self._memory_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""
        except Exception:
            return ""

    def _write_memory(self, content: str) -> None:
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._memory_path.write_text(content.strip() + "\n", encoding="utf-8")

    def _load_last_update_ts(self) -> float:
        try:
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
            return float(data.get("last_update_ts", 0.0))
        except Exception:
            return 0.0

    def _save_last_update_ts(self, ts: float) -> None:
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(
            json.dumps({"last_update_ts": ts, "saved_at": time.time()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
