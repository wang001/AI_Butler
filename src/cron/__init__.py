"""
cron — 定时任务模块。

当前包含：
  - MemoryUpdateService：app 级单例长期记忆更新器
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from agent.runner import AgentRunner
from tools.executor import ToolExecutor
from tools.memory_tools import MemoryTools, SearchHistoryTool, SearchMemoryTool
from tools.registry import ToolRegistry


class _MemoryOnlyDispatcher:
    """只给 MEMORY.md 更新暴露记忆相关工具，避免无关工具污染输出。"""

    def __init__(self, *, reme, history, memory_update_service):
        memory_tools = MemoryTools(
            reme=reme,
            history=history,
            memory_update_service=memory_update_service,
        )
        self._registry = ToolRegistry([
            SearchMemoryTool(memory_tools),
            SearchHistoryTool(memory_tools),
        ])
        self._executor = ToolExecutor(registry=self._registry)
        self.tools = self._registry.tools

    async def run(self, name: str, arguments: str) -> str:
        return await self._executor.run(name, arguments)


class MemoryUpdateService:
    """管理 MEMORY.md 的 app 级单例更新。"""

    _MAX_MEMORY_CHARS = 2000
    _UPDATE_INTERVAL_SECONDS = 60 * 60
    _DEFAULT_BATCH_SIZE = 500
    _DEFAULT_IDLE_WAIT_SECONDS = 60
    _META_VERSION = 2
    _RELEVANT_ROLES = ("user", "assistant")

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
        self._wake_event = asyncio.Event()

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(), name="memory-update-service")
        self.notify_new_messages()

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        try:
            self._history.close()
        except Exception:
            pass

    def notify_new_messages(self) -> None:
        """有新消息入库后唤醒调度器重新计算是否到期。"""
        self._wake_event.set()

    async def request_update(
        self,
        trigger: str,
        recent_messages: list[dict] | None = None,
        proposed_notes: str = "",
        force: bool = False,
    ) -> str:
        """由模型工具或后台调度触发的长期记忆更新入口。"""
        async with self._lock:
            return await self._run_update(
                trigger=trigger,
                recent_messages=recent_messages,
                proposed_notes=proposed_notes,
                force=force,
            )

    async def _loop(self) -> None:
        while True:
            try:
                await self.request_update(trigger="scheduled")
            except asyncio.CancelledError:
                raise
            except Exception:
                # 定时任务不能影响主服务运行
                pass

            timeout = self._compute_wait_timeout()
            self._wake_event.clear()
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

    def _compute_wait_timeout(self) -> float:
        try:
            meta = self._load_meta()
            pending = self._history.get_since_id(
                int(meta["last_processed_message_id"]),
                limit=1,
                roles=self._RELEVANT_ROLES,
            )
            if not pending:
                return self._DEFAULT_IDLE_WAIT_SECONDS

            due_at = self._compute_due_at(
                last_updated_at=float(meta["last_updated_at"]),
                oldest_pending_ts=float(pending[0]["ts"]),
            )
            remaining = max(1.0, due_at - time.time())
            return min(self._DEFAULT_IDLE_WAIT_SECONDS, remaining)
        except Exception:
            return self._DEFAULT_IDLE_WAIT_SECONDS

    async def _run_update(
        self,
        trigger: str,
        recent_messages: list[dict] | None = None,
        proposed_notes: str = "",
        force: bool = False,
    ) -> str:
        meta = self._load_meta()
        now = time.time()
        meta["last_checked_at"] = now

        pending_preview = self._history.get_since_id(
            int(meta["last_processed_message_id"]),
            limit=1,
            roles=self._RELEVANT_ROLES,
        )
        latest_message_id = self._history.get_latest_message_id(
            roles=self._RELEVANT_ROLES,
        )
        has_pending = bool(pending_preview)
        oldest_pending_ts = float(pending_preview[0]["ts"]) if pending_preview else 0.0

        if not force:
            if not has_pending:
                meta["next_due_at"] = (
                    float(meta["last_updated_at"]) + self._UPDATE_INTERVAL_SECONDS
                    if float(meta["last_updated_at"]) > 0
                    else 0.0
                )
                self._save_meta(meta)
                return "自上次长期记忆更新以来没有新的用户/助手消息。"

            due_at = self._compute_due_at(
                last_updated_at=float(meta["last_updated_at"]),
                oldest_pending_ts=oldest_pending_ts,
            )
            meta["next_due_at"] = due_at
            self._save_meta(meta)
            if now < due_at:
                remaining = int(max(1, due_at - now))
                return f"长期记忆更新尚未到期，约 {remaining} 秒后再检查。"

        if not has_pending and not proposed_notes.strip() and not recent_messages:
            meta["next_due_at"] = now + self._UPDATE_INTERVAL_SECONDS
            self._save_meta(meta)
            return "没有新的长期记忆候选信息，无需更新 MEMORY.md。"

        return await self._drain_updates(
            meta=meta,
            trigger=trigger,
            target_message_id=latest_message_id,
            recent_messages=recent_messages,
            proposed_notes=proposed_notes,
        )

    async def _drain_updates(
        self,
        *,
        meta: dict[str, Any],
        trigger: str,
        target_message_id: int,
        recent_messages: list[dict] | None = None,
        proposed_notes: str = "",
    ) -> str:
        last_processed_id = int(meta["last_processed_message_id"])
        current_memory = self._read_memory()
        extra_rows = self._normalize_recent_messages(recent_messages)
        processed_batches = 0
        processed_rows = 0
        memory_changed = False
        first_batch = True

        while True:
            batch_rows = []
            if target_message_id > last_processed_id:
                batch_rows = self._history.get_since_id(
                    since_id=last_processed_id,
                    limit=self._DEFAULT_BATCH_SIZE,
                    roles=self._RELEVANT_ROLES,
                    max_id=target_message_id,
                )

            prompt_rows = list(batch_rows)
            if first_batch and extra_rows:
                prompt_rows.extend(extra_rows)

            if not prompt_rows and not (first_batch and proposed_notes.strip()):
                break

            prompt = self._build_prompt(
                current_memory=current_memory,
                history_payload=self._build_history_payload(prompt_rows),
                trigger=trigger,
                proposed_notes=proposed_notes if first_batch else "",
            )

            updated_memory = await self._generate_memory(prompt)
            normalized = self._normalize_memory(updated_memory)

            if normalized != current_memory.strip():
                self._write_memory(normalized)
                current_memory = normalized
                memory_changed = True

            if batch_rows:
                last_processed_id = int(batch_rows[-1]["id"])
                meta["last_processed_message_id"] = last_processed_id
                processed_rows += len(batch_rows)

            processed_batches += 1
            first_batch = False

            if not batch_rows or last_processed_id >= target_message_id:
                break

        completed_at = time.time()
        meta["last_checked_at"] = completed_at
        meta["last_updated_at"] = completed_at
        meta["next_due_at"] = completed_at + self._UPDATE_INTERVAL_SECONDS
        self._save_meta(meta)

        if processed_batches == 0:
            return "没有新的长期记忆候选信息，无需更新 MEMORY.md。"

        if memory_changed:
            return (
                f"MEMORY.md 已更新（触发方式: {trigger}）。"
                f" 本次处理 {processed_rows} 条消息，分 {processed_batches} 批完成。"
            )
        return (
            f"已检查 {processed_rows} 条新消息，现有 MEMORY.md 无需更新。"
            f" 本次共执行 {processed_batches} 批。"
        )

    def _compute_due_at(self, *, last_updated_at: float, oldest_pending_ts: float) -> float:
        base_ts = last_updated_at if last_updated_at > 0 else oldest_pending_ts
        return base_ts + self._UPDATE_INTERVAL_SECONDS

    def _normalize_recent_messages(
        self,
        rows: list[dict] | None,
    ) -> list[dict]:
        result: list[dict] = []
        if not rows:
            return result
        for row in rows:
            role = (row or {}).get("role", "")
            if role not in self._RELEVANT_ROLES:
                continue
            content = ((row or {}).get("content") or "").strip()
            if not content:
                continue
            result.append(
                {
                    "id": int((row or {}).get("id") or 0),
                    "ts": float((row or {}).get("ts") or 0.0),
                    "role": role,
                    "content": content,
                    "session_id": (row or {}).get("session_id") or "external",
                    "channel": (row or {}).get("channel") or "unknown",
                }
            )
        return result

    def _build_history_payload(self, rows: list[dict]) -> str:
        parts: list[str] = []
        for row in rows:
            role = row.get("role", "unknown")
            if role not in self._RELEVANT_ROLES:
                continue
            content = (row.get("content") or "").strip()
            if not content:
                continue
            sid = str(row.get("session_id") or "")[:8] or "global"
            parts.append(f"[session:{sid}][{role}] {content}")
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
            "【本次待处理的最新对话历史】\n"
            f"{history_payload or '（无）'}\n"
        )

    async def _generate_memory(self, prompt: str) -> str:
        llm = AsyncOpenAI(base_url=self._cfg.llm_base_url, api_key=self._cfg.llm_api_key)
        dispatcher = _MemoryOnlyDispatcher(
            reme=self._reme,
            history=self._history,
            memory_update_service=self,
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

    def _default_meta(self) -> dict[str, Any]:
        return {
            "version": self._META_VERSION,
            "last_processed_message_id": 0,
            "last_checked_at": 0.0,
            "last_updated_at": 0.0,
            "next_due_at": 0.0,
        }

    def _load_meta(self) -> dict[str, Any]:
        meta = self._default_meta()
        try:
            raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                meta.update(raw)
        except Exception:
            return meta

        if "last_update_ts" in meta and "last_updated_at" not in meta:
            meta["last_updated_at"] = float(meta.get("last_update_ts") or 0.0)

        meta["version"] = self._META_VERSION
        meta["last_processed_message_id"] = int(meta.get("last_processed_message_id") or 0)
        meta["last_checked_at"] = float(meta.get("last_checked_at") or 0.0)
        meta["last_updated_at"] = float(meta.get("last_updated_at") or 0.0)
        meta["next_due_at"] = float(
            meta.get("next_due_at")
            or (
                meta["last_updated_at"] + self._UPDATE_INTERVAL_SECONDS
                if meta["last_updated_at"] > 0
                else 0.0
            )
        )
        return meta

    def _save_meta(self, meta: dict[str, Any]) -> None:
        clean = self._default_meta()
        clean.update(meta)
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(
            json.dumps(clean, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
