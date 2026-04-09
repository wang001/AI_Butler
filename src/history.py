# -*- coding: utf-8 -*-
"""
history.py — 对话历史日志系统

ChatHistory：双写 JSONL + SQLite FTS5
  - JSONL：append-only，完整原始日志，按日切分（chat_YYYY-MM-DD.jsonl）
  - SQLite FTS5：全文检索索引，支持 search(query, limit, role)

用法：
    history = ChatHistory(data_dir="/path/to/memory")
    history.append("user", "今天天气真好")
    history.append("assistant", "是啊，适合出门走走")
    results = history.search("天气", limit=5)
    history.close()
"""

import json
import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


CST = timezone(timedelta(hours=8))


class ChatHistory:
    """JSONL + SQLite FTS5 双写对话历史系统。"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # SQLite 数据库路径
        db_path = self.data_dir / "chat_history.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """初始化 SQLite FTS5 表。"""
        cur = self._conn.cursor()
        # 主表：存储消息元数据（不参与 FTS）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      REAL NOT NULL,
                role    TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)
        # FTS5 虚拟表：全文索引（内容来自 messages.content）
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
            USING fts5(
                content,
                content='messages',
                content_rowid='id',
                tokenize='unicode61'
            )
        """)
        # 触发器：insert 时自动同步到 FTS
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_ai
            AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content)
                VALUES (new.id, new.content);
            END
        """)
        self._conn.commit()

    def _jsonl_path(self) -> Path:
        """按日切分的 JSONL 文件路径。"""
        date_str = datetime.now(CST).strftime("%Y-%m-%d")
        return self.data_dir / f"chat_{date_str}.jsonl"

    def append(self, role: str, content: str):
        """追加一条消息到 JSONL 和 SQLite。"""
        if not content:
            return

        ts = time.time()

        # 写 JSONL
        record = {"ts": ts, "role": role, "content": content}
        with open(self._jsonl_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 写 SQLite（只索引 user/assistant/tool，跳过 tool_call 记录）
        if role in ("user", "assistant", "tool"):
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO messages (ts, role, content) VALUES (?, ?, ?)",
                (ts, role, content),
            )
            self._conn.commit()

    def search(
        self,
        query: str,
        limit: int = 8,
        role: str | None = None,
    ) -> list[dict]:
        """
        全文检索对话历史。

        Args:
            query: 搜索关键词，空格分隔多词（FTS5 OR 语义）
            limit: 最多返回条数
            role:  可选，只搜索指定角色（"user" 或 "assistant"）

        Returns:
            list of {"ts", "role", "content", "score"}，按相关度降序
        """
        if not query.strip():
            return []

        limit = min(max(1, limit), 20)

        # FTS5 bm25() 返回负数（越小越相关），转成正数分数
        try:
            cur = self._conn.cursor()
            if role:
                cur.execute(
                    """
                    SELECT m.ts, m.role, m.content,
                           -bm25(messages_fts) AS score
                    FROM messages_fts
                    JOIN messages m ON messages_fts.rowid = m.id
                    WHERE messages_fts MATCH ?
                      AND m.role = ?
                    ORDER BY score DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), role, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT m.ts, m.role, m.content,
                           -bm25(messages_fts) AS score
                    FROM messages_fts
                    JOIN messages m ON messages_fts.rowid = m.id
                    WHERE messages_fts MATCH ?
                    ORDER BY score DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), limit),
                )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # FTS 表可能还没有数据
            return []

        results = []
        max_score = max((r[3] for r in rows), default=1.0) or 1.0
        for ts, r, content, score in rows:
            results.append({
                "ts": ts,
                "role": r,
                "content": content,
                "score": score / max_score,  # 归一化到 0~1
            })
        return results

    @staticmethod
    def _fts_query(query: str) -> str:
        """
        把用户输入的关键词转成 FTS5 查询语法。
        多词之间用 OR 连接（任一词命中即可），不要求全部命中。
        """
        tokens = [t.strip() for t in query.split() if t.strip()]
        if not tokens:
            return '""'
        # FTS5 中每个 token 用双引号包裹防止特殊字符问题，OR 连接
        return " OR ".join(f'"{t}"' for t in tokens)

    def close(self):
        """关闭数据库连接。"""
        try:
            self._conn.close()
        except Exception:
            pass
