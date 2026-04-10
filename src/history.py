# -*- coding: utf-8 -*-
"""
history.py — 对话历史日志系统

ChatHistory：SQLite 单一存储（FTS5 全文检索 + 完整流水日志）
  - messages 表：存储所有消息，含 session_id / channel / role / content / ts
  - messages_fts：对 user / assistant / tool 角色建 FTS5 全文索引
  - tool_call 等元记录也写入 messages 表，但不参与 FTS 索引

每条记录携带 session_id 和 channel，解决多 Channel 并发写时的会话混杂问题：
  - session_id : UUID，每个 Butler 实例启动时生成，标识一次独立会话
  - channel    : 渠道标识，如 "cli" / "feishu" / "wecom" / "api"

并发安全：
  - WAL 模式：读不阻塞写，写不阻塞读
  - 类级别 threading.Lock（按 data_dir 共享）：保护同目录下多实例并发写

用法：
    history = ChatHistory(data_dir="/path/to/data", session_id="xxx", channel="cli")
    history.append("user", "今天天气真好")
    history.append("assistant", "是啊，适合出门走走")
    results = history.search("天气", limit=5)
    history.close()
"""

import sqlite3
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


CST = timezone(timedelta(hours=8))

# FTS 索引的角色白名单（其余角色写主表但不建索引）
_FTS_ROLES = {"user", "assistant", "tool"}


class ChatHistory:
    """SQLite FTS5 对话历史系统。"""

    # 类级别锁表：同一进程内相同 data_dir 的实例共享同一把锁
    _dir_locks: dict[str, threading.Lock] = {}
    _dir_locks_mutex = threading.Lock()

    @classmethod
    def _get_lock(cls, data_dir: Path) -> threading.Lock:
        key = str(data_dir.resolve())
        with cls._dir_locks_mutex:
            if key not in cls._dir_locks:
                cls._dir_locks[key] = threading.Lock()
            return cls._dir_locks[key]

    def __init__(self, data_dir: str, session_id: str, channel: str = "unknown"):
        """
        Args:
            data_dir   : 数据存储目录
            session_id : 当前会话唯一标识（UUID），由 Butler.create() 生成
            channel    : 渠道标识，如 "cli" / "feishu" / "wecom" / "api"
        """
        self.data_dir   = Path(data_dir)
        self.session_id = session_id
        self.channel    = channel
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = self._get_lock(self.data_dir)

        db_path = self.data_dir / "chat_history.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """初始化 SQLite 表结构，兼容旧表（自动补列）。"""
        cur = self._conn.cursor()

        # WAL 模式：读写并发，写不阻塞读
        cur.execute("PRAGMA journal_mode=WAL")

        # 主表：存储所有消息（含 tool_call 等元记录）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL    NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                session_id TEXT    NOT NULL DEFAULT '',
                channel    TEXT    NOT NULL DEFAULT 'unknown'
            )
        """)

        # 兼容旧库：若列不存在则补充
        existing = {row[1] for row in cur.execute("PRAGMA table_info(messages)")}
        for col, definition in [
            ("session_id", "TEXT NOT NULL DEFAULT ''"),
            ("channel",    "TEXT NOT NULL DEFAULT 'unknown'"),
        ]:
            if col not in existing:
                cur.execute(f"ALTER TABLE messages ADD COLUMN {col} {definition}")

        # FTS5 虚拟表（只索引白名单角色的内容）
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
            USING fts5(
                content,
                content='messages',
                content_rowid='id',
                tokenize='unicode61'
            )
        """)

        # 触发器：insert 后自动同步到 FTS（仅白名单角色）
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_ai
            AFTER INSERT ON messages
            WHEN new.role IN ('user', 'assistant', 'tool')
            BEGIN
                INSERT INTO messages_fts(rowid, content)
                VALUES (new.id, new.content);
            END
        """)

        self._conn.commit()

    def append(self, role: str, content: str):
        """
        追加一条消息到 SQLite。

        所有角色（含 tool_call）均写入 messages 主表；
        user / assistant / tool 角色额外触发 FTS 索引写入。
        加锁保证同目录下多实例并发安全。
        """
        if not content:
            return

        ts = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO messages (ts, role, content, session_id, channel) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, role, content, self.session_id, self.channel),
            )
            self._conn.commit()

    def get_by_date(
        self,
        date: str,
        role: str | None = None,
        channel: str | None = None,
    ) -> list[dict]:
        """
        按日期查询对话记录（不走 FTS，直接查主表）。

        Args:
            date    : 日期字符串，格式 "YYYY-MM-DD"，如 "2026-04-10"
            role    : 可选，只返回指定角色（"user" / "assistant" / "tool_call" 等）
            channel : 可选，只返回指定渠道

        Returns:
            list of {"ts", "role", "content", "session_id", "channel"}，按时间升序
        """
        try:
            day = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return []

        # 计算当天 0:00:00 ~ 次日 0:00:00 的 UTC 时间戳
        tz_day = day.replace(tzinfo=CST)
        ts_start = tz_day.timestamp()
        ts_end   = ts_start + 86400  # +24h

        conditions = ["ts >= ? AND ts < ?"]
        params: list = [ts_start, ts_end]

        if role:
            conditions.append("role = ?")
            params.append(role)
        if channel:
            conditions.append("channel = ?")
            params.append(channel)

        where = " AND ".join(conditions)

        try:
            cur = self._conn.cursor()
            cur.execute(
                f"SELECT ts, role, content, session_id, channel "
                f"FROM messages WHERE {where} ORDER BY ts ASC",
                params,
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            {"ts": ts, "role": role_, "content": content,
             "session_id": sid, "channel": ch}
            for ts, role_, content, sid, ch in rows
        ]

    def search(
        self,
        query: str,
        limit: int = 8,
        role: str | None = None,
        channel: str | None = None,
        session_id: str | None = None,
    ) -> list[dict]:
        """
        全文检索对话历史（不加锁，WAL 模式保证读写并发安全）。

        Args:
            query      : 搜索关键词，空格分隔多词（FTS5 OR 语义）
            limit      : 最多返回条数
            role       : 可选，只搜索指定角色（"user" 或 "assistant"）
            channel    : 可选，只搜索指定渠道
            session_id : 可选，只搜索指定会话

        Returns:
            list of {"ts", "role", "content", "session_id", "channel", "score"}，
            按相关度降序
        """
        if not query.strip():
            return []

        limit = min(max(1, limit), 20)

        conditions = ["messages_fts MATCH ?"]
        params: list = [self._fts_query(query)]

        if role:
            conditions.append("m.role = ?")
            params.append(role)
        if channel:
            conditions.append("m.channel = ?")
            params.append(channel)
        if session_id:
            conditions.append("m.session_id = ?")
            params.append(session_id)

        params.append(limit)
        where = " AND ".join(conditions)

        try:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                SELECT m.ts, m.role, m.content, m.session_id, m.channel,
                       -bm25(messages_fts) AS score
                FROM messages_fts
                JOIN messages m ON messages_fts.rowid = m.id
                WHERE {where}
                ORDER BY score DESC
                LIMIT ?
                """,
                params,
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return []

        if not rows:
            return []

        max_score = max((r[5] for r in rows), default=1.0) or 1.0
        return [
            {
                "ts":         ts,
                "role":       role_,
                "content":    content,
                "session_id": sid,
                "channel":    ch,
                "score":      score / max_score,
            }
            for ts, role_, content, sid, ch, score in rows
        ]

    @staticmethod
    def _fts_query(query: str) -> str:
        """
        把用户输入的关键词转成 FTS5 查询语法。
        多词之间用 OR 连接，命中词越多 bm25() 累加分越高，排序越靠前。
        """
        tokens = [t.strip() for t in query.split() if t.strip()]
        if not tokens:
            return '""'
        return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)

    def close(self):
        """关闭数据库连接。"""
        try:
            self._conn.close()
        except Exception:
            pass
