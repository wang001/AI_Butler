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

        # 会话元表：会话目录、快照状态与恢复边界
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id                  TEXT PRIMARY KEY,
                title               TEXT    NOT NULL DEFAULT '',
                channel             TEXT    NOT NULL DEFAULT 'unknown',
                status              TEXT    NOT NULL DEFAULT 'active',
                preview             TEXT    NOT NULL DEFAULT '',
                compressed_summary  TEXT    NOT NULL DEFAULT '',
                summary_history_id  INTEGER NOT NULL DEFAULT 0,
                tail_messages_json  TEXT    NOT NULL DEFAULT '[]',
                created_at          REAL    NOT NULL,
                updated_at          REAL    NOT NULL,
                last_active_at      REAL    NOT NULL,
                last_message_at     REAL    NOT NULL DEFAULT 0
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

        session_existing = {row[1] for row in cur.execute("PRAGMA table_info(sessions)")}
        for col, definition in [
            ("title",              "TEXT NOT NULL DEFAULT ''"),
            ("channel",            "TEXT NOT NULL DEFAULT 'unknown'"),
            ("status",             "TEXT NOT NULL DEFAULT 'active'"),
            ("preview",            "TEXT NOT NULL DEFAULT ''"),
            ("compressed_summary", "TEXT NOT NULL DEFAULT ''"),
            ("summary_history_id", "INTEGER NOT NULL DEFAULT 0"),
            ("tail_messages_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("created_at",         "REAL NOT NULL DEFAULT 0"),
            ("updated_at",         "REAL NOT NULL DEFAULT 0"),
            ("last_active_at",     "REAL NOT NULL DEFAULT 0"),
            ("last_message_at",    "REAL NOT NULL DEFAULT 0"),
        ]:
            if col not in session_existing:
                cur.execute(f"ALTER TABLE sessions ADD COLUMN {col} {definition}")

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

    def append(self, role: str, content: str) -> int:
        """
        追加一条消息到 SQLite。

        所有角色（含 tool_call）均写入 messages 主表；
        user / assistant / tool 角色额外触发 FTS 索引写入。
        加锁保证同目录下多实例并发安全。
        """
        if not content:
            return 0

        ts = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO messages (ts, role, content, session_id, channel) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, role, content, self.session_id, self.channel),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def create_session(
        self,
        session_id: str | None = None,
        channel: str | None = None,
        title: str = "",
        status: str = "active",
    ) -> dict:
        """
        创建会话元记录；若已存在则保持原记录并返回最新状态。
        """
        sid = session_id or self.session_id
        ch = channel or self.channel
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO sessions (
                    id, title, channel, status, created_at, updated_at, last_active_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (sid, title, ch, status, now, now, now),
            )
            self._conn.commit()
        return self.get_session(sid) or {
            "id": sid,
            "title": title,
            "channel": ch,
            "status": status,
            "preview": "",
            "compressed_summary": "",
            "summary_history_id": 0,
            "tail_messages_json": "[]",
            "created_at": now,
            "updated_at": now,
            "last_active_at": now,
            "last_message_at": 0.0,
        }

    def touch_session(
        self,
        session_id: str | None = None,
        *,
        title: str | None = None,
        status: str | None = None,
        preview: str | None = None,
        compressed_summary: str | None = None,
        summary_history_id: int | None = None,
        tail_messages_json: str | None = None,
        last_active_at: float | None = None,
        last_message_at: float | None = None,
    ) -> None:
        """
        更新会话元数据 / 状态快照。
        """
        sid = session_id or self.session_id
        now = time.time()
        self.create_session(sid)

        assignments = ["updated_at = ?"]
        params: list = [now]

        if title is not None:
            assignments.append("title = ?")
            params.append(title)
        if status is not None:
            assignments.append("status = ?")
            params.append(status)
        if preview is not None:
            assignments.append("preview = ?")
            params.append(preview)
        if compressed_summary is not None:
            assignments.append("compressed_summary = ?")
            params.append(compressed_summary)
        if summary_history_id is not None:
            assignments.append("summary_history_id = ?")
            params.append(summary_history_id)
        if tail_messages_json is not None:
            assignments.append("tail_messages_json = ?")
            params.append(tail_messages_json)

        assignments.append("last_active_at = ?")
        params.append(last_active_at if last_active_at is not None else now)

        if last_message_at is not None:
            assignments.append("last_message_at = ?")
            params.append(last_message_at)

        params.append(sid)

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"UPDATE sessions SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
            self._conn.commit()

    def get_session(self, session_id: str | None = None) -> dict | None:
        """读取单个会话元数据。"""
        sid = session_id or self.session_id
        try:
            self._conn.commit()
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id, title, channel, status, preview,
                       compressed_summary, summary_history_id, tail_messages_json,
                       created_at, updated_at, last_active_at, last_message_at
                FROM sessions
                WHERE id = ?
                """,
                (sid,),
            )
            row = cur.fetchone()
        except sqlite3.OperationalError:
            return None

        if not row:
            return None

        return {
            "id": row[0],
            "title": row[1],
            "channel": row[2],
            "status": row[3],
            "preview": row[4],
            "compressed_summary": row[5],
            "summary_history_id": row[6],
            "tail_messages_json": row[7],
            "created_at": row[8],
            "updated_at": row[9],
            "last_active_at": row[10],
            "last_message_at": row[11],
        }

    def list_sessions(
        self,
        limit: int = 50,
        channel: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """按最近活跃时间列出会话。"""
        limit = min(max(1, limit), 200)
        conditions: list[str] = []
        params: list = []
        if channel:
            conditions.append("channel = ?")
            params.append(channel)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        try:
            self._conn.commit()
            cur = self._conn.cursor()
            cur.execute(
                f"""
                SELECT id, title, channel, status, preview,
                       compressed_summary, summary_history_id, tail_messages_json,
                       created_at, updated_at, last_active_at, last_message_at
                FROM sessions
                {where}
                ORDER BY last_active_at DESC, updated_at DESC
                LIMIT ?
                """,
                params,
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            {
                "id": row[0],
                "title": row[1],
                "channel": row[2],
                "status": row[3],
                "preview": row[4],
                "compressed_summary": row[5],
                "summary_history_id": row[6],
                "tail_messages_json": row[7],
                "created_at": row[8],
                "updated_at": row[9],
                "last_active_at": row[10],
                "last_message_at": row[11],
            }
            for row in rows
        ]

    def get_session_messages(
        self,
        session_id: str | None = None,
        limit: int = 200,
        before_id: int | None = None,
    ) -> list[dict]:
        """按会话读取历史消息。"""
        sid = session_id or self.session_id
        limit = min(max(1, limit), 500)
        conditions = ["session_id = ?"]
        params: list = [sid]

        if before_id is not None:
            conditions.append("id < ?")
            params.append(before_id)

        params.append(limit)

        try:
            self._conn.commit()
            cur = self._conn.cursor()
            cur.execute(
                f"""
                SELECT id, ts, role, content, session_id, channel
                FROM messages
                WHERE {' AND '.join(conditions)}
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            {
                "id": row[0],
                "ts": row[1],
                "role": row[2],
                "content": row[3],
                "session_id": row[4],
                "channel": row[5],
            }
            for row in reversed(rows)
        ]

    def get_last_message_id(self, session_id: str | None = None) -> int:
        """读取会话最新一条消息的主键 id。"""
        sid = session_id or self.session_id
        try:
            self._conn.commit()
            cur = self._conn.cursor()
            cur.execute(
                "SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (sid,),
            )
            row = cur.fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row[0]) if row else 0

    def get_summary_boundary_id(
        self,
        tail_history_rows: int,
        session_id: str | None = None,
    ) -> int:
        """
        根据“当前 tail 覆盖的历史行数”推导摘要边界。

        返回值含义：
          当前 compressed_summary 至少已经覆盖到的最后一条 history.message.id。
        """
        sid = session_id or self.session_id
        if tail_history_rows <= 0:
            return self.get_last_message_id(sid)

        try:
            self._conn.commit()
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (sid, tail_history_rows),
            )
            tail_ids = [int(row[0]) for row in cur.fetchall()]
            if not tail_ids:
                return 0
            oldest_tail_id = tail_ids[-1]
            cur.execute(
                """
                SELECT id
                FROM messages
                WHERE session_id = ? AND id < ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (sid, oldest_tail_id),
            )
            row = cur.fetchone()
        except sqlite3.OperationalError:
            return 0

        return int(row[0]) if row else 0

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
            self._conn.commit()
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
            self._conn.commit()
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

    def get_since(
        self,
        since_ts: float,
        limit: int = 500,
        role: str | None = None,
        channel: str | None = None,
    ) -> list[dict]:
        """
        查询指定时间戳之后的对话记录（不走 FTS，直接查主表）。

        Args:
            since_ts : 起始时间戳（不包含该时刻）
            limit    : 最多返回条数，默认 500，最大 2000
            role     : 可选，只返回指定角色
            channel  : 可选，只返回指定渠道
        """
        limit = min(max(1, limit), 2000)
        conditions = ["ts > ?"]
        params: list = [since_ts]

        if role:
            conditions.append("role = ?")
            params.append(role)
        if channel:
            conditions.append("channel = ?")
            params.append(channel)

        where = " AND ".join(conditions)
        params.append(limit)

        try:
            self._conn.commit()
            cur = self._conn.cursor()
            cur.execute(
                f"SELECT ts, role, content, session_id, channel "
                f"FROM messages WHERE {where} ORDER BY ts ASC LIMIT ?",
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

    def get_since_id(
        self,
        since_id: int,
        limit: int = 500,
        roles: list[str] | tuple[str, ...] | None = None,
        channel: str | None = None,
        max_id: int | None = None,
    ) -> list[dict]:
        """
        查询指定 message.id 之后的对话记录（按主键增量读取）。

        Args:
            since_id : 起始 message.id（不包含该 id）
            limit    : 最多返回条数，默认 500，最大 2000
            roles    : 可选，只返回指定角色列表
            channel  : 可选，只返回指定渠道
            max_id   : 可选，只返回 <= 该 id 的记录

        Returns:
            list of {"id", "ts", "role", "content", "session_id", "channel"}，
            按 id 升序
        """
        limit = min(max(1, limit), 2000)
        conditions = ["id > ?"]
        params: list = [since_id]

        if max_id is not None:
            conditions.append("id <= ?")
            params.append(max_id)

        if roles:
            placeholders = ", ".join("?" for _ in roles)
            conditions.append(f"role IN ({placeholders})")
            params.extend(list(roles))

        if channel:
            conditions.append("channel = ?")
            params.append(channel)

        where = " AND ".join(conditions)
        params.append(limit)

        try:
            self._conn.commit()
            cur = self._conn.cursor()
            cur.execute(
                f"""
                SELECT id, ts, role, content, session_id, channel
                FROM messages
                WHERE {where}
                ORDER BY id ASC
                LIMIT ?
                """,
                params,
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            {
                "id": row[0],
                "ts": row[1],
                "role": row[2],
                "content": row[3],
                "session_id": row[4],
                "channel": row[5],
            }
            for row in rows
        ]

    def get_latest_message_id(
        self,
        *,
        roles: list[str] | tuple[str, ...] | None = None,
        channel: str | None = None,
    ) -> int:
        """
        读取全局最新一条消息的主键 id。

        可按角色 / 渠道过滤；不传 session_id，适合 app 级扫描全部会话。
        """
        conditions: list[str] = []
        params: list = []

        if roles:
            placeholders = ", ".join("?" for _ in roles)
            conditions.append(f"role IN ({placeholders})")
            params.extend(list(roles))

        if channel:
            conditions.append("channel = ?")
            params.append(channel)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        try:
            self._conn.commit()
            cur = self._conn.cursor()
            cur.execute(
                f"SELECT id FROM messages {where} ORDER BY id DESC LIMIT 1",
                params,
            )
            row = cur.fetchone()
        except sqlite3.OperationalError:
            return 0

        return int(row[0]) if row else 0

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
