"""
ChatHistory — 原始对话日志持久化

架构：
  - chat_history.jsonl  : append-only 原始记录，每行一个消息 turn
  - chat_index.db       : SQLite FTS5 索引 + 元数据，存文件偏移量引用

JSONL 每行格式：
  {
    "id": "<uuid4>",          # 唯一消息 ID
    "session_id": "<uuid4>",  # 本次启动的会话 ID
    "ts": 1234567890.123,     # Unix timestamp（float）
    "role": "user"|"assistant"|"tool",
    "content": "...",         # 消息正文
    "extra": {}               # 可选附加字段（tool_call_id、tool_name 等）
  }

SQLite 表：
  messages(
    id TEXT PRIMARY KEY,
    session_id TEXT,
    ts REAL,
    role TEXT,
    content_snippet TEXT,   -- content 前 500 字符，供快速预览
    jsonl_offset INTEGER,   -- 该行在 JSONL 文件中的字节偏移，用于精确读取原文
    jsonl_len INTEGER        -- 该行字节长度
  )
  messages_fts(content)     -- FTS5 虚表，content 列做全文索引
                               使用 trigram tokenizer，天然支持中文
"""

import json
import sqlite3
import uuid
import time
from pathlib import Path
from typing import Any


class ChatHistory:
    """管理 JSONL 原始日志 + SQLite FTS5 索引的对话历史系统。"""

    JSONL_FILENAME = "chat_history.jsonl"
    DB_FILENAME = "chat_index.db"

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.data_dir / self.JSONL_FILENAME
        self.db_path = self.data_dir / self.DB_FILENAME

        self._session_id = str(uuid.uuid4())
        self._conn = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")   # 并发友好
        conn.execute("PRAGMA synchronous=NORMAL")  # 性能与安全的平衡

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id              TEXT PRIMARY KEY,
                session_id      TEXT NOT NULL,
                ts              REAL NOT NULL,
                role            TEXT NOT NULL,
                content_snippet TEXT,
                jsonl_offset    INTEGER NOT NULL,
                jsonl_len       INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
            CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                id UNINDEXED,
                content,
                tokenize = 'trigram'
            );
        """)
        conn.commit()
        return conn

    # ── 写入 ──────────────────────────────────────────────────────────────────

    def append(
        self,
        role: str,
        content: str,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """
        追加一条消息到 JSONL 文件并同步更新 SQLite 索引。
        返回消息的 UUID。
        """
        if not content:
            return ""

        msg_id = str(uuid.uuid4())
        ts = time.time()
        record = {
            "id": msg_id,
            "session_id": self._session_id,
            "ts": ts,
            "role": role,
            "content": content,
        }
        if extra:
            record["extra"] = extra

        line = json.dumps(record, ensure_ascii=False) + "\n"
        line_bytes = line.encode("utf-8")

        # append 到 JSONL，记录偏移量
        with open(self.jsonl_path, "ab") as f:
            offset = f.seek(0, 2)   # 当前文件末尾 = 本行起始偏移
            f.write(line_bytes)

        snippet = content[:500]

        self._conn.execute(
            """
            INSERT OR IGNORE INTO messages
                (id, session_id, ts, role, content_snippet, jsonl_offset, jsonl_len)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (msg_id, self._session_id, ts, role, snippet, offset, len(line_bytes)),
        )
        # tool_call 是内务记录，不建 FTS5 索引，避免出现在用户搜索结果里
        if role != "tool_call":
            self._conn.execute(
                "INSERT INTO messages_fts(id, content) VALUES (?, ?)",
                (msg_id, content),
            )
        self._conn.commit()
        return msg_id

    def append_turn(self, user_content: str, assistant_content: str) -> None:
        """一次性追加一轮对话（user + assistant）。"""
        if user_content:
            self.append("user", user_content)
        if assistant_content:
            self.append("assistant", assistant_content)

    # ── 检索 ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 10,
        role: str | None = None,
        since_ts: float | None = None,
    ) -> list[dict]:
        """
        多关键词相关性搜索，按命中关键词数量打分排序。

        策略：
          1. 按空格拆分查询为多个关键词
          2. 每个关键词独立检索（>= 3 字符走 FTS5，< 3 字符走 LIKE）
          3. 合并结果，按 (命中词数 / 总词数) 打分，同分按时间降序
          4. 不要求所有关键词全部命中 — 命中 1 个也返回，只是排在后面

        Args:
            query:    搜索关键词（支持中文，多个词用空格分隔）
            limit:    最多返回条数
            role:     过滤角色，None 表示不限
            since_ts: 只返回该时间戳之后的记录（Unix float）

        Returns:
            list[dict]，每项包含：id, session_id, ts, role, content, snippet, score
            score = 命中关键词数 / 总关键词数（0~1）
        """
        # 拆分关键词（去空、去重、保序）
        keywords = list(dict.fromkeys(k for k in query.split() if k))
        if not keywords:
            return []

        # 公共过滤条件
        extra_filters: list[str] = []
        extra_params: list[Any] = []
        if role:
            extra_filters.append("m.role = ?")
            extra_params.append(role)
        if since_ts is not None:
            extra_filters.append("m.ts >= ?")
            extra_params.append(since_ts)
        where_extra = (" AND " + " AND ".join(extra_filters)) if extra_filters else ""

        # 每个关键词独立查询，收集 msg_id → 命中的关键词集合
        # 多取一些候选（limit * 3），最后统一排序截断
        candidate_limit = max(limit * 3, 30)
        hit_map: dict[str, set[str]] = {}      # msg_id → {命中的关键词}
        row_cache: dict[str, tuple] = {}        # msg_id → row tuple

        for kw in keywords:
            if len(kw) >= 3:
                # FTS5 trigram 路径
                fts_term = f'"{self._escape_fts_term(kw)}"'
                sql = f"""
                    SELECT m.id, m.session_id, m.ts, m.role,
                           m.content_snippet, m.jsonl_offset, m.jsonl_len
                    FROM messages_fts f
                    JOIN messages m ON f.id = m.id
                    WHERE messages_fts MATCH ?
                    {where_extra}
                    LIMIT ?
                """
                params: list[Any] = [fts_term] + extra_params + [candidate_limit]
            else:
                # 短词 LIKE 路径
                escaped = kw.replace("%", r"\%").replace("_", r"\_")
                sql = f"""
                    SELECT m.id, m.session_id, m.ts, m.role,
                           m.content_snippet, m.jsonl_offset, m.jsonl_len
                    FROM messages m
                    WHERE m.content_snippet LIKE ? ESCAPE '\\'
                    {where_extra}
                    LIMIT ?
                """
                params = [f"%{escaped}%"] + extra_params + [candidate_limit]

            for row in self._conn.execute(sql, params).fetchall():
                msg_id = row[0]
                hit_map.setdefault(msg_id, set()).add(kw)
                if msg_id not in row_cache:
                    row_cache[msg_id] = row

        if not hit_map:
            return []

        # 打分：score = 命中词数 / 总词数
        n_keywords = len(keywords)
        scored: list[tuple[float, float, str]] = []
        for msg_id, hits in hit_map.items():
            score = len(hits) / n_keywords
            ts = row_cache[msg_id][2]  # ts 字段
            scored.append((score, ts, msg_id))

        # 按 score 降序，同分按时间降序
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        results = []
        for score, _ts, msg_id in scored[:limit]:
            row = row_cache[msg_id]
            msg_id, session_id, ts, role_, snippet, offset, length = row
            content = self._read_jsonl_line(offset, length)
            results.append({
                "id": msg_id,
                "session_id": session_id,
                "ts": ts,
                "role": role_,
                "content": content,
                "snippet": snippet,
                "score": round(score, 2),
            })

        return results

    def _read_jsonl_line(self, offset: int, length: int) -> str:
        """从 JSONL 文件的指定偏移量读取一行，返回 content 字段。"""
        try:
            with open(self.jsonl_path, "rb") as f:
                f.seek(offset)
                raw = f.read(length).decode("utf-8").strip()
            record = json.loads(raw)
            return record.get("content", "")
        except Exception:
            return ""

    @staticmethod
    def _escape_fts_term(term: str) -> str:
        """
        将单个关键词转义为 FTS5 安全的短语片段（不含外层引号）。
        调用方负责用双引号包裹和 AND 组合。
        """
        # 去掉 FTS5 特殊字符（" 和 *），防止语法错误
        return term.replace('"', "").replace("*", "").strip()

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """返回简单统计信息。"""
        row = self._conn.execute(
            "SELECT COUNT(*), MIN(ts), MAX(ts) FROM messages"
        ).fetchone()
        total, min_ts, max_ts = row
        jsonl_size = self.jsonl_path.stat().st_size if self.jsonl_path.exists() else 0
        return {
            "total_messages": total,
            "earliest_ts": min_ts,
            "latest_ts": max_ts,
            "jsonl_size_bytes": jsonl_size,
            "session_id": self._session_id,
        }

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass
