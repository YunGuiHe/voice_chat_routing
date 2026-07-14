from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ConversationMessage:
    id: int
    session_id: str
    role: str
    content: str
    scene: str
    created_at: str


@dataclass
class ConversationSummary:
    session_id: str
    summary: str
    summarized_message_id: int
    updated_at: str


@dataclass
class LongTermMemory:
    id: int
    user_id: str
    memory_type: str
    content: str
    evidence: str
    confidence: float
    priority: int
    is_active: int
    created_at: str
    updated_at: str
    last_used_at: str | None


class LongTermMemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.setup()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def setup(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    scene TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    summarized_message_id INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON conversation_messages(session_id, id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    evidence TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    priority INTEGER NOT NULL DEFAULT 0,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_long_term_memories_user_active
                ON long_term_memories(user_id, is_active, priority, updated_at)
                """
            )

    def reset_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM conversation_summaries WHERE session_id = ?", (session_id,))

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        scene: str = "",
    ) -> int:
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported role: {role}")
        created_at = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversation_messages(session_id, role, content, scene, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, scene, created_at),
            )
            return int(cursor.lastrowid)

    def append_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        scene: str,
    ) -> None:
        self.append_message(session_id, "user", user_query, scene)
        self.append_message(session_id, "assistant", assistant_answer, scene)

    def get_messages(self, session_id: str) -> list[ConversationMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, scene, created_at
                FROM conversation_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [ConversationMessage(**dict(row)) for row in rows]

    def count_rounds(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM conversation_messages
                WHERE session_id = ? AND role = 'user'
                """,
                (session_id,),
            ).fetchone()
        return int(row["count"] if row else 0)

    def get_recent_messages(
        self,
        session_id: str,
        recent_rounds: int,
    ) -> list[ConversationMessage]:
        limit = max(recent_rounds, 0) * 2
        if limit == 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, scene, created_at
                FROM conversation_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [ConversationMessage(**dict(row)) for row in reversed(rows)]

    def get_summary(self, session_id: str) -> ConversationSummary | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, summary, summarized_message_id, updated_at
                FROM conversation_summaries
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return ConversationSummary(**dict(row)) if row else None

    def upsert_summary(
        self,
        session_id: str,
        summary: str,
        summarized_message_id: int,
    ) -> None:
        updated_at = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_summaries(
                    session_id,
                    summary,
                    summarized_message_id,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    summarized_message_id = excluded.summarized_message_id,
                    updated_at = excluded.updated_at
                """,
                (session_id, summary, summarized_message_id, updated_at),
            )

    def get_messages_between(
        self,
        session_id: str,
        after_message_id: int,
        through_message_id: int,
    ) -> list[ConversationMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, scene, created_at
                FROM conversation_messages
                WHERE session_id = ? AND id > ? AND id <= ?
                ORDER BY id ASC
                """,
                (session_id, after_message_id, through_message_id),
            ).fetchall()
        return [ConversationMessage(**dict(row)) for row in rows]

    def get_active(self, user_id: str, limit: int = 5) -> list[LongTermMemory]:
        limit = max(limit, 0)
        if limit == 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    user_id,
                    memory_type,
                    content,
                    evidence,
                    confidence,
                    priority,
                    is_active,
                    created_at,
                    updated_at,
                    last_used_at
                FROM long_term_memories
                WHERE user_id = ? AND is_active = 1
                ORDER BY priority DESC, updated_at DESC, id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [LongTermMemory(**dict(row)) for row in rows]

    def mark_used(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        now = datetime.now().isoformat(timespec="seconds")
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE long_term_memories
                SET last_used_at = ?
                WHERE id IN ({placeholders})
                """,
                (now, *memory_ids),
            )

    def upsert(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        evidence: str = "",
        confidence: float = 1.0,
        priority: int = 0,
        memory_id: int | None = None,
    ) -> int:
        clean_content = " ".join(content.split())
        if not clean_content:
            raise ValueError("长期记忆内容不能为空")

        now = datetime.now().isoformat(timespec="seconds")
        confidence = min(max(float(confidence), 0.0), 1.0)
        priority = min(max(int(priority), 0), 5)
        clean_type = memory_type.strip() or "preference"
        clean_evidence = " ".join(evidence.split())

        with self._connect() as conn:
            if memory_id is not None:
                row = conn.execute(
                    "SELECT id FROM long_term_memories WHERE id = ? AND user_id = ?",
                    (memory_id, user_id),
                ).fetchone()
                if row:
                    conn.execute(
                        """
                        UPDATE long_term_memories
                        SET
                            memory_type = ?,
                            content = ?,
                            evidence = ?,
                            confidence = ?,
                            priority = ?,
                            is_active = 1,
                            updated_at = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (
                            clean_type,
                            clean_content,
                            clean_evidence,
                            confidence,
                            priority,
                            now,
                            memory_id,
                            user_id,
                        ),
                    )
                    return memory_id

            row = conn.execute(
                """
                SELECT id
                FROM long_term_memories
                WHERE user_id = ? AND content = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id, clean_content),
            ).fetchone()
            if row:
                existing_id = int(row["id"])
                conn.execute(
                    """
                    UPDATE long_term_memories
                    SET
                        memory_type = ?,
                        evidence = ?,
                        confidence = ?,
                        priority = ?,
                        is_active = 1,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (clean_type, clean_evidence, confidence, priority, now, existing_id),
                )
                return existing_id

            cursor = conn.execute(
                """
                INSERT INTO long_term_memories(
                    user_id,
                    memory_type,
                    content,
                    evidence,
                    confidence,
                    priority,
                    is_active,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    user_id,
                    clean_type,
                    clean_content,
                    clean_evidence,
                    confidence,
                    priority,
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def deactivate(self, memory_id: int, user_id: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE long_term_memories
                SET is_active = 0, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (now, memory_id, user_id),
            )
