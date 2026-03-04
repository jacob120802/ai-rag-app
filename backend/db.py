"""SQLite helpers for sessions, chat history, and uploaded documents.

This module keeps DB logic separate from API routes to maintain clean architecture.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ChatDB:
    """Simple SQLite wrapper for local development.

    In production, replace this with PostgreSQL + SQLAlchemy/Alembic.
    """

    def __init__(self, sqlite_path: str) -> None:
        self.db_path = Path(sqlite_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL
                )
                """
            )

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions(session_id, created_at) VALUES(?, ?)",
                (session_id, datetime.now(timezone.utc).isoformat()),
            )
        return session_id

    def save_message(self, session_id: str, role: str, content: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES(?, ?, ?, ?)",
                (session_id, role, content, datetime.now(timezone.utc).isoformat()),
            )

    def get_messages(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def save_document(self, session_id: str, filename: str) -> str:
        doc_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO documents(doc_id, session_id, filename, uploaded_at) VALUES(?, ?, ?, ?)",
                (doc_id, session_id, filename, datetime.now(timezone.utc).isoformat()),
            )
        return doc_id

    def list_documents(self, session_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT doc_id, filename, uploaded_at FROM documents WHERE session_id = ? ORDER BY uploaded_at DESC",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]
