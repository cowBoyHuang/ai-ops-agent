"""Database interaction helpers for memory module."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class ChatDBStore:
    """SQLite-backed demo store for create/insert/delete/update operations."""

    def __init__(self, db_path: str = "src/.agent/memory.db") -> None:
        self.db_path = Path(db_path)
        self.create_database()
        self.create_tables()

    def create_database(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.close()

    def create_tables(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    total_message TEXT NOT NULL,
                    summary_message TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def insert_message(self, *, chat_id: str, user_id: str, total_message: str, summary_message: str) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                """
                INSERT INTO chat_memory(chat_id, user_id, total_message, summary_message)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, user_id, total_message, summary_message),
            )
            conn.commit()
            return int(cur.lastrowid or 0)
        finally:
            conn.close()

    def delete_message(self, row_id: int) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute("DELETE FROM chat_memory WHERE id = ?", (row_id,))
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()

    def update_message(self, row_id: int, *, total_message: str, summary_message: str) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                """
                UPDATE chat_memory
                SET total_message = ?, summary_message = ?
                WHERE id = ?
                """,
                (total_message, summary_message, row_id),
            )
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()

