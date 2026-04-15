"""Database interaction helpers for memory module (MySQL)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

try:
    import pymysql
    from pymysql.cursors import DictCursor
except Exception:  # pragma: no cover - optional dependency at runtime
    pymysql = None
    DictCursor = None  # type: ignore[assignment]

try:
    from sqlalchemy.engine import make_url
except Exception:  # pragma: no cover
    make_url = None  # type: ignore[assignment]


class ChatDBStore:
    """MySQL-backed store for total_message and summary_message tables."""

    def __init__(self, mysql_dsn: str | None = None) -> None:
        self.mysql_dsn = str(mysql_dsn or os.getenv("MYSQL_DSN", "")).strip()
        self._conn_args: dict[str, Any] = {}
        self._enabled = False

        if not self.mysql_dsn or pymysql is None or make_url is None:
            return

        try:
            self._conn_args = self._parse_mysql_dsn(self.mysql_dsn)
            self.create_tables()
            self._enabled = True
        except Exception:
            self._conn_args = {}
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _parse_mysql_dsn(self, dsn: str) -> dict[str, Any]:
        url = make_url(dsn)
        if not str(url.drivername).startswith("mysql"):
            raise ValueError("MYSQL_DSN must be a mysql dsn")
        if not url.database:
            raise ValueError("MYSQL_DSN database is required")

        return {
            "host": str(url.host or "127.0.0.1"),
            "port": int(url.port or 3306),
            "user": str(url.username or ""),
            "password": str(url.password or ""),
            "database": str(url.database),
            "charset": "utf8mb4",
            "autocommit": False,
            "cursorclass": DictCursor,
        }

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        if not self._conn_args or pymysql is None:
            raise RuntimeError("mysql not configured")
        conn = pymysql.connect(**self._conn_args)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_tables(self) -> bool:
        if not self._conn_args:
            return False
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS total_message (
                        id BIGINT NOT NULL AUTO_INCREMENT,
                        chat_id VARCHAR(128) NOT NULL,
                        role VARCHAR(32) NOT NULL,
                        content TEXT NOT NULL,
                        PRIMARY KEY (id),
                        KEY idx_total_message_chat_id (chat_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS summary_message (
                        id BIGINT NOT NULL AUTO_INCREMENT,
                        user_id VARCHAR(128) NOT NULL,
                        chat_id VARCHAR(128) NOT NULL,
                        content TEXT NOT NULL,
                        PRIMARY KEY (id),
                        KEY idx_summary_message_chat_id (chat_id),
                        KEY idx_summary_message_user_chat (user_id, chat_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
        return True

    # ===== total_message CRUD =====
    def create_total_message(self, *, chat_id: str, role: str, content: str) -> int:
        if not self._enabled:
            return 0
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO total_message(chat_id, role, content)
                    VALUES (%s, %s, %s)
                    """,
                    (chat_id, role, content),
                )
                return int(cur.lastrowid or 0)

    def get_total_message(self, row_id: int) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, chat_id, role, content FROM total_message WHERE id = %s",
                    (row_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def list_total_messages_by_chat_id(self, chat_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        if not self._enabled:
            return []
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, chat_id, role, content
                    FROM total_message
                    WHERE chat_id = %s
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (chat_id, max(1, int(limit))),
                )
                rows = cur.fetchall() or []
                return [dict(item) for item in rows]

    def update_total_message(self, row_id: int, *, role: str | None = None, content: str | None = None) -> int:
        if not self._enabled:
            return 0

        sets: list[str] = []
        args: list[Any] = []
        if role is not None:
            sets.append("role = %s")
            args.append(role)
        if content is not None:
            sets.append("content = %s")
            args.append(content)
        if not sets:
            return 0

        args.append(row_id)
        sql = f"UPDATE total_message SET {', '.join(sets)} WHERE id = %s"
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                return int(cur.rowcount or 0)

    def delete_total_message(self, row_id: int) -> int:
        if not self._enabled:
            return 0
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM total_message WHERE id = %s", (row_id,))
                return int(cur.rowcount or 0)

    # ===== summary_message CRUD =====
    def create_summary_message(self, *, user_id: str, chat_id: str, content: str) -> int:
        if not self._enabled:
            return 0
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO summary_message(user_id, chat_id, content)
                    VALUES (%s, %s, %s)
                    """,
                    (user_id, chat_id, content),
                )
                return int(cur.lastrowid or 0)

    def get_summary_message(self, row_id: int) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, user_id, chat_id, content FROM summary_message WHERE id = %s",
                    (row_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def list_summary_messages_by_chat_id(self, chat_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        if not self._enabled:
            return []
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, chat_id, content
                    FROM summary_message
                    WHERE chat_id = %s
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (chat_id, max(1, int(limit))),
                )
                rows = cur.fetchall() or []
                return [dict(item) for item in rows]

    def update_summary_message(self, row_id: int, *, content: str) -> int:
        if not self._enabled:
            return 0
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE summary_message SET content = %s WHERE id = %s",
                    (content, row_id),
                )
                return int(cur.rowcount or 0)

    def delete_summary_message(self, row_id: int) -> int:
        if not self._enabled:
            return 0
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM summary_message WHERE id = %s", (row_id,))
                return int(cur.rowcount or 0)

    # ===== backward-compatible methods =====
    def insert_message(self, *, chat_id: str, user_id: str, total_message: str, summary_message: str) -> int:
        """Compatibility wrapper used by historical memory module calls."""
        if not self._enabled:
            return 0

        user_content = ""
        assistant_content = ""
        for line in str(total_message or "").splitlines():
            if line.startswith("user:"):
                user_content = line[len("user:") :].strip()
            elif line.startswith("assistant:"):
                assistant_content = line[len("assistant:") :].strip()

        if user_content:
            self.create_total_message(chat_id=chat_id, role="user", content=user_content)
        if assistant_content:
            self.create_total_message(chat_id=chat_id, role="assistant", content=assistant_content)

        return self.create_summary_message(user_id=user_id, chat_id=chat_id, content=summary_message)

    def delete_message(self, row_id: int) -> int:
        return self.delete_summary_message(row_id)

    def update_message(self, row_id: int, *, total_message: str, summary_message: str) -> int:
        _ = total_message
        return self.update_summary_message(row_id, content=summary_message)
