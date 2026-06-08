import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SESSIONS_FILE = Path("sessions.json")
DATABASE_URL = os.getenv("DATABASE_URL")


def _connect():
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(
        DATABASE_URL,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )


def _ensure_sessions_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papertrail_sessions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_named BOOLEAN NOT NULL DEFAULT FALSE
        )
        """
    )


def _load_file_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_file_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def load_sessions() -> dict:
    if not DATABASE_URL:
        return _load_file_sessions()

    with _connect() as conn:
        _ensure_sessions_table(conn)
        rows = conn.execute(
            """
            SELECT id, name, created_at, is_named
            FROM papertrail_sessions
            ORDER BY created_at DESC
            """
        ).fetchall()
    return {
        row["id"]: {
            "id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "is_named": bool(row["is_named"]),
        }
        for row in rows
    }


def save_sessions(sessions_meta: dict) -> None:
    if not DATABASE_URL:
        _save_file_sessions(sessions_meta)
        return

    with _connect() as conn:
        _ensure_sessions_table(conn)
        with conn.cursor() as cur:
            for session in sessions_meta.values():
                cur.execute(
                    """
                    INSERT INTO papertrail_sessions (id, name, created_at, is_named)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET name = EXCLUDED.name,
                        created_at = EXCLUDED.created_at,
                        is_named = EXCLUDED.is_named
                    """,
                    (
                        session["id"],
                        session["name"],
                        session["created_at"],
                        bool(session.get("is_named", False)),
                    ),
                )
