"""JSON-line logs and SQLite persistence of runs."""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

from .config import data_dir

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(agent_id: str, status: str, message: str = "") -> None:
    """Append one JSON line per interaction. status: running | completed | error."""
    logs = data_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    entry = {"agent_id": agent_id, "timestamp": _now(), "status": status, "message": message}
    with _lock, open(logs / "agents.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def last_events() -> dict[str, dict]:
    """Last log entry per agent, for the /status endpoint."""
    path = data_dir() / "logs" / "agents.jsonl"
    if not path.exists():
        return {}
    latest: dict[str, dict] = {}
    with _lock, open(path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[entry["agent_id"]] = entry
    return latest


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(data_dir() / "agent_reports.db")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS runs (
               session_id TEXT PRIMARY KEY,
               status TEXT NOT NULL,
               payload TEXT NOT NULL,
               created_at TEXT NOT NULL,
               updated_at TEXT NOT NULL)"""
    )
    return conn


def save_run(session_id: str, status: str, payload_json: str) -> None:
    now = _now()
    with _lock:
        conn = _conn()
        try:
            conn.execute(
                """INSERT INTO runs (session_id, status, payload, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       status = excluded.status,
                       payload = excluded.payload,
                       updated_at = excluded.updated_at""",
                (session_id, status, payload_json, now, now),
            )
            conn.commit()
        finally:
            conn.close()


def load_run(session_id: str) -> Optional[str]:
    with _lock:
        conn = _conn()
        try:
            row = conn.execute("SELECT payload FROM runs WHERE session_id = ?", (session_id,)).fetchone()
        finally:
            conn.close()
    return row[0] if row else None
