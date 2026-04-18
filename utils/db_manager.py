"""SQLite database manager for SOC Copilot alert persistence.

Manages a local ``data/copilot_data.db`` database with an ``alerts`` table.
Provides both synchronous and async-compatible insert functions so that
the sync LangGraph pipeline and the async batch worker can both persist alerts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path — resolved from settings
# ---------------------------------------------------------------------------
_DB_PATH = Path(settings.sqlite_db_path)

# Thread-local storage for SQLite connections (SQLite objects can only be
# used in the thread that created them).
_local = threading.local()


def _get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it on first call."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        _local.conn = conn
    return conn


def init_db() -> None:
    """Create the ``alerts`` table if it doesn't exist."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id      TEXT    NOT NULL,
            timestamp       TEXT    NOT NULL,
            source_ip       TEXT    DEFAULT 'unknown',
            is_malicious    INTEGER NOT NULL DEFAULT 0,
            attack_type     TEXT,
            severity        TEXT    DEFAULT 'info',
            full_context    TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_request_id
        ON alerts (request_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_timestamp
        ON alerts (timestamp)
    """)
    conn.commit()
    logger.info("SQLite database initialised at %s", _DB_PATH)


def insert_alert(alert_json: dict) -> None:
    """Insert a single alert record (from an ``AIAnalysisResult`` dict)."""
    try:
        conn = _get_connection()
        conn.execute(
            """
            INSERT INTO alerts
                (request_id, timestamp, source_ip, is_malicious, attack_type, severity, full_context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert_json.get("request_id", ""),
                alert_json.get("timestamp", ""),
                alert_json.get("network", {}).get("srcip", "unknown"),
                int(alert_json.get("is_malicious", False)),
                alert_json.get("ai_analysis", {}).get("attack_type"),
                alert_json.get("ai_analysis", {}).get("severity", "info"),
                json.dumps(alert_json, default=str),
            ),
        )
        conn.commit()
        logger.debug("Alert inserted for request_id=%s", alert_json.get("request_id"))
    except sqlite3.Error as exc:
        logger.warning("SQLite insert failed: %s", exc)


def fetch_alerts(limit: int = 100, offset: int = 0) -> list[dict]:
    """Return the most recent alerts as a list of dicts (newest first)."""
    try:
        conn = _get_connection()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        conn.row_factory = None
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        logger.warning("SQLite fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Initialise on import
# ---------------------------------------------------------------------------
init_db()
