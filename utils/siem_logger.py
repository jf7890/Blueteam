"""SIEM Logger – Append AIAnalysisResult alerts to a JSONL file.

Writes one JSON object per line to ``logs/ai_alerts.jsonl`` for consumption
by Wazuh, ELK, or any SIEM that watches JSONL files.

Thread-safe: uses a module-level lock for file writes so that both the
sync LangGraph pipeline and async batch worker can log concurrently.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

_LOG_PATH = Path(settings.siem_log_path)
_write_lock = threading.Lock()

# Ensure the directory exists at import time.
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_alert(alert_json: dict) -> None:
    """Append a single alert dict as a JSON line to the SIEM log file."""
    try:
        line = json.dumps(alert_json, default=str, separators=(",", ":"))
        with _write_lock:
            with _LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        logger.debug("SIEM alert logged for request_id=%s", alert_json.get("request_id"))
    except OSError as exc:
        logger.warning("SIEM JSONL write failed: %s", exc)
