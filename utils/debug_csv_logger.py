"""CSV debug logger for payload-processing snapshots.

Writes one row per completed pipeline branch when ``DEBUG`` is enabled.
Intended for debugging payload transformations and RAG inputs/results.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
from pathlib import Path
from typing import Any

from config.settings import settings
from schema.state import GraphState

logger = logging.getLogger(__name__)

_CSV_PATH = Path(settings.debug_csv_path)
_WRITE_LOCK = threading.Lock()
_FIELDNAMES = [
    "raw_request",
    "normalized_data",
    "suspicious_payloads",
    "rag_query",
    "rag_query_result",
]

_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


def _to_json(value: Any) -> str:
    """Serialize values consistently for CSV columns."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def _extract_raw_request(state: GraphState) -> str:
    """Return the original raw HTTP payload when available."""
    raw_http_text = state.get("raw_http_text")
    if raw_http_text:
        return raw_http_text

    raw_request = state.get("raw_request")
    if raw_request is None:
        return ""

    dump = getattr(raw_request, "model_dump", None)
    if callable(dump):
        return _to_json(dump())

    return str(raw_request)


def _derive_rag_query(state: GraphState) -> list[str]:
    """Return the payloads that were or would be used as the RAG query."""
    return state.get("suspicious_payloads") or state.get("normalized_payloads", [])


def log_debug_snapshot(
    state: GraphState,
    *,
    rag_query: list[str] | None = None,
    rag_query_result: list[str] | None = None,
) -> None:
    """Append a debug snapshot row when ``DEBUG`` is enabled."""
    if not settings.debug:
        return

    row = {
        "raw_request": _extract_raw_request(state),
        "normalized_data": _to_json(state.get("normalized_payloads", [])),
        "suspicious_payloads": _to_json(state.get("suspicious_payloads", [])),
        "rag_query": _to_json(_derive_rag_query(state) if rag_query is None else rag_query),
        "rag_query_result": _to_json(
            state.get("rag_context", []) if rag_query_result is None else rag_query_result
        ),
    }

    if not any(row.values()):
        return

    try:
        with _WRITE_LOCK:
            write_header = not _CSV_PATH.exists() or _CSV_PATH.stat().st_size == 0
            with _CSV_PATH.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        logger.debug("Debug CSV snapshot logged to %s", _CSV_PATH)
    except OSError as exc:
        logger.warning("Debug CSV write failed: %s", exc)
