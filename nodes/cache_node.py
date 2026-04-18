"""Cache Node – Request-signature checksum generation and Redis lookup / update.

Responsibilities:
- Generate a SHA-256 hash over the request method, path, and normalised
  payload values.
- Check Redis for a cached verdict.
- On a HIT, populate ``final_result`` and set ``cache_hit = True``.
- On a MISS, set ``cache_hit = False`` and let the pipeline continue.
- ``update_cache_node`` writes the final verdict back into Redis.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from typing import Any

import redis

from config.settings import settings
from schema.state import GraphState
from utils.alert_builder import build_alert, persist_alert
from utils.debug_csv_logger import log_debug_snapshot
from utils.security import hash_payloads

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis client (lazy singleton)
# ---------------------------------------------------------------------------
_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    """Return a shared Redis client, creating it on first call."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
        )
    return _redis_client


_CACHE_PREFIX = "waf:verdict:"


def _get_header_value(headers: dict[str, str], key: str) -> str:
    """Return a header value with case-insensitive matching."""
    for header_name, header_value in headers.items():
        if header_name.lower() == key.lower():
            return header_value
    return ""


def _get_request_signature(raw_request: Any) -> tuple[str, str]:
    """Extract method + path for stable cache/request identifiers."""
    method = getattr(raw_request, "method", "") or ""
    url = getattr(raw_request, "url", "") or ""
    parsed_url = urllib.parse.urlparse(url)
    path = parsed_url.path
    if not path:
        path = "/" if (parsed_url.scheme or parsed_url.netloc) else url
    return method, path


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def cache_lookup_node(state: GraphState) -> dict[str, Any]:
    """Look up the payload hash in Redis.

    Returns ``cache_hit = True`` with ``final_result`` populated on a HIT,
    or ``cache_hit = False`` on a MISS.
    """
    payloads = state.get("normalized_payloads", [])
    raw_request = state.get("raw_request")
    method, path = _get_request_signature(raw_request)
    payload_hash = hash_payloads(payloads, method=method, path=path)
    cache_key = f"{_CACHE_PREFIX}{payload_hash}"

    try:
        r = _get_redis()
        cached = r.get(cache_key)
    except redis.RedisError as exc:
        logger.warning("Redis lookup failed (%s) – treating as MISS", exc)
        return {"cache_hit": False}

    if cached is not None:
        logger.info("Cache HIT for hash %s", payload_hash)
        final_result = json.loads(cached)

        # --- Dual-write even on cache HIT so every request is auditable ---
        try:
            headers = getattr(raw_request, "headers", {})
            alert = build_alert(
                request_id=payload_hash,
                is_malicious=final_result.get("verdict") == "malicious",
                confidence=float(final_result.get("confidence", 1.0)),
                attack_type=final_result.get("attack_type"),
                reasoning=final_result.get("reasoning", ""),
                source_node=f"cache ({final_result.get('source', 'unknown')})",
                matched_rule=final_result.get("matched_rule"),
                matched_payload=final_result.get("matched_payload"),
                method=getattr(raw_request, "method", "UNKNOWN"),
                uri=getattr(raw_request, "url", "/"),
                host=_get_header_value(headers, "host"),
                user_agent=_get_header_value(headers, "user-agent"),
                content_type=_get_header_value(headers, "content-type"),
                source_ip=state.get("source_ip", "unknown"),
                dest_ip=state.get("dest_ip", "unknown"),
                source_port=state.get("source_port", 0),
                dest_port=state.get("dest_port", 0),
            )
            persist_alert(alert)
        except Exception as exc:
            logger.warning("Cache HIT dual-write failed: %s", exc)

        log_debug_snapshot(state, rag_query=[], rag_query_result=[])

        return {"cache_hit": True, "final_result": final_result}

    logger.info("Cache MISS for hash %s", payload_hash)
    return {"cache_hit": False}


def update_cache_node(state: GraphState) -> dict[str, Any]:
    """Persist the final verdict into Redis with a configurable TTL.

    Also performs dual-write: SIEM JSONL + SQLite alert persistence.
    """
    payloads = state.get("normalized_payloads", [])
    raw_request = state.get("raw_request")
    method, path = _get_request_signature(raw_request)
    payload_hash = hash_payloads(payloads, method=method, path=path)
    cache_key = f"{_CACHE_PREFIX}{payload_hash}"
    final_result = state.get("final_result", {})

    try:
        r = _get_redis()
        r.set(
            cache_key,
            json.dumps(final_result, default=str),
            ex=settings.cache_ttl_seconds,
        )
        logger.info("Cache UPDATE for hash %s (TTL=%ds)", payload_hash, settings.cache_ttl_seconds)
    except redis.RedisError as exc:
        logger.warning("Redis write failed (%s) – verdict not cached", exc)

    # --- Dual-write: SIEM JSONL + SQLite ---
    try:
        headers = getattr(raw_request, "headers", {})
        alert = build_alert(
            request_id=payload_hash,
            is_malicious=final_result.get("verdict") == "malicious",
            confidence=float(final_result.get("confidence", 1.0)),
            attack_type=final_result.get("attack_type"),
            reasoning=final_result.get("reasoning", "Matched static rule"),
            source_node=final_result.get("source", "rule_engine"),
            matched_rule=final_result.get("matched_rule"),
            matched_payload=final_result.get("matched_payload"),
            method=getattr(raw_request, "method", "UNKNOWN"),
            uri=getattr(raw_request, "url", "/"),
            host=_get_header_value(headers, "host"),
            user_agent=_get_header_value(headers, "user-agent"),
            content_type=_get_header_value(headers, "content-type"),
            source_ip=state.get("source_ip", "unknown"),
            dest_ip=state.get("dest_ip", "unknown"),
            source_port=state.get("source_port", 0),
            dest_port=state.get("dest_port", 0),
        )
        persist_alert(alert)
    except Exception as exc:
        logger.warning("Alert persistence failed: %s", exc)

    log_debug_snapshot(state, rag_query=[], rag_query_result=[])

    return {}
