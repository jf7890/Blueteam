"""Queue Manager – Async Redis queue publisher and result store.

Provides coroutines for:
- Pushing analysis items onto the ``llm_analysis`` Redis list.
- Storing / retrieving per-request results keyed by ``request_id``.
- Reading results back so the API can poll for completion.
"""

from __future__ import annotations

import json
import logging

import redis.asyncio as aioredis

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Async Redis client (lazy singleton)
# ---------------------------------------------------------------------------
_async_redis: aioredis.Redis | None = None


async def _get_async_redis() -> aioredis.Redis:
    """Return a shared async Redis client, creating it on first call."""
    global _async_redis
    if _async_redis is None:
        _async_redis = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
        )
    return _async_redis


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------

async def enqueue_for_llm(
    request_id: str,
    normalized_payloads: list[str],
    rag_context: list[str] | None = None,
) -> None:
    """Push an analysis item onto the batch LLM queue (Redis list RPUSH)."""
    r = await _get_async_redis()
    item = json.dumps({
        "request_id": request_id,
        "normalized_payloads": normalized_payloads,
        "rag_context": rag_context or [],
        "retry_count": 0,
    })
    await r.rpush(settings.batch_queue_key, item)
    logger.info("Enqueued request_id=%s for batch LLM analysis", request_id)


async def enqueue_to_dlq(item_json: str) -> None:
    """Push a failed item onto the dead-letter queue."""
    r = await _get_async_redis()
    await r.rpush(settings.batch_dlq_key, item_json)
    logger.warning("Moved item to DLQ: %s", item_json[:120])


# ---------------------------------------------------------------------------
# Result store
# ---------------------------------------------------------------------------

async def store_result(request_id: str, result: dict) -> None:
    """Persist a per-request verdict result, keyed by ``request_id``."""
    r = await _get_async_redis()
    key = f"{settings.batch_result_prefix}{request_id}"
    await r.set(
        key,
        json.dumps(result, default=str),
        ex=settings.batch_result_ttl_seconds,
    )
    logger.info("Stored result for request_id=%s", request_id)


async def get_result(request_id: str) -> dict | None:
    """Retrieve a stored verdict for *request_id*, or ``None`` if not ready."""
    r = await _get_async_redis()
    key = f"{settings.batch_result_prefix}{request_id}"
    raw = await r.get(key)
    if raw is not None:
        return json.loads(raw)
    return None


# ---------------------------------------------------------------------------
# Sync wrappers for LangGraph node usage (non-async context)
# ---------------------------------------------------------------------------
_sync_redis = None


def _get_sync_redis():
    """Return a shared sync Redis client."""
    import redis as sync_redis

    global _sync_redis
    if _sync_redis is None:
        _sync_redis = sync_redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
        )
    return _sync_redis


def enqueue_for_llm_sync(
    request_id: str,
    normalized_payloads: list[str],
    rag_context: list[str] | None = None,
    metadata: dict | None = None,
) -> None:
    """Synchronous variant of :func:`enqueue_for_llm` for use in LangGraph nodes."""
    r = _get_sync_redis()
    item = {
        "request_id": request_id,
        "normalized_payloads": normalized_payloads,
        "rag_context": rag_context or [],
        "retry_count": 0,
    }
    if metadata:
        item.update(metadata)
    r.rpush(settings.batch_queue_key, json.dumps(item))
    logger.info("Enqueued (sync) request_id=%s for batch LLM analysis", request_id)


def get_result_sync(request_id: str) -> dict | None:
    """Synchronous variant of :func:`get_result` for use in Flask endpoints."""
    r = _get_sync_redis()
    key = f"{settings.batch_result_prefix}{request_id}"
    raw = r.get(key)
    if raw is not None:
        return json.loads(raw)
    return None
