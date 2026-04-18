"""Batch Processor Worker – Async tumbling-window consumer for LLM analysis.

Lifecycle:
1. Poll the ``llm_analysis`` Redis list (BLPOP with timeout).
2. Accumulate items until **batch_max_size** (default 20) OR
   **batch_window_seconds** (default 10 s) elapses – whichever comes first.
3. Invoke Gemini 2.5 Flash in a single batch call with a
   ``BatchEvaluationResult`` structured schema.  Each queued item already
   carries its RAG context (populated by the upstream RAG node).
4. Map each verdict back to its ``request_id``, update the verdict cache
   and the result store.
5. On failure: increment retry counter.  If below max retries → re-enqueue;
   otherwise → push to the dead-letter queue (DLQ).

Run with:
    python -m workers.batch_processor
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import redis.asyncio as aioredis

from config.settings import settings
from nodes.llm_analyzer import batch_llm_analyze
from schema.state import BatchItem
from utils.alert_builder import build_alert, persist_alert
from utils.queue_manager import (
    _get_async_redis,
    enqueue_to_dlq,
    store_result,
)
from utils.security import hash_payloads

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache helpers (async)
# ---------------------------------------------------------------------------

async def _update_verdict_cache(
    r: aioredis.Redis,
    request_id: str,
    result: dict[str, Any],
) -> None:
    """Write the verdict into the same Redis cache the sync pipeline uses."""
    cache_key = f"waf:verdict:{request_id}"
    await r.set(
        cache_key,
        json.dumps(result, default=str),
        ex=settings.cache_ttl_seconds,
    )


# ---------------------------------------------------------------------------
# Batch consumption loop
# ---------------------------------------------------------------------------

async def _drain_batch(r: aioredis.Redis) -> list[dict]:
    """Drain up to ``batch_max_size`` items within the tumbling window.

    Uses BLPOP for the first item (blocks until something arrives), then
    non-blocking LPOP for subsequent items up to the window budget.
    """
    batch: list[dict] = []
    window_deadline = time.monotonic() + settings.batch_window_seconds

    # Block-wait for the first item (timeout = window length)
    first = await r.blpop(
        settings.batch_queue_key,
        timeout=int(settings.batch_window_seconds),
    )
    if first is None:
        return batch  # timeout, nothing arrived

    # first is (key, value) tuple
    batch.append(json.loads(first[1]))

    # Fill the rest of the window with non-blocking pops
    while len(batch) < settings.batch_max_size:
        remaining = window_deadline - time.monotonic()
        if remaining <= 0:
            break

        raw = await r.lpop(settings.batch_queue_key)
        if raw is None:
            # Queue empty — wait a short tick then try again
            await asyncio.sleep(min(0.1, remaining))
            raw = await r.lpop(settings.batch_queue_key)
            if raw is None:
                break

        batch.append(json.loads(raw))

    return batch


async def _process_batch(batch_raw: list[dict]) -> None:
    """Process a single batch: batch LLM → store results.

    Each queued item already includes ``rag_context`` from the upstream
    RAG node, so no additional vector search is needed here.
    """
    r = await _get_async_redis()

    # 1. Parse into BatchItem models (RAG context already included)
    items: list[BatchItem] = []
    for raw in batch_raw:
        items.append(BatchItem(
            request_id=raw["request_id"],
            normalized_payloads=raw["normalized_payloads"],
            rag_context=raw.get("rag_context", []),
        ))

    logger.info("Processing batch of %d item(s)", len(items))

    # 2. Batch LLM call
    try:
        verdicts = batch_llm_analyze(items)
    except Exception as exc:
        logger.error("Batch LLM call failed: %s", exc)
        # Re-enqueue or DLQ each item
        for raw in batch_raw:
            await _handle_failure(raw)
        return

    # 3. Build a lookup of verdicts by request_id
    verdict_map: dict[str, dict] = {}
    for v in verdicts:
        verdict_map[v.request_id] = {
            "verdict": "malicious" if v.is_malicious else "benign",
            "source": "llm_batch",
            "confidence": v.confidence,
            "attack_type": v.attack_type,
            "reasoning": v.reasoning,
        }

    # 4. Store results + update verdict cache + dual-write alerts
    for item in items:
        result = verdict_map.get(item.request_id)
        if result is None:
            logger.warning(
                "No LLM verdict returned for request_id=%s", item.request_id
            )
            await _handle_failure(
                next(r for r in batch_raw if r["request_id"] == item.request_id)
            )
            continue

        await store_result(item.request_id, result)
        await _update_verdict_cache(r, item.request_id, result)

        # --- Dual-write: SIEM JSONL + SQLite ---
        try:
            raw = next(
                (b for b in batch_raw if b["request_id"] == item.request_id), {}
            )
            alert = build_alert(
                request_id=item.request_id,
                is_malicious=result["verdict"] == "malicious",
                confidence=result["confidence"],
                attack_type=result.get("attack_type"),
                reasoning=result.get("reasoning", ""),
                source_node="llm_batch",
                source_ip=raw.get("source_ip", "unknown"),
                dest_ip=raw.get("dest_ip", "unknown"),
                source_port=raw.get("source_port", 0),
                dest_port=raw.get("dest_port", 0),
                method=raw.get("method", "UNKNOWN"),
                uri=raw.get("uri", "/"),
                host=raw.get("host", ""),
                user_agent=raw.get("user_agent", ""),
                content_type=raw.get("content_type", ""),
            )
            persist_alert(alert)
        except Exception as exc:
            logger.warning(
                "Alert persistence failed for %s: %s", item.request_id, exc
            )

        logger.info(
            "Verdict for %s: %s (confidence=%.2f)",
            item.request_id,
            result["verdict"],
            result["confidence"],
        )


async def _handle_failure(raw_item: dict) -> None:
    """Retry or send to DLQ based on the retry counter."""
    retry_count = raw_item.get("retry_count", 0) + 1

    if retry_count <= settings.batch_max_retries:
        raw_item["retry_count"] = retry_count
        r = await _get_async_redis()
        await r.rpush(settings.batch_queue_key, json.dumps(raw_item))
        logger.info(
            "Re-enqueued request_id=%s (retry %d/%d)",
            raw_item["request_id"],
            retry_count,
            settings.batch_max_retries,
        )
    else:
        await enqueue_to_dlq(json.dumps(raw_item))
        try:
            alert = build_alert(
                request_id=raw_item.get("request_id", ""),
                is_malicious=False,
                confidence=0.0,
                attack_type="System Error",
                reasoning="Analysis failed and pushed to DLQ",
                source_node="dlq",
                method=raw_item.get("method", "UNKNOWN"),
                uri=raw_item.get("uri", "/"),
                host=raw_item.get("host", ""),
                user_agent=raw_item.get("user_agent", ""),
                content_type=raw_item.get("content_type", ""),
                source_ip=raw_item.get("source_ip", "unknown"),
                dest_ip=raw_item.get("dest_ip", "unknown"),
                source_port=raw_item.get("source_port", 0),
                dest_port=raw_item.get("dest_port", 0),
            )
            alert.verdict = "error"
            persist_alert(alert)
        except Exception as exc:
            logger.warning(
                "Failed to persist DLQ fallback alert for request_id=%s: %s",
                raw_item.get("request_id", ""),
                exc,
            )


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------

async def run_worker() -> None:
    """Continuously drain batches from the queue and process them."""
    logger.info(
        "Batch worker started (max_size=%d, window=%.1fs, queue=%s)",
        settings.batch_max_size,
        settings.batch_window_seconds,
        settings.batch_queue_key,
    )

    r = await _get_async_redis()

    while True:
        try:
            batch = await _drain_batch(r)
            if not batch:
                continue
            await _process_batch(batch)
        except asyncio.CancelledError:
            logger.info("Batch worker shutting down")
            break
        except Exception as exc:
            logger.exception("Unhandled error in batch worker loop: %s", exc)
            await asyncio.sleep(1)  # back-off before retrying


# ---------------------------------------------------------------------------
# Entry point: python -m workers.batch_processor
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
