"""RAG Node - Hybrid search against malicious payload intelligence."""

from __future__ import annotations

import logging
from typing import Any

import torch
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

from config.settings import settings
from schema.state import GraphState

logger = logging.getLogger(__name__)

_qdrant_client: QdrantClient | None = None
_dense_embedder: SentenceTransformer | None = None


def _rag_enabled() -> bool:
    """Return whether retrieval is enabled for the current process."""
    return bool(settings.rag_enabled)


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=10,
        )
    return _qdrant_client


def _get_dense_embedder() -> SentenceTransformer:
    global _dense_embedder
    if _dense_embedder is None:
        _dense_embedder = SentenceTransformer(
            settings.rag_dense_model,
            trust_remote_code=True,
        )
    return _dense_embedder


def _dense_vector(text: str) -> list[float]:
    """Encode text with the configured dense embedding model."""
    model = _get_dense_embedder()
    with torch.no_grad():
        vector = model.encode(text[:2048], normalize_embeddings=True)
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)


def _hybrid_query_points(client: QdrantClient, payload: str, limit: int) -> Any:
    dense_query = _dense_vector(payload)
    prefetch_limit = max(limit * 2, 10)

    return client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            models.Prefetch(
                query=dense_query,
                using=settings.qdrant_dense_vector_name,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.Document(text=payload, model="Qdrant/bm25"),
                using=settings.qdrant_sparse_vector_name,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )


def _collection_ready() -> bool:
    """Check whether the target collection exists and is accessible."""
    if not _rag_enabled():
        return False

    try:
        client = _get_qdrant()
        client.get_collection(settings.qdrant_collection)
        return True
    except (UnexpectedResponse, Exception) as exc:
        logger.warning("Qdrant collection '%s' not available: %s", settings.qdrant_collection, exc)
        return False


def search_similar_payloads(payload: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Search threat intel payloads using hybrid dense+sparse retrieval."""
    if not _rag_enabled():
        logger.info("RAG retrieval disabled via environment flag")
        return []
    if not payload or not _collection_ready():
        return []

    client = _get_qdrant()
    top_k = limit or settings.rag_top_k

    try:
        results = _hybrid_query_points(client, payload, top_k)
    except Exception as exc:
        logger.warning("Hybrid query failed; falling back to text query: %s", exc)
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=payload,
            limit=top_k,
            with_payload=True,
        )

    hits: list[dict[str, Any]] = []
    for point in results.points:
        record = point.payload or {}
        category = record.get("category") or record.get("attack_type") or "Unknown"
        raw_payload = (
            record.get("raw_payload")
            or record.get("payload")
            or record.get("content")
            or ""
        )
        text = record.get("text") or (
            f"Attack Type: {category} | Payload: {raw_payload}" if raw_payload else ""
        )
        if text:
            hits.append(
                {
                    "point_id": str(point.id),
                    "score": float(point.score),
                    "payload": text,
                    "record": {
                        "text": text,
                        "category": category,
                        "raw_payload": raw_payload,
                        "source_file": record.get("source_file"),
                        "line_no": record.get("line_no"),
                    },
                }
            )
    return hits


def merge_ranked_payload_hit_traces(
    payload_traces: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Merge per-payload hit traces into one globally ranked unique hit list."""
    all_hits: list[dict[str, Any]] = []
    for trace in payload_traces:
        all_hits.extend(trace.get("hits", []))

    all_hits.sort(key=lambda hit: float(hit.get("score", 0.0)), reverse=True)

    seen: set[str] = set()
    merged_hits: list[dict[str, Any]] = []
    for hit in all_hits:
        hit_key = str(hit.get("point_id") or hit.get("payload") or "")
        if not hit_key or hit_key in seen:
            continue

        seen.add(hit_key)
        merged_hits.append(hit)
        if len(merged_hits) >= limit:
            break

    return merged_hits


def collect_payload_hit_trace(
    payloads: list[str],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return a retrieval trace for every payload candidate."""
    if not payloads:
        return []

    if not _rag_enabled():
        return [
            {
                "payload_index": payload_index,
                "payload": payload,
                "eligible": False,
                "skip_reason": "rag_disabled",
                "hits": [],
            }
            for payload_index, payload in enumerate(payloads)
        ]

    if not _collection_ready():
        return [
            {
                "payload_index": payload_index,
                "payload": payload,
                "eligible": False,
                "skip_reason": "collection_not_ready",
                "hits": [],
            }
            for payload_index, payload in enumerate(payloads)
        ]

    top_k = limit or settings.rag_top_k
    payload_traces: list[dict[str, Any]] = []

    for payload_index, payload in enumerate(payloads):
        trace: dict[str, Any] = {
            "payload_index": payload_index,
            "payload": payload,
            "eligible": False,
            "skip_reason": "",
            "hits": [],
        }
        if not payload:
            trace["skip_reason"] = "empty_payload"
            payload_traces.append(trace)
            continue
        if len(payload) < 3:
            trace["skip_reason"] = "payload_too_short"
            payload_traces.append(trace)
            continue

        try:
            trace["hits"] = search_similar_payloads(payload, top_k)
            trace["eligible"] = True
        except Exception as exc:
            logger.warning("RAG search failed for payload chunk: %s", exc)
            trace["skip_reason"] = f"query_failed: {exc}"

        payload_traces.append(trace)

    return payload_traces


def collect_ranked_payload_hits(
    payloads: list[str],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return globally ranked unique retrieval hits across all payloads."""
    top_k = limit or settings.rag_top_k
    payload_traces = collect_payload_hit_trace(payloads, top_k)
    return merge_ranked_payload_hit_traces(payload_traces, top_k)


def _build_rag_context(payloads: list[str]) -> list[str]:
    """Return the globally top-ranked unique RAG hits across all payloads."""
    rag_hits = collect_ranked_payload_hits(payloads, settings.rag_top_k)
    rag_context: list[str] = []

    for hit in rag_hits:
        text = hit.get("payload")
        if text:
            rag_context.append(text)

    return rag_context


def rag_node(state: GraphState) -> dict[str, Any]:
    """Perform Hybrid RAG search and populate ``rag_context``."""
    payloads = state.get("suspicious_payloads") or state.get("normalized_payloads", [])

    if not payloads:
        return {"rag_context": []}

    if not _rag_enabled():
        logger.info("RAG node: disabled via environment flag")
        return {"rag_context": []}

    if not _collection_ready():
        logger.info("RAG node: collection not ready - returning empty context")
        return {"rag_context": []}

    rag_context = _build_rag_context(payloads)

    logger.info(
        "RAG node: retrieved %d context snippet(s) for %d suspicious payload(s)",
        len(rag_context),
        len(payloads),
    )
    return {"rag_context": rag_context}


def rag_search_for_payloads(payloads: list[str]) -> list[str]:
    """Run hybrid search for a list of payloads and return context strings."""
    if not _rag_enabled():
        return []
    return _build_rag_context(payloads)
