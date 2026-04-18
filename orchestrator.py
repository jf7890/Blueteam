"""LangGraph Orchestrator – Defence-in-Depth HTTP request analysis pipeline.

Compiles the following workflow:

    START
      │
      ▼
    Preprocess_Node
      │
      ▼
    Cache_Lookup_Node ──── HIT ──► END
      │
      MISS
      │
      ▼
    Rule_Engine_Node ──── MATCH ──► Update_Cache_Node ──► END
      │
      UNKNOWN
      │
      ▼
    Gatekeeper_Node ──── NO SUSPICIOUS PAYLOADS ──► Update_Cache_Node ──► END
      │
      SUSPICIOUS PAYLOADS
      │
      ▼
    RAG_Node  (hybrid search for known malicious patterns)
      │
      ▼
    Enqueue_Node ──► END  (async: result delivered via batch worker)
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

from langgraph.graph import END, StateGraph

from nodes.cache_node import cache_lookup_node, update_cache_node
from nodes.gatekeeper_node import gatekeeper_node
from nodes.preprocess import preprocess_node
from nodes.rag_node import rag_node
from nodes.rule_engine import rule_engine_node
from schema.state import GraphState
from utils.debug_csv_logger import log_debug_snapshot
from utils.queue_manager import enqueue_for_llm_sync
from utils.security import hash_payloads

logger = logging.getLogger(__name__)


def _get_header_value(headers: dict[str, str], key: str) -> str:
    """Return a header value with case-insensitive matching."""
    for header_name, header_value in headers.items():
        if header_name.lower() == key.lower():
            return header_value
    return ""

# ---------------------------------------------------------------------------
# Conditional-edge routing functions
# ---------------------------------------------------------------------------


def _route_after_cache(state: GraphState) -> str:
    """Route after the cache lookup node.

    - HIT  → go straight to END (result already populated).
    - MISS → continue to Rule_Engine_Node.
    """
    if state.get("cache_hit"):
        return "end"
    return "rule_engine"


def _route_after_rules(state: GraphState) -> str:
    """Route after the rule engine node.

    - MATCH (rule_verdict is not None) → Update_Cache_Node (short-circuit).
    - UNKNOWN (rule_verdict is None)   → Gatekeeper ML for payload triage.
    """
    if state.get("rule_verdict") is not None:
        return "update_cache"
    return "gatekeeper"


def _route_after_gatekeeper(state: GraphState) -> str:
    """Route after the Gatekeeper ML node.

    - No suspicious payloads → Update_Cache_Node (benign short-circuit).
    - Suspicious payloads    → RAG_Node for threat-intel enrichment.
    """
    if not state.get("suspicious_payloads"):
        return "update_cache"
    return "rag"


# ---------------------------------------------------------------------------
# Enqueue node — pushes item to Redis for batch LLM processing
# ---------------------------------------------------------------------------

def _enqueue_node(state: GraphState) -> dict[str, Any]:
    """Push the normalised payloads + RAG context onto the Redis batch queue.

    The request_id is a hash of the method, path, and normalised payloads
    so the API can poll for results.
    RAG context has already been attached by the upstream RAG node.
    Request metadata is included so the batch worker can build rich alerts.
    """
    payloads = state.get("normalized_payloads", [])
    rag_context = state.get("rag_context", [])

    # Gather HTTP/network metadata for the batch worker's alert builder
    raw_request = state.get("raw_request")
    metadata: dict[str, Any] = {}
    method = getattr(raw_request, "method", "") or ""
    url = getattr(raw_request, "url", "") or ""
    parsed_url = urllib.parse.urlparse(url)
    path = parsed_url.path
    if not path:
        path = "/" if (parsed_url.scheme or parsed_url.netloc) else url
    request_id = hash_payloads(payloads, method=method, path=path)

    if raw_request:
        metadata["method"] = method or "UNKNOWN"
        metadata["uri"] = url or "/"
        headers = getattr(raw_request, "headers", {})
        metadata["host"] = _get_header_value(headers, "host")
        metadata["user_agent"] = _get_header_value(headers, "user-agent")
        metadata["content_type"] = _get_header_value(headers, "content-type")
    metadata["source_ip"] = state.get("source_ip", "unknown")
    metadata["dest_ip"] = state.get("dest_ip", "unknown")
    metadata["source_port"] = state.get("source_port", 0)
    metadata["dest_port"] = state.get("dest_port", 0)

    try:
        enqueue_for_llm_sync(request_id, payloads, rag_context, metadata=metadata)
    except Exception as exc:
        logger.warning("Failed to enqueue request_id=%s: %s", request_id, exc)
        log_debug_snapshot(state, rag_query=state.get("suspicious_payloads") or payloads)
        return {
            "request_id": request_id,
            "enqueued": False,
            "final_result": {
                "verdict": "error",
                "source": "enqueue",
                "error": f"Queue unavailable: {exc}",
            },
        }

    log_debug_snapshot(
        state,
        rag_query=state.get("suspicious_payloads") or payloads,
        rag_query_result=rag_context,
    )

    return {
        "request_id": request_id,
        "enqueued": True,
        "final_result": {
            "verdict": "pending",
            "source": "enqueue",
            "request_id": request_id,
            "message": "Request queued for async batch analysis",
        },
    }


# ---------------------------------------------------------------------------
# Graph construction – Async (default)
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Construct the async-queue workflow.

    Unknown requests (not caught by cache or rule engine) go through
    Gatekeeper ML first. Payloads flagged as suspicious are enriched by
    RAG and then enqueued for the background batch LLM worker.
    """
    graph = StateGraph(GraphState)

    # --- Register nodes ---
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("cache_lookup", cache_lookup_node)
    graph.add_node("rule_engine", rule_engine_node)
    graph.add_node("gatekeeper", gatekeeper_node)
    graph.add_node("rag", rag_node)
    graph.add_node("enqueue", _enqueue_node)
    graph.add_node("update_cache", update_cache_node)

    # --- Edges ---
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "cache_lookup")

    # Conditional: after cache lookup
    graph.add_conditional_edges(
        "cache_lookup",
        _route_after_cache,
        {
            "end": END,
            "rule_engine": "rule_engine",
        },
    )

    # Conditional: after rule engine
    graph.add_conditional_edges(
        "rule_engine",
        _route_after_rules,
        {
            "update_cache": "update_cache",
            "gatekeeper": "gatekeeper",
        },
    )

    # Conditional: after Gatekeeper ML
    graph.add_conditional_edges(
        "gatekeeper",
        _route_after_gatekeeper,
        {
            "update_cache": "update_cache",
            "rag": "rag",
        },
    )

    # Linear: RAG → Enqueue → END
    graph.add_edge("rag", "enqueue")
    graph.add_edge("enqueue", END)
    graph.add_edge("update_cache", END)

    return graph


# Pre-compiled application – import and invoke this directly.
app = build_graph().compile()


# ---------------------------------------------------------------------------
# Quick smoke-test when running the module directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_raw_http = (
        "GET /search?q=%3Cscript%3Ealert(1)%3C/script%3E HTTP/1.1\r\n"
        "Host: example.com\r\n"
        "User-Agent: Mozilla/5.0\r\n"
        "\r\n"
    )

    result = app.invoke({"raw_http_text": sample_raw_http})
    print("=== Final Result ===")
    print(result.get("final_result"))
