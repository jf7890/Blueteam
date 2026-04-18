"""Pydantic models and TypedDict definitions for the LangGraph state machine.

Defines:
- HttpRequest: Pydantic model representing an incoming HTTP request.
- LLMVerdict: Pydantic model for structured LLM output (used for response_schema).
- AIAnalysisResult: Rich structured alert format for SIEM/SOC integration.
- GraphState: TypedDict that flows through every node in the LangGraph workflow.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Pydantic models – used for validation, serialization & LLM structured output
# ---------------------------------------------------------------------------

class HttpRequest(BaseModel):
    """Structured representation of an incoming HTTP request."""

    method: str = Field(..., description="HTTP method (GET, POST, PUT, …)")
    url: str = Field(..., description="Full request URL including query string")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    body: Optional[str] = Field(default=None, description="Raw request body, if any")


class LLMVerdict(BaseModel):
    """Schema returned by Gemini 2.5 Flash after analysing a request.

    Used as the `response_schema` so the model is forced to return
    well-structured JSON that can be parsed deterministically.
    """

    is_malicious: bool = Field(
        ..., description="True if the request is deemed malicious"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    attack_type: Optional[str] = Field(
        default=None,
        description="Attack category (e.g. SQLi, XSS, RCE) if malicious",
    )
    reasoning: str = Field(
        ..., description="Brief explanation of the verdict"
    )


# ---------------------------------------------------------------------------
# Batch processing schemas – used by the async batch worker & batch LLM call
# ---------------------------------------------------------------------------

class BatchItem(BaseModel):
    """A single item enqueued for batch LLM analysis."""

    request_id: str = Field(
        ..., description="Unique identifier (payload hash) for result mapping"
    )
    normalized_payloads: list[str] = Field(
        ..., description="Normalised payload strings extracted from the request"
    )
    rag_context: list[str] = Field(
        default_factory=list,
        description="RAG context snippets (populated by batch worker)",
    )


class BatchVerdictItem(BaseModel):
    """LLM verdict for one item inside a batch response."""

    request_id: str = Field(
        ..., description="Matching request_id from the BatchItem"
    )
    is_malicious: bool = Field(
        ..., description="True if the request is deemed malicious"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    attack_type: Optional[str] = Field(
        default=None,
        description="Attack category (e.g. SQLi, XSS, RCE) if malicious",
    )
    reasoning: str = Field(
        ..., description="Brief explanation of the verdict"
    )


class BatchEvaluationResult(BaseModel):
    """Top-level schema for the batch LLM response.

    Passed as ``response_schema`` to Gemini so the model returns a
    well-formed JSON array of verdicts, one per queued request.
    """

    items: list[BatchVerdictItem] = Field(
        ..., description="List of verdicts, one per BatchItem in the request"
    )


# ---------------------------------------------------------------------------
# Rich AIAnalysisResult – structured alert for SIEM / SOC integration
# ---------------------------------------------------------------------------

class NetworkInfo(BaseModel):
    """Network-level metadata extracted from the HTTP request context."""

    srcip: str = Field(default="unknown", description="Source IP address")
    dstip: str = Field(default="unknown", description="Destination IP address")
    srcport: int = Field(default=0, description="Source port")
    dstport: int = Field(default=0, description="Destination port (e.g. 80/443)")
    protocol: str = Field(default="HTTP", description="Network protocol")


class HttpInfo(BaseModel):
    """HTTP-level metadata from the parsed request."""

    method: str = Field(default="UNKNOWN", description="HTTP method")
    uri: str = Field(default="/", description="Request URI")
    user_agent: str = Field(default="", description="User-Agent header value")
    host: str = Field(default="", description="Host header value")
    content_type: str = Field(default="", description="Content-Type header value")


class AIAnalysis(BaseModel):
    """Core AI analysis verdict."""

    engine: str = Field(default="BlueAgent WAF", description="Analysis engine name")
    analyzer_node: str = Field(default="", description="Pipeline node that produced the verdict")
    model: str = Field(default="gemini-2.5-flash", description="LLM model used")
    attack_type: Optional[str] = Field(default=None, description="Detected attack category")
    severity: str = Field(default="info", description="Alert severity: critical/high/medium/low/info")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence 0-1")
    matched_payload: Optional[str] = Field(default=None, description="The payload that triggered detection")
    matched_rule: Optional[str] = Field(default=None, description="Rule name if rule engine matched")


class AIExplanation(BaseModel):
    """LLM-generated explanation of the analysis."""

    reasoning: str = Field(default="", description="Detailed reasoning for the verdict")


class ResponseRecommendation(BaseModel):
    """Recommended actions for incident response."""

    recommended_action: str = Field(default="allow", description="allow / block / monitor / rate_limit")
    mitigation_steps: list[str] = Field(default_factory=list, description="Suggested mitigation steps")


class ThreatClassification(BaseModel):
    """MITRE ATT&CK and threat intelligence classification."""

    technique: str = Field(default="N/A", description="MITRE ATT&CK technique name")
    mitre_attack_id: str = Field(default="N/A", description="MITRE ATT&CK technique ID (e.g. T1190)")
    mitre_attack_tactic: str = Field(default="N/A", description="MITRE ATT&CK tactic (e.g. Initial Access)")


class AIAnalysisResult(BaseModel):
    """Rich structured alert for SIEM logging (JSONL) and SQLite persistence.

    Combines network metadata, HTTP details, AI analysis verdict,
    explanation, response recommendations, and MITRE ATT&CK classification
    into a single comprehensive alert record.
    """

    timestamp: str = Field(
        default_factory=lambda: datetime.now().astimezone().isoformat(),
        description="ISO-8601 UTC timestamp",
    )
    event_type: str = Field(default="http_analysis", description="Event type identifier")
    event_category: str = Field(default="web_attack_detection", description="Event category")
    request_id: str = Field(default="", description="Unique request identifier (payload hash)")
    is_malicious: bool = Field(default=False, description="Whether the request is malicious")
    verdict: str = Field(default="benign", description="Final verdict string")
    network: NetworkInfo = Field(default_factory=NetworkInfo)
    http: HttpInfo = Field(default_factory=HttpInfo)
    ai_analysis: AIAnalysis = Field(default_factory=AIAnalysis)
    ai_explanation: AIExplanation = Field(default_factory=AIExplanation)
    response_recommendation: ResponseRecommendation = Field(default_factory=ResponseRecommendation)
    threat_classification: ThreatClassification = Field(default_factory=ThreatClassification)


# ---------------------------------------------------------------------------
# LangGraph state – a TypedDict that is passed through every node
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """Shared state flowing through the LangGraph Defense-in-Depth pipeline.

    Keys
    ----
    raw_http_text : str
        The full raw HTTP request as a single text blob, exactly as received
        (e.g. ``"GET /path?q=1 HTTP/1.1\r\nHost: …\r\n\r\nbody"``).
        This is the **primary input** to the pipeline.
    raw_request : HttpRequest
        Structured representation parsed from ``raw_http_text`` by the
        preprocess node.
    normalized_payloads : list[str]
        Recursively decoded and normalised payload strings extracted from the
        request (query params, body fields, header values, etc.).
    normalized_payload_details : list[dict[str, Any]]
        Audit-friendly per-payload metadata including source location,
        original value, and normalized value.
    suspicious_payloads : list[str]
        Payload values flagged as suspicious by the Gatekeeper ML classifier.
        This is populated after static rules miss and is used to narrow the
        downstream RAG search surface.
    cache_hit : bool | None
        Result of the Redis cache lookup.  ``True`` means a previous verdict
        was found; ``None`` means the lookup has not been performed yet.
    rule_verdict : str | None
        Outcome of the static rule-engine scan.
        ``"malicious"`` / ``"benign"`` / ``None`` (unknown).
    rag_context : list[str]
        Top-k relevant malicious payload examples retrieved via Hybrid RAG.
    llm_verdict : LLMVerdict | None
        Structured verdict returned by the Gemini 2.5 Flash LLM node.
    final_result : dict[str, Any]
        Consolidated final result written to the cache and returned to the
        caller.  Contains the verdict, source node, confidence, etc.
    """

    raw_http_text: str
    raw_request: HttpRequest
    normalized_payloads: list[str]
    normalized_payload_details: list[dict[str, Any]]
    suspicious_payloads: list[str]
    request_id: Optional[str]
    cache_hit: Optional[bool]
    rule_verdict: Optional[str]
    rag_context: list[str]
    llm_verdict: Optional[LLMVerdict]
    final_result: dict[str, Any]
    enqueued: Optional[bool]
    source_ip: Optional[str]
    dest_ip: Optional[str]
    source_port: Optional[int]
    dest_port: Optional[int]
