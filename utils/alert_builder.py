"""Alert builder & dual-write helper.

Constructs an ``AIAnalysisResult`` from pipeline / batch context and
persists it to both SIEM JSONL and SQLite.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from schema.state import (
    AIAnalysis,
    AIAnalysisResult,
    AIExplanation,
    HttpInfo,
    NetworkInfo,
    ResponseRecommendation,
    ThreatClassification,
)
from utils.db_manager import insert_alert
from utils.siem_logger import log_alert

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MITRE ATT&CK quick-lookup (heuristic mapping by attack type)
# ---------------------------------------------------------------------------
_MITRE_MAP: dict[str, tuple[str, str, str]] = {
    "sqli": ("Exploit Public-Facing Application", "T1190", "Initial Access"),
    "sql_injection": ("Exploit Public-Facing Application", "T1190", "Initial Access"),
    "xss": ("Drive-by Compromise", "T1189", "Initial Access"),
    "cross-site scripting": ("Drive-by Compromise", "T1189", "Initial Access"),
    "rce": ("Exploitation for Client Execution", "T1203", "Execution"),
    "command_injection": ("Command and Scripting Interpreter", "T1059", "Execution"),
    "cmd_injection": ("Command and Scripting Interpreter", "T1059", "Execution"),
    "path_traversal": ("File and Directory Discovery", "T1083", "Discovery"),
    "directory_traversal": ("File and Directory Discovery", "T1083", "Discovery"),
    "lfi": ("File and Directory Discovery", "T1083", "Discovery"),
    "rfi": ("Exploitation for Client Execution", "T1203", "Execution"),
    "ssrf": ("Server-Side Request Forgery", "T1210", "Lateral Movement"),
}


def _severity_from_confidence(is_malicious: bool, confidence: float) -> str:
    if not is_malicious:
        return "info"
    if confidence >= 0.9:
        return "critical"
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def _recommended_action(is_malicious: bool, severity: str) -> str:
    if not is_malicious:
        return "allow"
    if severity in ("critical", "high"):
        return "block"
    return "monitor"


def _mitigation_steps(attack_type: Optional[str]) -> list[str]:
    base = ["Log the request for forensic analysis"]
    if not attack_type:
        return base
    key = attack_type.lower().replace(" ", "_")
    if "sql" in key:
        return base + ["Use parameterised queries", "Apply WAF SQL injection ruleset"]
    if "xss" in key:
        return base + ["Sanitise and encode user output", "Implement Content-Security-Policy"]
    if "traversal" in key or "lfi" in key:
        return base + ["Restrict file access to allowed directories", "Validate file path inputs"]
    if "cmd" in key or "rce" in key:
        return base + ["Avoid passing user input to shell commands", "Use allowlist for commands"]
    return base


def build_alert(
    *,
    request_id: str,
    is_malicious: bool,
    confidence: float,
    attack_type: Optional[str],
    reasoning: str,
    source_node: str,
    matched_rule: Optional[str] = None,
    matched_payload: Optional[str] = None,
    method: str = "UNKNOWN",
    uri: str = "/",
    host: str = "",
    user_agent: str = "",
    content_type: str = "",
    source_ip: str = "unknown",
    dest_ip: str = "unknown",
    source_port: int = 0,
    dest_port: int = 0,
    model: str = "gemini-2.5-flash",
) -> AIAnalysisResult:
    """Construct a rich AIAnalysisResult from verdict + request context."""
    severity = _severity_from_confidence(is_malicious, confidence)
    action = _recommended_action(is_malicious, severity)

    mitre = _MITRE_MAP.get((attack_type or "").lower().replace(" ", "_"), ("N/A", "N/A", "N/A"))

    return AIAnalysisResult(
        request_id=request_id,
        is_malicious=is_malicious,
        verdict="malicious" if is_malicious else "benign",
        network=NetworkInfo(
            srcip=source_ip,
            dstip=dest_ip,
            srcport=source_port,
            dstport=dest_port,
        ),
        http=HttpInfo(
            method=method,
            uri=uri,
            host=host,
            user_agent=user_agent,
            content_type=content_type,
        ),
        ai_analysis=AIAnalysis(
            analyzer_node=source_node,
            model=model if source_node in ("llm_batch", "llm") else "",
            attack_type=attack_type,
            severity=severity,
            confidence_score=confidence,
            matched_payload=matched_payload,
            matched_rule=matched_rule,
        ),
        ai_explanation=AIExplanation(reasoning=reasoning),
        response_recommendation=ResponseRecommendation(
            recommended_action=action,
            mitigation_steps=_mitigation_steps(attack_type),
        ),
        threat_classification=ThreatClassification(
            technique=mitre[0],
            mitre_attack_id=mitre[1],
            mitre_attack_tactic=mitre[2],
        ),
    )


def persist_alert(alert: AIAnalysisResult) -> None:
    """Dual-write: SIEM JSONL + SQLite."""
    alert_dict = alert.model_dump()
    log_alert(alert_dict)
    insert_alert(alert_dict)
    logger.info(
        "Alert persisted for request_id=%s (verdict=%s)",
        alert.request_id,
        alert.verdict,
    )
