"""Rule Engine Node – Static regex / signature matching.

Responsibilities:
- Run a battery of regex patterns against every normalised payload.
- If any pattern matches, immediately flag the request and record the
  matched rule so the pipeline can short-circuit to caching.
- If nothing matches, yield ``rule_verdict = None`` (unknown) to hand
  off to the Gatekeeper ML triage path.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from schema.state import GraphState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quick-match regex signatures – expand this list as needed.
# ---------------------------------------------------------------------------
_SIGNATURES: list[tuple[str, str, re.Pattern[str]]] = [
    ("SQLi-basic", "SQLi", re.compile(
        r"(\b(union\s+select|or\s+1\s*=\s*1|drop\s+table|insert\s+into"
        r"|select\s+.*\s+from|delete\s+from|update\s+.*\s+set)\b)",
        re.IGNORECASE,
    )),
    ("XSS-basic", "XSS", re.compile(
        r"(<\s*script\b|javascript\s*:|on(error|load|click|mouseover)\s*=)",
        re.IGNORECASE,
    )),
    ("Path-traversal", "Path Traversal", re.compile(r"\.\./ |\.\.\\", re.IGNORECASE)),
    ("Command-injection", "Command Injection", re.compile(
        r"(;\s*(ls|cat|whoami|id|uname|curl|wget)\b|\|\s*\w+|\$\(.*\))",
        re.IGNORECASE,
    )),
]


def rule_engine_node(state: GraphState) -> dict[str, Any]:
    """Scan normalised payloads against static regex signatures.

    Returns ``rule_verdict = "malicious"`` with a ``final_result`` if a
    signature matches, otherwise ``rule_verdict = None`` (unknown) so
    Gatekeeper ML can decide whether deeper enrichment is warranted.
    """
    payloads = state.get("normalized_payloads", [])

    for payload in payloads:
        for rule_name, attack_type, pattern in _SIGNATURES:
            if pattern.search(payload):
                logger.warning(
                    "Rule engine MATCH: rule=%s payload=%r", rule_name, payload[:120]
                )
                return {
                    "rule_verdict": "malicious",
                    "final_result": {
                        "verdict": "malicious",
                        "source": "rule_engine",
                        "matched_rule": rule_name,
                        "attack_type": attack_type,
                        "confidence": 1.0,
                    },
                }

    logger.info("Rule engine: no match – forwarding to Gatekeeper ML")
    return {"rule_verdict": None}
