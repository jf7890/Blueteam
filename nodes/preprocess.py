"""Preprocess Node - Raw HTTP parsing, recursive decoding & attack surface extraction.

Responsibilities:
1. Parse the raw HTTP request text into a structured ``HttpRequest``.
2. Recursively decode each payload component (URL-decode, HTML-unescape, etc.).
3. Collect all user-controlled values (query params, body fields, header
   values, cookies) into ``normalized_payloads`` for downstream analysis.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from typing import Any

from schema.state import GraphState, HttpRequest
from utils.security import normalise_payload

logger = logging.getLogger(__name__)


def _parse_raw_http(raw_text: str) -> HttpRequest:
    """Parse a full raw HTTP request string into a structured ``HttpRequest``."""
    raw_text = raw_text.replace("\r\n", "\n").strip()

    if "\n\n" in raw_text:
        head, body = raw_text.split("\n\n", 1)
    else:
        head, body = raw_text, None

    lines = head.split("\n")
    request_line = lines[0].strip()
    parts = request_line.split(None, 2)
    if len(parts) < 2:
        raise ValueError(f"Malformed request line: {request_line!r}")

    method = parts[0].upper()
    path = parts[1]

    headers: dict[str, str] = {}
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()

    host = headers.get("Host", headers.get("host", ""))
    if path.startswith("http://") or path.startswith("https://"):
        url = path
    elif host:
        scheme = "https" if "443" in host else "http"
        url = f"{scheme}://{host}{path}"
    else:
        url = path

    body_str = body.strip() if body and body.strip() else None
    return HttpRequest(method=method, url=url, headers=headers, body=body_str)


def _payload_detail(source_type: str, source_name: str, raw_value: Any) -> dict[str, str]:
    """Build a normalized payload detail record for audit-friendly tracing."""
    raw_text = str(raw_value)
    return {
        "source_type": source_type,
        "source_name": source_name,
        "raw_value": raw_text,
        "normalized_value": normalise_payload(raw_text),
    }


def _extract_body_details(body: str, content_type: str) -> list[dict[str, str]]:
    """Extract individual body values with lightweight source metadata."""
    details: list[dict[str, str]] = []
    ct = content_type.lower()

    if "application/x-www-form-urlencoded" in ct:
        parsed = urllib.parse.parse_qs(body)
        for key, vals in parsed.items():
            for value in vals:
                details.append(_payload_detail("body_form", key, value))
    elif "application/json" in ct:
        try:
            obj = json.loads(body)
            _collect_json_details(obj, details)
        except (json.JSONDecodeError, TypeError):
            details.append(_payload_detail("body", "raw", body))
    else:
        details.append(_payload_detail("body", "raw", body))

    return details


def _collect_json_details(
    obj: Any,
    out: list[dict[str, str]],
    path: str = "$",
) -> None:
    """Recursively collect JSON leaf values together with their field paths."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path != "$" else f"$.{key}"
            _collect_json_details(value, out, next_path)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            _collect_json_details(item, out, f"{path}[{index}]")
    elif isinstance(obj, str):
        out.append(_payload_detail("body_json", path, obj))
    elif obj is not None:
        out.append(_payload_detail("body_json", path, str(obj)))


def preprocess_node(state: GraphState) -> dict[str, Any]:
    """Parse the raw HTTP text, extract and normalise all payload values."""
    raw_text: str = state["raw_http_text"]
    http_request = _parse_raw_http(raw_text)
    logger.info("Parsed request: %s %s", http_request.method, http_request.url)

    payload_details: list[dict[str, str]] = []

    parsed_url = urllib.parse.urlparse(http_request.url)
    path = urllib.parse.unquote(parsed_url.path or "")
    if path:
        payload_details.append(_payload_detail("path", "url_path", path))

    query_params = urllib.parse.parse_qs(parsed_url.query)
    for key, values in query_params.items():
        for value in values:
            payload_details.append(_payload_detail("query", key, value))

    skip_headers = {"host", "connection", "accept-encoding", "content-length"}
    for key, value in http_request.headers.items():
        if key.lower() not in skip_headers:
            payload_details.append(_payload_detail("header", key, value))

    if http_request.body:
        content_type = http_request.headers.get(
            "Content-Type", http_request.headers.get("content-type", "")
        )
        payload_details.extend(_extract_body_details(http_request.body, content_type))

    normalized = [detail["normalized_value"] for detail in payload_details]

    return {
        "raw_request": http_request,
        "normalized_payloads": normalized,
        "normalized_payload_details": payload_details,
    }
