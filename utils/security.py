"""Security utilities for hashing, text normalisation, and prompt delimiters.

Provides helpers used across multiple nodes to ensure consistent and safe
handling of payload data.
"""

from __future__ import annotations

import hashlib
import html
import re
import urllib.parse


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalise_payload(raw: str) -> str:
    """Recursively URL-decode and HTML-unescape a raw payload string.

    Decoding is applied in a loop until the output stabilises, which
    defeats multi-layer encoding evasion techniques.
    """
    max_depth = 5
    depth = 0
    previous = ""
    current = raw
    while current != previous and depth < max_depth:
        previous = current
        current = urllib.parse.unquote(current)
        current = html.unescape(current)
        depth += 1
    return current


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def hash_payloads(
    payloads: list[str],
    method: str = "",
    path: str = "",
) -> str:
    """Generate a deterministic SHA-256 checksum for a request signature.

    The request method and path are combined with sorted, lowercased,
    stripped payload values so semantically identical requests produce
    stable cache keys even if payload ordering changes.
    """
    canonical_payloads = sorted(p.strip().lower() for p in payloads)
    combined = "\n".join([method.strip().upper(), path.strip(), *canonical_payloads])
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Prompt-injection defence helpers
# ---------------------------------------------------------------------------

_XML_UNSAFE = re.compile(r"[<>&]")


def _escape_xml(text: str) -> str:
    """Escape characters that could break XML tag boundaries."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def wrap_in_xml_tags(tag: str, content: str) -> str:
    """Wrap *content* inside strict XML delimiters, escaping unsafe chars.

    This prevents user-controlled data from breaking out of the XML
    structure used in the LLM prompt.

    Example
    -------
    >>> wrap_in_xml_tags("USER_INPUT", "1 OR 1=1 --")
    '<USER_INPUT>1 OR 1=1 --</USER_INPUT>'
    """
    safe_content = _escape_xml(content)
    return f"<{tag}>{safe_content}</{tag}>"
