#!/usr/bin/env python3
"""Create small stratified benchmark subsets from a benchmark CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
import sys
from typing import Any
from urllib.parse import unquote_plus

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import normalize_attack_type  # noqa: E402

_CLASS_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "SQLi": [
        re.compile(r"(?:^|[\s'\"(])or(?:[\s(]+)?\d+\s*=\s*\d+", re.IGNORECASE),
        re.compile(r"union(?:\s+all)?\s+select", re.IGNORECASE),
        re.compile(r"sleep\s*\(", re.IGNORECASE),
        re.compile(r"benchmark\s*\(", re.IGNORECASE),
        re.compile(r"waitfor\s+delay", re.IGNORECASE),
        re.compile(r"xp_cmdshell", re.IGNORECASE),
        re.compile(r"information_schema", re.IGNORECASE),
        re.compile(r"@@version", re.IGNORECASE),
        re.compile(r"(?:--|#|/\*)", re.IGNORECASE),
        re.compile(r"'(?:\s*\)|\s*or\s+|\s*union\s+)", re.IGNORECASE),
    ],
    "XSS": [
        re.compile(r"<script", re.IGNORECASE),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"onerror\s*=", re.IGNORECASE),
        re.compile(r"onload\s*=", re.IGNORECASE),
        re.compile(r"<img", re.IGNORECASE),
        re.compile(r"<svg", re.IGNORECASE),
        re.compile(r"alert\s*\(", re.IGNORECASE),
        re.compile(r"document\.cookie", re.IGNORECASE),
        re.compile(r"%3cscript", re.IGNORECASE),
        re.compile(r"%3csvg", re.IGNORECASE),
        re.compile(r"%3cimg", re.IGNORECASE),
    ],
    "Path Traversal": [
        re.compile(r"\.\./", re.IGNORECASE),
        re.compile(r"\.\.\\", re.IGNORECASE),
        re.compile(r"%2e%2e%2f", re.IGNORECASE),
        re.compile(r"%2e%2e/", re.IGNORECASE),
        re.compile(r"/etc/passwd", re.IGNORECASE),
        re.compile(r"boot\.ini", re.IGNORECASE),
        re.compile(r"win\.ini", re.IGNORECASE),
        re.compile(r"system32", re.IGNORECASE),
    ],
    "Command Injection": [
        re.compile(r"(?:;|\|\||&&|\|)\s*(?:cat|ls|id|whoami|uname|pwd|nc|bash|sh|cmd|powershell|wget|curl)\b", re.IGNORECASE),
        re.compile(r"(?:/bin/sh|/bin/bash|cmd\.exe|powershell\.exe)", re.IGNORECASE),
        re.compile(r"xterm\s+-display", re.IGNORECASE),
        re.compile(r"%0a", re.IGNORECASE),
        re.compile(r"%0d", re.IGNORECASE),
        re.compile(r"\$\(", re.IGNORECASE),
        re.compile(r"`[^`]+`", re.IGNORECASE),
    ],
    "LDAP Injection": [
        re.compile(r"\(\|", re.IGNORECASE),
        re.compile(r"\(&", re.IGNORECASE),
        re.compile(r"\(objectclass", re.IGNORECASE),
        re.compile(r"\)\(", re.IGNORECASE),
        re.compile(r"cn=\*", re.IGNORECASE),
        re.compile(r"uid=\*", re.IGNORECASE),
        re.compile(r"mail=\*", re.IGNORECASE),
        re.compile(r"\*\)", re.IGNORECASE),
    ],
    "XPath Injection": [
        re.compile(r"/child::", re.IGNORECASE),
        re.compile(r"text\(\)", re.IGNORECASE),
        re.compile(r"position\(\)", re.IGNORECASE),
        re.compile(r"count\(", re.IGNORECASE),
        re.compile(r"local-name\(", re.IGNORECASE),
        re.compile(r"name\(", re.IGNORECASE),
        re.compile(r"//\*", re.IGNORECASE),
        re.compile(r"\bor\b.+child::", re.IGNORECASE),
        re.compile(r"\bor\b.+text\(\)", re.IGNORECASE),
    ],
    "SSI": [
        re.compile(r"<!--#(?:exec|include|echo|config|fsize|flastmod)", re.IGNORECASE),
        re.compile(r"%3c!--#(?:exec|include|echo|config|fsize|flastmod)", re.IGNORECASE),
    ],
}

_METHOD_PRIORITY = {
    "POST": 0,
    "PUT": 1,
    "GET": 2,
}

_ALL_PATTERNS: list[re.Pattern[str]] = []
for pattern_list in _CLASS_PATTERNS.values():
    _ALL_PATTERNS.extend(pattern_list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create small stratified benchmark subsets.",
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input benchmark CSV, for example data/pkdd_request_level_eval.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/pilot_subsets",
        help="Directory where subset CSV files will be written.",
    )
    parser.add_argument(
        "--subset-count",
        type=int,
        default=3,
        help="Number of subset files to create.",
    )
    parser.add_argument(
        "--benign-per-subset",
        type=int,
        default=100,
        help="Number of benign samples in each subset.",
    )
    parser.add_argument(
        "--malicious-per-subset",
        type=int,
        default=100,
        help="Number of malicious samples in each subset.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Label column name.",
    )
    parser.add_argument(
        "--attack-type-column",
        default="attack_type",
        help="Attack-type column name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--prefix",
        default="pkdd_pilot",
        help="Filename prefix for the generated subset files.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow the same sample to appear in multiple subsets.",
    )
    parser.add_argument(
        "--quality-mode",
        choices=["off", "basic"],
        default="basic",
        help="Whether to rank and filter rows using lightweight request-quality heuristics.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.70,
        help="Keep only the top fraction of each pool after quality ranking. Ignored when quality-mode=off.",
    )
    parser.add_argument(
        "--min-quality",
        type=int,
        default=None,
        help="Optional minimum quality score required after ranking.",
    )
    parser.add_argument(
        "--disable-dedupe",
        action="store_true",
        help="Keep exact duplicate raw_http rows instead of collapsing them.",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def _normalized_label(text: str) -> str:
    value = str(text).strip().lower()
    if value == "benign":
        return "benign"
    if value == "malicious":
        return "malicious"
    raise ValueError(f"Unsupported label value: {text!r}")


def _allocate_evenly(total: int, attack_types: list[str]) -> dict[str, int]:
    if not attack_types:
        return {}
    base = total // len(attack_types)
    remainder = total % len(attack_types)
    return {
        attack_type: base + (1 if index < remainder else 0)
        for index, attack_type in enumerate(sorted(attack_types))
    }


def _take_without_overlap(
    pool: list[dict[str, Any]],
    start: int,
    count: int,
) -> tuple[list[dict[str, Any]], int]:
    end = start + count
    if end > len(pool):
        raise ValueError(
            f"Requested {count} rows but only {len(pool) - start} remain in the pool."
        )
    return pool[start:end], end


def _parse_request(raw_http: str) -> tuple[str, str, str, dict[str, str], str]:
    text = str(raw_http or "").replace("\r\n", "\n").replace("\r", "\n")
    head, _, body = text.partition("\n\n")
    head_lines = [line for line in head.split("\n") if line]
    request_line = head_lines[0] if head_lines else ""
    parts = request_line.split(" ", 2)
    method = parts[0].upper() if parts else ""
    target = parts[1] if len(parts) >= 2 else ""
    proto = parts[2] if len(parts) >= 3 else ""
    headers: dict[str, str] = {}
    for line in head_lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return method, target, proto, headers, body


def _count_pattern_hits(patterns: list[re.Pattern[str]], texts: list[str]) -> int:
    hits = 0
    for pattern in patterns:
        if any(pattern.search(text) for text in texts):
            hits += 1
    return hits


def _score_row(
    row: dict[str, Any],
    label_column: str,
    attack_column: str,
) -> tuple[int, str, str]:
    label = _normalized_label(row.get(label_column, ""))
    attack_type = normalize_attack_type(row.get(attack_column))
    raw_http = str(row.get("raw_http", "") or "")
    method, target, proto, headers, body = _parse_request(raw_http)
    decoded = unquote_plus(raw_http).lower()
    raw_lower = raw_http.lower()
    texts = [raw_lower, decoded]
    notes: list[str] = []
    score = 0

    if method in {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}:
        score += 8
        notes.append("known_method")
    else:
        score -= 8
        notes.append("unknown_method")

    if proto.startswith("HTTP/"):
        score += 6
        notes.append("http_proto")
    else:
        score -= 10
        notes.append("bad_proto")

    if headers.get("host"):
        score += 4
        notes.append("has_host")
    else:
        score -= 6
        notes.append("missing_host")

    request_length = len(raw_http)
    if 80 <= request_length <= 5000:
        score += 6
        notes.append("good_length")
    elif request_length > 12000 or request_length < 30:
        score -= 10
        notes.append("extreme_length")

    if "?" in target or body.strip():
        score += 4
        notes.append("has_payload_area")

    if method in {"POST", "PUT"} and body.strip():
        score += 3
        notes.append("non_get_with_body")
    elif method in {"POST", "PUT"}:
        score -= 4
        notes.append("non_get_empty_body")

    if method in {"POST", "PUT"}:
        score += 1
        notes.append("method_diversity_bonus")

    if label == "malicious":
        patterns = _CLASS_PATTERNS.get(attack_type or "", [])
        class_hits = _count_pattern_hits(patterns, texts)
        any_hits = _count_pattern_hits(_ALL_PATTERNS, texts)
        if class_hits:
            score += 20 + (class_hits * 4)
            notes.append(f"class_hits={class_hits}")
        else:
            score -= 15
            notes.append("no_class_evidence")
        if any_hits:
            score += min(any_hits, 5) * 2
            notes.append(f"generic_hits={any_hits}")
    else:
        suspicious_hits = _count_pattern_hits(_ALL_PATTERNS, texts)
        if suspicious_hits == 0:
            score += 20
            notes.append("clean_benign")
        else:
            score -= 10 + (suspicious_hits * 3)
            notes.append(f"suspicious_hits={suspicious_hits}")

    return score, method, ",".join(notes[:8])


def _dedupe_rows(
    rows: list[dict[str, Any]],
    label_column: str,
    attack_column: str,
) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        label = _normalized_label(row.get(label_column, ""))
        attack_type = normalize_attack_type(row.get(attack_column)) or ""
        raw_http = str(row.get("raw_http", "") or "").strip()
        key = (label, attack_type, raw_http)
        current = best_by_key.get(key)
        if current is None or int(row.get("quality_score", 0)) > int(current.get("quality_score", 0)):
            best_by_key[key] = row
    return list(best_by_key.values())


def _interleave_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        buckets[str(row.get("quality_method", "") or "UNKNOWN")].append(row)

    ordered_methods = sorted(
        buckets,
        key=lambda method: (_METHOD_PRIORITY.get(method, 99), method),
    )
    interleaved: list[dict[str, Any]] = []
    while any(buckets.values()):
        for method in ordered_methods:
            if buckets[method]:
                interleaved.append(buckets[method].popleft())
    return interleaved


def _prepare_pool(
    pool: list[dict[str, Any]],
    required: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.quality_mode == "off":
        return pool

    ranked = sorted(
        pool,
        key=lambda row: (
            int(row.get("quality_score", 0)),
            str(row.get("sample_id", "")),
        ),
        reverse=True,
    )
    ranked = _interleave_by_method(ranked)

    if args.min_quality is not None:
        ranked = [row for row in ranked if int(row.get("quality_score", 0)) >= args.min_quality]

    keep_count = max(required, int(len(ranked) * args.top_fraction))
    keep_count = min(len(ranked), keep_count)
    return ranked[:keep_count]


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, fieldnames = _read_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    label_column = args.label_column
    attack_column = args.attack_type_column
    if label_column not in rows[0]:
        raise KeyError(f"Label column {label_column!r} not found in {input_path}")
    if attack_column not in rows[0]:
        raise KeyError(f"Attack-type column {attack_column!r} not found in {input_path}")

    if "raw_http" not in rows[0]:
        raise KeyError("This script expects a raw_http column in the input CSV.")

    if "quality_score" not in fieldnames:
        fieldnames.extend(["quality_score", "quality_method", "quality_notes"])

    original_row_count = len(rows)
    for row in rows:
        score, method, notes = _score_row(row, label_column, attack_column)
        row["quality_score"] = str(score)
        row["quality_method"] = method
        row["quality_notes"] = notes

    if not args.disable_dedupe:
        rows = _dedupe_rows(rows, label_column, attack_column)

    benign_pool: list[dict[str, Any]] = []
    malicious_pools: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        label = _normalized_label(row.get(label_column, ""))
        if label == "benign":
            benign_pool.append(row)
            continue

        attack_type = normalize_attack_type(row.get(attack_column)) or "Unknown"
        row[attack_column] = attack_type
        malicious_pools[attack_type].append(row)

    rng.shuffle(benign_pool)
    for pool in malicious_pools.values():
        rng.shuffle(pool)

    attack_types = sorted(malicious_pools)
    if not attack_types:
        raise ValueError("No malicious rows with attack_type were found.")

    per_attack_counts = _allocate_evenly(args.malicious_per_subset, attack_types)

    if args.allow_overlap:
        required_benign = args.benign_per_subset
    else:
        required_benign = args.benign_per_subset * args.subset_count
    if len(benign_pool) < required_benign:
        raise ValueError(
            f"Need {required_benign} benign rows but only {len(benign_pool)} are available."
        )

    benign_pool = _prepare_pool(benign_pool, required_benign, args)
    if len(benign_pool) < required_benign:
        raise ValueError(
            f"After quality filtering, only {len(benign_pool)} benign rows remain but {required_benign} are required."
        )

    for attack_type, needed in per_attack_counts.items():
        required = needed if args.allow_overlap else needed * args.subset_count
        malicious_pools[attack_type] = _prepare_pool(malicious_pools[attack_type], required, args)
        available = len(malicious_pools[attack_type])
        if available < required:
            raise ValueError(
                f"Need {required} malicious rows for {attack_type} but only {available} are available."
            )

    subset_summaries: list[dict[str, Any]] = []
    benign_offset = 0
    malicious_offsets = {attack_type: 0 for attack_type in attack_types}

    for subset_index in range(1, args.subset_count + 1):
        selected: list[dict[str, Any]] = []

        if args.allow_overlap:
            selected.extend(rng.sample(benign_pool, args.benign_per_subset))
        else:
            benign_rows, benign_offset = _take_without_overlap(
                benign_pool,
                benign_offset,
                args.benign_per_subset,
            )
            selected.extend(benign_rows)

        for attack_type in attack_types:
            count = per_attack_counts[attack_type]
            pool = malicious_pools[attack_type]
            if count == 0:
                continue
            if args.allow_overlap:
                selected.extend(rng.sample(pool, count))
            else:
                attack_rows, malicious_offsets[attack_type] = _take_without_overlap(
                    pool,
                    malicious_offsets[attack_type],
                    count,
                )
                selected.extend(attack_rows)

        rng.shuffle(selected)

        subset_name = f"{args.prefix}_{subset_index:02d}.csv"
        subset_path = output_dir / subset_name
        with subset_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in selected:
                writer.writerow(row)

        label_distribution = Counter(_normalized_label(row[label_column]) for row in selected)
        attack_distribution = Counter(
            normalize_attack_type(row.get(attack_column)) or "Unknown"
            for row in selected
            if _normalized_label(row[label_column]) == "malicious"
        )
        subset_summaries.append(
            {
                "file": subset_name,
                "total_rows": len(selected),
                "label_distribution": dict(sorted(label_distribution.items())),
                "attack_distribution": dict(sorted(attack_distribution.items())),
                "quality": {
                    "mean_score": (
                        sum(int(row.get("quality_score", 0)) for row in selected) / len(selected)
                        if selected else 0.0
                    ),
                    "min_score": min(int(row.get("quality_score", 0)) for row in selected),
                    "max_score": max(int(row.get("quality_score", 0)) for row in selected),
                    "method_distribution": dict(
                        sorted(Counter(str(row.get("quality_method", "") or "UNKNOWN") for row in selected).items())
                    ),
                },
            }
        )

    summary_path = output_dir / f"{args.prefix}_summary.json"
    summary = {
        "input_csv": str(input_path),
        "original_row_count": original_row_count,
        "row_count_after_dedupe": len(rows),
        "subset_count": args.subset_count,
        "benign_per_subset": args.benign_per_subset,
        "malicious_per_subset": args.malicious_per_subset,
        "allow_overlap": args.allow_overlap,
        "seed": args.seed,
        "quality_mode": args.quality_mode,
        "top_fraction": args.top_fraction,
        "min_quality": args.min_quality,
        "dedupe_enabled": not args.disable_dedupe,
        "per_attack_counts": dict(sorted(per_attack_counts.items())),
        "candidate_pool_sizes": {
            "benign": len(benign_pool),
            "malicious": {
                attack_type: len(pool)
                for attack_type, pool in sorted(malicious_pools.items())
            },
        },
        "subsets": subset_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {args.subset_count} subset file(s) to: {output_dir}")
    print(f"Subset summary: {summary_path}")
    print(f"Per-attack malicious allocation: {dict(sorted(per_attack_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
