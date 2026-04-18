#!/usr/bin/env python3
"""Audit category distribution and source hygiene inside a Qdrant collection."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qdrant_client import QdrantClient  # noqa: E402

from benchmark_common import normalize_attack_type  # noqa: E402
from config.settings import settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit attack-type/category distribution in a Qdrant collection.",
    )
    parser.add_argument(
        "--collection",
        default=settings.qdrant_collection,
        help="Collection name to inspect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of points to scan.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=512,
        help="Scroll page size.",
    )
    parser.add_argument(
        "--top-sources",
        type=int,
        default=20,
        help="How many source files to print.",
    )
    parser.add_argument(
        "--top-categories",
        type=int,
        default=50,
        help="How many categories to print.",
    )
    parser.add_argument(
        "--sample-per-category",
        type=int,
        default=3,
        help="How many raw payload examples to keep per category.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save full audit report as JSON.",
    )
    return parser.parse_args()


def _make_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=30,
    )


def _coerce_offset(value: Any) -> Any:
    if value in (None, "", 0):
        return None
    return value


def scan_collection(
    client: QdrantClient,
    collection: str,
    page_size: int,
    limit: int | None,
    sample_per_category: int,
) -> dict[str, Any]:
    offset: Any = None
    scanned = 0

    raw_category_counter: Counter[str] = Counter()
    normalized_category_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    category_samples: dict[str, list[dict[str, str]]] = defaultdict(list)

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=page_size,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break

        for point in points:
            payload = point.payload or {}
            raw_category = str(
                payload.get("category")
                or payload.get("attack_type")
                or "Unknown"
            )
            normalized = normalize_attack_type(raw_category) or raw_category
            source_file = str(payload.get("source_file") or "")
            raw_payload = str(
                payload.get("raw_payload")
                or payload.get("payload")
                or payload.get("text")
                or ""
            )

            raw_category_counter[raw_category] += 1
            normalized_category_counter[normalized] += 1
            if source_file:
                source_counter[source_file] += 1

            bucket = category_samples[raw_category]
            if len(bucket) < sample_per_category:
                bucket.append(
                    {
                        "point_id": str(point.id),
                        "raw_payload": raw_payload[:500],
                        "source_file": source_file,
                    }
                )

            scanned += 1
            if limit is not None and scanned >= limit:
                return {
                    "scanned_points": scanned,
                    "raw_category_counts": dict(raw_category_counter),
                    "normalized_category_counts": dict(normalized_category_counter),
                    "top_source_files": dict(source_counter),
                    "category_samples": dict(category_samples),
                    "truncated": True,
                }

        offset = _coerce_offset(next_offset)
        if offset is None:
            break

    return {
        "scanned_points": scanned,
        "raw_category_counts": dict(raw_category_counter),
        "normalized_category_counts": dict(normalized_category_counter),
        "top_source_files": dict(source_counter),
        "category_samples": dict(category_samples),
        "truncated": False,
    }


def print_report(report: dict[str, Any], args: argparse.Namespace) -> None:
    raw_counts = Counter(report["raw_category_counts"])
    normalized_counts = Counter(report["normalized_category_counts"])
    source_counts = Counter(report["top_source_files"])

    print(f"Collection           : {args.collection}")
    print(f"Scanned points       : {report['scanned_points']}")
    print(f"Truncated            : {report['truncated']}")

    print("\nRaw category counts")
    for category, count in raw_counts.most_common(args.top_categories):
        print(f"  {category:<35} {count}")

    print("\nNormalized category counts")
    for category, count in normalized_counts.most_common(args.top_categories):
        print(f"  {category:<35} {count}")

    print("\nTop source files")
    for source_file, count in source_counts.most_common(args.top_sources):
        print(f"  {count:<6} {source_file}")

    print("\nSample payloads by raw category")
    for category, samples in raw_counts.most_common(min(args.top_categories, 15)):
        print(f"  [{category}]")
        for sample in report["category_samples"].get(category, []):
            payload = sample["raw_payload"].replace("\n", "\\n")
            print(f"    - {payload}")


def main() -> int:
    args = parse_args()
    client = _make_client()
    report = scan_collection(
        client=client,
        collection=args.collection,
        page_size=args.page_size,
        limit=args.limit,
        sample_per_category=args.sample_per_category,
    )

    print_report(report, args)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved JSON report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
