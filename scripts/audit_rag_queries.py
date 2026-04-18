#!/usr/bin/env python3
"""Export audit-friendly RAG retrieval traces for manual inspection."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    REQUEST_COLUMN_CANDIDATES,
    add_dataset_arguments,
    coerce_query_text,
    ensure_output_dir,
    load_rows_from_dataset,
    normalize_attack_type,
    normalize_binary_label,
    pick_sample_id,
)
from config.settings import settings  # noqa: E402
from nodes.preprocess import preprocess_node  # noqa: E402
from nodes.rag_node import (  # noqa: E402
    collect_payload_hit_trace,
    merge_ranked_payload_hit_traces,
)
from utils.security import normalise_payload  # noqa: E402

AUDIT_CSV_COLUMNS = [
    # Request identity and ground truth.
    "dataset_index",
    "sample_id",
    "query_mode",
    "true_label",
    "true_attack_type",
    # Raw request and preprocess stage.
    "raw_query_text",
    "payload_count_before_limit",
    "payload_count",
    "skipped_payload_count",
    "normalized_payloads",
    "normalized_payload_details",
    "skipped_payloads",
    # Retrieval query stage.
    "retrieval_payload_count",
    "retrieval_payloads",
    "rag_query_text",
    # Qdrant top-k outputs.
    "top_k_returned",
    "retrieved_topk_categories",
    "retrieved_topk_scores",
    "retrieved_topk_payloads",
    "retrieved_topk_source_files",
    "retrieved_topk_line_nos",
    "top_categories",
    "top_results_compact",
    # Post-hoc evaluation.
    "true_attack_type_in_topk",
    "error",
]


@dataclass(slots=True)
class AuditSample:
    dataset_index: int
    sample_id: str
    query_text: str
    query_mode: str
    true_label: str
    true_attack_type: str | None


@dataclass(slots=True)
class AuditRecord:
    dataset_index: int
    sample_id: str
    query_mode: str
    true_label: str
    true_attack_type: str | None
    raw_query_text: str
    payload_count_before_limit: int
    payload_count: int
    normalized_payloads: str
    normalized_payload_details: str
    retrieval_payload_count: int
    retrieval_payloads: str
    skipped_payload_count: int
    skipped_payloads: str
    rag_query_text: str
    top_k_returned: int
    true_attack_type_in_topk: bool
    top_categories: str
    retrieved_topk_categories: str
    retrieved_topk_scores: str
    retrieved_topk_payloads: str
    retrieved_topk_source_files: str
    retrieved_topk_line_nos: str
    top_results_compact: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export detailed RAG retrieval traces for audit.",
    )
    add_dataset_arguments(parser)
    parser.add_argument(
        "--only-label",
        choices=("all", "benign", "malicious"),
        default="all",
        help="Keep only rows with this normalized label.",
    )
    parser.add_argument(
        "--only-attack-type",
        action="append",
        default=[],
        help="Keep only samples matching this normalized attack type. Can be repeated.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Keep only these sample ids. Can be repeated.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap after filtering. Use 0 to keep all samples.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-payloads-per-request",
        type=int,
        default=0,
        help="Optional cap on normalized payloads scanned inside each request. Use 0 for all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize_label_for_audit(
    row: dict[str, Any],
    label_column: str | None,
    label_feature: Any,
    attack_type_column: str | None,
) -> str:
    if label_column:
        return normalize_binary_label(row[label_column], label_feature)

    attack_type = normalize_attack_type(row.get(attack_type_column) if attack_type_column else None)
    return "malicious" if attack_type else ""


def load_audit_samples(args: argparse.Namespace) -> tuple[list[AuditSample], dict[str, Any]]:
    query_column = args.query_column or args.request_column
    loaded = load_rows_from_dataset(
        dataset_name=args.dataset,
        split=args.split,
        data_file=args.data_file,
        request_column=args.request_column,
        query_column=query_column,
        label_column=args.label_column,
        attack_type_column=args.attack_type_column,
        require_query=True,
        require_label=False,
    )
    rows = list(loaded.rows)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    wanted_ids = set(args.sample_id)
    wanted_attack_types = {
        normalized
        for normalized in (
            normalize_attack_type(value)
            for value in args.only_attack_type
        )
        if normalized
    }

    label_feature = loaded.features.get(loaded.label_column) if loaded.label_column else None
    samples: list[AuditSample] = []
    for index, row in enumerate(rows):
        query_text = coerce_query_text(row.get(loaded.query_column))
        if not query_text:
            continue

        sample_id = pick_sample_id(row, loaded.sample_id_column, index)
        if wanted_ids and sample_id not in wanted_ids:
            continue

        true_label = _normalize_label_for_audit(
            row,
            loaded.label_column,
            label_feature,
            loaded.attack_type_column,
        )
        if args.only_label != "all" and true_label != args.only_label:
            continue

        true_attack_type = normalize_attack_type(
            row.get(loaded.attack_type_column) if loaded.attack_type_column else None
        )
        if wanted_attack_types and true_attack_type not in wanted_attack_types:
            continue

        query_mode = "raw_http" if loaded.query_column in REQUEST_COLUMN_CANDIDATES else "text"
        samples.append(
            AuditSample(
                dataset_index=index,
                sample_id=sample_id,
                query_text=query_text,
                query_mode=query_mode,
                true_label=true_label,
                true_attack_type=true_attack_type,
            )
        )

    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    dataset_info = {
        "dataset": args.dataset,
        "split": args.split,
        "data_file": args.data_file,
        "query_column": loaded.query_column,
        "label_column": loaded.label_column,
        "attack_type_column": loaded.attack_type_column,
        "sample_id_column": loaded.sample_id_column,
        "column_names": loaded.column_names,
    }
    return samples, dataset_info


def _prepare_payload_trace(
    sample: AuditSample,
    max_payloads_per_request: int,
) -> tuple[list[str], list[dict[str, Any]], int]:
    if sample.query_mode == "raw_http":
        state = preprocess_node({"raw_http_text": sample.query_text})
        detail_pairs: list[tuple[str, dict[str, Any]]] = []
        for detail in state.get("normalized_payload_details", []):
            normalized_value = str(detail.get("normalized_value", "") or "")
            if not normalized_value:
                continue
            detail_pairs.append((normalized_value, detail))

        total_payload_count = len(detail_pairs)
        if max_payloads_per_request > 0:
            detail_pairs = detail_pairs[:max_payloads_per_request]

        payloads = [payload for payload, _detail in detail_pairs]
        details = [detail for _payload, detail in detail_pairs]
        return payloads, details, total_payload_count

    normalized = normalise_payload(sample.query_text)
    if not normalized:
        return [], [], 0
    return [normalized], [
        {
            "source_type": "query",
            "source_name": "text",
            "raw_value": sample.query_text,
            "normalized_value": normalized,
        }
    ], 1


def _simplify_hit(hit: dict[str, Any], true_attack_type: str | None) -> dict[str, Any]:
    record = hit.get("record") or {}
    category = normalize_attack_type(record.get("category")) or str(record.get("category") or "Unknown")
    return {
        "point_id": str(hit.get("point_id") or ""),
        "score": float(hit.get("score", 0.0)),
        "category": category,
        "payload": str(hit.get("payload", "") or ""),
        "source_file": str(record.get("source_file") or ""),
        "line_no": record.get("line_no"),
        "matches_true_attack_type": bool(true_attack_type and category == true_attack_type),
    }


def _compact_top_results(hits: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, hit in enumerate(hits, start=1):
        source_file = hit.get("source_file") or "-"
        lines.append(
            f"#{index} [{hit.get('category', 'Unknown')}] "
            f"score={float(hit.get('score', 0.0)):.4f} "
            f"source={source_file} payload={hit.get('payload', '')}"
        )
    return "\n".join(lines)


def evaluate_sample(
    sample: AuditSample,
    top_k: int,
    max_payloads_per_request: int,
) -> tuple[AuditRecord, dict[str, Any]]:
    try:
        payloads, payload_details, total_payload_count = _prepare_payload_trace(
            sample,
            max_payloads_per_request,
        )
        payload_traces = collect_payload_hit_trace(payloads, limit=top_k)
        merged_hits = merge_ranked_payload_hit_traces(payload_traces, top_k)
    except Exception as exc:
        record = AuditRecord(
            dataset_index=sample.dataset_index,
            sample_id=sample.sample_id,
            query_mode=sample.query_mode,
            true_label=sample.true_label,
            true_attack_type=sample.true_attack_type,
            raw_query_text=sample.query_text,
            payload_count_before_limit=0,
            payload_count=0,
            normalized_payloads="[]",
            normalized_payload_details="[]",
            retrieval_payload_count=0,
            retrieval_payloads="[]",
            skipped_payload_count=0,
            skipped_payloads="[]",
            rag_query_text="",
            top_k_returned=0,
            true_attack_type_in_topk=False,
            top_categories="",
            retrieved_topk_categories="[]",
            retrieved_topk_scores="[]",
            retrieved_topk_payloads="[]",
            retrieved_topk_source_files="[]",
            retrieved_topk_line_nos="[]",
            top_results_compact="",
            error=str(exc),
        )
        audit = {
            "dataset_index": sample.dataset_index,
            "sample_id": sample.sample_id,
            "query_mode": sample.query_mode,
            "true_label": sample.true_label,
            "true_attack_type": sample.true_attack_type,
            "raw_query_text": sample.query_text,
            "payload_count_before_limit": 0,
            "error": str(exc),
        }
        return record, audit

    retrieval_payloads = [
        trace["payload"]
        for trace in payload_traces
        if trace.get("eligible")
    ]
    skipped_payloads = [
        {
            "payload_index": trace.get("payload_index"),
            "payload": trace.get("payload"),
            "skip_reason": trace.get("skip_reason", ""),
        }
        for trace in payload_traces
        if not trace.get("eligible")
    ]
    simplified_hits = [_simplify_hit(hit, sample.true_attack_type) for hit in merged_hits]
    top_categories = [str(hit.get("category") or "Unknown") for hit in simplified_hits]

    record = AuditRecord(
        dataset_index=sample.dataset_index,
        sample_id=sample.sample_id,
        query_mode=sample.query_mode,
        true_label=sample.true_label,
        true_attack_type=sample.true_attack_type,
        raw_query_text=sample.query_text,
        payload_count_before_limit=total_payload_count,
        payload_count=len(payloads),
        normalized_payloads=_json_compact(payloads),
        normalized_payload_details=_json_compact(payload_details),
        retrieval_payload_count=len(retrieval_payloads),
        retrieval_payloads=_json_compact(retrieval_payloads),
        skipped_payload_count=len(skipped_payloads),
        skipped_payloads=_json_compact(skipped_payloads),
        rag_query_text="\n---\n".join(retrieval_payloads),
        top_k_returned=len(simplified_hits),
        true_attack_type_in_topk=bool(
            sample.true_attack_type and sample.true_attack_type in top_categories
        ),
        top_categories=",".join(top_categories),
        retrieved_topk_categories=_json_compact(top_categories),
        retrieved_topk_scores=_json_compact([float(hit.get("score", 0.0)) for hit in simplified_hits]),
        retrieved_topk_payloads=_json_compact([str(hit.get("payload", "") or "") for hit in simplified_hits]),
        retrieved_topk_source_files=_json_compact([
            str(hit.get("source_file", "") or "")
            for hit in simplified_hits
        ]),
        retrieved_topk_line_nos=_json_compact([hit.get("line_no") for hit in simplified_hits]),
        top_results_compact=_compact_top_results(simplified_hits),
        error="",
    )

    audit = {
        "dataset_index": sample.dataset_index,
        "sample_id": sample.sample_id,
        "query_mode": sample.query_mode,
        "true_label": sample.true_label,
        "true_attack_type": sample.true_attack_type,
        "raw_query_text": sample.query_text,
        "payload_count_before_limit": total_payload_count,
        "payload_count": len(payloads),
        "normalized_payloads": payloads,
        "normalized_payload_details": payload_details,
        "retrieval_payload_count": len(retrieval_payloads),
        "retrieval_payloads": retrieval_payloads,
        "skipped_payloads": skipped_payloads,
        "rag_query_text": "\n---\n".join(retrieval_payloads),
        "top_k_returned": len(simplified_hits),
        "true_attack_type_in_topk": record.true_attack_type_in_topk,
        "retrieved_topk": [
            {
                **hit,
                "rank": index + 1,
            }
            for index, hit in enumerate(simplified_hits)
        ],
        "per_payload_trace": [
            {
                "payload_index": trace.get("payload_index"),
                "payload": trace.get("payload"),
                "eligible": bool(trace.get("eligible")),
                "skip_reason": str(trace.get("skip_reason", "") or ""),
                "hits": [
                    _simplify_hit(hit, sample.true_attack_type)
                    for hit in trace.get("hits", [])
                ],
            }
            for trace in payload_traces
        ],
        "error": "",
    }
    return record, audit


def summarize(
    args: argparse.Namespace,
    dataset_info: dict[str, Any],
    records: list[AuditRecord],
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "dataset": dataset_info["dataset"],
        "split": dataset_info["split"],
        "data_file": dataset_info["data_file"],
        "dataset_columns": {
            "available": dataset_info["column_names"],
            "query_column": dataset_info["query_column"],
            "label_column": dataset_info["label_column"],
            "attack_type_column": dataset_info["attack_type_column"],
            "sample_id_column": dataset_info["sample_id_column"],
        },
        "config": {
            "top_k": args.top_k,
            "max_payloads_per_request": args.max_payloads_per_request,
            "rag_enabled": settings.rag_enabled,
            "qdrant_collection": settings.qdrant_collection,
            "only_label": args.only_label,
            "only_attack_type": args.only_attack_type,
            "sample_id_filters": args.sample_id,
            "max_samples": args.max_samples,
            "shuffle": args.shuffle,
            "seed": args.seed,
        },
        "counts": {
            "total_samples": len(records),
            "query_mode_distribution": dict(Counter(record.query_mode for record in records)),
            "label_distribution": dict(Counter(record.true_label for record in records if record.true_label)),
            "attack_type_distribution": dict(
                Counter(record.true_attack_type for record in records if record.true_attack_type)
            ),
            "errors": sum(1 for record in records if record.error),
            "with_any_context": sum(1 for record in records if record.top_k_returned > 0),
            "with_true_attack_in_topk": sum(1 for record in records if record.true_attack_type_in_topk),
        },
    }


def write_csv(records: list[AuditRecord], output_path: Path) -> None:
    if not records:
        output_path.write_text("", encoding="utf-8")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            raw = asdict(record)
            writer.writerow({column: raw.get(column, "") for column in AUDIT_CSV_COLUMNS})


def write_jsonl(items: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False))
            fh.write("\n")


def write_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(summary: dict[str, Any], csv_path: Path, jsonl_path: Path, summary_path: Path) -> None:
    counts = summary["counts"]
    print("\nRAG audit export summary")
    print(f"  Samples               : {counts['total_samples']}")
    print(f"  Errors                : {counts['errors']}")
    print(f"  RAG enabled           : {summary['config']['rag_enabled']}")
    print(f"  With any context      : {counts['with_any_context']}")
    print(f"  True type in top-k    : {counts['with_true_attack_in_topk']}")
    print(f"  CSV                   : {csv_path}")
    print(f"  JSONL                 : {jsonl_path}")
    print(f"  Summary               : {summary_path}")


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        print("--top-k must be at least 1.", file=sys.stderr)
        return 2

    samples, dataset_info = load_audit_samples(args)
    if not samples:
        print("No samples loaded after filtering.", file=sys.stderr)
        return 1

    records: list[AuditRecord] = []
    audits: list[dict[str, Any]] = []
    for sample in samples:
        record, audit = evaluate_sample(
            sample,
            args.top_k,
            args.max_payloads_per_request,
        )
        records.append(record)
        audits.append(audit)

    summary = summarize(args, dataset_info, records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_output_dir(args.output_dir)
    csv_path = output_dir / f"rag_audit_{timestamp}.csv"
    jsonl_path = output_dir / f"rag_audit_{timestamp}.jsonl"
    summary_path = output_dir / f"rag_audit_summary_{timestamp}.json"

    write_csv(records, csv_path)
    write_jsonl(audits, jsonl_path)
    write_json(summary, summary_path)
    print_summary(summary, csv_path, jsonl_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
