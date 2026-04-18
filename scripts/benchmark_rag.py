#!/usr/bin/env python3
"""Benchmark BlueAgent RAG retrieval quality."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
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
from nodes.rag_node import collect_ranked_payload_hits  # noqa: E402
from utils.debug_csv_logger import log_debug_snapshot  # noqa: E402
from utils.security import normalise_payload  # noqa: E402


@dataclass(slots=True)
class RetrievalSample:
    dataset_index: int
    sample_id: str
    query_text: str
    query_mode: str
    true_label: str
    true_attack_type: str | None


@dataclass(slots=True)
class RetrievalRecord:
    dataset_index: int
    sample_id: str
    query_mode: str
    true_label: str
    true_attack_type: str | None
    payload_count: int
    retrieval_payload_count: int
    normalized_payloads: str
    retrieval_payloads: str
    top_k_returned: int
    top1_score: float | None
    relevant_hits: int
    hit_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    returned_any_context: bool
    top_categories: str
    retrieved_topk_categories: str
    retrieved_topk_scores: str
    retrieved_topk_payloads: str
    retrieved_topk_source_files: str
    retrieved_topk_line_nos: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark BlueAgent RAG retrieval quality.",
    )
    add_dataset_arguments(parser)
    parser.add_argument("--benign-samples", type=int, default=200)
    parser.add_argument("--malicious-samples", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Drop retrieved hits below this score before computing metrics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_retrieval_samples(args: argparse.Namespace) -> tuple[list[RetrievalSample], dict[str, Any]]:
    print(
        f"Loading retrieval dataset '{args.dataset}' split '{args.split}'...",
        flush=True,
    )
    loaded = load_rows_from_dataset(
        dataset_name=args.dataset,
        split=args.split,
        data_file=args.data_file,
        query_column=args.query_column or args.request_column,
        label_column=args.label_column,
        attack_type_column=args.attack_type_column,
        require_query=True,
        require_label=False,
    )
    rows = list(loaded.rows)

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    label_feature = loaded.features.get(loaded.label_column) if loaded.label_column else None
    benign_pool: list[RetrievalSample] = []
    malicious_pool: list[RetrievalSample] = []

    for index, row in enumerate(rows):
        query_text = coerce_query_text(row.get(loaded.query_column))
        if not query_text:
            continue

        true_attack_type = normalize_attack_type(
            row.get(loaded.attack_type_column) if loaded.attack_type_column else None
        )
        if loaded.label_column:
            true_label = normalize_binary_label(row[loaded.label_column], label_feature)
        elif true_attack_type:
            true_label = "malicious"
        else:
            continue

        query_mode = "raw_http" if loaded.query_column in REQUEST_COLUMN_CANDIDATES else "text"
        sample = RetrievalSample(
            dataset_index=index,
            sample_id=pick_sample_id(row, loaded.sample_id_column, index),
            query_text=query_text,
            query_mode=query_mode,
            true_label=true_label,
            true_attack_type=true_attack_type,
        )
        if true_label == "benign":
            benign_pool.append(sample)
        elif true_label == "malicious":
            malicious_pool.append(sample)

    requested_benign = max(args.benign_samples, 0)
    requested_malicious = max(args.malicious_samples, 0)
    if requested_benign > len(benign_pool):
        raise ValueError(
            f"Requested {requested_benign} benign samples, but only {len(benign_pool)} are available."
        )
    if requested_malicious > len(malicious_pool):
        raise ValueError(
            f"Requested {requested_malicious} malicious samples, but only {len(malicious_pool)} are available."
        )

    samples = benign_pool[:requested_benign] + malicious_pool[:requested_malicious]
    if args.shuffle:
        random.Random(args.seed).shuffle(samples)

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


def _prepare_payloads(sample: RetrievalSample) -> tuple[list[str], list[str]]:
    if sample.query_mode == "raw_http":
        state = preprocess_node({"raw_http_text": sample.query_text})
        normalized_payloads = [
            payload for payload in state.get("normalized_payloads", []) if payload
        ]
        retrieval_payloads = _filter_with_gatekeeper(normalized_payloads)
        return normalized_payloads, retrieval_payloads

    text = normalise_payload(sample.query_text)
    normalized_payloads = [text] if text else []
    retrieval_payloads = _filter_with_gatekeeper(normalized_payloads)
    return normalized_payloads, retrieval_payloads


def _discounted_gain(relevance_flags: list[int]) -> float:
    score = 0.0
    for index, relevant in enumerate(relevance_flags):
        if relevant:
            score += 1.0 / math.log2(index + 2)
    return score


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _filter_with_gatekeeper(payloads: list[str]) -> list[str]:
    try:
        from nodes.gatekeeper_node import gatekeeper_node  # noqa: E402
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Gatekeeper ML is required for benchmark_rag.py but is not installed. "
            "Install the gatekeeper_ml dependency or run on the target server environment."
        ) from exc

    gatekeeper_state = gatekeeper_node({"normalized_payloads": payloads})
    return [
        payload for payload in gatekeeper_state.get("suspicious_payloads", []) if payload
    ]


def _log_benchmark_debug(
    sample: RetrievalSample,
    normalized_payloads: list[str],
    retrieval_payloads: list[str],
    filtered_hits: list[dict[str, Any]],
) -> None:
    """Emit the same debug CSV snapshot shape used by the live service."""
    log_debug_snapshot(
        {
            "raw_http_text": sample.query_text,
            "normalized_payloads": normalized_payloads,
            "suspicious_payloads": retrieval_payloads,
            "rag_context": [str(hit.get("payload", "") or "") for hit in filtered_hits],
        },
        rag_query=retrieval_payloads,
        rag_query_result=[str(hit.get("payload", "") or "") for hit in filtered_hits],
    )


def evaluate_sample(
    sample: RetrievalSample,
    top_k: int,
    min_score: float,
) -> RetrievalRecord:
    try:
        normalized_payloads, retrieval_payloads = _prepare_payloads(sample)
    except Exception as exc:
        return RetrievalRecord(
            dataset_index=sample.dataset_index,
            sample_id=sample.sample_id,
            query_mode=sample.query_mode,
            true_label=sample.true_label,
            true_attack_type=sample.true_attack_type,
            payload_count=0,
            retrieval_payload_count=0,
            normalized_payloads="[]",
            retrieval_payloads="[]",
            top_k_returned=0,
            top1_score=None,
            relevant_hits=0,
            hit_at_k=0.0,
            precision_at_k=0.0,
            reciprocal_rank=0.0,
            ndcg_at_k=0.0,
            returned_any_context=False,
            top_categories="",
            retrieved_topk_categories="[]",
            retrieved_topk_scores="[]",
            retrieved_topk_payloads="[]",
            retrieved_topk_source_files="[]",
            retrieved_topk_line_nos="[]",
            error=str(exc),
        )

    hits = collect_ranked_payload_hits(retrieval_payloads, limit=max(top_k * 3, top_k))
    filtered_hits = [
        hit for hit in hits
        if float(hit.get("score", 0.0)) >= min_score
    ][:top_k]

    normalized_categories = [
        normalize_attack_type((hit.get("record") or {}).get("category"))
        for hit in filtered_hits
    ]
    relevant_flags = [
        1 if sample.true_attack_type and category == sample.true_attack_type else 0
        for category in normalized_categories
    ]
    relevant_hits = sum(relevant_flags)

    reciprocal_rank = 0.0
    for index, relevant in enumerate(relevant_flags):
        if relevant:
            reciprocal_rank = 1.0 / (index + 1)
            break

    dcg = _discounted_gain(relevant_flags)
    ideal_dcg = _discounted_gain([1] * relevant_hits)
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    top_categories = ",".join(category or "Unknown" for category in normalized_categories[:3])
    _log_benchmark_debug(
        sample,
        normalized_payloads,
        retrieval_payloads,
        filtered_hits,
    )
    return RetrievalRecord(
        dataset_index=sample.dataset_index,
        sample_id=sample.sample_id,
        query_mode=sample.query_mode,
        true_label=sample.true_label,
        true_attack_type=sample.true_attack_type,
        payload_count=len(normalized_payloads),
        retrieval_payload_count=len(retrieval_payloads),
        normalized_payloads=_json_compact(normalized_payloads),
        retrieval_payloads=_json_compact(retrieval_payloads),
        top_k_returned=len(filtered_hits),
        top1_score=float(filtered_hits[0]["score"]) if filtered_hits else None,
        relevant_hits=relevant_hits,
        hit_at_k=1.0 if relevant_hits > 0 else 0.0,
        precision_at_k=(relevant_hits / len(filtered_hits)) if filtered_hits else 0.0,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=ndcg,
        returned_any_context=bool(filtered_hits),
        top_categories=top_categories,
        retrieved_topk_categories=_json_compact(normalized_categories),
        retrieved_topk_scores=_json_compact([float(hit.get("score", 0.0)) for hit in filtered_hits]),
        retrieved_topk_payloads=_json_compact([str(hit.get("payload", "") or "") for hit in filtered_hits]),
        retrieved_topk_source_files=_json_compact([
            str(((hit.get("record") or {}).get("source_file")) or "")
            for hit in filtered_hits
        ]),
        retrieved_topk_line_nos=_json_compact([
            ((hit.get("record") or {}).get("line_no"))
            for hit in filtered_hits
        ]),
        error="",
    )


def _safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _build_subset_metrics(records: list[RetrievalRecord]) -> dict[str, Any]:
    successful = [record for record in records if not record.error]
    top1_scores = [record.top1_score for record in successful if record.top1_score is not None]
    return {
        "count": len(records),
        "successful": len(successful),
        "errors": len(records) - len(successful),
        "hit_at_k": _safe_mean([record.hit_at_k for record in successful]),
        "precision_at_k": _safe_mean([record.precision_at_k for record in successful]),
        "mrr": _safe_mean([record.reciprocal_rank for record in successful]),
        "ndcg_at_k": _safe_mean([record.ndcg_at_k for record in successful]),
        "average_relevant_hits": _safe_mean([float(record.relevant_hits) for record in successful]),
        "average_returned_results": _safe_mean([float(record.top_k_returned) for record in successful]),
        "mean_top1_score": _safe_mean([float(score) for score in top1_scores]),
        "p95_top1_score": (
            sorted(top1_scores)[max(0, math.ceil(len(top1_scores) * 0.95) - 1)]
            if top1_scores else 0.0
        ),
    }


def summarize_records(
    args: argparse.Namespace,
    dataset_info: dict[str, Any],
    records: list[RetrievalRecord],
) -> dict[str, Any]:
    malicious_records = [
        record
        for record in records
        if record.true_label == "malicious" and record.true_attack_type
    ]
    benign_records = [record for record in records if record.true_label == "benign"]
    failures = [record for record in records if record.error]

    per_attack_type: dict[str, dict[str, Any]] = {}
    for attack_type in sorted({record.true_attack_type for record in malicious_records if record.true_attack_type}):
        subset = [record for record in malicious_records if record.true_attack_type == attack_type]
        per_attack_type[attack_type] = _build_subset_metrics(subset)

    summary = {
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
            "min_score": args.min_score,
            "benign_samples": args.benign_samples,
            "malicious_samples": args.malicious_samples,
            "use_gatekeeper": True,
            "debug_enabled": settings.debug,
            "debug_csv_path": settings.debug_csv_path,
            "rag_enabled": settings.rag_enabled,
            "qdrant_collection": settings.qdrant_collection,
            "shuffle": args.shuffle,
            "seed": args.seed,
        },
        "counts": {
            "total_samples": len(records),
            "malicious_samples": len(malicious_records),
            "benign_samples": len(benign_records),
            "errors": len(failures),
            "query_mode_distribution": dict(Counter(record.query_mode for record in records)),
            "attack_type_distribution": dict(
                Counter(record.true_attack_type for record in malicious_records if record.true_attack_type)
            ),
        },
        "malicious_retrieval": _build_subset_metrics(malicious_records),
        "benign_noise": {
            **_build_subset_metrics(benign_records),
            "context_return_rate": _safe_mean(
                [1.0 if record.returned_any_context else 0.0 for record in benign_records if not record.error]
            ),
        },
        "per_attack_type": per_attack_type,
        "failures": {
            "count": len(failures),
            "examples": [
                {
                    "sample_id": record.sample_id,
                    "query_mode": record.query_mode,
                    "true_label": record.true_label,
                    "error": record.error,
                }
                for record in failures[:20]
            ],
        },
        "example_misses": [
            {
                "sample_id": record.sample_id,
                "attack_type": record.true_attack_type,
                "top_categories": record.top_categories,
                "top1_score": record.top1_score,
                "retrieved_topk_payloads": json.loads(record.retrieved_topk_payloads or "[]"),
                "retrieved_topk_source_files": json.loads(record.retrieved_topk_source_files or "[]"),
            }
            for record in malicious_records
            if not record.error and record.hit_at_k == 0.0
        ][:20],
        "example_benign_noise": [
            {
                "sample_id": record.sample_id,
                "top_categories": record.top_categories,
                "top1_score": record.top1_score,
                "retrieved_topk_payloads": json.loads(record.retrieved_topk_payloads or "[]"),
                "retrieved_topk_source_files": json.loads(record.retrieved_topk_source_files or "[]"),
            }
            for record in benign_records
            if not record.error and record.returned_any_context
        ][:20],
    }
    return summary


def write_csv(records: list[RetrievalRecord], output_path: Path) -> None:
    if not records:
        output_path.write_text("", encoding="utf-8")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def print_summary(summary: dict[str, Any]) -> None:
    malicious = summary["malicious_retrieval"]
    benign = summary["benign_noise"]
    counts = summary["counts"]

    print("\nRAG retrieval benchmark summary")
    print(f"  Total measured samples : {counts['total_samples']}")
    print(f"  Malicious queries      : {counts['malicious_samples']}")
    print(f"  Benign queries         : {counts['benign_samples']}")
    print(f"  Errors                 : {counts['errors']}")
    print(f"  Gatekeeper used        : {summary['config']['use_gatekeeper']}")
    print(f"  Debug enabled          : {summary['config']['debug_enabled']}")
    print(f"  RAG enabled            : {summary['config']['rag_enabled']}")
    print(f"  Hit@{summary['config']['top_k']}           : {malicious['hit_at_k']:.4f}")
    print(f"  Precision@{summary['config']['top_k']}     : {malicious['precision_at_k']:.4f}")
    print(f"  MRR                    : {malicious['mrr']:.4f}")
    print(f"  nDCG@{summary['config']['top_k']}          : {malicious['ndcg_at_k']:.4f}")
    print(f"  Benign context rate    : {benign['context_return_rate']:.4f}")
    print(f"  Benign mean top1 score : {benign['mean_top1_score']:.4f}")
    if summary["config"]["debug_enabled"]:
        print(f"  Debug CSV              : {summary['config']['debug_csv_path']}")


def main() -> int:
    args = parse_args()
    samples, dataset_info = load_retrieval_samples(args)
    if not samples:
        print("No retrieval samples loaded.", file=sys.stderr)
        return 1

    records = [evaluate_sample(sample, args.top_k, args.min_score) for sample in samples]
    summary = summarize_records(args, dataset_info, records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_output_dir(args.output_dir)
    csv_path = output_dir / f"rag_details_{timestamp}.csv"
    json_path = output_dir / f"rag_summary_{timestamp}.json"

    write_csv(records, csv_path)
    write_json(summary, json_path)
    print_summary(summary)
    print(f"\nDetailed results: {csv_path}")
    print(f"Summary report  : {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
