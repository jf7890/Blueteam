#!/usr/bin/env python3
"""Async benchmark runner for BlueAgent HTTP classification."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    add_dataset_arguments,
    coerce_request_text,
    ensure_output_dir,
    load_rows_from_dataset,
    normalize_attack_type,
    normalize_binary_label,
    pick_sample_id,
)
from config.settings import settings  # noqa: E402
from nodes.preprocess import preprocess_node  # noqa: E402
from nodes.rag_node import collect_ranked_payload_hits  # noqa: E402

DEFAULT_BASE_URL = "http://localhost:5000"


@dataclass(slots=True)
class PreparedSample:
    dataset_index: int
    sample_id: str
    true_label: str
    true_attack_type: str | None
    raw_http: str


@dataclass(slots=True)
class BenchmarkRecord:
    dataset_index: int
    sample_id: str
    true_label: str
    true_attack_type: str | None
    predicted_label: str
    predicted_attack_type: str | None
    request_route: str
    initial_status: int
    final_status: int
    source: str
    attack_type: str
    request_id: str
    confidence: float | None
    predicted_reasoning: str
    latency_ms: float
    submit_latency_ms: float
    poll_count: int
    completed: bool
    normalized_payloads: str
    retrieval_trace_origin: str
    retrieved_topk_count: int
    retrieved_topk_categories: str
    retrieved_topk_scores: str
    retrieved_topk_payloads: str
    retrieved_topk_source_files: str
    retrieved_topk_line_nos: str
    true_attack_type_in_topk: bool
    predicted_attack_type_in_topk: bool
    label_correct: bool
    attack_type_correct: bool
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark BlueAgent HTTP classification.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    add_dataset_arguments(parser)
    parser.add_argument("--benign-samples", type=int, default=500)
    parser.add_argument("--malicious-samples", type=int, default=500)
    parser.add_argument("--warmup-samples", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--poll-interval", type=float, default=1.5)
    parser.add_argument("--poll-timeout", type=float, default=60.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument(
        "--retrieval-trace-top-k",
        type=int,
        default=5,
        help="Top-k retrieval hits to log per sample for groundedness analysis.",
    )
    parser.add_argument(
        "--skip-retrieval-trace",
        action="store_true",
        help="Disable benchmark-sidecar retrieval tracing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalize_api_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/api"):
        return base
    return f"{base}/api"


def load_sample_groups(args: argparse.Namespace) -> tuple[list[PreparedSample], list[PreparedSample], dict[str, Any]]:
    print(
        f"Loading dataset '{args.dataset}' split '{args.split}'...",
        flush=True,
    )

    loaded = load_rows_from_dataset(
        dataset_name=args.dataset,
        split=args.split,
        data_file=args.data_file,
        request_column=args.request_column,
        label_column=args.label_column,
        attack_type_column=args.attack_type_column,
        require_request=True,
        require_label=True,
    )
    rows = list(loaded.rows)

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    label_feature = loaded.features.get(loaded.label_column) if loaded.label_column else None
    benign_pool: list[PreparedSample] = []
    malicious_pool: list[PreparedSample] = []

    for index, row in enumerate(rows):
        true_label = normalize_binary_label(row[loaded.label_column], label_feature)
        raw_http = coerce_request_text(row.get(loaded.request_column))
        if not raw_http:
            continue

        sample = PreparedSample(
            dataset_index=index,
            sample_id=pick_sample_id(row, loaded.sample_id_column, index),
            true_label=true_label,
            true_attack_type=normalize_attack_type(
                row.get(loaded.attack_type_column) if loaded.attack_type_column else None
            ),
            raw_http=raw_http,
        )

        if true_label == "benign":
            benign_pool.append(sample)
        else:
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

    benchmark_samples = benign_pool[:requested_benign] + malicious_pool[:requested_malicious]
    remaining_samples = benign_pool[requested_benign:] + malicious_pool[requested_malicious:]

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(benchmark_samples)
        rng.shuffle(remaining_samples)

    warmup_count = min(max(args.warmup_samples, 0), len(remaining_samples))
    warmup_samples = remaining_samples[:warmup_count]

    dataset_info = {
        "dataset": args.dataset,
        "split": args.split,
        "data_file": args.data_file,
        "request_column": loaded.request_column,
        "label_column": loaded.label_column,
        "attack_type_column": loaded.attack_type_column,
        "sample_id_column": loaded.sample_id_column,
        "column_names": loaded.column_names,
    }
    return warmup_samples, benchmark_samples, dataset_info


async def safe_json_or_text(response: aiohttp.ClientResponse) -> dict[str, Any]:
    try:
        data = await response.json(content_type=None)
        if isinstance(data, dict):
            return data
        return {"raw": data}
    except (aiohttp.ContentTypeError, json.JSONDecodeError):
        return {"raw_text": await response.text()}


async def poll_result(
    session: aiohttp.ClientSession,
    api_base: str,
    request_id: str,
    poll_interval: float,
    poll_timeout: float,
) -> tuple[dict[str, Any] | None, int, int]:
    deadline = time.perf_counter() + poll_timeout
    poll_count = 0

    while time.perf_counter() < deadline:
        poll_count += 1
        try:
            async with session.get(
                f"{api_base}/result/{request_id}",
                timeout=aiohttp.ClientTimeout(total=max(5.0, poll_interval + 5.0)),
            ) as response:
                status = response.status
                if status == 200:
                    data = await safe_json_or_text(response)
                    if data.get("success"):
                        return data.get("result", {}), poll_count, status
                elif status != 404:
                    text = await response.text()
                    return {"error": f"poll returned {status}: {text[:200]}"}, poll_count, status
        except aiohttp.ClientError as exc:
            if time.perf_counter() >= deadline:
                return {"error": f"poll failed: {exc}"}, poll_count, 0

        await asyncio.sleep(poll_interval)

    return None, poll_count, 0


def _build_record(
    sample: PreparedSample,
    *,
    predicted_label: str,
    predicted_attack_type: str | None,
    request_route: str,
    initial_status: int,
    final_status: int,
    source: str,
    attack_type: str,
    request_id: str,
    confidence: float | None,
    predicted_reasoning: str,
    latency_ms: float,
    submit_latency_ms: float,
    poll_count: int,
    completed: bool,
    error: str,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        dataset_index=sample.dataset_index,
        sample_id=sample.sample_id,
        true_label=sample.true_label,
        true_attack_type=sample.true_attack_type,
        predicted_label=predicted_label,
        predicted_attack_type=predicted_attack_type,
        request_route=request_route,
        initial_status=initial_status,
        final_status=final_status,
        source=source,
        attack_type=attack_type,
        request_id=request_id,
        confidence=confidence,
        predicted_reasoning=predicted_reasoning,
        latency_ms=latency_ms,
        submit_latency_ms=submit_latency_ms,
        poll_count=poll_count,
        completed=completed,
        normalized_payloads="",
        retrieval_trace_origin="",
        retrieved_topk_count=0,
        retrieved_topk_categories="",
        retrieved_topk_scores="",
        retrieved_topk_payloads="",
        retrieved_topk_source_files="",
        retrieved_topk_line_nos="",
        true_attack_type_in_topk=False,
        predicted_attack_type_in_topk=False,
        label_correct=predicted_label == sample.true_label,
        attack_type_correct=(
            predicted_label == "malicious"
            and bool(sample.true_attack_type)
            and predicted_attack_type == sample.true_attack_type
        ),
        error=error,
    )


async def analyze_sample(
    session: aiohttp.ClientSession,
    api_base: str,
    sample: PreparedSample,
    semaphore: asyncio.Semaphore,
    poll_interval: float,
    poll_timeout: float,
) -> BenchmarkRecord:
    started_at = time.perf_counter()

    try:
        async with semaphore:
            async with session.post(
                f"{api_base}/analyze",
                json={"raw_http": sample.raw_http},
            ) as response:
                initial_status = response.status
                payload = await safe_json_or_text(response)
                submit_latency_ms = (time.perf_counter() - started_at) * 1000.0
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        total_latency_ms = (time.perf_counter() - started_at) * 1000.0
        return _build_record(
            sample,
            predicted_label="error",
            predicted_attack_type=None,
            request_route="error",
            initial_status=0,
            final_status=0,
            source="",
            attack_type="",
            request_id="",
            confidence=None,
            predicted_reasoning="",
            latency_ms=total_latency_ms,
            submit_latency_ms=total_latency_ms,
            poll_count=0,
            completed=False,
            error=str(exc),
        )

    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    predicted_attack_type = normalize_attack_type(result.get("attack_type"))

    if initial_status == 200:
        total_latency_ms = (time.perf_counter() - started_at) * 1000.0
        return _build_record(
            sample,
            predicted_label=result.get("verdict", "unknown"),
            predicted_attack_type=predicted_attack_type,
            request_route="immediate",
            initial_status=initial_status,
            final_status=initial_status,
            source=result.get("source", ""),
            attack_type=result.get("attack_type", "") or "",
            request_id=result.get("request_id", "") or "",
            confidence=result.get("confidence"),
            predicted_reasoning=str(result.get("reasoning", "") or ""),
            latency_ms=total_latency_ms,
            submit_latency_ms=submit_latency_ms,
            poll_count=0,
            completed=True,
            error="",
        )

    if initial_status == 202:
        request_id = str(result.get("request_id", "") or "")
        if not request_id:
            total_latency_ms = (time.perf_counter() - started_at) * 1000.0
            return _build_record(
                sample,
                predicted_label="error",
                predicted_attack_type=None,
                request_route="queued",
                initial_status=initial_status,
                final_status=202,
                source=result.get("source", ""),
                attack_type="",
                request_id="",
                confidence=None,
                predicted_reasoning=str(result.get("reasoning", "") or ""),
                latency_ms=total_latency_ms,
                submit_latency_ms=submit_latency_ms,
                poll_count=0,
                completed=False,
                error="202 response missing request_id",
            )

        polled_result, poll_count, final_status = await poll_result(
            session=session,
            api_base=api_base,
            request_id=request_id,
            poll_interval=poll_interval,
            poll_timeout=poll_timeout,
        )
        total_latency_ms = (time.perf_counter() - started_at) * 1000.0

        if polled_result is None:
            return _build_record(
                sample,
                predicted_label="timeout",
                predicted_attack_type=None,
                request_route="queued_timeout",
                initial_status=initial_status,
                final_status=final_status,
                source="timeout",
                attack_type="",
                request_id=request_id,
                confidence=None,
                predicted_reasoning="",
                latency_ms=total_latency_ms,
                submit_latency_ms=submit_latency_ms,
                poll_count=poll_count,
                completed=False,
                error=f"timed out after {poll_timeout:.1f}s",
            )

        if "error" in polled_result:
            return _build_record(
                sample,
                predicted_label="error",
                predicted_attack_type=None,
                request_route="queued_error",
                initial_status=initial_status,
                final_status=final_status,
                source="poll",
                attack_type="",
                request_id=request_id,
                confidence=None,
                predicted_reasoning="",
                latency_ms=total_latency_ms,
                submit_latency_ms=submit_latency_ms,
                poll_count=poll_count,
                completed=False,
                error=str(polled_result["error"]),
            )

        return _build_record(
            sample,
            predicted_label=polled_result.get("verdict", "unknown"),
            predicted_attack_type=normalize_attack_type(polled_result.get("attack_type")),
            request_route="queued",
            initial_status=initial_status,
            final_status=200,
            source=polled_result.get("source", ""),
            attack_type=polled_result.get("attack_type", "") or "",
            request_id=request_id,
            confidence=polled_result.get("confidence"),
            predicted_reasoning=str(polled_result.get("reasoning", "") or ""),
            latency_ms=total_latency_ms,
            submit_latency_ms=submit_latency_ms,
            poll_count=poll_count,
            completed=True,
            error="",
        )

    total_latency_ms = (time.perf_counter() - started_at) * 1000.0
    return _build_record(
        sample,
        predicted_label="error",
        predicted_attack_type=predicted_attack_type,
        request_route="error",
        initial_status=initial_status,
        final_status=initial_status,
        source=result.get("source", ""),
        attack_type=result.get("attack_type", "") or "",
        request_id=result.get("request_id", "") or "",
        confidence=result.get("confidence"),
        predicted_reasoning=str(result.get("reasoning", "") or ""),
        latency_ms=total_latency_ms,
        submit_latency_ms=submit_latency_ms,
        poll_count=0,
        completed=False,
        error=f"unexpected status {initial_status}",
    )


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _build_retrieval_trace(raw_http: str, top_k: int) -> dict[str, Any]:
    trace = {
        "normalized_payloads": [],
        "retrieval_trace_origin": (
            "benchmark_sidecar_disabled" if not settings.rag_enabled else "benchmark_sidecar"
        ),
        "retrieved_topk_count": 0,
        "retrieved_topk_categories": [],
        "retrieved_topk_scores": [],
        "retrieved_topk_payloads": [],
        "retrieved_topk_source_files": [],
        "retrieved_topk_line_nos": [],
        "error": "",
    }
    try:
        state = preprocess_node({"raw_http_text": raw_http})
        payloads = [payload for payload in state.get("normalized_payloads", []) if payload]
        trace["normalized_payloads"] = payloads
        hits = collect_ranked_payload_hits(payloads, limit=max(top_k, 0)) if top_k > 0 else []
        trace["retrieved_topk_count"] = len(hits)
        trace["retrieved_topk_categories"] = [
            normalize_attack_type((hit.get("record") or {}).get("category")) or "Unknown"
            for hit in hits
        ]
        trace["retrieved_topk_scores"] = [
            float(hit.get("score", 0.0))
            for hit in hits
        ]
        trace["retrieved_topk_payloads"] = [
            str(hit.get("payload", "") or "")
            for hit in hits
        ]
        trace["retrieved_topk_source_files"] = [
            str(((hit.get("record") or {}).get("source_file")) or "")
            for hit in hits
        ]
        trace["retrieved_topk_line_nos"] = [
            ((hit.get("record") or {}).get("line_no"))
            for hit in hits
        ]
    except Exception as exc:
        trace["error"] = str(exc)
    return trace


async def enrich_records_with_retrieval_trace(
    records: list[BenchmarkRecord],
    samples: list[PreparedSample],
    top_k: int,
) -> None:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    for record in records:
        sample = sample_by_id.get(record.sample_id)
        if sample is None:
            continue
        trace = await asyncio.to_thread(_build_retrieval_trace, sample.raw_http, top_k)
        categories = trace["retrieved_topk_categories"]
        record.normalized_payloads = _json_compact(trace["normalized_payloads"])
        record.retrieval_trace_origin = str(trace["retrieval_trace_origin"])
        record.retrieved_topk_count = int(trace["retrieved_topk_count"])
        record.retrieved_topk_categories = _json_compact(categories)
        record.retrieved_topk_scores = _json_compact(trace["retrieved_topk_scores"])
        record.retrieved_topk_payloads = _json_compact(trace["retrieved_topk_payloads"])
        record.retrieved_topk_source_files = _json_compact(trace["retrieved_topk_source_files"])
        record.retrieved_topk_line_nos = _json_compact(trace["retrieved_topk_line_nos"])
        record.true_attack_type_in_topk = bool(
            record.true_attack_type and record.true_attack_type in categories
        )
        record.predicted_attack_type_in_topk = bool(
            record.predicted_attack_type and record.predicted_attack_type in categories
        )
        if trace["error"]:
            record.error = f"{record.error} | retrieval_trace={trace['error']}".strip(" |")


async def run_phase(
    phase_name: str,
    samples: list[PreparedSample],
    api_base: str,
    concurrency: int,
    request_timeout: float,
    poll_interval: float,
    poll_timeout: float,
) -> tuple[list[BenchmarkRecord], float]:
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, concurrency))
    semaphore = asyncio.Semaphore(concurrency)

    started_at = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            asyncio.create_task(
                analyze_sample(
                    session=session,
                    api_base=api_base,
                    sample=sample,
                    semaphore=semaphore,
                    poll_interval=poll_interval,
                    poll_timeout=poll_timeout,
                )
            )
            for sample in samples
        ]
        results = await asyncio.gather(*tasks)
    duration_seconds = time.perf_counter() - started_at
    print(
        f"{phase_name}: completed {len(results)} request(s) in {duration_seconds:.2f}s",
        flush=True,
    )
    return results, duration_seconds


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * pct
    lower = int(idx)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def build_latency_summary(records: list[BenchmarkRecord]) -> dict[str, float]:
    values = sorted(r.latency_ms for r in records if r.latency_ms >= 0)
    if not values:
        return {}
    return {
        "count": float(len(values)),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": values[0],
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": values[-1],
    }


def compute_confusion(records: list[BenchmarkRecord]) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for record in records:
        if record.predicted_label not in {"benign", "malicious"}:
            continue
        if record.true_label == "malicious" and record.predicted_label == "malicious":
            tp += 1
        elif record.true_label == "benign" and record.predicted_label == "benign":
            tn += 1
        elif record.true_label == "benign" and record.predicted_label == "malicious":
            fp += 1
        elif record.true_label == "malicious" and record.predicted_label == "benign":
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_metrics(confusion: dict[str, int]) -> dict[str, float]:
    tp = confusion["tp"]
    tn = confusion["tn"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "f1_score": f1,
        "resolved_samples": float(total),
    }


def build_effectiveness_summary(records: list[BenchmarkRecord]) -> dict[str, Any]:
    confusion = compute_confusion(records)
    return {
        "confusion_matrix": confusion,
        **compute_metrics(confusion),
    }


def _normalize_predicted_attack_label(record: BenchmarkRecord) -> str:
    if record.predicted_label != "malicious":
        return "__missed__"
    if record.predicted_attack_type:
        return record.predicted_attack_type
    return "__unlabeled__"


def compute_attack_type_summary(records: list[BenchmarkRecord]) -> dict[str, Any] | None:
    eligible = [
        record
        for record in records
        if record.true_label == "malicious" and record.true_attack_type
    ]
    if not eligible:
        return None

    classes = sorted({record.true_attack_type for record in eligible if record.true_attack_type})
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    exact_matches = 0
    detected_malicious = 0
    labeled_predictions = 0

    for record in eligible:
        predicted_class = _normalize_predicted_attack_label(record)
        confusion[record.true_attack_type][predicted_class] += 1
        if record.predicted_label == "malicious":
            detected_malicious += 1
        if record.predicted_attack_type:
            labeled_predictions += 1
        if predicted_class == record.true_attack_type:
            exact_matches += 1

    per_class: dict[str, dict[str, float]] = {}
    macro_f1_values: list[float] = []
    for attack_type in classes:
        tp = confusion[attack_type][attack_type]
        fp = sum(
            confusion[other_true][attack_type]
            for other_true in classes
            if other_true != attack_type
        )
        fn = sum(
            count
            for predicted_label, count in confusion[attack_type].items()
            if predicted_label != attack_type
        )
        support = sum(confusion[attack_type].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        macro_f1_values.append(f1)
        per_class[attack_type] = {
            "support": float(support),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    predicted_distribution = Counter(_normalize_predicted_attack_label(record) for record in eligible)
    true_distribution = Counter(record.true_attack_type for record in eligible if record.true_attack_type)

    return {
        "eligible_samples": len(eligible),
        "detected_as_malicious": detected_malicious,
        "prediction_coverage": labeled_predictions / len(eligible),
        "joint_exact_match_accuracy": exact_matches / len(eligible),
        "conditional_exact_match_accuracy": (
            exact_matches / detected_malicious if detected_malicious else 0.0
        ),
        "macro_f1": statistics.fmean(macro_f1_values) if macro_f1_values else 0.0,
        "true_distribution": dict(true_distribution),
        "predicted_distribution": dict(predicted_distribution),
        "confusion_matrix": {
            true_label: dict(sorted(counter.items()))
            for true_label, counter in sorted(confusion.items())
        },
        "per_class": per_class,
    }


def compute_llm_groundedness_summary(records: list[BenchmarkRecord]) -> dict[str, Any] | None:
    llm_records = [
        record
        for record in records
        if record.completed and record.source in {"llm", "llm_batch"}
    ]
    if not llm_records:
        return None

    malicious_records = [
        record
        for record in llm_records
        if record.true_label == "malicious" and record.true_attack_type
    ]
    if not malicious_records:
        return {
            "eligible_llm_samples": len(llm_records),
            "eligible_malicious_samples": 0,
        }

    return {
        "eligible_llm_samples": len(llm_records),
        "eligible_malicious_samples": len(malicious_records),
        "label_accuracy": statistics.fmean(
            1.0 if record.label_correct else 0.0 for record in malicious_records
        ),
        "attack_type_accuracy": statistics.fmean(
            1.0 if record.attack_type_correct else 0.0 for record in malicious_records
        ),
        "true_attack_type_in_topk_rate": statistics.fmean(
            1.0 if record.true_attack_type_in_topk else 0.0 for record in malicious_records
        ),
        "predicted_attack_type_in_topk_rate": statistics.fmean(
            1.0 if record.predicted_attack_type_in_topk else 0.0 for record in malicious_records
        ),
        "grounded_correct_rate": statistics.fmean(
            1.0
            if record.label_correct and record.attack_type_correct and record.predicted_attack_type_in_topk
            else 0.0
            for record in malicious_records
        ),
        "examples_not_supported": [
            {
                "sample_id": record.sample_id,
                "true_attack_type": record.true_attack_type,
                "predicted_attack_type": record.predicted_attack_type,
                "retrieved_topk_categories": json.loads(record.retrieved_topk_categories or "[]"),
            }
            for record in malicious_records
            if not record.predicted_attack_type_in_topk
        ][:20],
    }


def build_partition_summary(
    name: str,
    records: list[BenchmarkRecord],
) -> dict[str, Any]:
    completed = [r for r in records if r.completed]
    unresolved = [r for r in records if not r.completed]
    return {
        "name": name,
        "counts": {
            "total_samples": len(records),
            "completed": len(completed),
            "unresolved": len(unresolved),
        },
        "effectiveness": build_effectiveness_summary(records),
        "attack_type": compute_attack_type_summary(records),
        "latency": build_latency_summary(completed),
    }


def summarize_results(
    args: argparse.Namespace,
    dataset_info: dict[str, Any],
    records: list[BenchmarkRecord],
    duration_seconds: float,
    warmup_samples: int,
) -> dict[str, Any]:
    completed = [r for r in records if r.completed]
    immediate = [r for r in records if r.request_route == "immediate"]
    queued = [r for r in records if r.request_route == "queued"]
    unresolved = [r for r in records if not r.completed]

    source_counts = Counter(r.source or "unknown" for r in records)
    route_counts = Counter(r.request_route for r in records)
    initial_status_counts = Counter(str(r.initial_status) for r in records)
    prediction_counts = Counter(r.predicted_label for r in records)
    label_counts = Counter(r.true_label for r in records)
    true_attack_type_counts = Counter(
        r.true_attack_type for r in records if r.true_attack_type
    )
    predicted_attack_type_counts = Counter(
        r.predicted_attack_type for r in records if r.predicted_attack_type
    )

    latency_by_source = {
        source: build_latency_summary([r for r in completed if (r.source or "unknown") == source])
        for source in sorted(source_counts)
    }
    effectiveness_by_source = {
        source: build_partition_summary(
            source,
            [r for r in records if (r.source or "unknown") == source],
        )
        for source in sorted(source_counts)
    }
    effectiveness_by_route = {
        route: build_partition_summary(route, [r for r in records if r.request_route == route])
        for route in sorted(route_counts)
    }

    summary = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "dataset": dataset_info["dataset"],
        "split": dataset_info["split"],
        "data_file": dataset_info["data_file"],
        "dataset_columns": {
            "available": dataset_info["column_names"],
            "request_column": dataset_info["request_column"],
            "label_column": dataset_info["label_column"],
            "attack_type_column": dataset_info["attack_type_column"],
            "sample_id_column": dataset_info["sample_id_column"],
        },
        "base_api_url": normalize_api_base(args.base_url),
        "config": {
            "benign_samples": args.benign_samples,
            "malicious_samples": args.malicious_samples,
            "warmup_samples": warmup_samples,
            "concurrency": args.concurrency,
            "poll_interval_seconds": args.poll_interval,
            "poll_timeout_seconds": args.poll_timeout,
            "request_timeout_seconds": args.request_timeout,
            "retrieval_trace_top_k": args.retrieval_trace_top_k,
            "retrieval_trace_enabled": not args.skip_retrieval_trace,
            "retrieval_trace_mode": "benchmark_sidecar",
            "rag_enabled": settings.rag_enabled,
            "qdrant_collection": settings.qdrant_collection,
            "shuffle": args.shuffle,
            "seed": args.seed,
        },
        "counts": {
            "total_samples": len(records),
            "completed": len(completed),
            "unresolved": len(unresolved),
            "immediate_path": len(immediate),
            "queued_path": len(queued),
            "warmup_samples": warmup_samples,
            "true_label_distribution": dict(label_counts),
            "prediction_distribution": dict(prediction_counts),
            "initial_status_distribution": dict(initial_status_counts),
            "route_distribution": dict(route_counts),
            "source_distribution": dict(source_counts),
            "true_attack_type_distribution": dict(true_attack_type_counts),
            "predicted_attack_type_distribution": dict(predicted_attack_type_counts),
        },
        "effectiveness": build_effectiveness_summary(records),
        "attack_type": compute_attack_type_summary(records),
        "llm_path_groundedness": compute_llm_groundedness_summary(records),
        "effectiveness_by_source": effectiveness_by_source,
        "effectiveness_by_route": effectiveness_by_route,
        "performance": {
            "wall_clock_seconds": duration_seconds,
            "throughput_rps": (len(records) / duration_seconds) if duration_seconds else 0.0,
            "overall_latency": build_latency_summary(completed),
            "immediate_latency": build_latency_summary(immediate),
            "queued_latency": build_latency_summary(queued),
            "latency_by_source": latency_by_source,
        },
        "failures": {
            "count": len(unresolved),
            "examples": [
                {
                    "dataset_index": r.dataset_index,
                    "sample_id": r.sample_id,
                    "request_id": r.request_id,
                    "route": r.request_route,
                    "error": r.error,
                }
                for r in unresolved[:20]
            ],
        },
    }
    return summary


def write_csv(records: list[BenchmarkRecord], output_path: Path) -> None:
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
    counts = summary["counts"]
    perf = summary["performance"]
    eff = summary["effectiveness"]

    print("\nClassification benchmark summary")
    print(f"  Total measured samples : {counts['total_samples']}")
    print(f"  Completed             : {counts['completed']}")
    print(f"  Unresolved            : {counts['unresolved']}")
    print(f"  Immediate path        : {counts['immediate_path']}")
    print(f"  Queued path           : {counts['queued_path']}")
    print(f"  RAG enabled           : {summary['config']['rag_enabled']}")
    print(f"  Throughput (RPS)      : {perf['throughput_rps']:.2f}")
    print(f"  Accuracy              : {eff['accuracy']:.4f}")
    print(f"  Precision             : {eff['precision']:.4f}")
    print(f"  Recall                : {eff['recall']:.4f}")
    print(f"  F1-score              : {eff['f1_score']:.4f}")
    print(f"  FPR / FNR             : {eff['false_positive_rate']:.4f} / {eff['false_negative_rate']:.4f}")

    overall = perf.get("overall_latency", {})
    if overall:
        print(f"  Avg latency           : {overall['mean_ms']:.2f} ms")
        print(f"  P95 latency           : {overall['p95_ms']:.2f} ms")

    attack_type = summary.get("attack_type")
    if attack_type:
        print("\nAttack-type summary")
        print(f"  Eligible malicious    : {attack_type['eligible_samples']}")
        print(f"  Joint exact match     : {attack_type['joint_exact_match_accuracy']:.4f}")
        print(f"  Conditional exact     : {attack_type['conditional_exact_match_accuracy']:.4f}")
        print(f"  Macro-F1              : {attack_type['macro_f1']:.4f}")
        print(f"  Coverage              : {attack_type['prediction_coverage']:.4f}")

    groundedness = summary.get("llm_path_groundedness")
    if groundedness and groundedness.get("eligible_malicious_samples"):
        print("\nLLM-path groundedness")
        print(f"  Eligible malicious    : {groundedness['eligible_malicious_samples']}")
        print(f"  Label accuracy        : {groundedness['label_accuracy']:.4f}")
        print(f"  Attack-type accuracy  : {groundedness['attack_type_accuracy']:.4f}")
        print(f"  True type in top-k    : {groundedness['true_attack_type_in_topk_rate']:.4f}")
        print(f"  Pred type in top-k    : {groundedness['predicted_attack_type_in_topk_rate']:.4f}")
        print(f"  Grounded correct      : {groundedness['grounded_correct_rate']:.4f}")

    source_sections = summary.get("effectiveness_by_source", {})
    if source_sections:
        print("\nPer-source effectiveness")
        for source, section in source_sections.items():
            metrics = section["effectiveness"]
            total = section["counts"]["total_samples"]
            print(
                "  "
                f"{source:<18} total={total:<5} acc={metrics['accuracy']:.4f} "
                f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
                f"f1={metrics['f1_score']:.4f}"
            )


async def async_main(args: argparse.Namespace) -> int:
    api_base = normalize_api_base(args.base_url)
    warmup_samples, benchmark_samples, dataset_info = load_sample_groups(args)

    if not benchmark_samples:
        print("No samples loaded.", file=sys.stderr)
        return 1

    if warmup_samples:
        print(f"Running warm-up with {len(warmup_samples)} sample(s)...", flush=True)
        await run_phase(
            phase_name="warmup",
            samples=warmup_samples,
            api_base=api_base,
            concurrency=args.concurrency,
            request_timeout=args.request_timeout,
            poll_interval=args.poll_interval,
            poll_timeout=args.poll_timeout,
        )

    if not benchmark_samples:
        print("No benchmark samples remain after warm-up.", file=sys.stderr)
        return 1

    print(f"Running benchmark with {len(benchmark_samples)} sample(s)...", flush=True)
    records, duration_seconds = await run_phase(
        phase_name="benchmark",
        samples=benchmark_samples,
        api_base=api_base,
        concurrency=args.concurrency,
        request_timeout=args.request_timeout,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
    )

    if not args.skip_retrieval_trace:
        print(
            f"Collecting benchmark-sidecar retrieval trace for {len(records)} sample(s)...",
            flush=True,
        )
        await enrich_records_with_retrieval_trace(
            records=records,
            samples=benchmark_samples,
            top_k=args.retrieval_trace_top_k,
        )

    summary = summarize_results(
        args=args,
        dataset_info=dataset_info,
        records=records,
        duration_seconds=duration_seconds,
        warmup_samples=len(warmup_samples),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_output_dir(args.output_dir)
    csv_path = output_dir / f"classification_details_{timestamp}.csv"
    json_path = output_dir / f"classification_summary_{timestamp}.json"

    write_csv(records, csv_path)
    write_json(summary, json_path)
    print_summary(summary)
    print(f"\nDetailed results: {csv_path}")
    print(f"Summary report  : {json_path}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
