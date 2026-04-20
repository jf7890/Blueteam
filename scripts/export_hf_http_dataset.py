#!/usr/bin/env python3
"""Download a Hugging Face HTTP dataset and export benchmark-ready CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path
import sys
from typing import Any

import duckdb
from huggingface_hub import HfApi, hf_hub_download

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import (  # noqa: E402
    coerce_request_text,
    normalize_attack_type,
    normalize_binary_label,
)

DEFAULT_LABEL_MAP = [
    "normal=benign",
    "anomalous=malicious",
    "benign=benign",
    "malicious=malicious",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a parquet-backed Hugging Face HTTP dataset to benchmark CSV.",
    )
    parser.add_argument(
        "--repo-id",
        default="vyykaaa/dataset_v5",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to export.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where benchmark CSV files will be written.",
    )
    parser.add_argument(
        "--request-column",
        default="raw-request",
        help="Source column containing the raw HTTP request text.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Source column containing the raw binary label.",
    )
    parser.add_argument(
        "--attack-type-column",
        default="attack_type",
        help="Source column containing the raw attack family label.",
    )
    parser.add_argument(
        "--label-map",
        nargs="+",
        default=DEFAULT_LABEL_MAP,
        help=(
            "Optional explicit label remapping in SRC=DST form. "
            "Defaults to normal=benign anomalous=malicious."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional filename prefix. Defaults to the dataset slug.",
    )
    return parser.parse_args()


def _parse_label_map(entries: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --label-map entry {entry!r}; expected SRC=DST."
            )
        source, target = entry.split("=", 1)
        source_key = source.strip().lower()
        target_value = target.strip().lower()
        if not source_key or not target_value:
            raise ValueError(
                f"Invalid --label-map entry {entry!r}; empty key/value is not allowed."
            )
        mapping[source_key] = target_value
    return mapping


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def _dataset_slug(repo_id: str) -> str:
    return repo_id.replace("/", "_").replace("-", "_")


def _list_split_files(repo_id: str, split: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    matches = sorted(
        path
        for path in files
        if fnmatch(path, f"data/{split}-*.parquet")
    )
    if not matches:
        raise FileNotFoundError(
            f"No parquet files found for split {split!r} in dataset {repo_id!r}."
        )
    return matches


def _download_split_files(repo_id: str, files: list[str]) -> list[str]:
    return [
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
        for filename in files
    ]


def _load_split_rows(
    repo_id: str,
    split: str,
    request_column: str,
    label_column: str,
    attack_type_column: str,
) -> list[tuple[Any, Any, Any]]:
    repo_files = _list_split_files(repo_id, split)
    local_files = _download_split_files(repo_id, repo_files)
    file_literals = ", ".join(_quote_literal(path) for path in local_files)

    query = f"""
        SELECT
            {_quote_ident(request_column)} AS request_text,
            {_quote_ident(label_column)} AS raw_label,
            {_quote_ident(attack_type_column)} AS raw_attack_type
        FROM read_parquet([{file_literals}])
    """
    con = duckdb.connect()
    try:
        return list(con.execute(query).fetchall())
    finally:
        con.close()


def _normalize_label(raw_label: Any, label_map: dict[str, str]) -> str:
    key = str(raw_label).strip().lower()
    if key in label_map:
        return label_map[key]
    return normalize_binary_label(raw_label)


def export_split(
    repo_id: str,
    split: str,
    output_dir: Path,
    output_prefix: str,
    request_column: str,
    label_column: str,
    attack_type_column: str,
    label_map: dict[str, str],
) -> Path:
    rows = _load_split_rows(
        repo_id=repo_id,
        split=split,
        request_column=request_column,
        label_column=label_column,
        attack_type_column=attack_type_column,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_prefix}_{split}_benchmark.csv"
    fieldnames = [
        "sample_id",
        "raw_http",
        "label",
        "attack_type",
        "source_dataset",
        "source_split",
        "original_label",
        "original_attack_type",
    ]

    label_counts: Counter[str] = Counter()
    attack_counts: Counter[str] = Counter()
    empty_request_rows = 0

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for index, (request_text, raw_label, raw_attack_type) in enumerate(rows):
            raw_http = coerce_request_text(request_text)
            if not raw_http:
                empty_request_rows += 1
                continue

            label = _normalize_label(raw_label, label_map)
            attack_type = normalize_attack_type(raw_attack_type) or ""
            sample_id = f"{output_prefix}-{split}-{index:06d}"

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "raw_http": raw_http,
                    "label": label,
                    "attack_type": attack_type,
                    "source_dataset": repo_id,
                    "source_split": split,
                    "original_label": "" if raw_label is None else str(raw_label),
                    "original_attack_type": (
                        "" if raw_attack_type is None else str(raw_attack_type)
                    ),
                }
            )

            label_counts[label] += 1
            if attack_type:
                attack_counts[attack_type] += 1

    print(f"Saved {split} CSV to: {output_path}")
    print(f"  rows written      : {sum(label_counts.values())}")
    print(f"  empty requests    : {empty_request_rows}")
    print(f"  label distribution: {dict(sorted(label_counts.items()))}")
    print(f"  attack top        : {dict(attack_counts.most_common(15))}")
    return output_path


def main() -> int:
    args = parse_args()
    label_map = _parse_label_map(args.label_map)
    output_dir = Path(args.output_dir)
    output_prefix = args.output_prefix or _dataset_slug(args.repo_id)

    for split in args.splits:
        export_split(
            repo_id=args.repo_id,
            split=split,
            output_dir=output_dir,
            output_prefix=output_prefix,
            request_column=args.request_column,
            label_column=args.label_column,
            attack_type_column=args.attack_type_column,
            label_map=label_map,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
