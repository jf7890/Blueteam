#!/usr/bin/env python3
"""Convert the flat ECML/PKDD 2007 dump into a benchmark-ready CSV.

The input format expected by this script looks like:

    Start - Id: 123
    class: SqlInjection
    GET /path?q=1 HTTP/1.1
    Host: example.com
    ...

    body=value

    End - Id: 123

The output CSV is suitable for ``scripts/run_benchmark.py`` and can also
be reused for request-level RAG evaluation.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import normalize_attack_type  # noqa: E402


@dataclass(slots=True)
class ParsedBlock:
    request_id: str
    original_class: str
    request_lines: list[str]
    source_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a flat ECML/PKDD 2007 dump into benchmark CSV.",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="One or more flat dump files, for example xml_test.txt.",
    )
    parser.add_argument(
        "--output",
        default="data/pkdd_request_level_eval.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--dataset-name",
        default="ECML/PKDD2007",
        help="Value written into the source_dataset column.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to export.",
    )
    return parser.parse_args()


def _iter_blocks(path: Path) -> Iterator[ParsedBlock]:
    current_id: str | None = None
    current_class: str | None = None
    current_lines: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")

            if line.startswith("Start - Id:"):
                if current_id is not None:
                    raise ValueError(f"Encountered nested block in {path}: {line}")
                current_id = line.split(":", 1)[1].strip()
                current_class = None
                current_lines = []
                continue

            if current_id is None:
                continue

            if line.startswith("class:"):
                current_class = line.split(":", 1)[1].strip()
                continue

            if line.startswith("End - Id:"):
                end_id = line.split(":", 1)[1].strip()
                if current_class is None:
                    raise ValueError(f"Missing class for block {current_id} in {path}")
                if end_id != current_id:
                    raise ValueError(
                        f"Mismatched end id in {path}: start={current_id} end={end_id}"
                    )
                yield ParsedBlock(
                    request_id=current_id,
                    original_class=current_class,
                    request_lines=current_lines,
                    source_file=path.name,
                )
                current_id = None
                current_class = None
                current_lines = []
                continue

            current_lines.append(line)

    if current_id is not None:
        raise ValueError(f"Unterminated block at end of file {path}: {current_id}")


def _build_raw_http(lines: list[str]) -> str:
    trimmed = list(lines)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    if not trimmed:
        return ""

    try:
        body_split = trimmed.index("")
    except ValueError:
        return "\r\n".join(trimmed)

    head = trimmed[:body_split]
    body = trimmed[body_split + 1 :]

    while body and not body[-1].strip():
        body.pop()

    if len(body) == 1 and body[0].strip().lower() == "null":
        body = []

    if not head:
        return ""
    if body:
        return "\r\n".join(head) + "\r\n\r\n" + "\r\n".join(body)
    return "\r\n".join(head) + "\r\n\r\n"


def _label_for_class(original_class: str) -> tuple[str, str]:
    if original_class.strip().lower() == "valid":
        return "benign", ""

    normalized = normalize_attack_type(original_class)
    if not normalized:
        raise ValueError(f"Could not normalize attack class: {original_class!r}")
    return "malicious", normalized


def _request_method(raw_http: str) -> str:
    first_line = raw_http.splitlines()[0] if raw_http else ""
    if not first_line:
        return ""
    return first_line.split(" ", 1)[0].strip().upper()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "request_id",
        "raw_http",
        "label",
        "attack_type",
        "source_dataset",
        "source_file",
        "source_split",
        "original_class",
        "request_method",
    ]

    class_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    attack_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    written = 0
    skipped = 0

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for input_name in args.input_files:
            input_path = Path(input_name)
            split_hint = input_path.stem

            for block in _iter_blocks(input_path):
                if args.limit is not None and written >= args.limit:
                    break

                raw_http = _build_raw_http(block.request_lines)
                if not raw_http:
                    skipped += 1
                    continue

                label, attack_type = _label_for_class(block.original_class)
                method = _request_method(raw_http)
                sample_id = f"{split_hint}-{block.request_id}"

                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "request_id": block.request_id,
                        "raw_http": raw_http,
                        "label": label,
                        "attack_type": attack_type,
                        "source_dataset": args.dataset_name,
                        "source_file": block.source_file,
                        "source_split": split_hint,
                        "original_class": block.original_class,
                        "request_method": method,
                    }
                )

                written += 1
                class_counts[block.original_class] += 1
                label_counts[label] += 1
                if attack_type:
                    attack_counts[attack_type] += 1
                if method:
                    method_counts[method] += 1

            if args.limit is not None and written >= args.limit:
                break

    print(f"Saved CSV to: {output_path}")
    print(f"Rows written: {written}")
    print(f"Rows skipped: {skipped}")
    print(f"Label distribution: {dict(sorted(label_counts.items()))}")
    print(f"Attack distribution: {dict(sorted(attack_counts.items()))}")
    print(f"Original class distribution: {dict(sorted(class_counts.items()))}")
    print(f"Method distribution: {dict(sorted(method_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
