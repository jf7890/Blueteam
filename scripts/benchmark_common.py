#!/usr/bin/env python3
"""Shared helpers for BlueAgent benchmarking scripts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import ClassLabel, load_dataset

DEFAULT_DATASET = "nquangit/CSIC2010_dataset_classification"
DEFAULT_OUTPUT_DIR = "benchmark_results"

REQUEST_COLUMN_CANDIDATES = [
    "requests",
    "raw_http",
    "raw_request",
    "http_request",
]
QUERY_COLUMN_CANDIDATES = [
    "query",
    "payload",
    "text",
    "content",
]
LABEL_COLUMN_CANDIDATES = [
    "label",
    "true_label",
    "verdict",
    "is_malicious",
    "malicious",
]
ATTACK_TYPE_COLUMN_CANDIDATES = [
    "attack_type",
    "true_attack_type",
    "category",
    "attack",
    "type",
]
SAMPLE_ID_COLUMN_CANDIDATES = [
    "sample_id",
    "id",
    "request_id",
    "uuid",
]

_ATTACK_TYPE_ALIASES = {
    "sqli": "SQLi",
    "sqlinjection": "SQLi",
    "sql injection": "SQLi",
    "sql injection attack": "SQLi",
    "s q l i": "SQLi",
    "xss": "XSS",
    "cross site scripting": "XSS",
    "cross site script": "XSS",
    "path traversal": "Path Traversal",
    "pathtransversal": "Path Traversal",
    "directory traversal": "Path Traversal",
    "dir traversal": "Path Traversal",
    "command injection": "Command Injection",
    "command execution": "Command Injection",
    "os commanding": "Command Injection",
    "oscommanding": "Command Injection",
    "os command injection": "Command Injection",
    "cmd injection": "Command Injection",
    "remote code execution": "RCE",
    "rce": "RCE",
    "ssti": "SSTI",
    "server side template injection": "SSTI",
    "xxe": "XXE",
    "xml external entity": "XXE",
    "xml external entity attack": "XXE",
    "csrf": "CSRF",
    "ssrf": "SSRF",
    "file upload": "File Upload",
    "lfi": "LFI",
    "local file inclusion": "LFI",
    "rfi": "RFI",
    "remote file inclusion": "RFI",
    "ldap injection": "LDAP Injection",
    "ldapinjection": "LDAP Injection",
    "xpath injection": "XPath Injection",
    "xpathinjection": "XPath Injection",
    "ssi": "SSI",
    "server side include": "SSI",
    "server side includes": "SSI",
    "log4shell": "Log4Shell",
    "shellshock": "Shellshock",
    "shell shock": "Shellshock",
    "crlf injection": "CRLF Injection",
    "parameter tampering": "Parameter Tampering",
    "information gathering": "Information Gathering",
    "buffer overflow": "Buffer Overflow",
}


@dataclass(slots=True)
class LoadedRows:
    rows: list[dict[str, Any]]
    features: dict[str, Any]
    request_column: str | None
    query_column: str | None
    label_column: str | None
    attack_type_column: str | None
    sample_id_column: str | None
    column_names: list[str]


def add_dataset_arguments(parser: Any) -> None:
    """Register dataset-related CLI arguments on *parser*."""
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=(
            "Hugging Face dataset name or dataset loader name such as "
            "'csv', 'json', or 'parquet'."
        ),
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help=(
            "Optional local dataset file. Use with --dataset csv/json/parquet, "
            "for example --dataset csv --data-file data/test.csv."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--request-column",
        default=None,
        help="Column containing raw HTTP requests for API classification benchmark.",
    )
    parser.add_argument(
        "--query-column",
        default=None,
        help="Column containing retrieval queries or payload text for RAG benchmark.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Column containing benign/malicious ground-truth labels.",
    )
    parser.add_argument(
        "--attack-type-column",
        default=None,
        help="Optional column containing attack-type ground truth.",
    )


def _resolve_column(
    explicit: str | None,
    candidates: list[str],
    available: list[str],
    label: str,
    required: bool,
) -> str | None:
    if explicit:
        if explicit not in available:
            raise KeyError(
                f"{label} column '{explicit}' was requested but available columns are: {available}"
            )
        return explicit

    for candidate in candidates:
        if candidate in available:
            return candidate

    if required:
        raise KeyError(
            f"Could not auto-detect {label} column. Available columns: {available}"
        )
    return None


def load_rows_from_dataset(
    dataset_name: str,
    split: str,
    data_file: str | None = None,
    *,
    request_column: str | None = None,
    query_column: str | None = None,
    label_column: str | None = None,
    attack_type_column: str | None = None,
    require_request: bool = False,
    require_query: bool = False,
    require_label: bool = True,
) -> LoadedRows:
    """Load a dataset and resolve important columns."""
    load_kwargs: dict[str, Any] = {"split": split}
    if data_file:
        load_kwargs["data_files"] = data_file

    dataset = load_dataset(dataset_name, **load_kwargs)
    rows = [dict(row) for row in dataset]
    column_names = list(dataset.column_names)
    features = getattr(dataset, "features", {}) or {}

    resolved_request = _resolve_column(
        request_column,
        REQUEST_COLUMN_CANDIDATES,
        column_names,
        "request",
        required=require_request,
    )
    resolved_query = _resolve_column(
        query_column,
        QUERY_COLUMN_CANDIDATES + REQUEST_COLUMN_CANDIDATES,
        column_names,
        "query",
        required=require_query,
    )
    resolved_label = _resolve_column(
        label_column,
        LABEL_COLUMN_CANDIDATES,
        column_names,
        "label",
        required=require_label,
    )
    resolved_attack_type = _resolve_column(
        attack_type_column,
        ATTACK_TYPE_COLUMN_CANDIDATES,
        column_names,
        "attack type",
        required=False,
    )
    resolved_sample_id = _resolve_column(
        None,
        SAMPLE_ID_COLUMN_CANDIDATES,
        column_names,
        "sample id",
        required=False,
    )

    return LoadedRows(
        rows=rows,
        features=features,
        request_column=resolved_request,
        query_column=resolved_query,
        label_column=resolved_label,
        attack_type_column=resolved_attack_type,
        sample_id_column=resolved_sample_id,
        column_names=column_names,
    )


def normalize_binary_label(raw_label: Any, feature: Any = None) -> str:
    """Normalize a label into 'benign' or 'malicious'."""
    decoded = raw_label

    if isinstance(feature, ClassLabel):
        try:
            decoded = feature.int2str(int(raw_label))
        except (TypeError, ValueError):
            decoded = raw_label

    if isinstance(decoded, bool):
        return "malicious" if decoded else "benign"

    if isinstance(decoded, (int, float)):
        numeric = int(decoded)
        if numeric == 0:
            return "benign"
        if numeric == 1:
            return "malicious"

    text = str(decoded).strip().lower()
    if text in {"0", "benign", "normal", "safe", "false"}:
        return "benign"
    if text in {"1", "malicious", "attack", "attacker", "true", "anomalous", "anomaly", "abnormal"}:
        return "malicious"

    raise ValueError(f"Unsupported label value: {raw_label!r}")


def normalize_attack_type(value: Any) -> str | None:
    """Normalize an attack-type label into a stable canonical string."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    collapsed = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if not collapsed:
        return None

    if collapsed in {"none", "n a", "na", "unknown", "benign"}:
        return None

    if collapsed in _ATTACK_TYPE_ALIASES:
        return _ATTACK_TYPE_ALIASES[collapsed]

    if collapsed.endswith(" attack") and collapsed[:-7] in _ATTACK_TYPE_ALIASES:
        return _ATTACK_TYPE_ALIASES[collapsed[:-7]]

    return " ".join(word.upper() if len(word) <= 4 else word.title() for word in collapsed.split())


def coerce_request_text(value: Any) -> str:
    """Convert a raw request field into HTTP text suitable for the API."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
    else:
        text = json.dumps(value, ensure_ascii=False)
    if not text:
        return ""
    if "\r\n" not in text and "\n" in text:
        text = text.replace("\n", "\r\n")
    return text


def coerce_query_text(value: Any) -> str:
    """Convert a retrieval query field into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def pick_sample_id(row: dict[str, Any], sample_id_column: str | None, index: int) -> str:
    """Return a stable sample identifier for reports."""
    if sample_id_column:
        raw = row.get(sample_id_column)
        if raw is not None:
            text = str(raw).strip()
            if text:
                return text
    return f"sample-{index}"


def ensure_output_dir(path_like: str | Path) -> Path:
    """Create and return an output directory path."""
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path
