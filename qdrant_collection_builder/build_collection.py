"""Build and load a hybrid (dense+sparse) Qdrant collection from payload folders.

Expected payload layout:

    payloads/
      SQL Injection/
        sqli.txt
      XSS/
        xss.txt

Each non-empty line from each file is treated as one payload sample.
Lines starting with '#' are ignored.

Dense vectors: jinaai/jina-embeddings-v2-base-code (configurable)
Sparse vectors: BM25 sparse embeddings for keyword/hybrid search

Usage:
    python qdrant_collection_builder/build_collection.py \
      --payload-dir payloads \
      --collection waf_payloads
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

try:
    from fastembed import SparseTextEmbedding
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "Missing dependency 'fastembed'. Install it with: pip install fastembed"
    ) from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("qdrant_collection_builder")


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")


@dataclass(frozen=True)
class BuilderSettings:
    qdrant_url: str
    qdrant_api_key: str
    collection: str
    dense_vector_name: str
    sparse_vector_name: str
    dense_model: str


@dataclass(frozen=True)
class PayloadRow:
    text: str
    attack_type: str
    source_file: str
    line_no: int


def _read_settings(collection_override: str | None) -> BuilderSettings:
    collection = collection_override or os.environ.get("QDRANT_COLLECTION", "waf_payloads")
    return BuilderSettings(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY", ""),
        collection=collection,
        dense_vector_name=os.environ.get("QDRANT_DENSE_VECTOR_NAME", "dense"),
        sparse_vector_name=os.environ.get("QDRANT_SPARSE_VECTOR_NAME", "sparse"),
        dense_model=os.environ.get(
            "RAG_DENSE_MODEL", "jinaai/jina-embeddings-v2-base-code"
        ),
    )


def _iter_payload_rows(payload_dir: Path) -> Iterable[PayloadRow]:
    if not payload_dir.exists() or not payload_dir.is_dir():
        raise FileNotFoundError(f"Payload directory does not exist: {payload_dir}")

    for attack_dir in sorted(payload_dir.iterdir()):
        if not attack_dir.is_dir():
            continue

        attack_type = attack_dir.name
        for file_path in sorted(attack_dir.rglob("*")):
            if not file_path.is_file():
                continue

            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line_no, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    yield PayloadRow(
                        text=line,
                        attack_type=attack_type,
                        source_file=str(file_path),
                        line_no=line_no,
                    )


def _make_point_id(row: PayloadRow) -> str:
    seed = f"{row.attack_type}|{row.source_file}|{row.line_no}|{row.text}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:32]


def _ensure_collection(
    client: QdrantClient,
    settings: BuilderSettings,
    dense_dim: int,
    recreate: bool,
) -> None:
    if recreate:
        logger.info("Recreating collection '%s'", settings.collection)
        client.recreate_collection(
            collection_name=settings.collection,
            vectors_config={
                settings.dense_vector_name: models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                settings.sparse_vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info("Collection '%s' recreated", settings.collection)
        return

    collections = client.get_collections().collections
    exists = any(c.name == settings.collection for c in collections)
    if exists:
        logger.info("Collection '%s' already exists; reusing", settings.collection)
        return

    logger.info("Creating collection '%s'", settings.collection)
    client.create_collection(
        collection_name=settings.collection,
        vectors_config={
            settings.dense_vector_name: models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            settings.sparse_vector_name: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )
    logger.info("Collection '%s' created", settings.collection)


def _batched(items: list[PayloadRow], size: int) -> Iterable[list[PayloadRow]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _upsert_payloads(
    client: QdrantClient,
    settings: BuilderSettings,
    rows: list[PayloadRow],
    batch_size: int,
) -> int:
    total = len(rows)
    logger.info(
        "Loading dense model '%s' and sparse model 'Qdrant/bm25'",
        settings.dense_model,
    )
    dense_model = SentenceTransformer(settings.dense_model, trust_remote_code=True)
    sparse_model = SparseTextEmbedding("Qdrant/bm25")

    inserted = 0
    for batch_idx, batch in enumerate(_batched(rows, batch_size), start=1):
        texts = [r.text for r in batch]

        logger.info(
            "Processing batch %d (%d rows) [%d/%d]",
            batch_idx,
            len(batch),
            inserted,
            total,
        )

        dense_vectors = dense_model.encode(texts, normalize_embeddings=True)
        if hasattr(dense_vectors, "tolist"):
            dense_vectors = dense_vectors.tolist()

        sparse_embs = list(sparse_model.embed(texts))

        points: list[models.PointStruct] = []
        for idx, row in enumerate(batch):
            sparse = sparse_embs[idx]
            category = row.attack_type
            text = f"Attack Type: {row.attack_type} | Payload: {row.text}"
            point = models.PointStruct(
                id=_make_point_id(row),
                vector={
                    settings.dense_vector_name: dense_vectors[idx],
                    settings.sparse_vector_name: models.SparseVector(
                        indices=sparse.indices.tolist()
                        if hasattr(sparse.indices, "tolist")
                        else list(sparse.indices),
                        values=sparse.values.tolist()
                        if hasattr(sparse.values, "tolist")
                        else list(sparse.values),
                    ),
                },
                payload={
                    # Canonical payload shape for hybrid retrieval.
                    "text": text,
                    "category": category,
                    "raw_payload": row.text,
                },
            )
            points.append(point)

        client.upsert(collection_name=settings.collection, points=points, wait=True)
        inserted += len(points)
        logger.info("Upserted batch %d, progress: %d/%d", batch_idx, inserted, total)

    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and load payload datasets into a hybrid Qdrant collection"
    )
    parser.add_argument(
        "--payload-dir",
        required=True,
        help="Path to payload folder (subfolders are attack types)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Override Qdrant collection name (else uses QDRANT_COLLECTION)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding+upsert batch size",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate collection before loading",
    )
    return parser.parse_args()


def main() -> None:
    _load_env()
    args = parse_args()

    settings = _read_settings(args.collection)
    payload_dir = Path(args.payload_dir).resolve()

    logger.info("Starting collection build")
    logger.info("Payload directory: %s", payload_dir)
    logger.info("Target collection: %s", settings.collection)

    rows = list(_iter_payload_rows(payload_dir))
    if not rows:
        raise SystemExit("No payload rows found. Check payload directory structure.")
    logger.info("Discovered %d payload rows", len(rows))

    logger.info("Probing dense model vector dimension")
    probe_model = SentenceTransformer(settings.dense_model, trust_remote_code=True)
    probe_vec = probe_model.encode(["dimension probe"], normalize_embeddings=True)
    if hasattr(probe_vec, "tolist"):
        probe_vec = probe_vec.tolist()
    dense_dim = len(probe_vec[0])
    logger.info("Dense vector dimension: %d", dense_dim)

    logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=60,
    )

    _ensure_collection(client, settings, dense_dim=dense_dim, recreate=args.recreate)
    inserted = _upsert_payloads(
        client,
        settings,
        rows,
        batch_size=max(1, args.batch_size),
    )

    logger.info("Collection build finished: %d rows loaded", inserted)

    summary = {
        "collection": settings.collection,
        "payload_dir": str(payload_dir),
        "rows_loaded": inserted,
        "dense_model": settings.dense_model,
        "dense_vector_name": settings.dense_vector_name,
        "sparse_vector_name": settings.sparse_vector_name,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
