"""Centralised configuration loaded from environment variables.

All credentials and service URLs are read from a ``.env`` file at the project
root via ``python-dotenv``.  Import ``settings`` from this module wherever
configuration values are needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from config/)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_flag(name: str) -> bool | None:
    """Parse an optional boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_rag_enabled(default: bool = True) -> bool:
    """Resolve RAG enablement from environment variables."""
    rag_enabled = _env_optional_flag("RAG_ENABLED")
    if rag_enabled is not None:
        return rag_enabled

    no_rag = _env_optional_flag("NO_RAG")
    if no_rag is not None:
        return not no_rag

    return default


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings sourced from environment variables."""

    # --- LLM -----------------------------------------------------------------
    google_api_key: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY", "")
    )
    gemini_model: str = "gemini-2.5-flash"

    # --- Redis ----------------------------------------------------------------
    redis_url: str = field(
        default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    )

    # --- Qdrant ---------------------------------------------------------------
    qdrant_url: str = field(
        default_factory=lambda: os.environ.get("QDRANT_URL", "http://localhost:6333")
    )
    qdrant_api_key: str = field(
        default_factory=lambda: os.environ.get("QDRANT_API_KEY", "")
    )
    qdrant_collection: str = field(
        default_factory=lambda: os.environ.get("QDRANT_COLLECTION", "waf_payloads")
    )
    qdrant_dense_vector_name: str = field(
        default_factory=lambda: os.environ.get("QDRANT_DENSE_VECTOR_NAME", "dense")
    )
    qdrant_sparse_vector_name: str = field(
        default_factory=lambda: os.environ.get("QDRANT_SPARSE_VECTOR_NAME", "sparse")
    )

    # --- General --------------------------------------------------------------
    cache_ttl_seconds: int = 3600
    rag_top_k: int = 3
    rag_dense_model: str = field(
        default_factory=lambda: os.environ.get(
            "RAG_DENSE_MODEL", "jinaai/jina-embeddings-v2-base-code"
        )
    )
    rag_enabled: bool = field(
        default_factory=lambda: _env_rag_enabled(default=True)
    )

    # --- Persistence / logging ------------------------------------------------
    sqlite_db_path: str = field(
        default_factory=lambda: os.environ.get("SQLITE_DB_PATH", "data/copilot_data.db")
    )
    siem_log_path: str = field(
        default_factory=lambda: os.environ.get("SIEM_LOG_PATH", "logs/ai_alerts.jsonl")
    )
    debug: bool = field(
        default_factory=lambda: _env_flag("DEBUG", default=False)
    )
    debug_csv_path: str = field(
        default_factory=lambda: os.environ.get(
            "DEBUG_CSV_PATH", "logs/debug_pipeline.csv"
        )
    )

    # --- Batch processing -----------------------------------------------------
    batch_max_size: int = field(
        default_factory=lambda: int(os.environ.get("BATCH_MAX_SIZE", "20"))
    )
    batch_window_seconds: float = field(
        default_factory=lambda: float(os.environ.get("BATCH_WINDOW_SECONDS", "10.0"))
    )
    batch_queue_key: str = field(
        default_factory=lambda: os.environ.get("BATCH_QUEUE_KEY", "waf:queue:llm_analysis")
    )
    batch_dlq_key: str = field(
        default_factory=lambda: os.environ.get("BATCH_DLQ_KEY", "waf:queue:llm_analysis_dlq")
    )
    batch_result_prefix: str = "waf:result:"
    batch_result_ttl_seconds: int = field(
        default_factory=lambda: int(os.environ.get("BATCH_RESULT_TTL", "3600"))
    )
    batch_max_retries: int = 3

    def validate(self) -> None:
        """Raise if critical secrets are missing."""
        if not self.google_api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )


# Module-level singleton – import this wherever settings are needed.
settings = Settings()
