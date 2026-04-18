"""Gatekeeper ML node – classify individual payload values before RAG.

Responsibilities:
- Lazily construct a singleton ``PayloadClassifier`` from ``gatekeeper_ml``.
- Score each batch of normalised payloads with ``predict_batch``.
- Forward only suspicious payloads to the downstream RAG node.
- Short-circuit clearly benign requests to cache persistence.
"""

from __future__ import annotations

import logging
from threading import Lock

from gatekeeper_ml import PayloadClassifier

from schema.state import GraphState

logger = logging.getLogger(__name__)

_classifier_instance: PayloadClassifier | None = None
_classifier_lock = Lock()


def _get_classifier() -> PayloadClassifier:
    """Return a shared classifier instance."""
    global _classifier_instance

    if _classifier_instance is None:
        with _classifier_lock:
            if _classifier_instance is None:
                _classifier_instance = PayloadClassifier()
    return _classifier_instance


def _predict_batch(classifier: PayloadClassifier, payloads: list[str]) -> list[int]:
    """Predict a batch of payloads using the documented Gatekeeper API."""
    predictions = classifier.predict_batch(payloads)
    if len(predictions) != len(payloads):
        raise ValueError(
            "Gatekeeper prediction count did not match payload count: "
            f"{len(predictions)} != {len(payloads)}"
        )
    return [int(prediction) for prediction in predictions]


def gatekeeper_node(state: GraphState) -> dict[str, Any]:
    """Classify payloads and retain only suspicious candidates for RAG."""
    payloads = state.get("normalized_payloads", [])
    if not payloads:
        return {
            "suspicious_payloads": [],
            "final_result": {
                "verdict": "benign",
                "source": "gatekeeper_ml",
                "confidence": 1.0,
                "reasoning": "No payload values were available for ML analysis.",
            },
        }

    try:
        classifier = _get_classifier()
    except Exception as exc:
        logger.warning("Gatekeeper classifier unavailable; forwarding to RAG: %s", exc)
        return {"suspicious_payloads": payloads}

    try:
        predictions = _predict_batch(classifier, payloads)
    except Exception as exc:
        logger.warning("Gatekeeper batch prediction failed; forwarding to RAG: %s", exc)
        return {"suspicious_payloads": payloads}

    suspicious_payloads = [
        payload for payload, prediction in zip(payloads, predictions) if prediction == 1
    ]

    if not suspicious_payloads:
        logger.info("Gatekeeper ML marked request benign")
        return {
            "suspicious_payloads": [],
            "final_result": {
                "verdict": "benign",
                "source": "gatekeeper_ml",
                "confidence": 1.0,
                "reasoning": "No payloads were flagged as suspicious by Gatekeeper ML.",
            },
        }

    logger.info(
        "Gatekeeper ML flagged %d suspicious payload(s) out of %d",
        len(suspicious_payloads),
        len(payloads),
    )
    return {"suspicious_payloads": suspicious_payloads}
