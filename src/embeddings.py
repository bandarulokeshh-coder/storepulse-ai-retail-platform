from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Cosine similarity at or above this value accepts an embedding-based category match.
# This is a semantic similarity score, NOT a calibrated classification confidence.
SEMANTIC_MATCH_THRESHOLD = 0.40

_model_cache: dict[str, Any] = {}
_attempted: set[str] = set()
_failure_reasons: dict[str, str] = {}


@dataclass(slots=True)
class EmbeddingStatus:
    available: bool
    model_name: str
    reason: str
    attempted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "model_name": self.model_name,
            "reason": self.reason,
            "attempted": self.attempted,
        }


def ensure_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Any | None:
    """Load (once) and return the multilingual sentence-transformers model.

    Returns None and caches the failure reason when the model is unavailable
    (offline, package missing, download failed). Never raises.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]
    if model_name in _attempted:
        return None
    _attempted.add(model_name)
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - depends on environment
        _failure_reasons[model_name] = f"sentence-transformers package unavailable ({exc.__class__.__name__})"
        return None
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - depends on network/environment
        _failure_reasons[model_name] = f"model could not be loaded ({exc.__class__.__name__}: {exc})"
        return None
    _model_cache[model_name] = model
    return model


def embedding_status(model_name: str = DEFAULT_MODEL_NAME) -> EmbeddingStatus:
    """Report embedding availability without forcing a (possibly slow) model download."""
    if model_name in _model_cache:
        return EmbeddingStatus(True, model_name, "loaded", True)
    if model_name in _attempted:
        return EmbeddingStatus(
            False,
            model_name,
            _failure_reasons.get(model_name, "model unavailable"),
            True,
        )
    return EmbeddingStatus(False, model_name, "model not loaded yet", False)


def reset_embedding_cache() -> None:
    """Allow a fresh load attempt (used by the Settings retry button)."""
    _model_cache.clear()
    _attempted.clear()
    _failure_reasons.clear()


def encode_texts(model: Any, texts: list[str]) -> np.ndarray:
    """Encode texts into L2-normalized embedding vectors."""
    vectors = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=float)


def build_category_prototypes(model: Any, prototype_texts: dict[str, str]) -> dict[str, np.ndarray]:
    """Build one prototype embedding per issue category from its name and keywords."""
    if not prototype_texts:
        return {}
    categories = list(prototype_texts.keys())
    vectors = encode_texts(model, [prototype_texts[category] for category in categories])
    return {category: vectors[idx] for idx, category in enumerate(categories)}


def best_semantic_match(
    model: Any,
    prototypes: dict[str, np.ndarray],
    text: str,
) -> tuple[str | None, float | None]:
    """Return (category, cosine similarity) for the closest category prototype.

    Returns (None, None) when no prototypes or text are available.
    """
    if not prototypes or not (text or "").strip():
        return None, None
    vector = encode_texts(model, [str(text)])[0]
    best_category: str | None = None
    best_similarity = -1.0
    for category, prototype in prototypes.items():
        similarity = float(np.dot(vector, prototype))
        if similarity > best_similarity:
            best_similarity = similarity
            best_category = category
    if best_category is None:
        return None, None
    return best_category, max(best_similarity, 0.0)