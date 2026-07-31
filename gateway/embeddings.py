"""
Stateless prompt embeddings for semantic cache lookups.

Uses a character n-gram HashingVectorizer so embeddings are:
  * deterministic  - the same prompt always maps to the same vector
  * stateless      - no fitted vocabulary to share between gateway replicas
  * zero-cost      - no embedding API call on the hot path

This is deliberate: paying an embedding API on every request would erase the
latency and cost savings the cache exists to create.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .config import settings

_vectorizer = HashingVectorizer(
    n_features=settings.embedding_dim,
    analyzer="char_wb",
    ngram_range=(3, 5),
    alternate_sign=False,
    norm="l2",
)


def embed(text: str) -> np.ndarray:
    """Return an L2-normalised float32 vector for `text`."""
    matrix = _vectorizer.transform([text or ""])
    vec = np.asarray(matrix.todense(), dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two already-normalised vectors."""
    return float(np.dot(a, b))


def cosine_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of `query` against every row of `matrix`."""
    if matrix.size == 0:
        return np.empty(0, dtype=np.float32)
    return matrix @ query
