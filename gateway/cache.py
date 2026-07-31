"""
Two-tier response cache.

Tier 1 - exact match:    SHA-256 of (model, normalised prompt, params). O(1).
Tier 2 - semantic match: cosine similarity over prompt embeddings, which
                         catches paraphrases that tier 1 misses.

Tier 1 runs first because it is far cheaper; tier 2 only runs on a tier 1 miss.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import settings
from .embeddings import cosine_batch, embed


def exact_key(model: str, prompt: str, params: dict) -> str:
    payload = json.dumps(
        {"model": model, "prompt": " ".join((prompt or "").split()).lower(), "params": params},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "ig:exact:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheHit:
    value: dict
    tier: str          # "exact" | "semantic"
    similarity: float  # 1.0 for exact


class SemanticIndex:
    """
    Bounded in-process vector index (FIFO eviction).

    Each replica keeps its own shortlist of recent vectors. Payloads live in the
    shared store, so a semantic hit on any replica resolves to the same cached
    response. Kept in-process because a full ANN service is unjustified below
    ~1e5 vectors, and the matrix multiply here is already sub-millisecond.
    """

    def __init__(self, dim: int, cap: int) -> None:
        self._dim = dim
        self._cap = cap
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._keys: list[str] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, key: str, vector: np.ndarray) -> None:
        with self._lock:
            if key in self._keys:
                return
            self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])
            self._keys.append(key)
            if len(self._keys) > self._cap:
                drop = len(self._keys) - self._cap
                self._vectors = self._vectors[drop:]
                self._keys = self._keys[drop:]

    def search(self, vector: np.ndarray) -> tuple[Optional[str], float]:
        with self._lock:
            if not self._keys:
                return None, 0.0
            scores = cosine_batch(vector, self._vectors)
            idx = int(np.argmax(scores))
            return self._keys[idx], float(scores[idx])


class ResponseCache:
    def __init__(self, store) -> None:
        self._store = store
        self._index = SemanticIndex(settings.embedding_dim, settings.semantic_index_cap)

    @property
    def index_size(self) -> int:
        return len(self._index)

    async def get(self, model: str, prompt: str, params: dict) -> Optional[CacheHit]:
        if not settings.cache_enabled:
            return None

        key = exact_key(model, prompt, params)
        raw = await self._store.get(key)
        if raw:
            return CacheHit(value=json.loads(raw), tier="exact", similarity=1.0)

        if not settings.semantic_enabled:
            return None

        vector = embed(prompt)
        near_key, score = self._index.search(vector)
        if near_key and score >= settings.semantic_threshold:
            raw = await self._store.get(near_key)
            if raw:
                return CacheHit(value=json.loads(raw), tier="semantic", similarity=score)
        return None

    async def put(self, model: str, prompt: str, params: dict, value: dict) -> None:
        if not settings.cache_enabled:
            return
        key = exact_key(model, prompt, params)
        await self._store.set(key, json.dumps(value), settings.cache_ttl_s)
        if settings.semantic_enabled:
            self._index.add(key, embed(prompt))
