"""Runtime configuration for the InferGate gateway."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _b(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- upstream providers, tried in order ---
    providers: list[str] = field(
        default_factory=lambda: os.getenv("PROVIDERS", "primary,secondary").split(",")
    )
    upstream_url: str = os.getenv("UPSTREAM_URL", "http://127.0.0.1:9100/v1/complete")
    upstream_fallback_url: str = os.getenv(
        "UPSTREAM_FALLBACK_URL", "http://127.0.0.1:9101/v1/complete"
    )
    upstream_timeout_s: float = _f("UPSTREAM_TIMEOUT_S", 20.0)

    # --- caching ---
    cache_enabled: bool = _b("CACHE_ENABLED", True)
    semantic_enabled: bool = _b("SEMANTIC_ENABLED", True)
    # Cosine threshold for a semantic hit. 0.78 is not a guess: bench/calibrate.py
    # sweeps it over 140 paraphrase pairs and 190 unrelated pairs, and 0.78 is the
    # highest-recall point that still yields zero false positives (precision 1.0,
    # recall 0.95). Below it, unrelated prompts start colliding.
    semantic_threshold: float = _f("SEMANTIC_THRESHOLD", 0.78)
    cache_ttl_s: int = _i("CACHE_TTL_S", 3600)
    # max vectors held in the in-process ANN shortlist per node
    semantic_index_cap: int = _i("SEMANTIC_INDEX_CAP", 20000)

    # --- request coalescing (single-flight) ---
    coalesce_enabled: bool = _b("COALESCE_ENABLED", True)
    coalesce_lock_ttl_s: int = _i("COALESCE_LOCK_TTL_S", 30)
    coalesce_poll_interval_s: float = _f("COALESCE_POLL_INTERVAL_S", 0.01)

    # --- circuit breaker ---
    breaker_enabled: bool = _b("BREAKER_ENABLED", True)
    breaker_fail_threshold: int = _i("BREAKER_FAIL_THRESHOLD", 5)
    breaker_reset_timeout_s: float = _f("BREAKER_RESET_TIMEOUT_S", 10.0)
    breaker_half_open_probes: int = _i("BREAKER_HALF_OPEN_PROBES", 2)

    # --- rate limiting (token bucket, per API key) ---
    limiter_enabled: bool = _b("LIMITER_ENABLED", True)
    limiter_rate_per_s: float = _f("LIMITER_RATE_PER_S", 200.0)
    limiter_burst: int = _i("LIMITER_BURST", 400)

    # --- infrastructure ---
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    redis_required: bool = _b("REDIS_REQUIRED", False)
    embedding_dim: int = _i("EMBEDDING_DIM", 512)

    # --- cost model, used to report savings in /metrics ---
    cost_per_1k_tokens_usd: float = _f("COST_PER_1K_TOKENS_USD", 0.00059)


settings = Settings()
