"""Prometheus metrics and the derived cost-savings model."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from .config import settings

REQUESTS = Counter(
    "infergate_requests_total", "Requests served", ["outcome"]
)
CACHE_EVENTS = Counter(
    "infergate_cache_events_total", "Cache lookups by result", ["result"]
)
COALESCED = Counter(
    "infergate_coalesced_total", "Requests resolved by coalescing", ["role"]
)
UPSTREAM_CALLS = Counter(
    "infergate_upstream_calls_total", "Upstream provider calls", ["provider"]
)
FAILOVERS = Counter("infergate_failovers_total", "Provider failovers")
RATE_LIMITED = Counter("infergate_rate_limited_total", "Rate-limited requests")

LATENCY = Histogram(
    "infergate_request_latency_seconds",
    "End-to-end gateway latency",
    ["path"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
UPSTREAM_LATENCY = Histogram(
    "infergate_upstream_latency_seconds",
    "Upstream provider latency",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)

SEMANTIC_INDEX = Gauge("infergate_semantic_index_size", "Vectors in semantic index")
TOKENS_SAVED = Counter("infergate_tokens_saved_total", "Tokens avoided via cache")


class Stats:
    """In-process counters used for the human-readable /stats endpoint."""

    def __init__(self) -> None:
        self.total = 0
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.upstream_calls = 0
        self.tokens_served = 0
        self.tokens_saved = 0

    @property
    def hits(self) -> int:
        return self.exact_hits + self.semantic_hits

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.total) if self.total else 0.0

    @property
    def usd_saved(self) -> float:
        return (self.tokens_saved / 1000.0) * settings.cost_per_1k_tokens_usd

    def as_dict(self) -> dict:
        return {
            "requests": self.total,
            "cache_hits": self.hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "upstream_calls": self.upstream_calls,
            "upstream_calls_avoided": max(0, self.total - self.upstream_calls),
            "tokens_served": self.tokens_served,
            "tokens_saved": self.tokens_saved,
            "estimated_usd_saved": round(self.usd_saved, 6),
        }


stats = Stats()
