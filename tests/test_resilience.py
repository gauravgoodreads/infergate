import asyncio

import pytest

from gateway.breaker import CircuitBreaker, State
from gateway.coalesce import Coalescer
from gateway.config import settings
from gateway.limiter import TokenBucket
from gateway.store import MemoryStore


# ---------------- circuit breaker ----------------

def test_breaker_opens_after_threshold():
    cb = CircuitBreaker("p")
    for _ in range(settings.breaker_fail_threshold):
        cb.record_failure()
    assert cb.state is State.OPEN
    assert cb.allows() is False


def test_breaker_half_opens_after_timeout(monkeypatch):
    cb = CircuitBreaker("p")
    for _ in range(settings.breaker_fail_threshold):
        cb.record_failure()
    cb.opened_at -= settings.breaker_reset_timeout_s + 1
    assert cb.allows() is True
    assert cb.state is State.HALF_OPEN


def test_breaker_closes_after_successful_probes():
    cb = CircuitBreaker("p")
    for _ in range(settings.breaker_fail_threshold):
        cb.record_failure()
    cb.opened_at -= settings.breaker_reset_timeout_s + 1
    cb.allows()
    for _ in range(settings.breaker_half_open_probes):
        cb.record_success()
    assert cb.state is State.CLOSED
    assert cb.failures == 0


def test_breaker_reopens_if_probe_fails():
    cb = CircuitBreaker("p")
    for _ in range(settings.breaker_fail_threshold):
        cb.record_failure()
    cb.opened_at -= settings.breaker_reset_timeout_s + 1
    cb.allows()
    cb.record_failure()
    assert cb.state is State.OPEN


# ---------------- coalescing ----------------

@pytest.mark.asyncio
async def test_coalescing_collapses_identical_concurrent_calls():
    store = MemoryStore()
    coalescer = Coalescer(store)
    upstream_calls = 0
    published: dict = {}

    async def do_work():
        nonlocal upstream_calls
        upstream_calls += 1
        await asyncio.sleep(0.15)
        published["v"] = {"text": "shared", "tokens": 7}
        return published["v"]

    async def read_cache():
        return published.get("v")

    results = await asyncio.gather(*(
        coalescer.run("same-key", do_work, read_cache) for _ in range(20)
    ))

    assert upstream_calls == 1, "20 identical requests must collapse to 1 upstream call"
    assert all(payload["text"] == "shared" for payload, _ in results)
    assert sum(1 for _, role in results if role == "follower") == 19


@pytest.mark.asyncio
async def test_distinct_keys_are_not_coalesced():
    store = MemoryStore()
    coalescer = Coalescer(store)
    calls = 0

    async def do_work():
        nonlocal calls
        calls += 1
        return {"text": "x", "tokens": 1}

    await asyncio.gather(*(
        coalescer.run(f"key-{i}", do_work, lambda: _none()) for i in range(5)
    ))
    assert calls == 5


async def _none():
    return None


# ---------------- rate limiter ----------------

@pytest.mark.asyncio
async def test_token_bucket_allows_burst_then_rejects():
    original_rate, original_burst = settings.limiter_rate_per_s, settings.limiter_burst
    settings.limiter_rate_per_s, settings.limiter_burst = 0.0001, 5
    try:
        limiter = TokenBucket(MemoryStore())
        allowed = [(await limiter.allow("k"))[0] for _ in range(8)]
        assert allowed[:5] == [True] * 5
        assert allowed[5:] == [False] * 3
        assert limiter.rejected == 3
    finally:
        settings.limiter_rate_per_s = original_rate
        settings.limiter_burst = original_burst


@pytest.mark.asyncio
async def test_rate_limit_is_per_api_key():
    original_rate, original_burst = settings.limiter_rate_per_s, settings.limiter_burst
    settings.limiter_rate_per_s, settings.limiter_burst = 0.0001, 2
    try:
        limiter = TokenBucket(MemoryStore())
        assert (await limiter.allow("tenant-a"))[0] is True
        assert (await limiter.allow("tenant-a"))[0] is True
        assert (await limiter.allow("tenant-a"))[0] is False
        assert (await limiter.allow("tenant-b"))[0] is True, "keys must not share budget"
    finally:
        settings.limiter_rate_per_s = original_rate
        settings.limiter_burst = original_burst
