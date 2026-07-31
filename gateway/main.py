"""
InferGate - distributed LLM inference gateway.

Request path:
    rate limit -> exact cache -> semantic cache -> coalesce -> provider failover

Every stage exists to remove work from the upstream provider, which is both the
slowest and the only metered component in the path.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import metrics as M
from .cache import ResponseCache, exact_key
from .coalesce import Coalescer
from .config import settings
from .limiter import TokenBucket
from .providers import ProviderPool, UpstreamError
from .store import build_store

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = await build_store()
    app_state["store"] = store
    app_state["cache"] = ResponseCache(store)
    app_state["coalescer"] = Coalescer(store)
    app_state["limiter"] = TokenBucket(store)
    app_state["pool"] = ProviderPool.from_env()
    app_state["client"] = httpx.AsyncClient()
    app_state["started"] = time.time()
    yield
    await app_state["client"].aclose()
    await store.close()


app = FastAPI(
    title="InferGate",
    description="Distributed LLM inference gateway with semantic caching, "
                "cross-replica request coalescing, and provider failover.",
    version="1.0.0",
    lifespan=lifespan,
)


class CompleteRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_tokens: int = 512


@app.get("/health")
async def health():
    store = app_state.get("store")
    return {
        "status": "ok",
        "backend": getattr(store, "backend", "unknown"),
        "uptime_s": round(time.time() - app_state.get("started", time.time()), 2),
        "providers": app_state["pool"].snapshot(),
    }


@app.get("/stats")
async def stats():
    data = M.stats.as_dict()
    coalescer = app_state["coalescer"]
    data["coalesced_followers"] = coalescer.followers_served
    data["semantic_index_size"] = app_state["cache"].index_size
    data["provider_failovers"] = app_state["pool"].failovers
    data["rate_limited"] = app_state["limiter"].rejected
    data["backend"] = getattr(app_state.get("store"), "backend", "unknown")
    return data


@app.post("/admin/reset")
async def reset_stats():
    """Reset in-process counters so a benchmark run starts from a clean slate."""
    M.stats.__init__()
    app_state["coalescer"].followers_served = 0
    app_state["coalescer"].leader_calls = 0
    app_state["pool"].failovers = 0
    for p in app_state["pool"].providers:
        p.calls = 0
    return {"status": "reset"}


@app.get("/metrics")
async def prometheus_metrics():
    M.SEMANTIC_INDEX.set(app_state["cache"].index_size)
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/complete")
async def complete(
    body: CompleteRequest,
    request: Request,
    x_api_key: str = Header(default="anonymous"),
):
    started = time.perf_counter()
    cache: ResponseCache = app_state["cache"]
    pool: ProviderPool = app_state["pool"]
    client: httpx.AsyncClient = app_state["client"]
    params = {"temperature": body.temperature, "max_tokens": body.max_tokens}

    allowed, retry_after = await app_state["limiter"].allow(x_api_key)
    if not allowed:
        M.REQUESTS.labels(outcome="rate_limited").inc()
        M.RATE_LIMITED.inc()
        return JSONResponse(
            {"error": "rate_limited", "retry_after_s": round(retry_after, 3)},
            status_code=429,
            headers={"Retry-After": str(max(1, int(retry_after)))},
        )

    M.stats.total += 1

    hit = await cache.get(body.model, body.prompt, params)
    if hit:
        if hit.tier == "exact":
            M.stats.exact_hits += 1
        else:
            M.stats.semantic_hits += 1
        saved = int(hit.value.get("tokens", 0))
        M.stats.tokens_saved += saved
        M.TOKENS_SAVED.inc(saved)
        M.CACHE_EVENTS.labels(result=hit.tier).inc()
        M.REQUESTS.labels(outcome="cache_hit").inc()
        elapsed = time.perf_counter() - started
        M.LATENCY.labels(path="/v1/complete").observe(elapsed)
        return {
            **hit.value,
            "cached": True,
            "cache_tier": hit.tier,
            "similarity": round(hit.similarity, 4),
            "gateway_ms": round(elapsed * 1000, 3),
        }

    M.stats.misses += 1
    M.CACHE_EVENTS.labels(result="miss").inc()

    async def do_upstream() -> dict:
        completion = await pool.complete(client, body.model, body.prompt)
        M.stats.upstream_calls += 1
        M.stats.tokens_served += completion.tokens
        M.UPSTREAM_CALLS.labels(provider=completion.provider).inc()
        M.UPSTREAM_LATENCY.observe(completion.upstream_ms / 1000.0)
        payload = {
            "text": completion.text,
            "tokens": completion.tokens,
            "provider": completion.provider,
            "upstream_ms": round(completion.upstream_ms, 3),
        }
        await cache.put(body.model, body.prompt, params, payload)
        return payload

    async def read_cache():
        again = await cache.get(body.model, body.prompt, params)
        return again.value if again else None

    try:
        payload, role = await app_state["coalescer"].run(
            exact_key(body.model, body.prompt, params), do_upstream, read_cache
        )
    except UpstreamError as exc:
        M.REQUESTS.labels(outcome="upstream_error").inc()
        return JSONResponse({"error": "upstream_unavailable", "detail": str(exc)}, 503)

    M.COALESCED.labels(role=role).inc()
    if role == "follower":
        M.REQUESTS.labels(outcome="coalesced").inc()
    else:
        M.REQUESTS.labels(outcome="upstream").inc()

    elapsed = time.perf_counter() - started
    M.LATENCY.labels(path="/v1/complete").observe(elapsed)
    return {
        **payload,
        "cached": False,
        "coalesce_role": role,
        "gateway_ms": round(elapsed * 1000, 3),
    }
