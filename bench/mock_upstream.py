"""
Instrumented mock LLM provider.

Used instead of a live provider for benchmarking so results are reproducible and
isolated from provider-side rate limits and network variance. Latency is drawn
from a log-normal distribution centred on UPSTREAM_MEAN_MS, which matches the
long-tailed shape of real chat-completion latency far better than a constant
sleep. It also counts every call it receives, which is how the benchmark proves
cache and coalescing actually removed upstream work.

Run:  python -m bench.mock_upstream --port 9100 --mean-ms 200
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="mock-upstream")

STATE = {"calls": 0, "mean_ms": 200.0, "sigma": 0.35, "fail_rate": 0.0, "started": time.time()}


class Req(BaseModel):
    prompt: str
    model: str = "mock"


@app.post("/v1/complete")
async def complete(body: Req):
    STATE["calls"] += 1

    if STATE["fail_rate"] > 0 and random.random() < STATE["fail_rate"]:
        await asyncio.sleep(0.005)
        return {"error": "injected_failure"}, 503

    mean = STATE["mean_ms"]
    sigma = STATE["sigma"]
    mu = __import__("math").log(max(mean, 1.0)) - (sigma ** 2) / 2.0
    delay_ms = min(random.lognormvariate(mu, sigma), mean * 12)
    await asyncio.sleep(delay_ms / 1000.0)

    tokens = 40 + len(body.prompt.split()) * 3
    return {
        "text": f"[mock completion for {len(body.prompt)} chars]",
        "tokens": tokens,
        "latency_ms": round(delay_ms, 3),
    }


@app.get("/calls")
async def calls():
    return {"calls": STATE["calls"], "mean_ms": STATE["mean_ms"], "fail_rate": STATE["fail_rate"]}


@app.post("/reset")
async def reset():
    STATE["calls"] = 0
    return {"calls": 0}


@app.post("/configure")
async def configure(mean_ms: float | None = None, fail_rate: float | None = None):
    if mean_ms is not None:
        STATE["mean_ms"] = mean_ms
    if fail_rate is not None:
        STATE["fail_rate"] = fail_rate
    return {"mean_ms": STATE["mean_ms"], "fail_rate": STATE["fail_rate"]}


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--mean-ms", type=float, default=200.0)
    args = ap.parse_args()
    STATE["mean_ms"] = args.mean_ms
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
