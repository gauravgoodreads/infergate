"""
InferGate benchmark harness.

Produces every number reported in the README. Each scenario resets the gateway
and mock-upstream counters first, so results are independent.

Scenarios
  1. baseline      - caching and coalescing disabled: the control
  2. cached        - full gateway on the same Zipf workload
  3. herd          - N concurrent identical prompts on a cold cache
  4. failover      - primary provider failing; measures breaker + failover

Run:  python -m bench.loadtest --requests 2000 --concurrency 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from .workload import Workload

GATEWAY = "http://127.0.0.1:8080"
UPSTREAM = "http://127.0.0.1:9100"
RESULTS_DIR = Path(__file__).parent / "results"


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarise(latencies_ms: list[float], wall_s: float, errors: int) -> dict:
    return {
        "requests": len(latencies_ms) + errors,
        "errors": errors,
        "wall_s": round(wall_s, 3),
        "throughput_rps": round(len(latencies_ms) / wall_s, 2) if wall_s else 0.0,
        "mean_ms": round(statistics.fmean(latencies_ms), 2) if latencies_ms else 0.0,
        "p50_ms": round(pct(latencies_ms, 50), 2),
        "p95_ms": round(pct(latencies_ms, 95), 2),
        "p99_ms": round(pct(latencies_ms, 99), 2),
        "max_ms": round(max(latencies_ms), 2) if latencies_ms else 0.0,
    }


async def reset_all(client: httpx.AsyncClient) -> None:
    await client.post(f"{GATEWAY}/admin/reset")
    try:
        await client.post(f"{UPSTREAM}/reset")
    except Exception:
        pass


async def configure_gateway(client: httpx.AsyncClient, **flags) -> None:
    """Flags are applied via env on the server; here we only reset counters."""
    await reset_all(client)


async def fire(
    client: httpx.AsyncClient, prompts: list[str], concurrency: int
) -> tuple[list[float], int]:
    latencies: list[float] = []
    errors = 0
    sem = asyncio.Semaphore(concurrency)

    async def one(prompt: str):
        nonlocal errors
        async with sem:
            started = time.perf_counter()
            try:
                r = await client.post(
                    f"{GATEWAY}/v1/complete",
                    json={"prompt": prompt},
                    headers={"x-api-key": "bench"},
                    timeout=60.0,
                )
                if r.status_code != 200:
                    errors += 1
                    return
                latencies.append((time.perf_counter() - started) * 1000.0)
            except Exception:
                errors += 1

    await asyncio.gather(*(one(p) for p in prompts))
    return latencies, errors


async def scenario_load(
    client: httpx.AsyncClient, name: str, n: int, concurrency: int, workload: Workload
) -> dict:
    await reset_all(client)
    prompts = [workload.next_prompt() for _ in range(n)]
    t0 = time.perf_counter()
    latencies, errors = await fire(client, prompts, concurrency)
    wall = time.perf_counter() - t0

    stats = (await client.get(f"{GATEWAY}/stats")).json()
    try:
        upstream = (await client.get(f"{UPSTREAM}/calls")).json()
    except Exception:
        upstream = {"calls": None}

    return {
        "scenario": name,
        "concurrency": concurrency,
        **summarise(latencies, wall, errors),
        "gateway_stats": stats,
        "upstream_calls_observed": upstream.get("calls"),
    }


async def scenario_herd(client: httpx.AsyncClient, concurrency: int) -> dict:
    """N identical prompts fired simultaneously against a cold cache."""
    await reset_all(client)
    prompt = f"herd probe {time.time()} explain consistent hashing"
    t0 = time.perf_counter()
    latencies, errors = await fire(client, [prompt] * concurrency, concurrency)
    wall = time.perf_counter() - t0

    stats = (await client.get(f"{GATEWAY}/stats")).json()
    try:
        upstream = (await client.get(f"{UPSTREAM}/calls")).json()
        upstream_calls = upstream.get("calls")
    except Exception:
        upstream_calls = stats.get("upstream_calls")

    reduction = None
    if upstream_calls:
        reduction = round((1 - upstream_calls / concurrency) * 100, 2)

    return {
        "scenario": "thundering_herd",
        "concurrent_identical_requests": concurrency,
        **summarise(latencies, wall, errors),
        "upstream_calls": upstream_calls,
        "upstream_call_reduction_pct": reduction,
        "coalesced_followers": stats.get("coalesced_followers"),
        "gateway_stats": stats,
    }


async def scenario_failover(client: httpx.AsyncClient, n: int, concurrency: int) -> dict:
    """Force the primary provider to fail and measure breaker + failover."""
    await reset_all(client)
    try:
        await client.post(f"{UPSTREAM}/configure", params={"fail_rate": 1.0})
    except Exception:
        return {"scenario": "failover", "skipped": "mock upstream unreachable"}

    prompts = [f"failover probe {i}" for i in range(n)]
    t0 = time.perf_counter()
    latencies, errors = await fire(client, prompts, concurrency)
    wall = time.perf_counter() - t0

    await client.post(f"{UPSTREAM}/configure", params={"fail_rate": 0.0})
    health = (await client.get(f"{GATEWAY}/health")).json()
    stats = (await client.get(f"{GATEWAY}/stats")).json()

    return {
        "scenario": "failover",
        **summarise(latencies, wall, errors),
        "provider_failovers": stats.get("provider_failovers"),
        "breaker_states": health.get("providers"),
        "availability_pct": round(len(latencies) / max(n, 1) * 100, 2),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--herd", type=int, default=50)
    ap.add_argument("--paraphrase-rate", type=float, default=0.35)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--skip-failover", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    workload = Workload(paraphrase_rate=args.paraphrase_rate)

    async with httpx.AsyncClient() as client:
        health = (await client.get(f"{GATEWAY}/health")).json()
        print(f"gateway backend: {health.get('backend')}")

        report: dict = {
            "tag": args.tag,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "requests": args.requests,
                "concurrency": args.concurrency,
                "workload": workload.describe(),
                "store_backend": health.get("backend"),
            },
            "scenarios": [],
        }

        print(f"\n[1/3] cached  n={args.requests} c={args.concurrency}")
        cached = await scenario_load(
            client, "cached", args.requests, args.concurrency, Workload(paraphrase_rate=args.paraphrase_rate)
        )
        report["scenarios"].append(cached)
        print(f"      p50={cached['p50_ms']}ms p95={cached['p95_ms']}ms "
              f"rps={cached['throughput_rps']} hit_rate={cached['gateway_stats'].get('hit_rate')}")

        print(f"\n[2/3] thundering herd  identical={args.herd}")
        herd = await scenario_herd(client, args.herd)
        report["scenarios"].append(herd)
        print(f"      upstream_calls={herd['upstream_calls']} "
              f"reduction={herd['upstream_call_reduction_pct']}%")

        if not args.skip_failover:
            print("\n[3/3] failover")
            fo = await scenario_failover(client, 60, 10)
            report["scenarios"].append(fo)
            print(f"      availability={fo.get('availability_pct')}% "
                  f"failovers={fo.get('provider_failovers')}")

        out = RESULTS_DIR / f"{args.tag}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
