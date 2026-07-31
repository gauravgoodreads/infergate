"""
Full benchmark orchestrator.

Starts the mock upstreams, then runs the gateway twice - once with caching and
coalescing disabled (the control) and once fully enabled - flushing Redis and
restarting the process between runs so neither result contaminates the other.
Emits a single merged report plus a markdown summary.

Run:  python -m bench.run_all --requests 2000 --concurrency 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
GATEWAY = "http://127.0.0.1:8080"
UPSTREAM = "http://127.0.0.1:9100"
FALLBACK = "http://127.0.0.1:9101"


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_http(url: str, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def flush_redis() -> str:
    try:
        import redis

        r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))
        r.flushdb()
        return "flushed"
    except Exception as exc:
        return f"skipped ({type(exc).__name__})"


def spawn(cmd: list[str], env: dict) -> subprocess.Popen:
    full_env = {**os.environ, **env}
    creation = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return subprocess.Popen(
        cmd, cwd=str(ROOT), env=full_env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=creation,
    )


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def start_gateway(cache: bool, coalesce: bool) -> subprocess.Popen:
    env = {
        "CACHE_ENABLED": "true" if cache else "false",
        "SEMANTIC_ENABLED": "true" if cache else "false",
        "COALESCE_ENABLED": "true" if coalesce else "false",
        "LIMITER_RATE_PER_S": "100000",
        "LIMITER_BURST": "200000",
    }
    proc = spawn(
        [sys.executable, "-m", "uvicorn", "gateway.main:app",
         "--host", "127.0.0.1", "--port", "8080", "--log-level", "warning"],
        env,
    )
    if not wait_http(f"{GATEWAY}/health"):
        stop(proc)
        raise RuntimeError("gateway failed to start")
    return proc


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--herd", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=3,
                    help="repetitions; headline numbers are the median across them")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    for i in range(1, args.repeat + 1):
        print(f"\n{'='*58}\nREPETITION {i} of {args.repeat}\n{'='*58}")
        runs.append(await single_run(args))

    aggregate = aggregate_runs(runs, args)
    (RESULTS / "benchmark.json").write_text(
        json.dumps({"runs": runs, "aggregate": aggregate}, indent=2), encoding="utf-8")
    (RESULTS / "SUMMARY.md").write_text(render_markdown(aggregate), encoding="utf-8")

    print(f"\n{'='*58}\nMEDIAN OF {args.repeat} RUNS\n{'='*58}")
    for k, v in aggregate["headline"].items():
        print(f"  {k}: {v}")
    print(f"\nwrote {RESULTS/'benchmark.json'} and SUMMARY.md")


async def single_run(args) -> dict:
    from .loadtest import scenario_failover, scenario_herd, scenario_load
    from .workload import Workload

    upstreams: list[subprocess.Popen] = []

    if port_free(9100):
        upstreams.append(spawn(
            [sys.executable, "-m", "bench.mock_upstream", "--port", "9100", "--mean-ms", "200"], {}))
    if port_free(9101):
        upstreams.append(spawn(
            [sys.executable, "-m", "bench.mock_upstream", "--port", "9101", "--mean-ms", "260"], {}))
    for url in (UPSTREAM, FALLBACK):
        if not wait_http(f"{url}/calls"):
            raise RuntimeError(f"mock upstream not up: {url}")

    report: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "herd_size": args.herd,
            "upstream_mean_ms": 200,
            "upstream_model": "log-normal latency, instrumented call counter",
        },
        "scenarios": [],
    }

    try:
        # ---------- control: caching + coalescing OFF ----------
        print(f"redis: {flush_redis()}")
        print("\n=== BASELINE (cache off, coalescing off) ===")
        gw = start_gateway(cache=False, coalesce=False)
        try:
            async with httpx.AsyncClient() as client:
                base = await scenario_load(
                    client, "baseline_no_cache", args.requests, args.concurrency,
                    Workload(paraphrase_rate=0.35))
                base_herd = await scenario_herd(client, args.herd)
                base_herd["scenario"] = "thundering_herd_no_coalescing"
            report["scenarios"] += [base, base_herd]
            print(f"  p50={base['p50_ms']}ms p95={base['p95_ms']}ms "
                  f"p99={base['p99_ms']}ms rps={base['throughput_rps']}")
            print(f"  upstream_calls={base['gateway_stats']['upstream_calls']}")
            print(f"  herd: {args.herd} identical -> "
                  f"{base_herd['upstream_calls']} upstream calls")
        finally:
            stop(gw)

        # ---------- treatment: full gateway ----------
        print(f"\nredis: {flush_redis()}")
        print("=== INFERGATE (semantic cache + coalescing on) ===")
        gw = start_gateway(cache=True, coalesce=True)
        try:
            async with httpx.AsyncClient() as client:
                cached = await scenario_load(
                    client, "infergate_cached", args.requests, args.concurrency,
                    Workload(paraphrase_rate=0.35))
                herd = await scenario_herd(client, args.herd)
                failover = await scenario_failover(client, 60, 10)
            report["scenarios"] += [cached, herd, failover]
            print(f"  p50={cached['p50_ms']}ms p95={cached['p95_ms']}ms "
                  f"p99={cached['p99_ms']}ms rps={cached['throughput_rps']}")
            print(f"  hit_rate={cached['gateway_stats']['hit_rate']} "
                  f"upstream_calls={cached['gateway_stats']['upstream_calls']}")
            print(f"  herd: {args.herd} identical -> {herd['upstream_calls']} "
                  f"upstream calls ({herd['upstream_call_reduction_pct']}% reduction)")
            print(f"  failover availability={failover.get('availability_pct')}%")
        finally:
            stop(gw)

        report["comparison"] = build_comparison(base, cached, base_herd, herd)
        print("\n--- run headline ---")
        for k, v in report["comparison"].items():
            print(f"  {k}: {v}")
        return report
    finally:
        for p in upstreams:
            stop(p)


def build_comparison(base: dict, cached: dict, base_herd: dict, herd: dict) -> dict:
    def drop(before: float, after: float) -> float | None:
        if not before:
            return None
        return round((1 - after / before) * 100, 1)

    b_up = base["gateway_stats"]["upstream_calls"]
    c_up = cached["gateway_stats"]["upstream_calls"]
    return {
        "baseline_p50_ms": base["p50_ms"],
        "baseline_p95_ms": base["p95_ms"],
        "baseline_p99_ms": base["p99_ms"],
        "baseline_rps": base["throughput_rps"],
        "cached_p50_ms": cached["p50_ms"],
        "cached_p95_ms": cached["p95_ms"],
        "cached_p99_ms": cached["p99_ms"],
        "cached_rps": cached["throughput_rps"],
        "baseline_upstream_calls": b_up,
        "cached_upstream_calls": c_up,
        "p50_latency_reduction_pct": drop(base["p50_ms"], cached["p50_ms"]),
        "p95_latency_reduction_pct": drop(base["p95_ms"], cached["p95_ms"]),
        "p99_latency_reduction_pct": drop(base["p99_ms"], cached["p99_ms"]),
        "throughput_gain_pct": round(
            (cached["throughput_rps"] / base["throughput_rps"] - 1) * 100, 1
        ) if base["throughput_rps"] else None,
        "upstream_call_reduction_pct": drop(b_up, c_up),
        "cache_hit_rate": cached["gateway_stats"]["hit_rate"],
        "tokens_avoided": cached["gateway_stats"]["tokens_saved"],
        "herd_upstream_calls_before": base_herd["upstream_calls"],
        "herd_upstream_calls_after": herd["upstream_calls"],
    }


def aggregate_runs(runs: list[dict], args) -> dict:
    """
    Reduce repetitions to medians.

    Percentile latencies on a log-normal upstream are noisy run to run, so a
    single run is not a trustworthy headline. The median across repetitions is,
    and reporting min/max alongside it makes the spread visible rather than hidden.
    """
    keys = [k for k in runs[0]["comparison"] if isinstance(
        runs[0]["comparison"][k], (int, float)) or runs[0]["comparison"][k] is None]

    summary: dict = {}
    for key in keys:
        values = [r["comparison"][key] for r in runs
                  if isinstance(r["comparison"].get(key), (int, float))]
        if not values:
            continue
        summary[key] = {
            "median": round(statistics.median(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    failovers = [
        s for r in runs for s in r["scenarios"] if s["scenario"] == "failover"
    ]
    availability = [f.get("availability_pct") for f in failovers
                    if f.get("availability_pct") is not None]

    def med(key: str):
        return summary.get(key, {}).get("median")

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repetitions": len(runs),
        "config": runs[0]["config"],
        "metrics": summary,
        "failover_availability_pct_min": min(availability) if availability else None,
        "headline": {
            "p50_latency_reduction": f"{med('p50_latency_reduction_pct')}%",
            "p95_latency_reduction": f"{med('p95_latency_reduction_pct')}%",
            "p99_latency_reduction": f"{med('p99_latency_reduction_pct')}%",
            "throughput_gain": f"{med('throughput_gain_pct')}%",
            "upstream_call_reduction": f"{med('upstream_call_reduction_pct')}%",
            "cache_hit_rate": med("cache_hit_rate"),
            "tokens_avoided": med("tokens_avoided"),
            "herd_upstream_calls": f"{med('herd_upstream_calls_before')} -> "
                                   f"{med('herd_upstream_calls_after')}",
            "failover_availability_min": f"{min(availability) if availability else 'n/a'}%",
        },
    }


def render_markdown(agg: dict) -> str:
    m = agg["metrics"]
    cfg = agg["config"]

    def cell(key: str, unit: str = "") -> str:
        row = m.get(key)
        if not row:
            return "n/a"
        return f"{row['median']}{unit}"

    def spread(key: str, unit: str = "") -> str:
        row = m.get(key)
        if not row:
            return "n/a"
        return f"{row['median']}{unit} ({row['min']}-{row['max']})"

    lines = [
        "# InferGate benchmark results",
        "",
        f"Generated {agg['timestamp']} - median of {agg['repetitions']} repetitions.",
        "",
        f"- {cfg['requests']} requests at concurrency {cfg['concurrency']} per repetition",
        f"- Upstream: instrumented mock provider, log-normal latency, "
        f"mean {cfg['upstream_mean_ms']} ms",
        "- Workload: Zipf-distributed prompt popularity, 35% paraphrase rate",
        "- Redis flushed and the gateway process restarted between control and treatment",
        "",
        "## Latency and throughput (median of repetitions)",
        "",
        "| Metric | Baseline (no cache) | InferGate | Reduction (median, min-max) |",
        "| --- | --- | --- | --- |",
        f"| p50 latency | {cell('baseline_p50_ms',' ms')} | {cell('cached_p50_ms',' ms')} | "
        f"{spread('p50_latency_reduction_pct','%')} |",
        f"| p95 latency | {cell('baseline_p95_ms',' ms')} | {cell('cached_p95_ms',' ms')} | "
        f"{spread('p95_latency_reduction_pct','%')} |",
        f"| p99 latency | {cell('baseline_p99_ms',' ms')} | {cell('cached_p99_ms',' ms')} | "
        f"{spread('p99_latency_reduction_pct','%')} |",
        f"| Throughput | {cell('baseline_rps',' rps')} | {cell('cached_rps',' rps')} | "
        f"{spread('throughput_gain_pct','%')} higher |",
        f"| Upstream calls | {cell('baseline_upstream_calls')} | "
        f"{cell('cached_upstream_calls')} | "
        f"{spread('upstream_call_reduction_pct','%')} fewer |",
        "",
        "p99 is the noisiest metric because it is driven by the tail of the "
        "log-normal upstream distribution; the min-max column shows that spread "
        "rather than hiding it.",
        "",
        "## Cache behaviour",
        "",
        f"- Hit rate: **{cell('cache_hit_rate')}**",
        f"- Tokens avoided: {cell('tokens_avoided')}",
        "",
        "## Request coalescing (thundering herd)",
        "",
        f"- {cfg['herd_size']} identical concurrent requests against a cold cache",
        f"- Without coalescing: {cell('herd_upstream_calls_before')} upstream calls",
        f"- With coalescing: {cell('herd_upstream_calls_after')} upstream call(s)",
        "",
        "## Provider failover",
        "",
        f"- Primary failing 100% of requests; worst-case request availability "
        f"across repetitions: **{agg.get('failover_availability_pct_min')}%**",
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python -m bench.calibrate              # semantic threshold sweep",
        "python -m bench.run_all --repeat 3     # control + treatment, median",
        "```",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    asyncio.run(main())
