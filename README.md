# InferGate

A distributed LLM inference gateway that removes redundant work from the most
expensive component in an AI request path: the upstream provider call.

It sits in front of any OpenAI-compatible provider and applies four independent
optimisations - a two-tier response cache, cross-replica request coalescing,
per-tenant rate limiting, and circuit-breaker-guarded provider failover.

Every number below was produced by `python -m bench.run_all` against a control
run with caching and coalescing disabled. The harness is in this repo.

---

## Measured results

Median of 3 repetitions, 2,000 requests at concurrency 50 each. Upstream is an
instrumented provider with log-normal latency (mean 200 ms). Redis is flushed and
the gateway process restarted between the control and treatment runs, so neither
contaminates the other.

| Metric | Baseline (no cache) | InferGate | Reduction (median, min-max) |
| --- | --- | --- | --- |
| p50 latency | 238.34 ms | 99.60 ms | **58.5%** (49.5-59.8) |
| p95 latency | 393.93 ms | 187.75 ms | **53.4%** (52.3-57.5) |
| p99 latency | 485.18 ms | 385.99 ms | 20.4% (14.4-27.4) |
| Throughput | 196.23 rps | 421.50 rps | **116.5% higher** (81.8-123.9) |
| Upstream provider calls | 2,000 | 43 | **97.9% fewer** (97.6-97.9) |

Cache behaviour: **96.0% hit rate**, **124,525 tokens avoided** per run.

Percentile latency is noisy on a log-normal upstream, so single runs are not a
trustworthy headline - hence the median across repetitions, with the min-max
spread shown rather than hidden. p99 is the noisiest metric because it is driven
entirely by the upstream tail, which caching cannot compress.

**Thundering herd.** 50 identical prompts fired simultaneously at a cold cache:

| | Upstream calls |
| --- | --- |
| Without coalescing | 50 |
| With coalescing | **1** |

**Failover.** With the primary provider returning errors on 100% of requests,
the breaker opened and traffic shifted to the secondary with **100% request
availability** maintained in every repetition.

Raw output: [`bench/results/benchmark.json`](bench/results/benchmark.json),
[`bench/results/SUMMARY.md`](bench/results/SUMMARY.md).

### Why the hit rate is high

The benchmark workload models FAQ-shaped traffic: 20 base prompts with
Zipf-distributed popularity and a 35% paraphrase rate. Hit rate is a function of
prompt repetition, so a workload of entirely unique prompts would approach 0%.
The number that does not depend on workload shape is the **thundering-herd
result** (50 -> 1) and the **latency reduction on cache misses**, which comes from
coalescing rather than caching.

---

## The semantic threshold is calibrated, not guessed

A false semantic cache hit serves the wrong answer to a user, so the cosine
threshold is chosen from measured data. `bench/calibrate.py` scores 140
paraphrase pairs (should hit) against 190 unrelated pairs (must never hit):

| Threshold | Precision | Recall | False positives |
| --- | --- | --- | --- |
| 0.70 | 0.986 | 1.000 | 2 |
| 0.76 | 0.993 | 0.993 | 1 |
| **0.78** | **1.000** | **0.950** | **0** |
| 0.80 | 1.000 | 0.914 | 0 |
| 0.84 | 1.000 | 0.779 | 0 |
| 0.92 | 1.000 | 0.179 | 0 |

The distributions separate cleanly - unrelated pairs top out at 0.778 while
paraphrases average 0.873 - so **0.78 is the highest-recall point that still
produces zero false positives: 100% precision at 95.0% recall**. Below it,
unrelated prompts begin to collide; above it, precision is already saturated and
each increment only discards real hits.

Reproduce: `python -m bench.calibrate`

---

## Architecture

```
                       +--------------+
   client ------------>|    Nginx     |  least_conn, keepalive
                       +------+-------+
                              |
        +---------------------+---------------------+
        v                     v                     v
   +---------+          +---------+           +---------+
   |gateway 1|          |gateway 2|           |gateway 3|   stateless replicas
   +----+----+          +----+----+           +----+----+
        +---------------+----+---------------------+
                        v
                  +----------+
                  |  Redis   |  cache payloads, coalescing locks, rate buckets
                  +----------+
                        |
                        v
        +-------------------------------+
        |  provider pool (ordered)      |
        |  primary --breaker--> fallback|
        +-------------------------------+
```

Request path, cheapest stage first:

1. **Rate limit** - token bucket per API key, budget shared through Redis so the
   limit is global rather than per-replica.
2. **Exact cache** - SHA-256 over (model, normalised prompt, params). O(1).
3. **Semantic cache** - cosine similarity over prompt embeddings, catching
   paraphrases the exact tier misses. Only runs on an exact miss.
4. **Coalescing** - the first caller for a key wins a Redis `SET NX` lock and
   makes the single upstream call; concurrent callers wait for the result.
5. **Provider failover** - ordered pool, each provider guarded by its own
   circuit breaker so a dead provider is skipped without paying its timeout.

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions and trade-offs.

---

## Quick start

### Docker (full stack: Nginx + 3 replicas + Redis + Prometheus)

```bash
docker compose up --build
curl localhost:8080/health
```

### Local

```bash
pip install -r requirements.txt

python -m bench.mock_upstream --port 9100 --mean-ms 200 &
python -m bench.mock_upstream --port 9101 --mean-ms 260 &
uvicorn gateway.main:app --port 8080
```

Point it at a real provider by exporting a key - the gateway then speaks the
OpenAI-compatible chat completions API and keeps the fallback as secondary:

```bash
export GROQ_API_KEY=...
```

### Reproduce the benchmark

```bash
python -m bench.calibrate           # threshold sweep
python -m bench.run_all --repeat 3  # control + treatment, writes bench/results/
pytest                              # 21 tests
```

`run_all` manages the whole experiment itself: it starts both mock upstreams,
runs the control with caching and coalescing disabled, flushes Redis, restarts
the gateway with them enabled, repeats the pair N times, and writes the medians.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/complete` | Completion. Returns `cached`, `cache_tier`, `similarity`, `gateway_ms`. |
| `GET` | `/health` | Liveness, store backend, per-provider breaker state. |
| `GET` | `/stats` | Hit rate, tier split, upstream calls avoided, tokens saved. |
| `GET` | `/metrics` | Prometheus exposition. |
| `POST` | `/admin/reset` | Reset counters (benchmark isolation). |

```bash
curl -X POST localhost:8080/v1/complete \
  -H 'content-type: application/json' \
  -H 'x-api-key: demo' \
  -d '{"prompt":"Explain eventual consistency"}'
```

```json
{
  "text": "...",
  "tokens": 61,
  "provider": "primary",
  "cached": true,
  "cache_tier": "semantic",
  "similarity": 0.9717,
  "gateway_ms": 1.334
}
```

---

## Configuration

All settings are environment variables (see `gateway/config.py`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Shared state. Falls back to in-process if unreachable. |
| `REDIS_REQUIRED` | `false` | Set `true` in production to fail fast instead of degrading. |
| `SEMANTIC_THRESHOLD` | `0.80` | Cosine threshold; see calibration above. |
| `CACHE_TTL_S` | `3600` | Cached response lifetime. |
| `BREAKER_FAIL_THRESHOLD` | `5` | Consecutive failures before opening. |
| `LIMITER_RATE_PER_S` | `200` | Token refill rate per API key. |

---

## Engineering notes

**Embeddings are local and stateless.** Prompt vectors come from a character
n-gram `HashingVectorizer`, not an embedding API. Calling a remote embedding
model on every request would add a network round trip to the hot path and a
per-request cost - erasing the latency and spend the cache exists to save.
Hashing is also stateless, so replicas need no shared vocabulary.

**Redis is an optimisation, not a hard dependency.** The gateway degrades to an
in-process store when Redis is unreachable, which keeps local development and the
benchmark runnable without Docker. Redis is what makes the cache and coalescing
work *across* replicas, so `REDIS_REQUIRED=true` is correct for production.

**Coalescing followers cannot deadlock.** The lock carries a TTL, so a crashed
leader cannot wedge waiters. If the leader never publishes within the deadline,
followers fall through to their own upstream call - degrading to baseline
behaviour rather than dropping the request.

**The benchmark uses a mock upstream deliberately.** A live provider would make
results irreproducible (network variance, provider-side rate limits) and would
not expose a trustworthy call counter. The mock counts every call it receives,
which is how the 97.7% upstream reduction is verified rather than inferred.

---

## Tests

21 tests covering cache key normalisation, semantic hit/miss boundaries, index
eviction, all four circuit-breaker transitions, coalescing collapse, per-tenant
rate-limit isolation, and the HTTP surface.

```bash
pytest
```

## Licence

MIT
