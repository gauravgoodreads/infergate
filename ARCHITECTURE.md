# Architecture and design decisions

## The problem

In an LLM-backed application the upstream provider call dominates everything
else. It is two to three orders of magnitude slower than any local work, it is
the only metered component, and it is the component you control least - rate
limits, tail latency, and outages are all imposed on you.

So the design goal is not "make the gateway fast." A gateway that adds 1 ms to a
200 ms call is irrelevant either way. The goal is **to not make the upstream call
at all**, and when it must be made, to make it once and to make it survive
provider failure.

Each stage below removes a different class of redundant upstream call.

---

## Request path

```
POST /v1/complete
  |
  1. rate limit        token bucket per API key, budget shared via Redis
  |                    reject early: a 429 costs nothing downstream
  2. exact cache       SHA-256(model, normalised prompt, params) -> O(1) lookup
  |
  3. semantic cache    cosine similarity over prompt embeddings
  |                    only on exact miss; catches paraphrases
  4. coalescing        Redis SET NX lock; one leader calls upstream,
  |                    concurrent duplicates wait for its result
  5. provider pool     ordered failover, per-provider circuit breaker
  |
  response + cache write
```

Ordering is deliberate: each stage is more expensive than the one before it, so
the cheapest possible rejection or hit happens first. Rate limiting precedes
caching because a request that should not be served at all should not consume
cache lookups either.

---

## Decisions and trade-offs

### Two cache tiers instead of one

An exact-match cache is a hash lookup - effectively free, but it misses
"explain X" vs "Explain X?". A semantic cache catches those, but every lookup
costs an embedding plus a similarity scan, so running it first would tax the
common case to serve the rarer one.

Running exact first and semantic only on a miss means the frequent path stays
O(1) and the expensive path is only paid when it can actually help. On the
benchmark workload the split is roughly 70% exact / 30% semantic of all hits.

### Local hashing embeddings, not an embedding API

The semantic tier needs a vector per request. Calling a hosted embedding model
would add a network round trip and a per-request charge to the hot path - which
defeats the purpose, since the cache exists to remove exactly those two costs.

A character n-gram `HashingVectorizer` is used instead. It is:

- **stateless** - no fitted vocabulary, so replicas need nothing shared
- **deterministic** - the same prompt always yields the same vector
- **free and local** - sub-millisecond, no network

The trade-off is real and worth stating plainly: hashing n-grams capture surface
form, not deep meaning. It reliably catches rewordings, punctuation, casing, and
filler phrasing. It will *not* recognise that "How do I make my queries faster?"
and "What causes slow SELECTs?" want the same answer. For a cache that is the
right side of the trade - a conservative matcher that never serves a wrong answer
beats an aggressive one that occasionally does. The embedder is swappable behind
`embed()` for anyone who wants to pay for stronger recall.

### The similarity threshold is measured, not chosen

A false semantic hit is not a performance problem, it is a correctness bug: the
user gets an answer to a question they did not ask. So the threshold is derived
from data (`bench/calibrate.py`) rather than picked by feel.

Scoring 140 paraphrase pairs against 190 unrelated pairs shows clean separation:
unrelated pairs peak at 0.778, paraphrases average 0.873. At **0.78** precision
is 1.000 with recall 0.950 and zero false positives. At 0.76 a false positive
appears; above 0.80 precision is already saturated so raising it only discards
real hits. 0.78 is therefore the maximum-recall point subject to zero false
positives.

### Coalescing is cross-replica, and cannot deadlock

Caching alone does not help the cold-start burst: N identical requests arriving
before the first response is cached still produce N upstream calls. Under
load-balanced replicas an in-process lock does not fix this either, because the
duplicates land on different processes.

The lock therefore lives in Redis (`SET NX EX`). Two failure modes had to be
handled explicitly:

- **Leader crashes.** The lock carries a TTL, so it expires rather than wedging
  every follower until timeout.
- **Leader finishes without publishing** (upstream error). Followers notice the
  lock has disappeared, stop waiting, and issue their own call. The system
  degrades to baseline behaviour instead of dropping requests.

Measured effect: 50 identical concurrent requests against a cold cache produce
**1** upstream call instead of 50.

### Circuit breaker per provider, not per gateway

A single global breaker would be wrong: one provider failing says nothing about
the health of the others. Each provider owns its own breaker, so an open breaker
means "skip this provider," not "stop serving traffic."

The value is in what an open breaker *avoids*. Without it, every request to a
dead provider pays the full timeout before failing over - turning a provider
outage into a latency outage. With it, a dead provider is skipped in constant
time and traffic reaches the fallback immediately. Measured: 100% request
availability with the primary failing 100% of requests.

Half-open recovery requires N consecutive successful probes before closing, so a
flapping provider does not oscillate back into the rotation.

### Redis is an optimisation with a documented failure mode

Redis holds cache payloads, coalescing locks, and rate-limit buckets - it is what
makes those three features work *across* replicas rather than per-process.

The gateway still starts without it, degrading to an in-process store. That is a
deliberate developer-experience choice: tests and the benchmark run on a laptop
with no Docker. It is also a correctness hazard in production, because each
replica would then keep a private cache and a private rate-limit budget. Hence
`REDIS_REQUIRED=true`, which makes the process fail fast at startup instead of
silently serving degraded semantics.

### Bounded in-process vector index

The semantic shortlist is a NumPy matrix per replica with FIFO eviction, capped
at 20,000 vectors. A brute-force matrix multiply at that size is sub-millisecond,
so an approximate-nearest-neighbour service would add an operational dependency
and a network hop to save time that is not being spent. Payloads live in Redis,
so a hit on any replica resolves to the same cached response.

Past roughly 1e5 vectors this stops being true and the index should move to a
real ANN backend (pgvector or FAISS) - recorded here as a known scaling boundary
rather than discovered later.

---

## Benchmark methodology

Claims about performance are only as good as the control they are measured
against, so the harness enforces the comparison rather than asserting it.

**Control.** The same binary runs with `CACHE_ENABLED=false` and
`COALESCE_ENABLED=false`. The difference between runs is the feature under test,
not a different code path or a different machine.

**Isolation.** Redis is flushed and the gateway process restarted between control
and treatment. Without this, cached entries leak forward and the "baseline"
silently measures a warm cache. This is not hypothetical - it happened during
development when a port collision left the previous gateway serving, producing an
impossible 100% baseline hit rate. The orchestrator now owns process lifecycle
end to end so the mistake cannot recur.

**Repetition.** Percentile latency on a log-normal upstream varies materially
between runs (p50 reduction ranged 49.5-59.8% across three repetitions).
Headline figures are medians, and min-max is reported alongside.

**Instrumented upstream, not a live provider.** The mock counts every call it
receives, which is what makes the 97.9% upstream reduction *verified* rather than
inferred from the gateway's own bookkeeping. Its latency is drawn from a
log-normal distribution rather than a fixed sleep, because a constant-latency
upstream makes tail-latency results meaningless. A live provider would also make
the benchmark irreproducible and expose it to provider-side rate limits. Real
providers are supported at runtime via `GROQ_API_KEY`; they are simply not used
to generate numbers.

**Workload shape is declared.** 20 base prompts, Zipf-distributed popularity,
35% paraphrase rate - modelling FAQ-shaped traffic. Hit rate is a direct function
of prompt repetition, so this number is a property of the workload as much as of
the cache, and quoting it without the workload would be meaningless. The
workload-independent results are the thundering-herd collapse (50 -> 1) and the
failover availability.

---

## What this does not do

Stated explicitly, because a design doc that lists only strengths is not useful:

- **No streaming.** Token streaming and response caching conflict; supporting
  both means caching the assembled response while replaying it chunk-wise, which
  is not implemented.
- **No semantic cache invalidation.** Entries expire by TTL only. There is no way
  to say "the answer to this class of question changed."
- **Cost savings are modelled, not billed.** `/stats` multiplies tokens avoided
  by a configured per-1k rate. It is an estimate from a real token count, not a
  reconciled invoice.
- **Single-region.** No cross-region replication or cache locality strategy.
- **Rate limiting is per API key only.** No per-model or per-endpoint budgets.
