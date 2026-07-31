# InferGate benchmark results

Generated 2026-08-01T00:13:44 - median of 3 repetitions.

- 2000 requests at concurrency 50 per repetition
- Upstream: instrumented mock provider, log-normal latency, mean 200 ms
- Workload: Zipf-distributed prompt popularity, 35% paraphrase rate
- Redis flushed and the gateway process restarted between control and treatment

## Latency and throughput (median of repetitions)

| Metric | Baseline (no cache) | InferGate | Reduction (median, min-max) |
| --- | --- | --- | --- |
| p50 latency | 238.34 ms | 99.6 ms | 58.5% (49.5-59.8) |
| p95 latency | 393.93 ms | 187.75 ms | 53.4% (52.3-57.5) |
| p99 latency | 485.18 ms | 385.99 ms | 20.4% (14.4-27.4) |
| Throughput | 196.23 rps | 421.5 rps | 116.5% (81.8-123.9) higher |
| Upstream calls | 2000 | 43 | 97.9% (97.6-97.9) fewer |

p99 is the noisiest metric because it is driven by the tail of the log-normal upstream distribution; the min-max column shows that spread rather than hiding it.

## Cache behaviour

- Hit rate: **0.96**
- Tokens avoided: 124525

## Request coalescing (thundering herd)

- 50 identical concurrent requests against a cold cache
- Without coalescing: 50 upstream calls
- With coalescing: 1 upstream call(s)

## Provider failover

- Primary failing 100% of requests; worst-case request availability across repetitions: **100.0%**

## Reproduce

```bash
pip install -r requirements.txt
python -m bench.calibrate              # semantic threshold sweep
python -m bench.run_all --repeat 3     # control + treatment, median
```
