"""
Benchmark workload generator.

Real LLM traffic is heavily skewed: a small set of popular prompts accounts for
most requests, and many of those arrive as paraphrases rather than byte-identical
strings. A uniform-random workload would understate caching; an all-identical
workload would overstate it. This generator models both effects explicitly:

  * prompt popularity follows a Zipf distribution over `n_unique` base prompts
  * a `paraphrase_rate` share of requests are reworded variants of a base prompt,
    which only the semantic tier can catch
"""
from __future__ import annotations

import random

BASE_PROMPTS = [
    "Explain the difference between a process and a thread",
    "What is the time complexity of binary search and why",
    "Summarise how HTTP caching headers work",
    "How does a database index speed up queries",
    "Explain eventual consistency in distributed systems",
    "What is the CAP theorem",
    "Describe how TLS handshake works",
    "What causes the N+1 query problem in ORMs",
    "Explain how a bloom filter works",
    "What is the difference between TCP and UDP",
    "How does garbage collection work in the JVM",
    "Explain what a circuit breaker does in microservices",
    "What is idempotency in REST APIs",
    "Describe how Kafka partitions provide ordering",
    "Explain the difference between latency and throughput",
    "How does Redis implement expiry of keys",
    "What is a write-ahead log used for",
    "Explain vector similarity search in one paragraph",
    "What is the difference between authn and authz",
    "How does connection pooling improve performance",
]

_PARAPHRASE_TEMPLATES = [
    "{p}?",
    "Could you explain: {p}",
    "I want to understand {p}",
    "In simple terms, {p}",
    "{p} - explain briefly",
    "Please describe {p} for a beginner",
    "Help me understand {p}",
]


def _zipf_weights(n: int, s: float) -> list[float]:
    raw = [1.0 / ((i + 1) ** s) for i in range(n)]
    total = sum(raw)
    return [r / total for r in raw]


class Workload:
    def __init__(
        self,
        n_unique: int = 20,
        zipf_s: float = 1.1,
        paraphrase_rate: float = 0.35,
        seed: int = 1337,
    ) -> None:
        self.prompts = BASE_PROMPTS[:n_unique]
        self.weights = _zipf_weights(len(self.prompts), zipf_s)
        self.paraphrase_rate = paraphrase_rate
        self.rng = random.Random(seed)

    def next_prompt(self) -> str:
        base = self.rng.choices(self.prompts, weights=self.weights, k=1)[0]
        if self.rng.random() < self.paraphrase_rate:
            template = self.rng.choice(_PARAPHRASE_TEMPLATES)
            return template.format(p=base[0].lower() + base[1:])
        return base

    def describe(self) -> dict:
        return {
            "unique_base_prompts": len(self.prompts),
            "zipf_s": self.weights and round(1.1, 2),
            "paraphrase_rate": self.paraphrase_rate,
        }
