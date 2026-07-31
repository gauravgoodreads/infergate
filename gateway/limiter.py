"""
Token-bucket rate limiter, keyed per API key and shared through the store so
the budget is global across replicas rather than per-process.
"""
from __future__ import annotations

import time

from .config import settings


class TokenBucket:
    def __init__(self, store) -> None:
        self._store = store
        self.rejected = 0

    async def allow(self, api_key: str, cost: int = 1) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        if not settings.limiter_enabled:
            return True, 0.0

        key = f"ig:rl:{api_key}"
        now = time.time()
        raw = await self._store.get(key)

        if raw:
            tokens_s, ts_s = raw.split("|")
            tokens, last = float(tokens_s), float(ts_s)
        else:
            tokens, last = float(settings.limiter_burst), now

        # refill
        tokens = min(
            float(settings.limiter_burst),
            tokens + (now - last) * settings.limiter_rate_per_s,
        )

        if tokens >= cost:
            tokens -= cost
            await self._store.set(key, f"{tokens}|{now}", 3600)
            return True, 0.0

        self.rejected += 1
        deficit = cost - tokens
        retry_after = deficit / max(settings.limiter_rate_per_s, 1e-6)
        await self._store.set(key, f"{tokens}|{now}", 3600)
        return False, retry_after
