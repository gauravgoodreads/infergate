"""
Cross-replica request coalescing (single-flight).

Without this, N identical concurrent prompts on a cold cache produce N upstream
calls - the thundering-herd problem. Here the first caller wins a Redis NX lock
and performs the single upstream call; the rest wait for the cache to fill and
read the shared result.

Correctness notes:
  * the lock carries a TTL, so a crashed leader cannot wedge followers
  * followers fall through to their own upstream call if the leader never
    publishes within the deadline, so a failed leader degrades to today's
    behaviour rather than dropping the request
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Optional

from .config import settings


class Coalescer:
    def __init__(self, store) -> None:
        self._store = store
        self.followers_served = 0
        self.leader_calls = 0

    async def run(
        self,
        key: str,
        do_work: Callable[[], Awaitable[dict]],
        read_cache: Callable[[], Awaitable[Optional[dict]]],
    ) -> tuple[dict, str]:
        """
        Returns (payload, role) where role is "leader" | "follower" | "direct".
        """
        if not settings.coalesce_enabled:
            self.leader_calls += 1
            return await do_work(), "direct"

        lock_key = f"ig:lock:{key}"
        got_lock = await self._store.setnx(
            lock_key, "1", settings.coalesce_lock_ttl_s
        )

        if got_lock:
            try:
                self.leader_calls += 1
                return await do_work(), "leader"
            finally:
                await self._store.delete(lock_key)

        # Follower: wait for the leader to publish into the cache.
        deadline = time.monotonic() + settings.upstream_timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(settings.coalesce_poll_interval_s)
            cached = await read_cache()
            if cached is not None:
                self.followers_served += 1
                return cached, "follower"
            if await self._store.get(lock_key) is None:
                break  # leader finished without publishing -> stop waiting

        self.leader_calls += 1
        return await do_work(), "direct"
