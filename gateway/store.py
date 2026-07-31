"""
Shared key/value + lock backend.

Redis when reachable, in-process fallback otherwise. The fallback keeps the
gateway runnable (and the benchmark reproducible) on a single node without
Docker, while Redis is what makes cache and coalescing work *across* replicas.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from .config import settings

try:
    import redis.asyncio as aioredis
except Exception:  # pragma: no cover
    aioredis = None


class MemoryStore:
    """Single-process stand-in for Redis. Not shared across replicas."""

    backend = "memory"

    def __init__(self) -> None:
        self._kv: dict[str, tuple[str, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            hit = self._kv.get(key)
            if not hit:
                return None
            value, expires = hit
            if expires and expires < time.time():
                self._kv.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        async with self._lock:
            expires = time.time() + ttl_s if ttl_s else 0.0
            self._kv[key] = (value, expires)

    async def setnx(self, key: str, value: str, ttl_s: int) -> bool:
        async with self._lock:
            hit = self._kv.get(key)
            if hit:
                _, expires = hit
                if not expires or expires >= time.time():
                    return False
            self._kv[key] = (value, time.time() + ttl_s)
            return True

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._kv.pop(key, None)

    async def incr(self, key: str, amount: int = 1) -> int:
        async with self._lock:
            current = 0
            hit = self._kv.get(key)
            if hit:
                try:
                    current = int(hit[0])
                except ValueError:
                    current = 0
            current += amount
            self._kv[key] = (str(current), 0.0)
            return current

    async def close(self) -> None:
        self._kv.clear()


class RedisStore:
    backend = "redis"

    def __init__(self, client) -> None:
        self._r = client

    async def get(self, key: str) -> Optional[str]:
        return await self._r.get(key)

    async def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        if ttl_s:
            await self._r.setex(key, ttl_s, value)
        else:
            await self._r.set(key, value)

    async def setnx(self, key: str, value: str, ttl_s: int) -> bool:
        return bool(await self._r.set(key, value, nx=True, ex=ttl_s))

    async def delete(self, key: str) -> None:
        await self._r.delete(key)

    async def incr(self, key: str, amount: int = 1) -> int:
        return int(await self._r.incrby(key, amount))

    async def close(self) -> None:
        closer = getattr(self._r, "aclose", None) or self._r.close
        await closer()


async def build_store():
    """Prefer Redis; fall back to in-process unless REDIS_REQUIRED is set."""
    if aioredis is not None:
        try:
            client = aioredis.from_url(
                settings.redis_url, encoding="utf-8", decode_responses=True
            )
            await asyncio.wait_for(client.ping(), timeout=1.5)
            return RedisStore(client)
        except Exception:
            if settings.redis_required:
                raise
    if settings.redis_required:
        raise RuntimeError("Redis required but unreachable")
    return MemoryStore()
