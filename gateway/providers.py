"""
Upstream provider pool with circuit-breaker-guarded ordered failover.

Providers are attempted in configured order. A provider whose breaker is open is
skipped without paying its timeout. `Groq` speaks the OpenAI-compatible chat
completions API; `HttpProvider` targets the instrumented benchmark upstream.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from .breaker import CircuitBreaker
from .config import settings


class UpstreamError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    tokens: int
    provider: str
    upstream_ms: float


class HttpProvider:
    """Generic JSON upstream: {prompt, model} -> {text, tokens}."""

    def __init__(self, name: str, url: str) -> None:
        self.name = name
        self.url = url
        self.breaker = CircuitBreaker(name)
        self.calls = 0

    async def complete(self, client: httpx.AsyncClient, model: str, prompt: str) -> Completion:
        started = time.perf_counter()
        self.calls += 1
        resp = await client.post(
            self.url,
            json={"model": model, "prompt": prompt},
            timeout=settings.upstream_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return Completion(
            text=data.get("text", ""),
            tokens=int(data.get("tokens", 0)),
            provider=self.name,
            upstream_ms=(time.perf_counter() - started) * 1000.0,
        )


class GroqProvider:
    """OpenAI-compatible chat completions (Groq)."""

    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        self.api_key = api_key
        self.model = model
        self.breaker = CircuitBreaker(self.name)
        self.calls = 0

    async def complete(self, client: httpx.AsyncClient, model: str, prompt: str) -> Completion:
        started = time.perf_counter()
        self.calls += 1
        resp = await client.post(
            self.URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model or self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=settings.upstream_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"],
            tokens=int(usage.get("total_tokens", 0)),
            provider=self.name,
            upstream_ms=(time.perf_counter() - started) * 1000.0,
        )


class ProviderPool:
    def __init__(self, providers: list) -> None:
        self.providers = providers
        self.failovers = 0

    @classmethod
    def from_env(cls) -> "ProviderPool":
        key = os.getenv("GROQ_API_KEY", "").strip()
        if key:
            return cls([
                GroqProvider(key),
                HttpProvider("fallback", settings.upstream_fallback_url),
            ])
        return cls([
            HttpProvider("primary", settings.upstream_url),
            HttpProvider("secondary", settings.upstream_fallback_url),
        ])

    async def complete(self, client: httpx.AsyncClient, model: str, prompt: str) -> Completion:
        last_error: Exception | None = None
        for i, provider in enumerate(self.providers):
            if not provider.breaker.allows():
                continue
            try:
                result = await provider.complete(client, model, prompt)
                provider.breaker.record_success()
                if i > 0:
                    self.failovers += 1
                return result
            except Exception as exc:  # noqa: BLE001 - breaker treats all failures alike
                provider.breaker.record_failure()
                last_error = exc
        raise UpstreamError(f"all providers unavailable: {last_error}")

    def snapshot(self) -> list[dict]:
        return [
            {**p.breaker.snapshot(), "upstream_calls": p.calls}
            for p in self.providers
        ]
