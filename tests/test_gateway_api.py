"""End-to-end API tests against a stubbed provider (no network)."""
import httpx
import pytest
from fastapi.testclient import TestClient

from gateway import main as gw
from gateway.providers import Completion


class StubProvider:
    name = "stub"

    def __init__(self) -> None:
        self.calls = 0

        class _AlwaysClosed:
            def allows(self):
                return True

            def record_success(self):
                pass

            def record_failure(self):
                pass

            def snapshot(self):
                return {"provider": "stub", "state": "closed", "failures": 0, "trips": 0}

        self.breaker = _AlwaysClosed()

    async def complete(self, client, model, prompt) -> Completion:
        self.calls += 1
        return Completion(text=f"stub:{prompt[:20]}", tokens=42,
                          provider=self.name, upstream_ms=1.0)


@pytest.fixture()
def client():
    with TestClient(gw.app) as c:
        stub = StubProvider()
        gw.app_state["pool"].providers = [stub]
        c.stub = stub
        c.post("/admin/reset")
        yield c


def test_health_reports_backend(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] in {"redis", "memory"}


def test_first_call_misses_then_second_hits(client):
    prompt = "unique gateway api test prompt about consistent hashing"
    first = client.post("/v1/complete", json={"prompt": prompt}).json()
    assert first["cached"] is False
    assert first["tokens"] == 42

    second = client.post("/v1/complete", json={"prompt": prompt}).json()
    assert second["cached"] is True
    assert second["cache_tier"] == "exact"


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.post("/v1/complete", json={"prompt": "metrics probe prompt"})
    text = client.get("/metrics").text
    assert "infergate_requests_total" in text
    assert "infergate_request_latency_seconds" in text


def test_stats_endpoint_tracks_hit_rate(client):
    prompt = "stats probe prompt about write ahead logs"
    client.post("/v1/complete", json={"prompt": prompt})
    client.post("/v1/complete", json={"prompt": prompt})
    stats = client.get("/stats").json()
    assert stats["requests"] >= 2
    assert stats["cache_hits"] >= 1
    assert 0.0 < stats["hit_rate"] <= 1.0


def test_empty_prompt_is_rejected(client):
    assert client.post("/v1/complete", json={"prompt": ""}).status_code == 422
