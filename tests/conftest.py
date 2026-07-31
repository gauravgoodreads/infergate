"""
Pin runtime settings for the test suite.

`gateway.config` reads environment variables at import time, which is right for a
deployed process but makes tests depend on ambient shell state. This fixture
forces the flags every test assumes, then restores them, so results are identical
on a laptop and in CI.
"""
import pytest

from gateway.config import settings

_FORCED = {
    "cache_enabled": True,
    "semantic_enabled": True,
    "coalesce_enabled": True,
    "breaker_enabled": True,
    "limiter_enabled": True,
    "semantic_threshold": 0.78,
    "coalesce_poll_interval_s": 0.005,
    "upstream_timeout_s": 5.0,
}


@pytest.fixture(autouse=True)
def _pinned_settings():
    original = {k: getattr(settings, k) for k in _FORCED}
    for key, value in _FORCED.items():
        setattr(settings, key, value)
    yield
    for key, value in original.items():
        setattr(settings, key, value)
