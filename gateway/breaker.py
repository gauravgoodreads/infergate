"""
Per-provider circuit breaker (closed -> open -> half-open -> closed).

Purpose: stop paying the full upstream timeout on every request once a provider
is clearly down, and fail over to the next provider immediately instead.
"""
from __future__ import annotations

import time
from enum import Enum

from .config import settings


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.state = State.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self.half_open_successes = 0
        self.trips = 0

    def allows(self) -> bool:
        if not settings.breaker_enabled:
            return True
        if self.state is State.CLOSED:
            return True
        if self.state is State.OPEN:
            if time.monotonic() - self.opened_at >= settings.breaker_reset_timeout_s:
                self.state = State.HALF_OPEN
                self.half_open_successes = 0
                return True
            return False
        return True  # HALF_OPEN admits probe traffic

    def record_success(self) -> None:
        if self.state is State.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= settings.breaker_half_open_probes:
                self._close()
        else:
            self._close()

    def record_failure(self) -> None:
        self.failures += 1
        if self.state is State.HALF_OPEN:
            self._open()
        elif self.failures >= settings.breaker_fail_threshold:
            self._open()

    def _open(self) -> None:
        if self.state is not State.OPEN:
            self.trips += 1
        self.state = State.OPEN
        self.opened_at = time.monotonic()

    def _close(self) -> None:
        self.state = State.CLOSED
        self.failures = 0
        self.half_open_successes = 0

    def snapshot(self) -> dict:
        return {
            "provider": self.name,
            "state": self.state.value,
            "failures": self.failures,
            "trips": self.trips,
        }
