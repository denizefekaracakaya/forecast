"""Tests for the rate limiter."""

from __future__ import annotations

import time

import pytest

from forecast.data.rate_limiter import RateLimiter, shared_limiter


class TestRateLimiter:
    def test_no_delay_when_cold(self) -> None:
        limiter = RateLimiter(min_interval=0.5)
        t0 = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1

    def test_enforces_min_interval(self) -> None:
        limiter = RateLimiter(min_interval=0.3)
        limiter.wait()
        t0 = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.25

    def test_multiple_calls_respect_interval(self) -> None:
        limiter = RateLimiter(min_interval=0.1)
        t0 = time.monotonic()
        for _ in range(3):
            limiter.wait()
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.2

    def test_shared_limiter_singleton(self) -> None:
        a = shared_limiter()
        b = shared_limiter()
        assert a is b


class TestSharedLimiter:
    def test_uses_config_min_interval(self) -> None:
        limiter = shared_limiter()
        assert isinstance(limiter, RateLimiter)
