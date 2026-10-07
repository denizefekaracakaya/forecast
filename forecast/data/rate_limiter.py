"""Simple rate limiter for external API calls."""

from __future__ import annotations

import time
import threading


class RateLimiter:
    def __init__(self, min_interval: float = 2.0) -> None:
        self._min_interval = min_interval
        self._last_call: float = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                sleep_for = self._min_interval - elapsed
                time.sleep(sleep_for)
            self._last_call = time.monotonic()


_shared_limiter: RateLimiter | None = None


def shared_limiter(min_interval: float | None = None) -> RateLimiter:
    global _shared_limiter
    if _shared_limiter is None:
        from forecast.config import settings

        interval = min_interval if min_interval is not None else settings.YAHOO_RATE_MIN_INTERVAL
        _shared_limiter = RateLimiter(interval)
    return _shared_limiter
