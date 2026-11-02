"""Price cache — thread-safe in-memory store with per-entry TTL (Redis drop-in later)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pandas as pd


class PriceCache:
    """In-memory DataFrame cache.

    Entries expire ``ttl_seconds`` after they are written. A ``ttl_seconds`` of
    ``None`` (or <= 0) disables expiry.
    """

    def __init__(
        self,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds if ttl_seconds and ttl_seconds > 0 else None
        self._clock = clock
        self._store: dict[str, tuple[float, pd.DataFrame]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(key: str) -> str:
        return key.upper().strip()

    def _expired(self, written_at: float) -> bool:
        return self._ttl is not None and self._clock() - written_at >= self._ttl

    def set(self, key: str, value: pd.DataFrame) -> None:
        with self._lock:
            self._store[self._key(key)] = (self._clock(), value)

    def get(self, key: str) -> pd.DataFrame | None:
        k = self._key(key)
        with self._lock:
            entry = self._store.get(k)
            if entry is None:
                return None
            written_at, value = entry
            if self._expired(written_at):
                del self._store[k]
                return None
            return value

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def keys(self) -> list[str]:
        with self._lock:
            return [k for k, (ts, _) in self._store.items() if not self._expired(ts)]
