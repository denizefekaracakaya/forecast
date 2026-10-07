"""Price cache — in-memory dict (Redis drop-in later)."""

from __future__ import annotations

import pandas as pd


class PriceCache:
    def __init__(self) -> None:
        self._store: dict[str, pd.DataFrame] = {}

    def set(self, key: str, value: pd.DataFrame) -> None:
        self._store[key.upper().strip()] = value

    def get(self, key: str) -> pd.DataFrame | None:
        return self._store.get(key.upper().strip())

    def has(self, key: str) -> bool:
        return key.upper().strip() in self._store

    def clear(self) -> None:
        self._store.clear()

    def keys(self) -> list[str]:
        return list(self._store.keys())
