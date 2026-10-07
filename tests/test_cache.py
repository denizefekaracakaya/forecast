"""Tests for the price cache."""

from __future__ import annotations

import pandas as pd
import pytest

from forecast.data.cache import PriceCache


class TestPriceCache:
    def test_set_and_get(self) -> None:
        cache = PriceCache()
        df = pd.DataFrame({"close": [100.0]})
        cache.set("AAPL", df)
        result = cache.get("AAPL")
        assert result is not None
        assert result["close"].iloc[0] == 100.0

    def test_get_missing(self) -> None:
        cache = PriceCache()
        assert cache.get("NONEXISTENT") is None

    def test_has(self) -> None:
        cache = PriceCache()
        cache.set("SPY", pd.DataFrame())
        assert cache.has("SPY") is True
        assert cache.has("QQQ") is False

    def test_case_insensitive_key(self) -> None:
        cache = PriceCache()
        cache.set("aapl", pd.DataFrame({"close": [150.0]}))
        assert cache.has("AAPL") is True
        assert cache.get("AAPL") is not None

    def test_clear(self) -> None:
        cache = PriceCache()
        cache.set("A", pd.DataFrame())
        cache.set("B", pd.DataFrame())
        cache.clear()
        assert cache.keys() == []

    def test_keys(self) -> None:
        cache = PriceCache()
        cache.set("A", pd.DataFrame())
        cache.set("B", pd.DataFrame())
        keys = cache.keys()
        assert "A" in keys
        assert "B" in keys
