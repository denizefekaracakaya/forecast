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


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestPriceCacheTTL:
    def test_entry_expires_after_ttl(self) -> None:
        clock = FakeClock()
        cache = PriceCache(ttl_seconds=10, clock=clock)
        cache.set("AAPL", pd.DataFrame({"close": [1.0]}))
        clock.now = 9.9
        assert cache.get("AAPL") is not None
        clock.now = 10.0
        assert cache.get("AAPL") is None
        assert cache.keys() == []

    def test_rewrite_refreshes_ttl(self) -> None:
        clock = FakeClock()
        cache = PriceCache(ttl_seconds=10, clock=clock)
        cache.set("AAPL", pd.DataFrame())
        clock.now = 8
        cache.set("AAPL", pd.DataFrame())
        clock.now = 15
        assert cache.has("AAPL")

    @pytest.mark.parametrize("ttl", [None, 0, -1])
    def test_no_ttl_never_expires(self, ttl) -> None:
        clock = FakeClock()
        cache = PriceCache(ttl_seconds=ttl, clock=clock)
        cache.set("AAPL", pd.DataFrame())
        clock.now = 1e9
        assert cache.has("AAPL")
