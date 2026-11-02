"""Tests for the ingestion worker."""

from __future__ import annotations

from unittest.mock import MagicMock

from forecast.data.cache import PriceCache
from forecast.data.normalization import NORMALIZED_COLUMNS
from forecast.ingestion.worker import IngestionWorker


def test_run_once_caches_normalized_frames(sample_ohlcv_single) -> None:
    provider = MagicMock()
    provider.fetch_history.return_value = sample_ohlcv_single
    cache = PriceCache()

    results = IngestionWorker(provider, cache).run_once(["AAPL"])

    assert results == {"AAPL": True}
    cached = cache.get("AAPL")
    assert list(cached.columns) == NORMALIZED_COLUMNS
    assert cached["ticker"].iloc[0] == "AAPL"


def test_run_once_reports_failures() -> None:
    provider = MagicMock()
    provider.fetch_history.side_effect = RuntimeError("boom")
    cache = PriceCache()

    results = IngestionWorker(provider, cache).run_once(["AAPL"])

    assert results == {"AAPL": False}
    assert cache.get("AAPL") is None


def test_run_loop_respects_max_iterations(sample_ohlcv_single) -> None:
    provider = MagicMock()
    provider.fetch_history.return_value = sample_ohlcv_single
    worker = IngestionWorker(provider, PriceCache(), poll_interval=0)

    worker.run_loop(["AAPL", "SPY"], max_iterations=2)

    assert provider.fetch_history.call_count == 4
