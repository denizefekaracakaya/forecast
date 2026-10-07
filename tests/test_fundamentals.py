"""Tests for stock fundamentals and computed statistics."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from forecast.data.fundamentals import (
    compute_price_stats,
    fetch_fundamentals,
)


@pytest.fixture
def normalized_df() -> pd.DataFrame:
    n = 252
    close = 100 + np.arange(n, dtype=float) * 0.5
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "ds": dates,
        "ticker": "TEST",
        "market": "ABD (US)",
        "open": close - 0.5,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": np.full(n, 1_000_000, dtype=float),
        "source": "yahoo",
    })


class TestComputePriceStats:
    def test_returns_expected_keys(self, normalized_df: pd.DataFrame) -> None:
        stats = compute_price_stats(normalized_df)
        assert "last_close" in stats
        assert "daily_change" in stats
        assert "daily_change_pct" in stats
        assert "rsi" in stats
        assert "sma_20" in stats
        assert "sma_50" in stats
        assert "momentum_5" in stats
        assert "week52_low" in stats
        assert "week52_high" in stats
        assert "week52_position_pct" in stats

    def test_empty_dataframe_returns_empty(self) -> None:
        df = pd.DataFrame()
        stats = compute_price_stats(df)
        assert stats == {}

    def test_single_row_dataframe(self) -> None:
        df = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=1),
            "close": [100.0],
            "volume": [1_000_000],
            "open": [99.0],
            "high": [101.0],
            "low": [98.0],
        })
        stats = compute_price_stats(df)
        assert stats["last_close"] == 100.0
        assert stats["daily_change"] == 0.0
        assert stats["rsi"] == 50.0

    def test_two_rows_computes_daily_change(self) -> None:
        df = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=2, freq="B"),
            "close": [100.0, 105.0],
            "volume": [1_000_000, 1_200_000],
            "open": [99.0, 104.0],
            "high": [101.0, 106.0],
            "low": [98.0, 103.0],
        })
        stats = compute_price_stats(df)
        assert stats["last_close"] == 105.0
        assert stats["daily_change"] == 5.0
        assert stats["daily_change_pct"] == pytest.approx(5.0)
        # avg_volume_30 over 2 rows = (1M + 1.2M) / 2 = 1.1M
        # last_volume / avg = 1.2M / 1.1M - 1 = 9.09%
        assert stats["volume_vs_avg_pct"] == pytest.approx(9.0909, rel=1e-3)

    def test_sma_values(self, normalized_df: pd.DataFrame) -> None:
        stats = compute_price_stats(normalized_df)
        expected_sma20 = float(normalized_df["close"].tail(20).mean())
        assert stats["sma_20"] == pytest.approx(expected_sma20, rel=1e-3)

    def test_week52_position(self) -> None:
        close = np.linspace(50, 150, 252)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        df = pd.DataFrame({
            "ds": dates,
            "close": close,
            "volume": np.full(252, 1_000_000),
            "open": close,
            "high": close + 2,
            "low": close - 2,
        })
        stats = compute_price_stats(df)
        assert stats["week52_low"] == 50.0
        assert stats["week52_high"] == 150.0
        assert stats["week52_position_pct"] == pytest.approx(100.0, rel=1e-2)

    def test_no_open_high_low_falls_back_to_close(self) -> None:
        df = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=5, freq="B"),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1_000_000] * 5,
        })
        stats = compute_price_stats(df)
        assert stats["last_close"] == 104.0
        # when open column is missing, fallback is the close series -> last close
        assert stats["last_open"] == 104.0
        assert stats["last_high"] == 104.0


class TestFetchFundamentals:
    @patch("forecast.data.fundamentals.yf.Ticker")
    def test_returns_known_keys(self, mock_ticker_class) -> None:
        mock_ticker = mock_ticker_class.return_value
        mock_ticker.info = {
            "marketCap": 2_000_000_000_000,
            "trailingPE": 28.5,
            "beta": 1.15,
            "dividendYield": 0.0052,
            "fiftyTwoWeekHigh": 200.0,
            "fiftyTwoWeekLow": 150.0,
        }

        result = fetch_fundamentals("AAPL")
        assert result["marketCap"] == 2_000_000_000_000
        assert result["trailingPE"] == 28.5
        assert result["beta"] == 1.15
        assert result["fiftyTwoWeekHigh"] == 200.0

    @patch("forecast.data.fundamentals.yf.Ticker")
    def test_missing_keys_omitted(self, mock_ticker_class) -> None:
        mock_ticker = mock_ticker_class.return_value
        mock_ticker.info = {"marketCap": 1_000_000_000}

        result = fetch_fundamentals("TEST")
        assert "marketCap" in result
        assert "trailingPE" not in result
        assert "beta" not in result

    @patch("forecast.data.fundamentals.yf.Ticker")
    def test_api_error_returns_empty(self, mock_ticker_class) -> None:
        mock_ticker_class.side_effect = ConnectionError("API Error")

        result = fetch_fundamentals("FAIL")
        assert result == {}
