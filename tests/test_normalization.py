"""Tests for OHLCV normalization."""

from __future__ import annotations

import pandas as pd
import pytest

from forecast.data.normalization import normalize_ohlcv, NORMALIZED_COLUMNS


class TestNormalizeOHLCV:
    def test_normalize_multiindex(self, sample_ohlcv_multiindex: pd.DataFrame) -> None:
        result = normalize_ohlcv(sample_ohlcv_multiindex, "AAPL", source="yahoo")
        assert isinstance(result, pd.DataFrame)
        for col in NORMALIZED_COLUMNS:
            assert col in result.columns
        assert result["ticker"].iloc[0] == "AAPL"
        assert result["source"].iloc[0] == "yahoo"
        assert len(result) == 10

    def test_normalize_single_level(self, sample_ohlcv_single: pd.DataFrame) -> None:
        result = normalize_ohlcv(sample_ohlcv_single, "SPY", source="yahoo")
        assert isinstance(result, pd.DataFrame)
        for col in NORMALIZED_COLUMNS:
            assert col in result.columns
        assert result["ticker"].iloc[0] == "SPY"

    def test_market_resolution_bist(self) -> None:
        df = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = normalize_ohlcv(df, "THYAO.IS")
        assert result["market"].iloc[0] == "BIST (Türkiye)"

    def test_market_resolution_us(self) -> None:
        df = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = normalize_ohlcv(df, "AAPL")
        assert result["market"].iloc[0] == "ABD (US)"

    def test_empty_dataframe_raises(self) -> None:
        empty = pd.DataFrame()
        with pytest.raises(ValueError, match="Empty"):
            normalize_ohlcv(empty, "AAPL")

    def test_values_preserved(self, sample_ohlcv_single: pd.DataFrame) -> None:
        result = normalize_ohlcv(sample_ohlcv_single, "SPY")
        assert result["close"].iloc[0] == 100.0
        assert result["open"].iloc[0] == 99.0
        assert result["high"].iloc[0] == 101.0
        assert result["low"].iloc[0] == 98.0
        assert result["volume"].iloc[0] == 1_000_000
