"""Tests for Prophet model helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.forecasting.model import (
    ModelError,
    _mape,
    _rmse,
    currency_symbol,
    is_supported_ticker,
    market_of,
    naive_baseline_metrics,
    prepare_prophet_df,
    validate_ticker,
)


class TestHelpers:
    def test_is_supported_ticker(self) -> None:
        assert is_supported_ticker("AAPL") is True
        assert is_supported_ticker("THYAO.IS") is True
        assert is_supported_ticker("INVALID") is False

    def test_validate_ticker_supported(self) -> None:
        assert validate_ticker("aapl") == "AAPL"

    def test_validate_ticker_unsupported(self) -> None:
        with pytest.raises(ValueError, match="desteklenmiyor"):
            validate_ticker("ZYXW")

    def test_currency_symbol_bist(self) -> None:
        assert currency_symbol("THYAO.IS") == "₺"

    def test_currency_symbol_us(self) -> None:
        assert currency_symbol("AAPL") == "$"

    def test_market_of_bist(self) -> None:
        assert "BIST" in market_of("GARAN.IS")

    def test_market_of_us(self) -> None:
        assert "ABD" in market_of("MSFT")

    def test_mape_zero_actual(self) -> None:
        result = _mape(np.array([0, 0]), np.array([1, 2]))
        assert np.isnan(result)

    def test_mape_normal(self) -> None:
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 180.0])
        result = _mape(actual, predicted)
        expected = (abs(10 / 100) + abs(-20 / 200)) / 2 * 100
        assert result == pytest.approx(expected)

    def test_rmse(self) -> None:
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 180.0])
        result = _rmse(actual, predicted)
        expected = np.sqrt((10**2 + (-20) ** 2) / 2)
        assert result == pytest.approx(expected)


class TestNaiveBaseline:
    def test_short_df_returns_nan(self) -> None:
        df = pd.DataFrame({"y": [100.0]})
        result = naive_baseline_metrics(df)
        assert np.isnan(result["mape"])
        assert np.isnan(result["rmse"])

    def test_baseline_matches_expected(self) -> None:
        df = pd.DataFrame({"y": [100.0, 110.0, 121.0]})
        result = naive_baseline_metrics(df)
        assert result["mape"] == pytest.approx(
            (abs(110 - 100) / 110 + abs(121 - 110) / 121) / 2 * 100
        )


class TestPrepareProphetDF:
    def test_has_expected_columns(self, sample_ohlcv_single: pd.DataFrame) -> None:
        from forecast.data.normalization import normalize_ohlcv

        normalized = normalize_ohlcv(sample_ohlcv_single, "SPY")
        result = prepare_prophet_df(normalized)
        for col in ("ds", "y", "volume", "rolling_mean_7", "rsi"):
            assert col in result.columns

    def test_y_matches_close(self, sample_ohlcv_single: pd.DataFrame) -> None:
        from forecast.data.normalization import normalize_ohlcv

        normalized = normalize_ohlcv(sample_ohlcv_single, "SPY")
        result = prepare_prophet_df(normalized)
        assert result["y"].iloc[0] == 100.0
