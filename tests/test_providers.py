"""Tests for Yahoo Finance provider."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from forecast.data.providers.yahoo import (
    YahooProvider,
    DataFetchError,
    BACKOFF_SECONDS,
    MAX_RETRIES,
)


class TestYahooProvider:
    def test_backoff_length(self) -> None:
        assert len(BACKOFF_SECONDS) == MAX_RETRIES

    def test_backoff_values_positive(self) -> None:
        assert all(s > 0 for s in BACKOFF_SECONDS)

    @patch("forecast.data.providers.yahoo.yf.download")
    def test_fetch_history_success(self, mock_download) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        mock_df = pd.DataFrame({"Close": range(100, 105)}, index=dates)
        mock_download.return_value = mock_df

        provider = YahooProvider()
        result = provider.fetch_history("AAPL")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        mock_download.assert_called_once()

    @patch("forecast.data.providers.yahoo.yf.download")
    def test_fetch_history_empty_retries(self, mock_download) -> None:
        mock_download.return_value = pd.DataFrame()

        provider = YahooProvider()
        with pytest.raises(DataFetchError):
            provider.fetch_history("INVALID")
        assert mock_download.call_count == MAX_RETRIES

    @patch("forecast.data.providers.yahoo.yf.download")
    def test_fetch_history_exception_retries(self, mock_download) -> None:
        mock_download.side_effect = ConnectionError("timeout")

        provider = YahooProvider()
        with pytest.raises(DataFetchError, match="timeout"):
            provider.fetch_history("AAPL")
        assert mock_download.call_count == MAX_RETRIES

    @patch("forecast.data.providers.yahoo.yf.download")
    def test_fetch_history_succeeds_on_retry(self, mock_download) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        good_df = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)
        mock_download.side_effect = [ConnectionError("fail"), ConnectionError("fail"), good_df]

        provider = YahooProvider()
        result = provider.fetch_history("AAPL")
        assert len(result) == 3
        assert mock_download.call_count == 3

    @patch("forecast.data.providers.yahoo.yf.download")
    def test_fetch_and_normalize(self, mock_download) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        mock_df = pd.DataFrame({"Close": range(100, 105)}, index=dates)
        mock_download.return_value = mock_df

        provider = YahooProvider()
        result = provider.fetch_and_normalize("AAPL")

        assert "ds" in result.columns
        assert "close" in result.columns
        assert "ticker" in result.columns
        assert result["ticker"].iloc[0] == "AAPL"
