"""Tests for the FastAPI gateway."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from forecast.gateway.app import app, _cache
from forecast.data.providers.yahoo import DataFetchError

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    _cache.clear()


class TestHealth:
    def test_health_returns_ok(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestMarkets:
    def test_list_markets(self) -> None:
        resp = client.get("/api/v1/markets")
        assert resp.status_code == 200
        data = resp.json()
        assert "BIST (Türkiye)" in data
        assert "ABD (US)" in data


class TestPrices:
    @patch("forecast.gateway.app._provider.fetch_history")
    def test_get_prices_supported_ticker(self, mock_fetch) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        mock_fetch.return_value = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)

        resp = client.get("/api/v1/prices/AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "AAPL"
        assert data["source"] == "yahoo"
        assert data["count"] == 3

    @patch("forecast.gateway.app._provider.fetch_history")
    def test_get_prices_with_date_range(self, mock_fetch) -> None:
        dates = pd.date_range("2024-06-01", periods=2, freq="B")
        mock_fetch.return_value = pd.DataFrame({"Close": [150, 151]}, index=dates)

        resp = client.get("/api/v1/prices/MSFT?start=2024-06-01&end=2024-06-05")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "MSFT"
        assert data["count"] == 2
        mock_fetch.assert_called_with("MSFT", start="2024-06-01", end="2024-06-05")

    def test_get_prices_unsupported_ticker(self) -> None:
        resp = client.get("/api/v1/prices/INVALID")
        assert resp.status_code == 404

    @patch("forecast.gateway.app._provider.fetch_history")
    def test_get_prices_provider_error(self, mock_fetch) -> None:
        mock_fetch.side_effect = DataFetchError("Yahoo error")

        resp = client.get("/api/v1/prices/GOOGL")
        assert resp.status_code == 502
        assert "Yahoo error" in resp.json()["detail"]

    @patch("forecast.gateway.app._provider.fetch_history")
    def test_get_prices_uses_cache(self, mock_fetch) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        mock_fetch.return_value = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)

        resp1 = client.get("/api/v1/prices/AAPL")
        assert resp1.status_code == 200
        call_count_after_first = mock_fetch.call_count

        resp2 = client.get("/api/v1/prices/AAPL")
        assert resp2.status_code == 200
        assert mock_fetch.call_count == call_count_after_first
