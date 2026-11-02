"""Tests for the FastAPI gateway."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from forecast.data.providers.yahoo import DataFetchError
from forecast.gateway.app import _cache, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    _cache.clear()


class TestHealth:
    def test_health_returns_ok(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["version"]


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


class TestPriceCacheIsolation:
    @patch("forecast.gateway.app._provider.fetch_history")
    def test_ranged_query_does_not_poison_default_cache(self, mock_fetch) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        full = pd.DataFrame({"Close": [1, 2, 3, 4, 5]}, index=dates)
        mock_fetch.return_value = full.iloc[:2]
        resp = client.get("/api/v1/prices/AAPL?start=2024-01-01&end=2024-01-02")
        assert resp.json()["count"] == 2

        mock_fetch.return_value = full
        resp = client.get("/api/v1/prices/AAPL")
        assert resp.json()["count"] == 5

    def test_invalid_date_returns_422(self) -> None:
        resp = client.get("/api/v1/prices/AAPL?start=yesterday")
        assert resp.status_code == 422

    def test_start_after_end_returns_422(self) -> None:
        resp = client.get("/api/v1/prices/AAPL?start=2024-02-01&end=2024-01-01")
        assert resp.status_code == 422


class TestNews:
    def test_ticker_news_serialization(self) -> None:
        from datetime import UTC, datetime

        from forecast.data.news.models import NewsItem

        item = NewsItem(
            id="n1",
            title="Apple beats",
            summary="s",
            source="Reuters",
            url="https://example.com/a",
            published_at=datetime(2024, 1, 2, tzinfo=UTC),
            ticker="AAPL",
        )
        with patch("forecast.gateway.app._news_provider.fetch_news", return_value=[item]):
            resp = client.get("/api/v1/news/AAPL")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["data"][0]["id"] == "n1"
        assert body["data"][0]["published_at"].startswith("2024-01-02")

    def test_unsupported_ticker_news_404(self) -> None:
        assert client.get("/api/v1/news/NOPE").status_code == 404
