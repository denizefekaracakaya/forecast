"""NewsAPI provider — fetches stock news via newsapi.org."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from forecast.config import settings

logger = logging.getLogger(__name__)


@dataclass
class NewsAPIResult:
    title: str
    description: str
    source_name: str
    url: str
    published_at: datetime
    ticker: str


class NewsAPIProvider:
    def __init__(self) -> None:
        self._api_key = settings.NEWSAPI_API_KEY
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            return None
        try:
            from newsapi import NewsApiClient
            self._client = NewsApiClient(api_key=self._api_key)
        except ImportError:
            logger.warning("newsapi-python not installed")
            return None
        return self._client

    def fetch_news(self, ticker: str, max_results: int = 10) -> list[NewsAPIResult]:
        client = self._ensure_client()
        if client is None:
            return []

        company_name = ticker.replace(".IS", "")
        try:
            response = client.get_everything(
                q=f"({company_name}) AND (stock OR share OR market OR earnings)",
                language="en",
                sort_by="publishedAt",
                page_size=min(max_results, 100),
            )
        except Exception as exc:
            logger.warning("NewsAPI fetch failed for %s: %s", ticker, exc)
            return []

        articles = response.get("articles", [])
        results: list[NewsAPIResult] = []
        for art in articles[:max_results]:
            if not isinstance(art, dict):
                continue
            pub_str = art.get("publishedAt", "")
            pub_date = _parse_iso(pub_str) if pub_str else datetime.now(timezone.utc)
            results.append(
                NewsAPIResult(
                    title=art.get("title", "") or "",
                    description=art.get("description", "") or "",
                    source_name=(art.get("source") or {}).get("name", "") or "",
                    url=art.get("url", "") or "",
                    published_at=pub_date,
                    ticker=ticker,
                )
            )
        return results

    def fetch_market_news(self, max_results: int = 20) -> list[NewsAPIResult]:
        client = self._ensure_client()
        if client is None:
            return []

        try:
            response = client.get_top_headlines(
                category="business",
                language="en",
                page_size=min(max_results, 100),
            )
        except Exception as exc:
            logger.warning("NewsAPI market news fetch failed: %s", exc)
            return []

        articles = response.get("articles", [])
        results: list[NewsAPIResult] = []
        for art in articles[:max_results]:
            if not isinstance(art, dict):
                continue
            pub_str = art.get("publishedAt", "")
            pub_date = _parse_iso(pub_str) if pub_str else datetime.now(timezone.utc)
            results.append(
                NewsAPIResult(
                    title=art.get("title", "") or "",
                    description=art.get("description", "") or "",
                    source_name=(art.get("source") or {}).get("name", "") or "",
                    url=art.get("url", "") or "",
                    published_at=pub_date,
                    ticker="MARKET",
                )
            )
        return results


def _parse_iso(iso_str: str) -> datetime:
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)
