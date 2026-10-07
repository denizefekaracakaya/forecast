"""News fetching and analysis package."""

from __future__ import annotations

from forecast.data.news.models import NewsItem
from forecast.data.news.provider import NewsProvider, get_news_provider

__all__ = [
    "NewsItem",
    "NewsProvider",
    "get_news_provider",
]
