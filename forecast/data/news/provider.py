"""News provider — fetches stock news from Yahoo Finance (+ optionally NewsAPI), enriches with spaCy NLP."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import yfinance as yf

from forecast.config import settings, SUPPORTED_TICKERS
from forecast.data.news.models import NewsItem
from forecast.data.news.nlp import (
    analyze_sentiment_spacy,
    classify_sectors,
    extract_entities,
    extract_ticker_candidates,
)
from forecast.data.news.providers.newsapi import NewsAPIProvider
from forecast.data.news.sentiment_keywords import BEARISH_KEYWORDS, BULLISH_KEYWORDS
from forecast.data.rate_limiter import shared_limiter

logger = logging.getLogger(__name__)

HIGH_IMPORTANCE_KEYWORDS: set[str] = {
    "earnings", "revenue", "quarterly", "fiscal", "guidance",
    "acquisition", "merger", "takeover", "ipo", "buyout",
    "ceo", "chief executive", "chairman", "management change",
    "restructuring", "spinoff", "divestiture",
}

MEDIUM_IMPORTANCE_KEYWORDS: set[str] = {
    "analyst", "price target", "rating", "sector", "industry",
    "regulatory", "sec", "fda", "approval", "patent",
    "lawsuit", "settlement", "partnership", "joint venture",
    "expansion", "contract", "dividend", "buyback",
}


def _analyze_sentiment(text: str) -> tuple[str, float]:
    text_lower = text.lower()
    bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    total = bullish_count + bearish_count
    if total == 0:
        return "neutral", 0.0
    score = (bullish_count - bearish_count) / total
    if score > 0.2:
        return "bullish", score
    if score < -0.2:
        return "bearish", score
    return "neutral", score


def _analyze_sentiment_hybrid(text: str) -> tuple[str, float]:
    kw_sent, kw_score = _analyze_sentiment(text)
    spacy_sent, spacy_score = analyze_sentiment_spacy(text)
    if spacy_score == 0.0:
        return kw_sent, kw_score
    combined = (kw_score + spacy_score) / 2
    if combined > 0.2:
        return "bullish", combined
    if combined < -0.2:
        return "bearish", combined
    return "neutral", combined


def _compute_importance(title: str, summary: str) -> float:
    text = f"{title} {summary}".lower()
    score = 0.0
    for kw in HIGH_IMPORTANCE_KEYWORDS:
        if kw in text:
            score += 2.0
    for kw in MEDIUM_IMPORTANCE_KEYWORDS:
        if kw in text:
            score += 1.0
    return min(score, 10.0)


def _parse_date(raw: str | int | None) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, int | float):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt
    except (ValueError, TypeError):
        return None


def _truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


class NewsProvider:
    def __init__(self) -> None:
        self._limiter = shared_limiter()
        self._newsapi = NewsAPIProvider() if settings.NEWSAPI_API_KEY else None

    def _enrich_item(self, item: NewsItem) -> NewsItem:
        text = f"{item.title} {item.summary}"
        item.sectors = classify_sectors(text)
        ents = extract_entities(text)
        item.topics = list(set(e["label"] for e in ents))
        if settings.NEWS_SENTIMENT_ENABLED:
            item.sentiment, item.sentiment_score = _analyze_sentiment_hybrid(text)
        extra_tickers = extract_ticker_candidates(text, set(SUPPORTED_TICKERS))
        for t in extra_tickers:
            if t not in item.related_tickers and t != item.ticker:
                item.related_tickers.append(t)
        return item

    def _yfinance_news(self, ticker: str, max_results: int) -> list[NewsItem]:
        self._limiter.wait()
        try:
            t = yf.Ticker(ticker)
            raw_items = t.news
        except Exception as exc:
            logger.warning("Failed to fetch yfinance news for %s: %s", ticker, exc)
            return []

        if not raw_items or not isinstance(raw_items, list):
            return []

        items: list[NewsItem] = []
        for raw in raw_items[:max_results]:
            if not isinstance(raw, dict):
                continue
            content = raw.get("content") or raw
            if isinstance(content, dict):
                title = content.get("title", "") or ""
                summary = content.get("summary", "") or content.get("description", "") or ""
                provider_raw = content.get("provider") or {}
                provider_name = provider_raw.get("displayName", "") if isinstance(provider_raw, dict) else str(provider_raw) if provider_raw else ""
                published = _parse_date(content.get("pubDate"))
                canonical = content.get("canonicalUrl") or {}
                url = str(canonical.get("url", "") if isinstance(canonical, dict) else "") or str(raw.get("link", "") or raw.get("url", ""))
            else:
                title = raw.get("title", "") or ""
                summary = raw.get("summary", "") or raw.get("description", "") or ""
                provider_raw = raw.get("provider") or {}
                provider_name = provider_raw.get("displayName", "") if isinstance(provider_raw, dict) else str(provider_raw) if provider_raw else ""
                published = _parse_date(raw.get("pubDate") or raw.get("providerPublishTime") or raw.get("publishedAt"))
                url = str(raw.get("link", "") or raw.get("url", ""))

            combined = f"{title} {summary}"
            sentiment, sentiment_score = _analyze_sentiment_hybrid(combined)
            importance = _compute_importance(title, summary)

            item = NewsItem(
                id=str(raw.get("id", "") or content.get("id", "") if isinstance(content, dict) else ""),
                title=title,
                summary=_truncate(summary),
                source=provider_name,
                url=url,
                published_at=published or datetime.now(timezone.utc),
                ticker=ticker,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                importance_score=importance,
                related_tickers=raw.get("relatedTickers", []),
                source_provider="yahoo",
                raw=raw,
            )
            items.append(self._enrich_item(item))

        items.sort(key=lambda x: x.importance_score, reverse=True)
        return items

    def _newsapi_news(self, ticker: str, max_results: int) -> list[NewsItem]:
        if self._newsapi is None:
            return []
        api_results = self._newsapi.fetch_news(ticker, max_results)
        items: list[NewsItem] = []
        for i, r in enumerate(api_results):
            combined = f"{r.title} {r.description}"
            sentiment, sentiment_score = _analyze_sentiment_hybrid(combined)
            importance = _compute_importance(r.title, r.description)
            item = NewsItem(
                id=f"newsapi-{ticker}-{i}",
                title=r.title,
                summary=_truncate(r.description),
                source=r.source_name,
                url=r.url,
                published_at=r.published_at,
                ticker=r.ticker,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                importance_score=importance,
                source_provider="newsapi",
            )
            items.append(self._enrich_item(item))
        items.sort(key=lambda x: x.importance_score, reverse=True)
        return items

    def fetch_news(self, ticker: str, max_results: int = 10) -> list[NewsItem]:
        ticker = ticker.upper().strip()
        yahoo_items = self._yfinance_news(ticker, max_results)
        newsapi_items = self._newsapi_news(ticker, max_results)
        merged = _merge_news(yahoo_items, newsapi_items, max_results)
        return merged

    def fetch_market_news(self, max_results: int = 20) -> list[NewsItem]:
        self._limiter.wait()
        try:
            search = yf.Search(query="stock market")
            raw_items = search.news
        except Exception as exc:
            logger.warning("Failed to fetch market news: %s", exc)
            return []

        if not raw_items or not isinstance(raw_items, list):
            return []

        items: list[NewsItem] = []
        for raw in raw_items[:max_results]:
            if not isinstance(raw, dict):
                continue
            content = raw.get("content") or raw
            if isinstance(content, dict):
                title = content.get("title", "") or ""
                summary = content.get("summary", "") or content.get("description", "") or ""
                provider_raw = content.get("provider") or {}
                provider_name = provider_raw.get("displayName", "") if isinstance(provider_raw, dict) else str(provider_raw) if provider_raw else ""
                published = _parse_date(content.get("pubDate"))
                canonical = content.get("canonicalUrl") or {}
                url = str(canonical.get("url", "") if isinstance(canonical, dict) else "") or str(raw.get("link", "") or raw.get("url", ""))
            else:
                title = raw.get("title", "") or ""
                summary = raw.get("summary", "") or ""
                provider_raw = raw.get("provider")
                provider_name = provider_raw.get("name", "") if isinstance(provider_raw, dict) else str(provider_raw) if provider_raw else ""
                published = _parse_date(raw.get("providerPublishTime") or raw.get("publishedAt") or raw.get("pubDate"))
                url = str(raw.get("link", "") or raw.get("url", ""))

            uuid = raw.get("uuid", "") or raw.get("id", "") or (content.get("id", "") if isinstance(content, dict) else "")
            related = raw.get("relatedTickers", [])

            combined = f"{title} {summary}"
            sentiment, sentiment_score = _analyze_sentiment_hybrid(combined)
            importance = _compute_importance(title, summary)

            item = NewsItem(
                id=str(uuid),
                title=title,
                summary=_truncate(summary),
                source=provider_name,
                url=url,
                published_at=published or datetime.now(timezone.utc),
                ticker=related[0] if related else "MARKET",
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                importance_score=importance,
                related_tickers=related,
                source_provider="yahoo",
                raw=raw,
            )
            items.append(self._enrich_item(item))

        if self._newsapi:
            api_market = self._newsapi.fetch_market_news(max_results)
            for i, r in enumerate(api_market):
                combined = f"{r.title} {r.description}"
                sentiment, sentiment_score = _analyze_sentiment_hybrid(combined)
                importance = _compute_importance(r.title, r.description)
                item = NewsItem(
                    id=f"newsapi-market-{i}",
                    title=r.title,
                    summary=_truncate(r.description),
                    source=r.source_name,
                    url=r.url,
                    published_at=r.published_at,
                    ticker="MARKET",
                    sentiment=sentiment,
                    sentiment_score=sentiment_score,
                    importance_score=importance,
                    source_provider="newsapi",
                )
                items.append(self._enrich_item(item))

        items.sort(key=lambda x: x.importance_score, reverse=True)
        return items


def _merge_news(yahoo: list[NewsItem], newsapi: list[NewsItem], max_results: int) -> list[NewsItem]:
    seen_urls: set[str] = set()
    merged: list[NewsItem] = []
    for item in yahoo + newsapi:
        if item.url and item.url in seen_urls:
            continue
        if item.url:
            seen_urls.add(item.url)
        merged.append(item)
    merged.sort(key=lambda x: (x.importance_score, x.published_at.timestamp() if x.published_at else 0), reverse=True)
    return merged[:max_results]


def get_news_provider() -> NewsProvider:
    return NewsProvider()
