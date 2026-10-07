"""News ranking engine — scores and sorts news items per user preferences."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from forecast.data.news.models import NewsItem


def score_news_item(
    item: NewsItem,
    watchlist: set[str],
    preferred_sectors: list[str],
    sentiment_bias: str,
    feedback_signals: dict[str, int],
    now: datetime | None = None,
) -> float:
    if now is None:
        now = datetime.now(timezone.utc)

    score = 0.0

    # Watchlist match (weight: 3.0)
    if item.ticker in watchlist:
        score += 3.0
    if any(t in watchlist for t in item.related_tickers):
        score += 2.0

    # Sector preference match (weight: 2.0)
    if preferred_sectors and item.sectors:
        overlap = len(set(item.sectors) & set(preferred_sectors))
        score += overlap * 2.0

    # Sentiment bias (weight: 1.5)
    if sentiment_bias != "neutral" and item.sentiment == sentiment_bias:
        score += 1.5

    # Feedback signal (weight: 1.0)
    if item.id in feedback_signals:
        score += feedback_signals[item.id] * 1.0

    # Importance score from NLP (weight: 0.5)
    score += item.importance_score * 0.5

    # Recency decay (halve after 48 hours)
    if item.published_at:
        hours_ago = (now - item.published_at).total_seconds() / 3600
        recency_factor = math.exp(-hours_ago / 48.0 * math.log(2))
        score *= 0.5 + 0.5 * recency_factor

    return score


def rank_news(
    items: list[NewsItem],
    watchlist: set[str],
    preferred_sectors: list[str],
    sentiment_bias: str,
    feedback_signals: dict[str, int] | None = None,
) -> list[tuple[NewsItem, float]]:
    if feedback_signals is None:
        feedback_signals = {}

    scored = [
        (item, score_news_item(item, watchlist, preferred_sectors, sentiment_bias, feedback_signals))
        for item in items
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
