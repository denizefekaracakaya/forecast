"""News data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class NewsItem:
    id: str
    title: str
    summary: str
    source: str
    url: str
    published_at: datetime
    ticker: str
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    importance_score: float = 0.0
    related_tickers: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None
    source_provider: str = "yahoo"
