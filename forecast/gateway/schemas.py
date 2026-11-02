"""Request/response schemas for the gateway.

Typed bodies give clients a 422 with a precise message instead of a 500 when a
field is missing or has the wrong type.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from forecast.config import settings

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

Language = Literal["tr", "en"]
SentimentBias = Literal["bearish", "neutral", "bullish"]
RiskLevel = Literal["low", "medium", "high"]


def _check_password(value: str) -> str:
    if len(value) < settings.PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters"
        )
    return value


def _check_email(value: str) -> str:
    value = value.strip().lower()
    if not _EMAIL_RE.match(value):
        raise ValueError("Valid email is required")
    return value


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


# ── Auth ──────────────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=100)
    email: str = Field(max_length=255)
    password: str = Field(max_length=128)
    language: Language = "tr"

    _username = field_validator("username", mode="before")(_strip)
    _email = field_validator("email")(_check_email)
    _password = field_validator("password")(_check_password)


class LoginRequest(BaseModel):
    login: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=128)


class TokenRequest(BaseModel):
    token: str = Field(min_length=1, max_length=256)


class EmailRequest(BaseModel):
    email: str = Field(max_length=255)

    _email = field_validator("email")(_check_email)


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=1, max_length=256)
    password: str = Field(max_length=128)

    _password = field_validator("password")(_check_password)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(max_length=128)

    _password = field_validator("new_password")(_check_password)


class AddPasswordRequest(BaseModel):
    email: str = Field(max_length=255)
    password: str = Field(max_length=128)

    _email = field_validator("email")(_check_email)
    _password = field_validator("password")(_check_password)


# ── Personalization ───────────────────────────────────────────────────────


class PreferencesUpdate(BaseModel):
    preferred_sectors: list[str] | None = Field(default=None, max_length=50)
    sentiment_bias: SentimentBias | None = None
    risk_level: RiskLevel | None = None
    max_news_per_ticker: int | None = Field(default=None, ge=1, le=50)
    dark_theme: bool | None = None


class WatchlistAdd(BaseModel):
    ticker: str = Field(min_length=1, max_length=20)


class FeedbackRequest(BaseModel):
    news_id: str = Field(min_length=1, max_length=100)
    ticker: str = Field(default="", max_length=20)
    rating: Literal[-1, 1]


# ── News ──────────────────────────────────────────────────────────────────


def news_to_dict(item: Any, **extra: Any) -> dict[str, Any]:
    """Serialize a NewsItem consistently across endpoints."""
    published = item.published_at
    data = {
        "id": item.id,
        "title": item.title,
        "summary": item.summary,
        "source": item.source,
        "url": item.url,
        "published_at": published.isoformat() if isinstance(published, datetime) else str(published),
        "sentiment": item.sentiment,
        "sentiment_score": item.sentiment_score,
        "importance_score": item.importance_score,
        "related_tickers": item.related_tickers,
    }
    data.update(extra)
    return data
