"""Personalization API — watchlists, preferences, feedback, ranked news with JWT ownership."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from forecast.config import SUPPORTED_TICKERS
from forecast.data.database import get_session
from forecast.data.news import get_news_provider
from forecast.data.orm_models import NewsFeedback, User, UserPreference, Watchlist
from forecast.gateway.dependencies import get_current_user
from forecast.personalization.ranker import rank_news

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/users", tags=["personalization"])

_news_provider = get_news_provider()


def _check_ownership(current_user: User, user_id: int) -> None:
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")


# ── Legacy user lookup (for migration, requires auth) ─────────────────────


@router.get("/{user_id}")
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    _check_ownership(current_user, user_id)
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "is_verified": current_user.is_verified,
        "language": current_user.language,
    }


# ── Preferences ───────────────────────────────────────────────────────────


@router.get("/{user_id}/preferences")
async def get_preferences(
    user_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    _check_ownership(current_user, user_id)
    prefs = await session.get(UserPreference, user_id)
    if prefs is None:
        return {"preferred_sectors": [], "sentiment_bias": "neutral", "risk_level": "medium", "max_news_per_ticker": 5, "dark_theme": False}
    return {
        "preferred_sectors": prefs.preferred_sectors or [],
        "sentiment_bias": prefs.sentiment_bias or "neutral",
        "risk_level": prefs.risk_level or "medium",
        "max_news_per_ticker": prefs.max_news_per_ticker or 5,
        "dark_theme": bool(prefs.dark_theme),
    }


@router.put("/{user_id}/preferences")
async def update_preferences(
    user_id: int,
    body: dict[str, Any],
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    _check_ownership(current_user, user_id)
    prefs = await session.get(UserPreference, user_id)
    if prefs is None:
        prefs = UserPreference(user_id=user_id)
        session.add(prefs)

    if "preferred_sectors" in body:
        prefs.preferred_sectors = body["preferred_sectors"]
    if "sentiment_bias" in body:
        prefs.sentiment_bias = body["sentiment_bias"]
    if "risk_level" in body:
        prefs.risk_level = body["risk_level"]
    if "max_news_per_ticker" in body:
        prefs.max_news_per_ticker = body["max_news_per_ticker"]
    if "dark_theme" in body:
        prefs.dark_theme = 1 if body["dark_theme"] else 0

    await session.commit()
    return await get_preferences(user_id, current_user, session)


# ── Watchlist ─────────────────────────────────────────────────────────────


@router.get("/{user_id}/watchlist")
async def get_watchlist(
    user_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[str]:
    _check_ownership(current_user, user_id)
    result = await session.execute(select(Watchlist.ticker).where(Watchlist.user_id == user_id))
    return [row[0] for row in result.all()]


@router.post("/{user_id}/watchlist")
async def add_to_watchlist(
    user_id: int,
    body: dict[str, str],
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    _check_ownership(current_user, user_id)
    ticker = body.get("ticker", "").upper().strip()
    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker: {ticker}")

    existing = await session.execute(
        select(Watchlist).where(Watchlist.user_id == user_id, Watchlist.ticker == ticker)
    )
    if existing.scalar_one_or_none():
        return {"status": "already_exists", "ticker": ticker}

    entry = Watchlist(user_id=user_id, ticker=ticker)
    session.add(entry)
    await session.commit()
    return {"status": "added", "ticker": ticker}


@router.delete("/{user_id}/watchlist/{ticker}")
async def remove_from_watchlist(
    user_id: int,
    ticker: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    _check_ownership(current_user, user_id)
    ticker = ticker.upper().strip()
    result = await session.execute(
        delete(Watchlist).where(Watchlist.user_id == user_id, Watchlist.ticker == ticker)
    )
    await session.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Ticker not in watchlist")
    return {"status": "removed", "ticker": ticker}


# ── Feedback ──────────────────────────────────────────────────────────────


@router.post("/{user_id}/feedback")
async def submit_feedback(
    user_id: int,
    body: dict[str, Any],
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    _check_ownership(current_user, user_id)
    feedback = NewsFeedback(
        user_id=user_id,
        news_id=body.get("news_id", ""),
        ticker=body.get("ticker", "").upper().strip(),
        rating=body.get("rating", 0),
    )
    session.add(feedback)
    await session.commit()
    return {"status": "recorded", "news_id": feedback.news_id, "rating": feedback.rating}


@router.get("/{user_id}/feedback")
async def get_feedback(
    user_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    _check_ownership(current_user, user_id)
    result = await session.execute(
        select(NewsFeedback).where(NewsFeedback.user_id == user_id).order_by(NewsFeedback.created_at.desc())
    )
    return [
        {
            "id": f.id,
            "news_id": f.news_id,
            "ticker": f.ticker,
            "rating": f.rating,
            "created_at": f.created_at.isoformat() if f.created_at else "",
        }
        for f in result.scalars().all()
    ]


# ── Ranked News ───────────────────────────────────────────────────────────


@router.get("/{user_id}/news")
async def get_personalized_news(
    user_id: int,
    max_results: int = Query(20, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    _check_ownership(current_user, user_id)
    prefs = await session.get(UserPreference, user_id)
    preferred_sectors = prefs.preferred_sectors if prefs and prefs.preferred_sectors else []
    sentiment_bias = prefs.sentiment_bias if prefs and prefs.sentiment_bias else "neutral"

    wl_result = await session.execute(select(Watchlist.ticker).where(Watchlist.user_id == user_id))
    watchlist = set(row[0] for row in wl_result.all())

    fb_result = await session.execute(
        select(NewsFeedback).where(NewsFeedback.user_id == user_id)
    )
    feedback_signals: dict[str, int] = {}
    for fb in fb_result.scalars().all():
        feedback_signals[fb.news_id] = fb.rating

    market_news = _news_provider.fetch_market_news(max_results=max_results)
    ticker_news: list = []
    for ticker in watchlist:
        ticker_news.extend(_news_provider.fetch_news(ticker, max_results=5))

    all_news = list({n.id: n for n in market_news + ticker_news}.values())

    scored = rank_news(
        items=all_news,
        watchlist=watchlist,
        preferred_sectors=preferred_sectors,
        sentiment_bias=sentiment_bias,
        feedback_signals=feedback_signals,
    )

    return {
        "count": len(scored[:max_results]),
        "data": [
            {
                "id": item.id,
                "title": item.title,
                "summary": item.summary,
                "source": item.source,
                "url": item.url,
                "published_at": item.published_at.isoformat() if hasattr(item.published_at, "isoformat") else str(item.published_at),
                "ticker": item.ticker,
                "sentiment": item.sentiment,
                "sentiment_score": item.sentiment_score,
                "importance_score": item.importance_score,
                "related_tickers": item.related_tickers,
                "sectors": item.sectors,
                "personalized_score": round(score, 4),
                "source_provider": item.source_provider,
            }
            for item, score in scored[:max_results]
        ],
    }
