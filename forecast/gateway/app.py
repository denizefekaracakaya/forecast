"""FastAPI REST gateway — serves normalized price data."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from forecast.config import SUPPORTED_TICKERS, TICKER_GROUPS, settings
from forecast.data.cache import PriceCache
from forecast.data.database import create_tables
from forecast.data.news import get_news_provider
from forecast.data.normalization import normalize_ohlcv
from forecast.data.providers.yahoo import YahooProvider, DataFetchError
from forecast.gateway.routes.personalization import router as personalization_router
from forecast.gateway.routes.auth import router as auth_router

logger = logging.getLogger(__name__)

_db_initialized = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_initialized
    if not _db_initialized:
        try:
            await create_tables()
            _db_initialized = True
            logger.info("Database tables created")
        except Exception as exc:
            logger.warning("Database not available: %s — personalization features disabled", exc)
    yield


app = FastAPI(
    title="E-Forecast API",
    version="0.1.0",
    description="E-Forecast market data gateway — prices, fundamentals, and news.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = PriceCache()
_provider = YahooProvider()
_news_provider = get_news_provider()

app.include_router(auth_router)
app.include_router(personalization_router)


def _records_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = df.copy()
    if "ds" in df.columns:
        df["ds"] = df["ds"].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x))
    return df.to_dict(orient="records")


def _fetch_or_cache(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    cached = _cache.get(ticker)
    if cached is not None and start is None and end is None:
        return cached

    try:
        raw = _provider.fetch_history(ticker, start=start, end=end)
        df = normalize_ohlcv(raw, ticker, source="yahoo")
        _cache.set(ticker, df)
        return df
    except DataFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/markets")
async def list_markets() -> dict[str, list[str]]:
    return TICKER_GROUPS


@app.get("/api/v1/prices/{ticker}")
async def get_prices(
    ticker: str,
    start: str | None = Query(None, description="Start date YYYY-MM-DD"),
    end: str | None = Query(None, description="End date YYYY-MM-DD"),
) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(status_code=404, detail=f"Unsupported ticker: {ticker}")

    df = _fetch_or_cache(ticker, start, end)
    records = _records_from_df(df)

    return {
        "ticker": ticker,
        "source": "yahoo",
        "count": len(records),
        "data": records,
    }


@app.get("/api/v1/news/{ticker}")
async def get_ticker_news(
    ticker: str,
    max_results: int = Query(10, ge=1, le=50, description="Maximum news items"),
) -> dict:
    ticker = ticker.upper().strip()
    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(status_code=404, detail=f"Unsupported ticker: {ticker}")

    items = _news_provider.fetch_news(ticker, max_results=max_results)
    return {
        "ticker": ticker,
        "count": len(items),
        "data": [
            {
                "id": n.id,
                "title": n.title,
                "summary": n.summary,
                "source": n.source,
                "url": n.url,
                "published_at": n.published_at.isoformat() if hasattr(n.published_at, "isoformat") else str(n.published_at),
                "sentiment": n.sentiment,
                "sentiment_score": n.sentiment_score,
                "importance_score": n.importance_score,
                "related_tickers": n.related_tickers,
            }
            for n in items
        ],
    }


@app.get("/api/v1/news")
async def get_market_news(
    max_results: int = Query(20, ge=1, le=50, description="Maximum news items"),
) -> dict:
    items = _news_provider.fetch_market_news(max_results=max_results)
    return {
        "count": len(items),
        "data": [
            {
                "id": n.id,
                "title": n.title,
                "summary": n.summary,
                "source": n.source,
                "url": n.url,
                "published_at": n.published_at.isoformat() if hasattr(n.published_at, "isoformat") else str(n.published_at),
                "sentiment": n.sentiment,
                "sentiment_score": n.sentiment_score,
                "importance_score": n.importance_score,
                "related_tickers": n.related_tickers,
            }
            for n in items
        ],
    }
