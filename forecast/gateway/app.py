"""FastAPI REST gateway — serves normalized price data."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from forecast import __version__
from forecast.config import SUPPORTED_TICKERS, TICKER_GROUPS, cors_origins, settings
from forecast.data.cache import PriceCache
from forecast.data.database import create_tables
from forecast.data.news import get_news_provider
from forecast.data.normalization import normalize_ohlcv
from forecast.data.providers.yahoo import DataFetchError, YahooProvider
from forecast.gateway.routes.auth import router as auth_router
from forecast.gateway.routes.personalization import router as personalization_router
from forecast.gateway.schemas import news_to_dict

logger = logging.getLogger(__name__)

_db_initialized = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_initialized
    if not _db_initialized:
        try:
            await create_tables()
            _db_initialized = True
        except Exception as exc:
            logger.warning("Database not available: %s — personalization features disabled", exc)
    yield


app = FastAPI(
    title="E-Forecast API",
    version=__version__,
    description="E-Forecast market data gateway — prices, fundamentals, and news.",
    lifespan=lifespan,
)

_origins = cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # Credentialed CORS with a wildcard origin is rejected by browsers anyway.
    allow_credentials="*" not in _origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = PriceCache(ttl_seconds=settings.CACHE_TTL_SECONDS)
_provider = YahooProvider()
_news_provider = get_news_provider()

app.include_router(auth_router)
app.include_router(personalization_router)


def _records_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = df.copy()
    if "ds" in df.columns:
        df["ds"] = df["ds"].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x))
    return df.to_dict(orient="records")


def _require_supported(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(status_code=404, detail=f"Unsupported ticker: {ticker}")
    return ticker


def _parse_date(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"'{name}' must be YYYY-MM-DD") from exc


def _fetch_or_cache(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    # Only the default (full-history) window is cached: caching a custom range
    # under the bare ticker key would serve that slice to later default requests.
    use_cache = start is None and end is None
    if use_cache:
        cached = _cache.get(ticker)
        if cached is not None:
            return cached

    try:
        raw = _provider.fetch_history(ticker, start=start, end=end)
        df = normalize_ohlcv(raw, ticker, source="yahoo")
    except DataFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if use_cache:
        _cache.set(ticker, df)
    return df


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/v1/markets")
def list_markets() -> dict[str, list[str]]:
    return TICKER_GROUPS


# Endpoints below are plain ``def``: FastAPI runs them in a worker thread, so the
# blocking yfinance / news HTTP calls no longer stall the event loop.


@app.get("/api/v1/prices/{ticker}")
def get_prices(
    ticker: str,
    start: str | None = Query(None, description="Start date YYYY-MM-DD"),
    end: str | None = Query(None, description="End date YYYY-MM-DD"),
) -> dict[str, Any]:
    ticker = _require_supported(ticker)
    start = _parse_date(start, "start")
    end = _parse_date(end, "end")
    if start and end and start > end:
        raise HTTPException(status_code=422, detail="'start' must not be after 'end'")

    records = _records_from_df(_fetch_or_cache(ticker, start, end))
    return {
        "ticker": ticker,
        "source": "yahoo",
        "count": len(records),
        "data": records,
    }


@app.get("/api/v1/news/{ticker}")
def get_ticker_news(
    ticker: str,
    max_results: int = Query(10, ge=1, le=50, description="Maximum news items"),
) -> dict:
    ticker = _require_supported(ticker)
    items = _news_provider.fetch_news(ticker, max_results=max_results)
    return {
        "ticker": ticker,
        "count": len(items),
        "data": [news_to_dict(n) for n in items],
    }


@app.get("/api/v1/news")
def get_market_news(
    max_results: int = Query(20, ge=1, le=50, description="Maximum news items"),
) -> dict:
    items = _news_provider.fetch_market_news(max_results=max_results)
    return {
        "count": len(items),
        "data": [news_to_dict(n) for n in items],
    }
