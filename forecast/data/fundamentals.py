"""Stock fundamentals and computed statistics for UI display."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from forecast.data.rate_limiter import shared_limiter
from forecast.forecasting.model import compute_rsi

logger = logging.getLogger(__name__)

FUNDAMENTAL_KEYS = [
    "marketCap",
    "trailingPE",
    "trailingEps",
    "dividendYield",
    "beta",
    "sharesOutstanding",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "previousClose",
    "dayLow",
    "dayHigh",
    "volume",
    "averageVolume",
    "averageVolume10days",
    "forwardPE",
    "priceToBook",
    "debtToEquity",
    "returnOnEquity",
    "revenuePerShare",
    "profitMargins",
]


def fetch_fundamentals(ticker: str) -> dict:
    """Fetch fundamental data from Yahoo Finance via yf.Ticker.info.

    Returns a flat dict with only the keys listed in FUNDAMENTAL_KEYS.
    Missing keys are silently omitted (caller checks ``is not None``).
    """
    limiter = shared_limiter()
    limiter.wait()
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        logger.warning("Failed to fetch fundamentals for %s: %s", ticker, exc)
        return {}

    result: dict = {}
    for key in FUNDAMENTAL_KEYS:
        val = info.get(key)
        if val is not None:
            result[key] = val
    return result


def compute_price_stats(df: pd.DataFrame) -> dict:
    """Compute derived statistics from a normalized OHLCV DataFrame.

    Expects columns: ds, close, volume, open, high, low (as produced by
    ``normalize_ohlcv``).  Returns a dict of computed values.
    """
    if df.empty:
        return {}

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    open_p = df["open"].astype(float) if "open" in df else close
    high = df["high"].astype(float) if "high" in df else close
    low = df["low"].astype(float) if "low" in df else close

    stats: dict = {}

    # Latest values
    stats["last_close"] = float(close.iloc[-1])
    stats["last_open"] = float(open_p.iloc[-1])
    stats["last_high"] = float(high.iloc[-1])
    stats["last_low"] = float(low.iloc[-1])
    stats["last_volume"] = int(volume.iloc[-1]) if pd.notna(volume.iloc[-1]) else 0

    # Daily change
    if len(close) >= 2:
        prev_close = float(close.iloc[-2])
        stats["daily_change"] = float(close.iloc[-1]) - prev_close
        stats["daily_change_pct"] = (
            (float(close.iloc[-1]) / prev_close - 1) * 100 if prev_close != 0 else 0.0
        )
    else:
        stats["daily_change"] = 0.0
        stats["daily_change_pct"] = 0.0

    # Average volume (last 30 days)
    vol_last_30 = volume.tail(30)
    stats["avg_volume_30"] = int(vol_last_30.mean()) if len(vol_last_30) > 0 else 0

    # RSI (14) — latest value
    rsi_series = compute_rsi(close)
    stats["rsi"] = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 else 50.0

    # SMA 20 / SMA 50
    sma20 = close.rolling(window=20, min_periods=20).mean()
    sma50 = close.rolling(window=50, min_periods=50).mean()
    stats["sma_20"] = float(sma20.iloc[-1]) if len(sma20) > 0 and pd.notna(sma20.iloc[-1]) else None
    stats["sma_50"] = float(sma50.iloc[-1]) if len(sma50) > 0 and pd.notna(sma50.iloc[-1]) else None

    # Momentum 5-day
    if len(close) >= 6:
        mom5 = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100
        stats["momentum_5"] = mom5
    else:
        stats["momentum_5"] = 0.0

    # SMA crossover signal
    if stats["sma_20"] is not None and stats["sma_50"] is not None:
        stats["sma_cross_pct"] = (stats["sma_20"] / stats["sma_50"] - 1) * 100
    else:
        stats["sma_cross_pct"] = None

    # YTD change
    try:
        year_start = df["ds"].min()
        # normalize_ohlcv resets to a RangeIndex, so the current year always comes from wall-clock time
        ytd_mask = df["ds"] >= pd.Timestamp(year=datetime.now(timezone.utc).year, month=1, day=1)
        if ytd_mask.any():
            first_ytd_idx = ytd_mask.idxmax()
            ytd_start_close = float(close.loc[first_ytd_idx])
            stats["ytd_change_pct"] = (float(close.iloc[-1]) / ytd_start_close - 1) * 100 if ytd_start_close != 0 else 0.0
        else:
            stats["ytd_change_pct"] = None
    except Exception:
        stats["ytd_change_pct"] = None

    # 52-week position (percentage from 52-week low to high)
    if len(close) >= 2:
        year_ago_idx = max(0, len(close) - 252)
        segment = close.iloc[year_ago_idx:]
        low_52 = float(segment.min())
        high_52 = float(segment.max())
        stats["week52_low"] = low_52
        stats["week52_high"] = high_52
        if high_52 > low_52:
            stats["week52_position_pct"] = (
                (float(close.iloc[-1]) - low_52) / (high_52 - low_52) * 100
            )
        else:
            stats["week52_position_pct"] = 50.0
    else:
        stats["week52_low"] = None
        stats["week52_high"] = None
        stats["week52_position_pct"] = None

    # Volume change vs average
    if stats["avg_volume_30"] > 0:
        stats["volume_vs_avg_pct"] = (
            (stats["last_volume"] / stats["avg_volume_30"] - 1) * 100
        )
    else:
        stats["volume_vs_avg_pct"] = None

    return stats
