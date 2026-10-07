"""Centralized OHLCV normalization — single schema for all providers."""

from __future__ import annotations

import pandas as pd

NORMALIZED_COLUMNS = [
    "ds", "ticker", "market", "open", "high", "low", "close", "volume", "source",
]


def _resolve_market(ticker: str) -> str:
    t = ticker.upper().strip()
    return "BIST (Türkiye)" if t.endswith(".IS") else "ABD (US)"


def _extract_col(raw: pd.DataFrame, name: str) -> pd.Series:
    col = raw[name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col


def normalize_ohlcv(
    raw: pd.DataFrame,
    ticker: str,
    source: str = "yahoo",
) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Empty DataFrame — nothing to normalize.")

    ticker_norm = ticker.upper().strip()

    if isinstance(raw.columns, pd.MultiIndex):
        close = _extract_col(raw, "Close")
        volume = _extract_col(raw, "Volume")
        open_ = _extract_col(raw, "Open")
        high = _extract_col(raw, "High")
        low = _extract_col(raw, "Low")
    else:
        close = raw["Close"]
        volume = raw.get("Volume", pd.Series(index=raw.index, dtype=float))
        open_ = raw.get("Open", close)
        high = raw.get("High", close)
        low = raw.get("Low", close)

    df = pd.DataFrame({
        "ds": close.index,
        "ticker": ticker_norm,
        "market": _resolve_market(ticker_norm),
        "open": open_.values.astype(float),
        "high": high.values.astype(float),
        "low": low.values.astype(float),
        "close": close.values.astype(float),
        "volume": volume.reindex(close.index).fillna(0).astype(float).values,
        "source": source,
    })
    return df.reset_index(drop=True)
