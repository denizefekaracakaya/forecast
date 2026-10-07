"""Shared fixtures for tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_multiindex() -> pd.DataFrame:
    n = 10
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {
        ("Close", ""): range(100, 100 + n),
        ("Volume", ""): range(1_000_000, 1_000_000 + n),
        ("Open", ""): range(99, 99 + n),
        ("High", ""): range(101, 101 + n),
        ("Low", ""): range(98, 98 + n),
    }
    df = pd.DataFrame(data, index=idx)
    df.index.name = "Date"
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def sample_ohlcv_single() -> pd.DataFrame:
    n = 10
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Close": range(100, 100 + n),
            "Volume": range(1_000_000, 1_000_000 + n),
            "Open": range(99, 99 + n),
            "High": range(101, 101 + n),
            "Low": range(98, 98 + n),
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def normalized_prophet_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    close = 100 + pd.Series(range(100)) * 0.5
    volume = 1_000_000 + pd.Series(range(100)) * 1000
    rolling_7 = close.rolling(7, min_periods=1).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)

    return pd.DataFrame({
        "ds": dates,
        "y": close.values.astype(float),
        "volume": volume.values.astype(float),
        "rolling_mean_7": rolling_7.values.astype(float),
        "rsi": rsi.values.astype(float),
    })
