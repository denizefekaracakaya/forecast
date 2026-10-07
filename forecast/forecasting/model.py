"""Prophet model training, forecasting, and evaluation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from forecast.config import DEFAULT_PARAMS, SUPPORTED_TICKERS

logger = logging.getLogger(__name__)

REGRESSOR_COLUMNS = [
    "volume",
    "volume_change",
    "rolling_mean_7",
    "rsi",
    "momentum_5",
    "sma_cross",
]


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((loss == 0) & (gain == 0)), 50.0)
    rsi = rsi.where(~((loss == 0) & (gain > 0)), 100.0)
    return rsi.fillna(50.0)


def prepare_prophet_df(normalized: pd.DataFrame) -> pd.DataFrame:
    df = normalized.copy()
    close = df["close"]
    volume = df["volume"]

    df["rolling_mean_7"] = close.rolling(window=7, min_periods=1).mean()
    df["rsi"] = compute_rsi(close)

    volume_pct = volume.pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    df["volume_change"] = volume_pct.clip(-1, 10)

    momentum_5 = close.pct_change(periods=5).fillna(0).replace([np.inf, -np.inf], 0)
    df["momentum_5"] = momentum_5.clip(-1, 1)

    sma_20 = close.rolling(window=20, min_periods=20).mean()
    sma_50 = close.rolling(window=50, min_periods=50).mean()
    cross = (sma_20 / sma_50.replace(0, np.nan)) - 1.0
    df["sma_cross"] = cross.fillna(0).clip(-0.5, 0.5)

    prophet = pd.DataFrame({
        "ds": df["ds"],
        "y": close.values.astype(float),
    })
    for col in REGRESSOR_COLUMNS:
        prophet[col] = df[col].values.astype(float)
    return prophet


class ModelError(Exception):
    ...


def is_supported_ticker(ticker: str) -> bool:
    return ticker.upper().strip() in SUPPORTED_TICKERS


def validate_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if not is_supported_ticker(ticker):
        supported = ", ".join(SUPPORTED_TICKERS)
        raise ValueError(
            f"'{ticker}' desteklenmiyor. Desteklenen semboller: {supported}"
        )
    return ticker


def currency_symbol(ticker: str) -> str:
    return "₺" if ticker.upper().strip().endswith(".IS") else "$"


def market_of(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if ticker.endswith(".IS"):
        return "BIST (Türkiye)"
    return "ABD (US)"


def train_model(df: pd.DataFrame, params: dict[str, Any] | None = None) -> Prophet:
    params = {**DEFAULT_PARAMS, **(params or {})}
    use_regressors = params.pop("use_regressors", True)

    model = Prophet(
        changepoint_prior_scale=params.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=params.get("seasonality_prior_scale", 10.0),
        seasonality_mode=params.get("seasonality_mode", "multiplicative"),
        changepoint_range=params.get("changepoint_range", 0.8),
        weekly_seasonality=params.get("weekly_seasonality", True),
        yearly_seasonality=params.get("yearly_seasonality", True),
        daily_seasonality=False,
    )

    if use_regressors:
        for reg in REGRESSOR_COLUMNS:
            if reg in df.columns:
                model.add_regressor(reg)

    try:
        model.fit(df)
    except Exception as exc:
        raise ModelError(f"Model eğitimi başarısız: {exc}") from exc

    return model


def forecast(
    model: Prophet, days: int, history_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    if days < 1:
        raise ValueError("Tahmin süresi en az 1 gün olmalıdır.")

    future = model.make_future_dataframe(periods=days, freq="B")

    if history_df is not None and any(c in history_df.columns for c in REGRESSOR_COLUMNS):
        hist = history_df.set_index("ds")
        for reg in REGRESSOR_COLUMNS:
            if reg in hist.columns:
                last_val = hist[reg].iloc[-1]
                future[reg] = future["ds"].map(hist[reg]).fillna(last_val)

    try:
        return model.predict(future)
    except Exception as exc:
        raise ModelError(f"Tahmin üretilemedi: {exc}") from exc


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def naive_baseline_metrics(df: pd.DataFrame) -> dict[str, float]:
    if len(df) < 2:
        return {"mape": float("nan"), "rmse": float("nan")}
    actual = df["y"].values[1:]
    predicted = df["y"].values[:-1]
    return {"mape": _mape(actual, predicted), "rmse": _rmse(actual, predicted)}


def evaluate(model: Prophet, df: pd.DataFrame) -> dict[str, float]:
    n = len(df)
    if n < 60:
        raise ModelError("Backtest için en az 60 günlük veri gerekir.")

    initial = f"{max(30, int(n * 0.6))} days"
    period = f"{max(7, int(n * 0.05))} days"
    horizon = f"{max(5, int(n * 0.05))} days"

    try:
        cv = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel="threads",
        )
        metrics = performance_metrics(cv, rolling_window=1)
        return {
            "mape": float(metrics["mape"].mean() * 100),
            "rmse": float(metrics["rmse"].mean()),
        }
    except Exception as exc:
        raise ModelError(f"Backtest başarısız: {exc}") from exc
