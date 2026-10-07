"""Hyperparameter tuning and experiment tracking.

Uses Optuna for Bayesian hyperparameter optimization.
Falls back to a simple grid when Optuna is unavailable (e.g. minimal env).
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from typing import Any

import pandas as pd

from forecast.config import DEFAULT_PARAMS, EXPERIMENTS_CSV, MAPE_THRESHOLD
from forecast.forecasting.model import (
    ModelError,
    REGRESSOR_COLUMNS,
    evaluate,
    forecast,
    naive_baseline_metrics,
    prepare_prophet_df,
    train_model,
)
from forecast.data.normalization import normalize_ohlcv
from forecast.data.providers.yahoo import YahooProvider

logger = logging.getLogger(__name__)


def log_experiment(
    ticker: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    baseline: dict[str, float],
    notes: str = "",
) -> None:
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "changepoint_prior_scale": params.get("changepoint_prior_scale"),
        "seasonality_prior_scale": params.get("seasonality_prior_scale"),
        "seasonality_mode": params.get("seasonality_mode"),
        "changepoint_range": params.get("changepoint_range"),
        "weekly_seasonality": params.get("weekly_seasonality"),
        "yearly_seasonality": params.get("yearly_seasonality"),
        "use_regressors": params.get("use_regressors"),
        "mape": round(metrics.get("mape", float("nan")), 4),
        "rmse": round(metrics.get("rmse", float("nan")), 4),
        "baseline_mape": round(baseline.get("mape", float("nan")), 4),
        "baseline_rmse": round(baseline.get("rmse", float("nan")), 4),
        "notes": notes,
    }
    EXPERIMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not EXPERIMENTS_CSV.exists()
    with EXPERIMENTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def tune_hyperparameters(
    df: pd.DataFrame,
    ticker: str = "SPCX",
    n_trials: int = 30,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    """Run Optuna-based hyperparameter optimization, with grid fallback."""
    try:
        return _tune_optuna(df, ticker, n_trials)
    except ImportError:
        logger.info("Optuna not available — falling back to grid search.")
        return _tune_grid(df, ticker)


def _tune_grid(
    df: pd.DataFrame,
    ticker: str = "SPCX",
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    grid = [
        {
            "changepoint_prior_scale": cps,
            "seasonality_prior_scale": sps,
            "seasonality_mode": sm,
        }
        for cps in [0.01, 0.05, 0.1, 0.5]
        for sps in [1.0, 10.0]
        for sm in ["additive", "multiplicative"]
    ]

    baseline = naive_baseline_metrics(df)
    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None

    for params in grid:
        full_params = {**DEFAULT_PARAMS, **params}
        try:
            model = train_model(df, full_params)
            metrics = evaluate(model, df)
            log_experiment(ticker, full_params, metrics, baseline, notes="grid_search")
            if best_metrics is None or metrics["mape"] < best_metrics["mape"]:
                best_params = full_params
                best_metrics = metrics
        except ModelError as exc:
            logger.warning("Grid point failed: %s %s", params, exc)

    if best_params is None or best_metrics is None:
        raise ModelError("Hiçbir hiperparametre kombinasyonu backtest geçemedi.")

    return best_params, best_metrics, baseline


def _tune_optuna(
    df: pd.DataFrame,
    ticker: str = "SPCX",
    n_trials: int = 30,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    import optuna  # type: ignore[import-untyped]

    baseline = naive_baseline_metrics(df)
    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_params, best_metrics

        params = {
            "changepoint_prior_scale": trial.suggest_float(
                "changepoint_prior_scale", 0.001, 0.5, log=True
            ),
            "seasonality_prior_scale": trial.suggest_float(
                "seasonality_prior_scale", 0.1, 20.0, log=True
            ),
            "seasonality_mode": trial.suggest_categorical(
                "seasonality_mode", ["additive", "multiplicative"]
            ),
            "changepoint_range": trial.suggest_float(
                "changepoint_range", 0.6, 0.95
            ),
            "weekly_seasonality": trial.suggest_categorical(
                "weekly_seasonality", [True, False]
            ),
            "yearly_seasonality": trial.suggest_categorical(
                "yearly_seasonality", [True, False]
            ),
            "use_regressors": trial.suggest_categorical(
                "use_regressors", [True, False]
            ),
        }

        full_params = {**DEFAULT_PARAMS, **params}
        try:
            model = train_model(df, full_params)
            metrics = evaluate(model, df)
            log_experiment(ticker, full_params, metrics, baseline, notes="optuna")
            if best_metrics is None or metrics["mape"] < best_metrics["mape"]:
                best_params = full_params
                best_metrics = metrics
            return metrics["mape"]
        except ModelError as exc:
            logger.warning("Optuna trial failed: %s %s", params, exc)
            return float("inf")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, timeout=600, show_progress_bar=False)

    if best_params is None or best_metrics is None:
        raise ModelError("Hiçbir Optuna denemesi backtest geçemedi.")

    logger.info(
        "Optuna done — best MAPE=%.2f%%, params=%s",
        best_metrics["mape"], best_params,
    )
    return best_params, best_metrics, baseline


def fetch_data(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    from forecast.forecasting.model import validate_ticker

    ticker = validate_ticker(ticker)
    provider = YahooProvider()
    raw = provider.fetch_history(ticker, start=start, end=end)
    normalized = normalize_ohlcv(raw, ticker, source="yahoo")
    return prepare_prophet_df(normalized)


def run_backtest_check(
    ticker: str = "SPCX", threshold: float = MAPE_THRESHOLD
) -> dict[str, Any]:
    df = fetch_data(ticker)
    params, metrics, baseline = tune_hyperparameters(df, ticker=ticker)
    passed = metrics["mape"] <= threshold
    return {
        "ticker": ticker,
        "params": params,
        "metrics": metrics,
        "baseline": baseline,
        "threshold": threshold,
        "passed": passed,
    }
