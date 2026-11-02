#!/usr/bin/env python3
"""Weekly backtest runner for CI.

Usage:
    python scripts/run_backtest.py [TICKER] [--trials N] [--threshold PCT]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python scripts/<name>.py` from anywhere without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecast.config import MAPE_THRESHOLD, OPTUNA_N_TRIALS
from forecast.forecasting.tuning import run_backtest_check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune and backtest the Prophet model.")
    parser.add_argument("ticker", nargs="?", default="SPCX")
    parser.add_argument("--trials", type=int, default=OPTUNA_N_TRIALS, help="Optuna trials")
    parser.add_argument(
        "--threshold", type=float, default=MAPE_THRESHOLD, help="Max allowed MAPE (%%)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = run_backtest_check(ticker=args.ticker, threshold=args.threshold, n_trials=args.trials)

    m = result["metrics"]
    b = result["baseline"]
    print(f"Ticker: {result['ticker']}")
    print(f"Model MAPE: {m['mape']:.2f}% | Baseline MAPE: {b['mape']:.2f}%")
    print(f"Model RMSE: {m['rmse']:.2f} | Baseline RMSE: {b['rmse']:.2f}")
    print(f"Threshold: {result['threshold']}% | Passed: {result['passed']}")
    print(f"Best params: {result['params']}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
