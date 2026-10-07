#!/usr/bin/env python3
"""Manual ingestion runner — poll provider, normalize, cache."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from forecast.config import SUPPORTED_TICKERS, TICKER_GROUPS
from forecast.data.cache import PriceCache
from forecast.data.normalization import normalize_ohlcv
from forecast.data.providers import get_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_ingestion")


def resolve_tickers(market: str | None) -> list[str]:
    if market is None:
        return list(SUPPORTED_TICKERS)
    market_lower = market.lower().replace(" ", "_")
    for group_name, tickers in TICKER_GROUPS.items():
        if market_lower in group_name.lower().replace(" ", "_"):
            return list(tickers)
    raise ValueError(f"Unknown market: {market}. Available: {list(TICKER_GROUPS.keys())}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ingestion for one or more tickers.")
    parser.add_argument(
        "--market", "-m",
        help="Market group (e.g. 'BIST', 'US'). If omitted, fetches all.",
    )
    parser.add_argument(
        "--tickers", "-t", nargs="+",
        help="Specific tickers to fetch (overrides --market).",
    )
    parser.add_argument(
        "--source", "-s", default="yahoo",
        help="Provider source (default: yahoo).",
    )
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper().strip() for t in args.tickers]
    else:
        tickers = resolve_tickers(args.market)

    provider = get_provider(args.source)
    cache = PriceCache()

    success = 0
    failed = 0
    for ticker in tickers:
        try:
            raw = provider.fetch_history(ticker)
            normalized = normalize_ohlcv(raw, ticker, source=args.source)
            cache.set(ticker, normalized)
            logger.info(
                "OK %s — %d rows (%.2f–%.2f)",
                ticker, len(normalized),
                normalized["close"].min(), normalized["close"].max(),
            )
            success += 1
        except Exception as exc:
            logger.error("FAIL %s: %s", ticker, exc)
            failed += 1

    print(f"\nDone — {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
