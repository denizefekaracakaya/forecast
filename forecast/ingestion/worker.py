"""Base ingestion worker — poll provider, normalize, cache."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Protocol

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

import pandas as pd

from forecast.config import INGESTION_INTERVAL, SUPPORTED_TICKERS
from forecast.data.cache import PriceCache
from forecast.data.providers.base import PriceProvider

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(
        self,
        provider: PriceProvider,
        cache: PriceCache,
        market: str = "",
        poll_interval: int = INGESTION_INTERVAL,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.market = market
        self.poll_interval = poll_interval

    def run_once(self, tickers: list[str]) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for ticker in tickers:
            try:
                raw = self.provider.fetch_history(ticker)
                self.cache.set(ticker, raw)
                results[ticker] = True
                logger.info("Cached %s (%d rows)", ticker, len(raw))
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", ticker, exc)
                results[ticker] = False
        return results

    def run_loop(self, tickers: list[str]) -> None:
        while True:
            self.run_once(tickers)
            time.sleep(self.poll_interval)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from forecast.data.providers import get_provider

    provider = get_provider("yahoo")
    cache = PriceCache()
    worker = IngestionWorker(provider, cache)
    logger.warning(
        "PriceCache is in-memory and process-local: entries cached here are NOT "
        "visible to the gateway or app processes. This worker currently only "
        "validates data availability and exercises the fetch path; a shared "
        "backend (e.g. Redis) is required before ingestion can pre-warm the "
        "gateway's cache — see project roadmap."
    )
    logger.info(
        "Starting ingestion loop for %d tickers (interval=%ds)...",
        len(SUPPORTED_TICKERS), INGESTION_INTERVAL,
    )
    worker.run_loop(SUPPORTED_TICKERS)
