"""Base ingestion worker — poll provider, normalize, cache."""

from __future__ import annotations

import logging
import time

from forecast.config import INGESTION_INTERVAL, SUPPORTED_TICKERS
from forecast.data.cache import PriceCache
from forecast.data.normalization import normalize_ohlcv
from forecast.data.providers.base import PriceProvider

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(
        self,
        provider: PriceProvider,
        cache: PriceCache,
        market: str = "",
        poll_interval: int = INGESTION_INTERVAL,
        source: str = "yahoo",
    ) -> None:
        self.provider = provider
        self.source = source
        self.cache = cache
        self.market = market
        self.poll_interval = poll_interval

    def run_once(self, tickers: list[str]) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for ticker in tickers:
            try:
                raw = self.provider.fetch_history(ticker)
                normalized = normalize_ohlcv(raw, ticker, source=self.source)
                self.cache.set(ticker, normalized)
                results[ticker] = True
                logger.info("Cached %s (%d rows)", ticker, len(normalized))
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", ticker, exc)
                results[ticker] = False
        return results

    def run_loop(self, tickers: list[str], max_iterations: int | None = None) -> None:
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            started = time.monotonic()
            results = self.run_once(tickers)
            ok = sum(results.values())
            logger.info("Ingestion pass done: %d/%d tickers OK", ok, len(results))
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break
            # Keep a steady cadence regardless of how long the pass took.
            time.sleep(max(0.0, self.poll_interval - (time.monotonic() - started)))


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
