"""Yahoo Finance provider via yfinance."""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from forecast.config import settings
from forecast.data.normalization import normalize_ohlcv
from forecast.data.providers.base import PriceProvider
from forecast.data.rate_limiter import shared_limiter

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    ...


MAX_RETRIES = settings.YAHOO_MAX_RETRIES


def _backoff_seconds() -> list[float]:
    return [2.0 * (1.5**i) for i in range(MAX_RETRIES)]


BACKOFF_SECONDS = _backoff_seconds()


class YahooProvider(PriceProvider):
    def __init__(self) -> None:
        self._limiter = shared_limiter()

    def fetch_history(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        ticker = ticker.upper().strip()
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        raw = None
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES):
            self._limiter.wait()

            try:
                raw = yf.download(
                    ticker, start=start, end=end, progress=False, auto_adjust=True
                )
                if raw is not None and not raw.empty:
                    return raw
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Yahoo fetch attempt %d/%d failed for %s: %s",
                    attempt + 1, MAX_RETRIES, ticker, exc,
                )

            if attempt < MAX_RETRIES - 1:
                jitter = random.uniform(0.5, 1.5)
                sleep_for = BACKOFF_SECONDS[attempt] * jitter
                logger.info(
                    "Retrying %s in %.1fs (attempt %d/%d)...",
                    ticker, sleep_for, attempt + 2, MAX_RETRIES,
                )
                time.sleep(sleep_for)

        if last_exc is not None:
            raise DataFetchError(
                f"'{ticker}' için veri indirilemedi ({MAX_RETRIES} deneme): {last_exc}"
            ) from last_exc
        raise DataFetchError(
            f"'{ticker}' için geçerli fiyat verisi bulunamadı. "
            "Yahoo Finance geçici olarak yanıt vermiyor olabilir; birkaç dakika sonra tekrar deneyin."
        )

    def fetch_fundamentals(self, ticker: str) -> dict:
        from forecast.data.fundamentals import fetch_fundamentals as _fetch_fundamentals

        self._limiter.wait()
        return _fetch_fundamentals(ticker)

    def fetch_and_normalize(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        raw = self.fetch_history(ticker, start=start, end=end)
        return normalize_ohlcv(raw, ticker, source="yahoo")
