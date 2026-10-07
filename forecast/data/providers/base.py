"""Abstract provider interface for market data."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PriceProvider(ABC):
    @abstractmethod
    def fetch_history(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def fetch_fundamentals(self, ticker: str) -> dict:
        ...
