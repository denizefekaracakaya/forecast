"""Provider factory."""

from __future__ import annotations

from forecast.data.providers.base import PriceProvider
from forecast.data.providers.yahoo import YahooProvider


def get_provider(source: str = "yahoo") -> PriceProvider:
    if source == "yahoo":
        return YahooProvider()
    raise ValueError(f"Unknown provider source: {source}")
