"""Shared keyword sets for lexicon-based sentiment scoring.

Used by both the substring-based scorer in ``provider.py`` and the
spaCy lemma-based scorer in ``nlp.py`` so the two techniques agree on
what counts as bullish/bearish language.
"""

from __future__ import annotations

BULLISH_KEYWORDS: set[str] = {
    "beat", "raised", "upgrade", "upgraded", "positive", "growth",
    "profit", "record", "surge", "rally", "bull", "bullish", "buy",
    "strong", "outperform", "exceed", "soar", "jump", "gain",
    "breakthrough", "optimistic", "upward", "recovery", "boom",
    "climb", "climbs", "climbed", "rise", "rises", "rose", "rising",
    "momentum", "uptrend", "break out", "breakout", "outlook",
}

BEARISH_KEYWORDS: set[str] = {
    "miss", "missed", "downgrade", "downgraded", "negative", "loss",
    "decline", "drop", "fall", "fell", "sell", "bear", "bearish",
    "weak", "cut", "warning", "caution", "downturn", "slump",
    "plunge", "crash", "risk", "uncertainty", "investigation",
    "layoff", "lawsuit", "probe", "debt", "default",
    "slide", "slides", "tumble", "tumbles", "tumbled", "tumbling",
    "sink", "sinks", "sank", "sinking", "retreat", "retreats",
    "bubble", "correction", "overvalued", "overhang",
}
