"""spaCy NLP pipeline for news enrichment — NER, sector classification, sentiment."""

from __future__ import annotations

import logging
import re

from forecast.data.news.sentiment_keywords import BEARISH_KEYWORDS, BULLISH_KEYWORDS

logger = logging.getLogger(__name__)

_SECTOR_MAP: dict[str, str] = {
    "technology": "Teknoloji",
    "software": "Teknoloji",
    "hardware": "Teknoloji",
    "semiconductor": "Teknoloji",
    "chip": "Teknoloji",
    "ai": "Teknoloji",
    "artificial intelligence": "Teknoloji",
    "cloud": "Teknoloji",
    "cyber": "Teknoloji",
    "finance": "Finans",
    "bank": "Finans",
    "banking": "Finans",
    "insurance": "Finans",
    "investment": "Finans",
    "healthcare": "Sağlık",
    "pharma": "Sağlık",
    "pharmaceutical": "Sağlık",
    "biotech": "Sağlık",
    "medical": "Sağlık",
    "drug": "Sağlık",
    "energy": "Enerji",
    "oil": "Enerji",
    "gas": "Enerji",
    "renewable": "Enerji",
    "solar": "Enerji",
    "wind": "Enerji",
    "automotive": "Otomotiv",
    "auto": "Otomotiv",
    "electric vehicle": "Otomotiv",
    "ev": "Otomotiv",
    "retail": "Perakende",
    "e-commerce": "Perakende",
    "consumer": "Tüketim",
    "telecom": "Telekom",
    "telecommunications": "Telekom",
    "real estate": "Gayrimenkul",
    "realestate": "Gayrimenkul",
    "manufacturing": "Üretim",
    "industrial": "Sanayi",
    "defense": "Savunma",
    "aerospace": "Savunma",
    "media": "Medya",
    "entertainment": "Medya",
    "food": "Gıda",
    "beverage": "Gıda",
    "mining": "Madencilik",
    "metal": "Madencilik",
    "transportation": "Ulaşım",
    "logistics": "Ulaşım",
    "airline": "Ulaşım",
}

_NLP = None


def _load_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        logger.warning("spaCy model 'en_core_web_sm' not found — NLP features disabled")
        _NLP = None
    return _NLP


def extract_entities(text: str) -> list[dict]:
    nlp = _load_nlp()
    if nlp is None:
        return []
    doc = nlp(text[:5000])
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
        if ent.label_ in {"ORG", "PERSON", "GPE", "PRODUCT"}
    ]


def classify_sectors(text: str) -> list[str]:
    text_lower = text.lower()
    matched = set()
    for keyword, sector in _SECTOR_MAP.items():
        if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
            matched.add(sector)
    return sorted(matched)


def analyze_sentiment_spacy(text: str) -> tuple[str, float]:
    nlp = _load_nlp()
    if nlp is None:
        return "neutral", 0.0

    doc = nlp(text[:5000])
    bullish_score = 0.0
    bearish_score = 0.0
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in BULLISH_KEYWORDS:
            bullish_score += 1.0
        elif lemma in BEARISH_KEYWORDS:
            bearish_score += 1.0

    total = bullish_score + bearish_score
    if total == 0:
        return "neutral", 0.0
    score = (bullish_score - bearish_score) / total
    if score > 0.2:
        return "bullish", score
    if score < -0.2:
        return "bearish", score
    return "neutral", score


def extract_ticker_candidates(text: str, known_tickers: set[str]) -> list[str]:
    found: list[str] = []
    for t in known_tickers:
        base = t.replace(".IS", "").replace(".", "")
        if re.search(rf"\b{re.escape(base)}\b", text):
            found.append(t)
    return sorted(set(found))
