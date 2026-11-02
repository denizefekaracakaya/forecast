"""Global configuration — ticker lists, default params, thresholds.

Settings are loaded from environment variables (via .env file or system env)
with sensible defaults.  See .env.example for all available options.
"""

from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Project-root .env works regardless of the current working directory;
        # a .env in the CWD (if different) takes precedence.
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    APP_TITLE: str = "E-Forecast"
    APP_DEBUG: bool = False

    # Cache
    CACHE_TTL_SECONDS: int = 3600

    # Rate limiter (Yahoo Finance)
    YAHOO_RATE_MIN_INTERVAL: float = 2.0
    YAHOO_MAX_RETRIES: int = 5

    # Ingestion
    INGESTION_INTERVAL: int = 600

    # News
    NEWS_MAX_RESULTS: int = 10
    NEWS_MARKET_MAX_RESULTS: int = 20
    NEWS_SENTIMENT_ENABLED: bool = True

    # Model defaults
    MODEL_CHANGEPOINT_PRIOR_SCALE: float = 0.05
    MODEL_SEASONALITY_PRIOR_SCALE: float = 10.0
    MODEL_SEASONALITY_MODE: str = "multiplicative"
    MODEL_USE_REGRESSORS: bool = True

    # Backtest threshold
    MAPE_THRESHOLD: float = 15.0

    # Hyperparameter tuning
    OPTUNA_N_TRIALS: int = 30

    # Gateway server
    GATEWAY_HOST: str = "localhost"
    GATEWAY_PORT: int = 8000
    # Comma-separated list of allowed CORS origins ("*" allows any origin)
    CORS_ORIGINS: str = "*"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://forecast:forecast@localhost:5432/forecast"
    DB_POOL_SIZE: int = 5
    DB_ECHO: bool = False

    # NewsAPI
    NEWSAPI_API_KEY: str = ""

    # Auth
    # Leave empty (or at a placeholder) to get a random per-process secret —
    # tokens then stop working whenever the gateway restarts.
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 24
    AUTH_REQUIRE_VERIFICATION: bool = True
    PASSWORD_MIN_LENGTH: int = 8
    # Brute-force protection: failed logins allowed per identifier per window
    LOGIN_MAX_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_SECONDS: int = 300

    # Resend (email)
    RESEND_API_KEY: str = ""
    EMAIL_FROM: str = "onboarding@resend.dev"
    EMAIL_FROM_NAME: str = "E-Forecast"

    # App base URL (for email links)
    APP_BASE_URL: str = "http://localhost:8501"

    # Logging
    LOG_LEVEL: str = "INFO"


settings = Settings()

_INSECURE_JWT_SECRETS = {
    "",
    "change-me-in-production",
    "eforecast-local-dev-secret-key-2024",
}
_ephemeral_jwt_secret: str | None = None


def jwt_secret() -> str:
    """Signing key for auth tokens.

    Falls back to a random per-process key (with a warning) when JWT_SECRET_KEY
    is unset or a well-known placeholder, so a forgotten setting can never
    produce forgeable tokens.
    """
    global _ephemeral_jwt_secret
    if settings.JWT_SECRET_KEY not in _INSECURE_JWT_SECRETS:
        return settings.JWT_SECRET_KEY
    if _ephemeral_jwt_secret is None:
        logging.getLogger(__name__).warning(
            "JWT_SECRET_KEY is unset or a known placeholder; using a random per-process "
            "secret. Tokens will be invalidated on restart. Set JWT_SECRET_KEY in .env."
        )
        _ephemeral_jwt_secret = secrets.token_urlsafe(48)
    return _ephemeral_jwt_secret


def cors_origins() -> list[str]:
    return [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]

TICKERS_BIST: list[str] = [
    "AEFES.IS",
    "AGHOL.IS",
    "AGROT.IS",
    "AHGAZ.IS",
    "AKBNK.IS",
    "AKSA.IS",
    "AKSEN.IS",
    "ALARK.IS",
    "ALFAS.IS",
    "ALTNY.IS",
    "ANHYT.IS",
    "ANSGR.IS",
    "ARCLK.IS",
    "ARDYZ.IS",
    "ASELS.IS",
    "ASTOR.IS",
    "AVPGY.IS",
    "BERA.IS",
    "BIMAS.IS",
    "BRSAN.IS",
    "BRYAT.IS",
    "BSOKE.IS",
    "BTCIM.IS",
    "CANTE.IS",
    "CCOLA.IS",
    "CIMSA.IS",
    "CLEBI.IS",
    "CWENE.IS",
    "DOAS.IS",
    "DOHOL.IS",
    "ECILC.IS",
    "EFORC.IS",
    "EGEEN.IS",
    "EKGYO.IS",
    "ENJSA.IS",
    "ENKAI.IS",
    "EREGL.IS",
    "EUPWR.IS",
    "FROTO.IS",
    "GARAN.IS",
    "GESAN.IS",
    "GOLTS.IS",
    "GRTHO.IS",
    "GSRAY.IS",
    "GUBRF.IS",
    "HALKB.IS",
    "HEKTS.IS",
    "IEYHO.IS",
    "ISCTR.IS",
    "ISMEN.IS",
    "KARSN.IS",
    "KCAER.IS",
    "KCHOL.IS",
    "KONTR.IS",
    "KONYA.IS",
    "KOZAA.IS",
    "KOZAL.IS",
    "KRDMD.IS",
    "KTLEV.IS",
    "LMKDC.IS",
    "MAGEN.IS",
    "MAVI.IS",
    "MGROS.IS",
    "MIATK.IS",
    "MPARK.IS",
    "OBAMS.IS",
    "ODAS.IS",
    "OTKAR.IS",
    "OYAKC.IS",
    "PASEU.IS",
    "PETKM.IS",
    "PGSUS.IS",
    "RALYH.IS",
    "REEDR.IS",
    "RYGYO.IS",
    "SAHOL.IS",
    "SASA.IS",
    "SELEC.IS",
    "SISE.IS",
    "SKBNK.IS",
    "SMRTG.IS",
    "SOKM.IS",
    "TABGD.IS",
    "TAVHL.IS",
    "TCELL.IS",
    "THYAO.IS",
    "TKFEN.IS",
    "TOASO.IS",
    "TSKB.IS",
    "TTKOM.IS",
    "TTRAK.IS",
    "TUPRS.IS",
    "TURSG.IS",
    "ULKER.IS",
    "VAKBN.IS",
    "VESTL.IS",
    "YEOTK.IS",
    "YKBNK.IS",
    "ZOREN.IS",
]

TICKERS_US: list[str] = [
    "SPCX",
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "GOOGL",
    "GOOG",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "AVGO",
    "COST",
    "NFLX",
    "AMD",
    "ADBE",
    "INTC",
    "CSCO",
    "AMGN",
    "TMUS",
    "QCOM",
    "PEP",
    "ADI",
    "INTU",
    "AMAT",
    "ISRG",
    "BKNG",
    "VRTX",
    "SBUX",
    "MDLZ",
    "HON",
    "GILD",
    "ADP",
    "CMCSA",
    "TXN",
    "LIN",
    "MU",
    "PANW",
    "KLAC",
    "LRCX",
    "MCHP",
    "SNPS",
    "CDNS",
    "CRWD",
    "FTNT",
    "MAR",
    "ABNB",
    "WDAY",
    "ADSK",
    "CTSH",
    "EA",
    "FAST",
    "BKR",
    "ODFL",
    "PAYX",
    "ROST",
    "PCAR",
    "CTAS",
    "CSX",
    "XEL",
    "AEP",
    "EXC",
    "CEG",
    "FANG",
    "WBD",
    "KDP",
    "MNST",
    "DASH",
    "MELI",
    "MSTR",
    "PLTR",
    "SHOP",
    "ARM",
    "APP",
    "MRVL",
    "WDC",
    "STX",
    "TTWO",
    "DXCM",
    "IDXX",
    "ALNY",
    "REGN",
    "VRSK",
    "ROP",
    "CPRT",
    "CHTR",
    "CSGP",
    "ZS",
    "TEAM",
    "DDOG",
    "INSM",
    "GEHC",
    "KHC",
    "NXPI",
    "MPWR",
    "ORLY",
    "PYPL",
    "PDD",
    "TRI",
    "CCEP",
    "FER",
    "AXON",
    "ASML",
    "WMT",
    "JPM",
]

TICKER_GROUPS: dict[str, list[str]] = {
    "BIST (Türkiye)": TICKERS_BIST,
    "ABD (US)": TICKERS_US,
}

SUPPORTED_TICKERS = TICKERS_BIST + TICKERS_US

DEFAULT_PARAMS: dict[str, Any] = {
    "changepoint_prior_scale": settings.MODEL_CHANGEPOINT_PRIOR_SCALE,
    "seasonality_prior_scale": settings.MODEL_SEASONALITY_PRIOR_SCALE,
    "seasonality_mode": settings.MODEL_SEASONALITY_MODE,
    "use_regressors": settings.MODEL_USE_REGRESSORS,
}

EXPERIMENTS_CSV = PROJECT_ROOT / "data" / "experiments.csv"

MAPE_THRESHOLD = settings.MAPE_THRESHOLD

OPTUNA_N_TRIALS = settings.OPTUNA_N_TRIALS

INGESTION_INTERVAL = settings.INGESTION_INTERVAL
