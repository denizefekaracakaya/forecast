#!/usr/bin/env python3
"""Run the FastAPI gateway server (settings come from .env via forecast.config)."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from forecast.config import settings

    uvicorn.run(
        "forecast.gateway.app:app",
        host=settings.GATEWAY_HOST,
        port=settings.GATEWAY_PORT,
        reload=settings.APP_DEBUG,
        app_dir=str(ROOT),
    )
