#!/usr/bin/env python3
"""Run the FastAPI gateway server."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

import uvicorn

if __name__ == "__main__":
    from forecast.config import settings
    uvicorn.run(
        "forecast.gateway.app:app",
        host=settings.GATEWAY_HOST,
        port=settings.GATEWAY_PORT,
        reload=settings.APP_DEBUG,
    )
