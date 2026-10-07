"""Database engine, session, and base — async SQLAlchemy with PostgreSQL or SQLite.

Engine is lazily initialized. Falls back gracefully if database is unavailable.
Use SQLite for local development without Docker:
  DATABASE_URL=sqlite+aiosqlite:///./forecast.db
"""

from __future__ import annotations

import logging

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from forecast.config import settings

logger = logging.getLogger(__name__)

_engine = None
_async_session_factory = None


def _ensure_engine():
    global _engine, _async_session_factory
    if _engine is not None:
        return _engine, _async_session_factory
    try:
        url = settings.DATABASE_URL
        is_sqlite = url.startswith("sqlite")
        kw = {"echo": settings.DB_ECHO}
        if is_sqlite:
            kw["connect_args"] = {"check_same_thread": False}
        else:
            kw["pool_size"] = settings.DB_POOL_SIZE
        _engine = create_async_engine(url, **kw)
        _async_session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    except Exception as exc:
        logger.warning("Database engine initialization failed: %s — personalization disabled", exc)
        _engine = None
        _async_session_factory = None
    return _engine, _async_session_factory


class Base(DeclarativeBase):
    pass


async def get_session():
    _, factory = _ensure_engine()
    if factory is None:
        raise HTTPException(status_code=503, detail="Database not available — personalization features disabled")
    async with factory() as session:
        yield session


async def create_tables() -> None:
    engine, _ = _ensure_engine()
    if engine is None:
        logger.warning("Cannot create tables — database engine unavailable")
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")


async def drop_tables() -> None:
    engine, _ = _ensure_engine()
    if engine is None:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
