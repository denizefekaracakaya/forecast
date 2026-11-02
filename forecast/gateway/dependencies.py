"""FastAPI dependencies — JWT auth, ownership checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from forecast.config import jwt_secret, settings
from forecast.data.database import get_session
from forecast.data.orm_models import User


def create_jwt(user: User) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(user.id),
        "email": user.email,
        "username": user.username,
        "exp": now + timedelta(hours=settings.JWT_EXPIRE_HOURS),
        "iat": now,
    }
    return jwt.encode(payload, jwt_secret(), algorithm=settings.JWT_ALGORITHM)


class _AuthError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


async def _resolve_user(authorization: str | None, session: AsyncSession) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise _AuthError("Missing or invalid token")
    token = authorization[7:]
    try:
        payload = jwt.decode(
            token,
            jwt_secret(),
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
        user_id = int(payload["sub"])
    except jwt.ExpiredSignatureError as exc:
        raise _AuthError("Token expired") from exc
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError) as exc:
        raise _AuthError("Invalid token") from exc

    user = await session.get(User, user_id)
    if user is None or not user.is_active:
        raise _AuthError("User not found or deactivated")
    return user


async def get_current_user(
    authorization: str | None = Header(None),
    session: AsyncSession = Depends(get_session),
) -> User:
    try:
        return await _resolve_user(authorization, session)
    except _AuthError as exc:
        raise HTTPException(
            status_code=401, detail=exc.detail, headers={"WWW-Authenticate": "Bearer"}
        ) from exc


async def get_current_user_optional(
    authorization: str | None = Header(None),
    session: AsyncSession = Depends(get_session),
) -> User | None:
    try:
        return await _resolve_user(authorization, session)
    except _AuthError:
        return None
