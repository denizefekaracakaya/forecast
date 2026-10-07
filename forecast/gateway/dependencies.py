"""FastAPI dependencies — JWT auth, ownership checks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import jwt
from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from forecast.config import settings
from forecast.data.database import get_session
from forecast.data.orm_models import User

ALGORITHM = settings.JWT_ALGORITHM
SECRET = settings.JWT_SECRET_KEY


def create_jwt(user: User) -> str:
    payload: dict[str, Any] = {
        "sub": str(user.id),
        "email": user.email,
        "username": user.username,
        "exp": datetime.now(timezone.utc).timestamp() + settings.JWT_EXPIRE_HOURS * 3600,
        "iat": datetime.now(timezone.utc).timestamp(),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


async def get_current_user(
    authorization: str | None = Header(None),
    session: AsyncSession = Depends(get_session),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization[7:]
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await session.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or deactivated")
    return user


async def get_current_user_optional(
    authorization: str | None = Header(None),
    session: AsyncSession = Depends(get_session),
) -> User | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
    except Exception:
        return None
    user = await session.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        return None
    return user
