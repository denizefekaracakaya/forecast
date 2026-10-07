"""Auth API — register, login, verify email, password reset, profile."""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from forecast.config import settings
from forecast.data.database import get_session
from forecast.data.email import send_password_reset_email, send_verification_email
from forecast.data.orm_models import User, UserPreference
from forecast.gateway.dependencies import create_jwt, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, pwd_hash: str | None) -> bool:
    if not pwd_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), pwd_hash.encode("utf-8"))
    except Exception:
        return False


def _generate_token() -> str:
    return secrets.token_urlsafe(48)


# ── Register ──────────────────────────────────────────────────────────────


@router.post("/register")
async def register(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> dict:
    username = body.get("username", "").strip()
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    if not username or len(username) < 2:
        raise HTTPException(400, "Username must be at least 2 characters")
    if not email or "@" not in email:
        raise HTTPException(400, "Valid email is required")
    if not password or len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    existing_user = await session.execute(select(User).where(User.username == username))
    if existing_user.scalar_one_or_none():
        raise HTTPException(409, "Username already taken")

    existing_email = await session.execute(select(User).where(User.email == email))
    if existing_email.scalar_one_or_none():
        raise HTTPException(409, "Email already registered")

    verification_token = _generate_token()
    verification_expires = datetime.now(timezone.utc) + timedelta(hours=24)

    user = User(
        username=username,
        email=email,
        pwd_hash=_hash_password(password),
        is_verified=False,
        verification_token=verification_token,
        verification_token_expires=verification_expires,
        language=body.get("language", "tr"),
    )
    session.add(user)
    await session.flush()

    prefs = UserPreference(user_id=user.id)
    session.add(prefs)
    await session.commit()
    await session.refresh(user)

    send_verification_email(email, username, verification_token, lang=user.language)

    token = create_jwt(user)
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_verified": user.is_verified,
        "token": token,
    }


# ── Login ─────────────────────────────────────────────────────────────────


@router.post("/login")
async def login(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> dict:
    login_id = body.get("login", "").strip()
    password = body.get("password", "")

    if not login_id or not password:
        raise HTTPException(400, "Login and password are required")

    user = await session.execute(
        select(User).where(
            (User.email == login_id.lower()) | (User.username == login_id)
        )
    )
    user = user.scalar_one_or_none()

    if user is None or not _verify_password(password, user.pwd_hash):
        raise HTTPException(401, "Invalid email/username or password")

    if not user.is_active:
        raise HTTPException(403, "Account is deactivated")

    if settings.AUTH_REQUIRE_VERIFICATION and not user.is_verified and user.email:
        raise HTTPException(403, "Email not verified")

    user.last_login = datetime.now(timezone.utc)
    await session.commit()

    token = create_jwt(user)
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_verified": user.is_verified,
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "language": user.language,
        "token": token,
    }


# ── Verify Email ──────────────────────────────────────────────────────────


@router.post("/verify-email")
async def verify_email(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> dict:
    token = body.get("token", "")
    if not token:
        raise HTTPException(400, "Token is required")

    user = await session.execute(
        select(User).where(User.verification_token == token)
    )
    user = user.scalar_one_or_none()

    if user is None:
        raise HTTPException(404, "Invalid verification token")
    if user.is_verified:
        return {"status": "already_verified"}
    if user.verification_token_expires and user.verification_token_expires < datetime.now(timezone.utc):
        raise HTTPException(410, "Verification token expired")

    user.is_verified = True
    user.verification_token = None
    user.verification_token_expires = None
    await session.commit()

    return {"status": "verified"}


# ── Resend Verification ───────────────────────────────────────────────────


@router.post("/resend-verification")
async def resend_verification(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> dict:
    email = body.get("email", "").strip().lower()
    if not email:
        raise HTTPException(400, "Email is required")

    user = await session.execute(select(User).where(User.email == email))
    user = user.scalar_one_or_none()

    if user is None:
        return {"status": "sent"}
    if user.is_verified:
        return {"status": "already_verified"}

    verification_token = _generate_token()
    user.verification_token = verification_token
    user.verification_token_expires = datetime.now(timezone.utc) + timedelta(hours=24)
    await session.commit()

    send_verification_email(user.email, user.username, verification_token, lang=user.language)
    return {"status": "sent"}


# ── Forgot Password ───────────────────────────────────────────────────────


@router.post("/forgot-password")
async def forgot_password(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> dict:
    email = body.get("email", "").strip().lower()
    if not email:
        raise HTTPException(400, "Email is required")

    user = await session.execute(select(User).where(User.email == email))
    user = user.scalar_one_or_none()

    if user is None:
        return {"status": "sent"}

    reset_token = _generate_token()
    user.reset_token = reset_token
    user.reset_token_expires = datetime.now(timezone.utc) + timedelta(hours=1)
    await session.commit()

    send_password_reset_email(user.email, user.username, reset_token, lang=user.language)
    return {"status": "sent"}


# ── Reset Password ────────────────────────────────────────────────────────


@router.post("/reset-password")
async def reset_password(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> dict:
    token = body.get("token", "")
    password = body.get("password", "")

    if not token or not password:
        raise HTTPException(400, "Token and password are required")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    user = await session.execute(select(User).where(User.reset_token == token))
    user = user.scalar_one_or_none()

    if user is None:
        raise HTTPException(404, "Invalid reset token")
    if user.reset_token_expires and user.reset_token_expires < datetime.now(timezone.utc):
        raise HTTPException(410, "Reset token expired")

    user.pwd_hash = _hash_password(password)
    user.reset_token = None
    user.reset_token_expires = None
    await session.commit()

    return {"status": "password_reset"}


# ── Me (get current user) ────────────────────────────────────────────────


@router.get("/me")
async def get_me(
    current_user: User = Depends(get_current_user),
) -> dict:
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "is_verified": current_user.is_verified,
        "language": current_user.language,
        "has_password": current_user.pwd_hash is not None,
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
    }


# ── Change Password ────────────────────────────────────────────────────────


@router.post("/change-password")
async def change_password(
    body: dict[str, Any],
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    current_password = body.get("current_password", "")
    new_password = body.get("new_password", "")

    if not current_password or not new_password:
        raise HTTPException(400, "Both current and new passwords are required")
    if len(new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    if not _verify_password(current_password, current_user.pwd_hash):
        raise HTTPException(401, "Current password is incorrect")

    current_user.pwd_hash = _hash_password(new_password)
    await session.commit()

    return {"status": "password_changed"}


# ── Add Password (migration for existing users) ───────────────────────────


@router.post("/add-password")
async def add_password(
    body: dict[str, Any],
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    if not email or "@" not in email:
        raise HTTPException(400, "Valid email is required")
    if not password or len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    if current_user.email:
        raise HTTPException(400, "User already has an email set")

    existing_email = await session.execute(select(User).where(User.email == email))
    if existing_email.scalar_one_or_none():
        raise HTTPException(409, "Email already registered by another user")

    current_user.email = email
    current_user.pwd_hash = _hash_password(password)
    current_user.is_verified = False
    verification_token = _generate_token()
    current_user.verification_token = verification_token
    current_user.verification_token_expires = datetime.now(timezone.utc) + timedelta(hours=24)
    await session.commit()

    send_verification_email(email, current_user.username, verification_token, lang=current_user.language)

    new_token = create_jwt(current_user)
    return {
        "status": "password_added",
        "email": email,
        "is_verified": False,
        "token": new_token,
    }
