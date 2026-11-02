"""Auth API — register, login, verify email, password reset, profile."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from forecast.config import settings
from forecast.data.database import get_session
from forecast.data.email import send_password_reset_email, send_verification_email
from forecast.data.orm_models import User, UserPreference
from forecast.gateway.dependencies import create_jwt, get_current_user
from forecast.gateway.schemas import (
    AddPasswordRequest,
    ChangePasswordRequest,
    EmailRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenRequest,
)
from forecast.gateway.security import (
    LoginThrottle,
    generate_token,
    hash_password,
    hash_token,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

VERIFICATION_TTL = timedelta(hours=24)
RESET_TTL = timedelta(hours=1)

login_throttle = LoginThrottle(
    max_attempts=settings.LOGIN_MAX_ATTEMPTS,
    window_seconds=settings.LOGIN_LOCKOUT_SECONDS,
)

# Checked against when the login identifier is unknown so that response time
# does not reveal which accounts exist.
_DUMMY_HASH = hash_password("timing-equalizer")


def _now() -> datetime:
    return datetime.now(UTC)


def _is_expired(expires: datetime | None) -> bool:
    if expires is None:
        return False
    if expires.tzinfo is None:  # SQLite drops tzinfo
        expires = expires.replace(tzinfo=UTC)
    return expires < _now()


def _issue_verification_token(user: User) -> str:
    token = generate_token()
    user.verification_token = hash_token(token)
    user.verification_token_expires = _now() + VERIFICATION_TTL
    return token


async def _user_by_email(session: AsyncSession, email: str) -> User | None:
    result = await session.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


# ── Register ──────────────────────────────────────────────────────────────


@router.post("/register")
async def register(
    body: RegisterRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    existing_user = await session.execute(select(User).where(User.username == body.username))
    if existing_user.scalar_one_or_none():
        raise HTTPException(409, "Username already taken")

    if await _user_by_email(session, body.email):
        raise HTTPException(409, "Email already registered")

    user = User(
        username=body.username,
        email=body.email,
        pwd_hash=hash_password(body.password),
        is_verified=False,
        language=body.language,
    )
    verification_token = _issue_verification_token(user)
    session.add(user)
    await session.flush()

    session.add(UserPreference(user_id=user.id))
    await session.commit()
    await session.refresh(user)

    send_verification_email(body.email, body.username, verification_token, lang=user.language)

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_verified": user.is_verified,
        "token": create_jwt(user),
    }


# ── Login ─────────────────────────────────────────────────────────────────


@router.post("/login")
async def login(
    body: LoginRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    login_id = body.login.strip()

    retry_after = login_throttle.retry_after(login_id)
    if retry_after > 0:
        raise HTTPException(
            429,
            "Too many failed login attempts. Try again later.",
            headers={"Retry-After": str(math.ceil(retry_after))},
        )

    result = await session.execute(
        select(User).where((User.email == login_id.lower()) | (User.username == login_id))
    )
    user = result.scalar_one_or_none()

    password_ok = verify_password(body.password, user.pwd_hash if user else _DUMMY_HASH)
    if user is None or not password_ok:
        login_throttle.record_failure(login_id)
        raise HTTPException(401, "Invalid email/username or password")

    login_throttle.reset(login_id)

    if not user.is_active:
        raise HTTPException(403, "Account is deactivated")

    if settings.AUTH_REQUIRE_VERIFICATION and not user.is_verified and user.email:
        raise HTTPException(403, "Email not verified")

    user.last_login = _now()
    await session.commit()

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_verified": user.is_verified,
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "language": user.language,
        "token": create_jwt(user),
    }


# ── Verify Email ──────────────────────────────────────────────────────────


@router.post("/verify-email")
async def verify_email(
    body: TokenRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await session.execute(
        select(User).where(User.verification_token == hash_token(body.token))
    )
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(404, "Invalid verification token")
    if user.is_verified:
        return {"status": "already_verified"}
    if _is_expired(user.verification_token_expires):
        raise HTTPException(410, "Verification token expired")

    user.is_verified = True
    user.verification_token = None
    user.verification_token_expires = None
    await session.commit()

    return {"status": "verified"}


# ── Resend Verification ───────────────────────────────────────────────────


@router.post("/resend-verification")
async def resend_verification(
    body: EmailRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    user = await _user_by_email(session, body.email)

    # Same response for unknown and known addresses — avoids account enumeration.
    if user is None or user.is_verified:
        return {"status": "sent"}

    verification_token = _issue_verification_token(user)
    await session.commit()

    send_verification_email(user.email, user.username, verification_token, lang=user.language)
    return {"status": "sent"}


# ── Forgot Password ───────────────────────────────────────────────────────


@router.post("/forgot-password")
async def forgot_password(
    body: EmailRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    user = await _user_by_email(session, body.email)
    if user is None:
        return {"status": "sent"}

    reset_token = generate_token()
    user.reset_token = hash_token(reset_token)
    user.reset_token_expires = _now() + RESET_TTL
    await session.commit()

    send_password_reset_email(user.email, user.username, reset_token, lang=user.language)
    return {"status": "sent"}


# ── Reset Password ────────────────────────────────────────────────────────


@router.post("/reset-password")
async def reset_password(
    body: ResetPasswordRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await session.execute(select(User).where(User.reset_token == hash_token(body.token)))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(404, "Invalid reset token")
    if _is_expired(user.reset_token_expires):
        raise HTTPException(410, "Reset token expired")

    user.pwd_hash = hash_password(body.password)
    user.reset_token = None
    user.reset_token_expires = None
    await session.commit()
    login_throttle.reset(user.username)
    if user.email:
        login_throttle.reset(user.email)

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
    body: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    if not verify_password(body.current_password, current_user.pwd_hash):
        raise HTTPException(401, "Current password is incorrect")

    current_user.pwd_hash = hash_password(body.new_password)
    await session.commit()

    return {"status": "password_changed"}


# ── Add Password (migration for existing users) ───────────────────────────


@router.post("/add-password")
async def add_password(
    body: AddPasswordRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    if current_user.email:
        raise HTTPException(400, "User already has an email set")

    if await _user_by_email(session, body.email):
        raise HTTPException(409, "Email already registered by another user")

    current_user.email = body.email
    current_user.pwd_hash = hash_password(body.password)
    current_user.is_verified = False
    verification_token = _issue_verification_token(current_user)
    await session.commit()

    send_verification_email(
        body.email, current_user.username, verification_token, lang=current_user.language
    )

    return {
        "status": "password_added",
        "email": body.email,
        "is_verified": False,
        "token": create_jwt(current_user),
    }
