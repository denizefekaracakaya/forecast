"""End-to-end tests for auth + personalization routes against a temporary SQLite DB."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from forecast.config import jwt_secret, settings
from forecast.data import orm_models  # noqa: F401  (registers tables on Base)
from forecast.data.database import Base, get_session
from forecast.data.orm_models import User
from forecast.gateway.app import app
from forecast.gateway.routes.auth import login_throttle

PASSWORD = "s3cret-pass"


@pytest.fixture
def db_url(tmp_path) -> str:
    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    async def _create() -> None:
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create())
    return url


@pytest.fixture
def client(db_url, monkeypatch):
    # Tests must not depend on the developer's local .env.
    monkeypatch.setattr(settings, "AUTH_REQUIRE_VERIFICATION", True)
    # NullPool: TestClient runs requests on its own event loop, so connections
    # must not be shared across loops.
    engine = create_async_engine(db_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_session] = _session
    login_throttle.reset()
    sent: dict[str, str] = {}

    def _capture_verify(email, username, token, lang="tr"):
        sent["verify"] = token
        return True

    def _capture_reset(email, username, token, lang="tr"):
        sent["reset"] = token
        return True

    with (
        patch("forecast.gateway.routes.auth.send_verification_email", _capture_verify),
        patch("forecast.gateway.routes.auth.send_password_reset_email", _capture_reset),
    ):
        c = TestClient(app)
        c.sent = sent  # type: ignore[attr-defined]
        c.db_url = db_url  # type: ignore[attr-defined]
        yield c

    app.dependency_overrides.clear()
    login_throttle.reset()


def _fetch_user(db_url: str, username: str) -> User:
    async def _get() -> User:
        engine = create_async_engine(db_url)
        async with AsyncSession(engine) as s:
            user = (await s.execute(select(User).where(User.username == username))).scalar_one()
        await engine.dispose()
        return user

    return asyncio.run(_get())


def _register(client, username="alice", email="alice@example.com", password=PASSWORD):
    return client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "password": password},
    )


def _verified_login(client, username="alice") -> tuple[int, dict[str, str]]:
    assert _register(client, username=username, email=f"{username}@example.com").status_code == 200
    assert client.post("/api/v1/auth/verify-email", json={"token": client.sent["verify"]}).json() == {
        "status": "verified"
    }
    resp = client.post("/api/v1/auth/login", json={"login": username, "password": PASSWORD})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body["id"], {"Authorization": f"Bearer {body['token']}"}


class TestRegister:
    def test_register_success(self, client) -> None:
        resp = _register(client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["username"] == "alice"
        assert body["is_verified"] is False
        assert body["token"]

    @pytest.mark.parametrize(
        "payload",
        [
            {"username": "a", "email": "a@example.com", "password": PASSWORD},
            {"username": "alice", "email": "not-an-email", "password": PASSWORD},
            {"username": "alice", "email": "a@example.com", "password": "short"},
            {"username": 123, "email": "a@example.com", "password": PASSWORD},
            {"email": "a@example.com", "password": PASSWORD},
        ],
    )
    def test_invalid_payload_is_422_not_500(self, client, payload) -> None:
        assert client.post("/api/v1/auth/register", json=payload).status_code == 422

    def test_duplicate_username_and_email(self, client) -> None:
        _register(client)
        assert _register(client, email="other@example.com").status_code == 409
        assert _register(client, username="bob").status_code == 409

    def test_email_is_normalized(self, client) -> None:
        resp = _register(client, email="  Alice@Example.COM ")
        assert resp.json()["email"] == "alice@example.com"


class TestLoginAndVerification:
    def test_unverified_login_blocked(self, client) -> None:
        _register(client)
        resp = client.post("/api/v1/auth/login", json={"login": "alice", "password": PASSWORD})
        assert resp.status_code == 403

    def test_verification_token_is_stored_hashed(self, client) -> None:
        _register(client)
        user = _fetch_user(client.db_url, "alice")
        assert user.verification_token
        assert user.verification_token != client.sent["verify"]

    def test_verify_then_login_and_me(self, client) -> None:
        _, headers = _verified_login(client)
        me = client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["is_verified"] is True

    def test_login_by_email(self, client) -> None:
        _verified_login(client)
        resp = client.post(
            "/api/v1/auth/login", json={"login": "ALICE@example.com", "password": PASSWORD}
        )
        assert resp.status_code == 200

    def test_wrong_password(self, client) -> None:
        _verified_login(client)
        resp = client.post("/api/v1/auth/login", json={"login": "alice", "password": "nope-nope"})
        assert resp.status_code == 401

    def test_brute_force_lockout(self, client) -> None:
        _verified_login(client)
        for _ in range(settings.LOGIN_MAX_ATTEMPTS):
            client.post("/api/v1/auth/login", json={"login": "alice", "password": "bad-pass"})
        resp = client.post("/api/v1/auth/login", json={"login": "alice", "password": PASSWORD})
        assert resp.status_code == 429
        assert int(resp.headers["Retry-After"]) > 0

    def test_resend_verification_does_not_reveal_accounts(self, client) -> None:
        unknown = client.post("/api/v1/auth/resend-verification", json={"email": "x@example.com"})
        assert unknown.json() == {"status": "sent"}


class TestPasswordReset:
    def test_reset_flow(self, client) -> None:
        _verified_login(client)
        assert client.post(
            "/api/v1/auth/forgot-password", json={"email": "alice@example.com"}
        ).json() == {"status": "sent"}
        token = client.sent["reset"]

        resp = client.post(
            "/api/v1/auth/reset-password", json={"token": token, "password": "brand-new-pass"}
        )
        assert resp.json() == {"status": "password_reset"}

        # Token is single-use
        again = client.post(
            "/api/v1/auth/reset-password", json={"token": token, "password": "another-pass"}
        )
        assert again.status_code == 404

        login = client.post(
            "/api/v1/auth/login", json={"login": "alice", "password": "brand-new-pass"}
        )
        assert login.status_code == 200

    def test_change_password_requires_current(self, client) -> None:
        _, headers = _verified_login(client)
        bad = client.post(
            "/api/v1/auth/change-password",
            json={"current_password": "wrong-one", "new_password": "brand-new-pass"},
            headers=headers,
        )
        assert bad.status_code == 401
        ok = client.post(
            "/api/v1/auth/change-password",
            json={"current_password": PASSWORD, "new_password": "brand-new-pass"},
            headers=headers,
        )
        assert ok.json() == {"status": "password_changed"}


class TestTokens:
    def test_missing_token(self, client) -> None:
        assert client.get("/api/v1/auth/me").status_code == 401

    def test_token_with_non_numeric_sub_is_401(self, client) -> None:
        token = jwt.encode(
            {"sub": "not-a-number", "exp": 9_999_999_999}, jwt_secret(), algorithm="HS256"
        )
        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_token_signed_with_other_key_is_401(self, client) -> None:
        token = jwt.encode({"sub": "1", "exp": 9_999_999_999}, "a-completely-different-signing-key-0123456789", algorithm="HS256")
        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401


class TestPersonalization:
    def test_preferences_can_be_updated_repeatedly(self, client) -> None:
        # Regression: preferences used to be looked up by their own PK instead of
        # user_id, so a second user's update tried to insert a duplicate row.
        _verified_login(client, "first")
        uid, headers = _verified_login(client, "second")
        for bias in ("bullish", "bearish"):
            resp = client.put(
                f"/api/v1/users/{uid}/preferences",
                json={"sentiment_bias": bias, "preferred_sectors": ["technology"]},
                headers=headers,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["sentiment_bias"] == bias
            assert resp.json()["preferred_sectors"] == ["technology"]

    def test_invalid_preference_value_is_422(self, client) -> None:
        uid, headers = _verified_login(client)
        resp = client.put(
            f"/api/v1/users/{uid}/preferences", json={"risk_level": "yolo"}, headers=headers
        )
        assert resp.status_code == 422

    def test_watchlist_roundtrip(self, client) -> None:
        uid, headers = _verified_login(client)
        base = f"/api/v1/users/{uid}/watchlist"
        assert client.post(base, json={"ticker": "aapl"}, headers=headers).json()["status"] == "added"
        assert (
            client.post(base, json={"ticker": "AAPL"}, headers=headers).json()["status"]
            == "already_exists"
        )
        assert client.get(base, headers=headers).json() == ["AAPL"]
        assert client.post(base, json={"ticker": "NOPE"}, headers=headers).status_code == 400
        assert client.delete(f"{base}/AAPL", headers=headers).status_code == 200
        assert client.delete(f"{base}/AAPL", headers=headers).status_code == 404

    def test_other_users_data_is_forbidden(self, client) -> None:
        uid_a, _ = _verified_login(client, "first")
        _, headers_b = _verified_login(client, "second")
        assert client.get(f"/api/v1/users/{uid_a}/watchlist", headers=headers_b).status_code == 403

    def test_feedback_is_upserted_and_validated(self, client) -> None:
        uid, headers = _verified_login(client)
        base = f"/api/v1/users/{uid}/feedback"
        for rating in (1, -1):
            resp = client.post(
                base, json={"news_id": "n1", "ticker": "aapl", "rating": rating}, headers=headers
            )
            assert resp.status_code == 200
        items = client.get(base, headers=headers).json()
        assert len(items) == 1
        assert items[0]["rating"] == -1
        bad = client.post(base, json={"news_id": "n1", "rating": 5}, headers=headers)
        assert bad.status_code == 422
