"""Tests for auth security helpers."""

from __future__ import annotations

from forecast.gateway.security import (
    LoginThrottle,
    generate_token,
    hash_password,
    hash_token,
    verify_password,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestPasswords:
    def test_roundtrip(self) -> None:
        h = hash_password("correct horse")
        assert verify_password("correct horse", h)
        assert not verify_password("wrong", h)

    def test_none_hash_never_verifies(self) -> None:
        assert not verify_password("anything", None)

    def test_garbage_hash_does_not_raise(self) -> None:
        assert not verify_password("anything", "not-a-bcrypt-hash")

    def test_password_longer_than_72_bytes(self) -> None:
        long_pw = "ş" * 100  # 200 UTF-8 bytes; bcrypt>=5 would raise without truncation
        assert verify_password(long_pw, hash_password(long_pw))


class TestTokens:
    def test_tokens_are_unique_and_long(self) -> None:
        a, b = generate_token(), generate_token()
        assert a != b
        assert len(a) >= 60

    def test_hash_is_deterministic_and_not_identity(self) -> None:
        t = generate_token()
        assert hash_token(t) == hash_token(t)
        assert hash_token(t) != t
        assert len(hash_token(t)) == 64


class TestLoginThrottle:
    def test_locks_after_max_failures(self) -> None:
        clock = FakeClock()
        t = LoginThrottle(max_attempts=3, window_seconds=60, clock=clock)
        for _ in range(2):
            t.record_failure("alice")
        assert t.retry_after("alice") == 0
        t.record_failure("alice")
        assert t.retry_after("alice") == 60

    def test_unlocks_after_window(self) -> None:
        clock = FakeClock()
        t = LoginThrottle(max_attempts=1, window_seconds=60, clock=clock)
        t.record_failure("alice")
        clock.now = 59
        assert t.retry_after("alice") == 1
        clock.now = 60
        assert t.retry_after("alice") == 0

    def test_keys_are_case_insensitive_and_independent(self) -> None:
        t = LoginThrottle(max_attempts=1, window_seconds=60, clock=FakeClock())
        t.record_failure(" Alice ")
        assert t.retry_after("alice") > 0
        assert t.retry_after("bob") == 0

    def test_reset_clears_failures(self) -> None:
        t = LoginThrottle(max_attempts=1, window_seconds=60, clock=FakeClock())
        t.record_failure("alice")
        t.reset("alice")
        assert t.retry_after("alice") == 0
