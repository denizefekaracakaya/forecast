"""Auth security helpers — password hashing, one-time token hashing, login throttling."""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections.abc import Callable

import bcrypt

# bcrypt only looks at the first 72 bytes; bcrypt>=5 raises instead of truncating.
# Truncate explicitly so long passphrases work and old hashes keep verifying.
_BCRYPT_MAX_BYTES = 72


def _pw_bytes(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_pw_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, pwd_hash: str | None) -> bool:
    if not pwd_hash:
        return False
    try:
        return bcrypt.checkpw(_pw_bytes(password), pwd_hash.encode("utf-8"))
    except Exception:
        return False


def generate_token() -> str:
    """Return a URL-safe one-time token to send to the user (never stored as-is)."""
    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:
    """Digest stored in the DB for verification/reset tokens.

    A DB leak then does not hand out usable reset links. SHA-256 is fine here
    (unlike for passwords) because the tokens carry 384 bits of entropy.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class LoginThrottle:
    """Fixed-window failed-login counter keyed by login identifier.

    After ``max_attempts`` failures within ``window_seconds`` the key is locked
    until the window that started with the first failure has elapsed.
    Process-local; swap for Redis when running several gateway replicas.
    """

    def __init__(
        self,
        max_attempts: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_attempts
        self._window = window_seconds
        self._clock = clock
        self._failures: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(identifier: str) -> str:
        return identifier.strip().lower()

    def retry_after(self, identifier: str) -> float:
        """Seconds until the identifier may try again (0 when not locked)."""
        key = self._key(identifier)
        with self._lock:
            entry = self._failures.get(key)
            if entry is None:
                return 0.0
            started, count = entry
            remaining = self._window - (self._clock() - started)
            if remaining <= 0:
                del self._failures[key]
                return 0.0
            return remaining if count >= self._max else 0.0

    def record_failure(self, identifier: str) -> None:
        key = self._key(identifier)
        now = self._clock()
        with self._lock:
            started, count = self._failures.get(key, (now, 0))
            if now - started >= self._window:
                started, count = now, 0
            self._failures[key] = (started, count + 1)

    def reset(self, identifier: str | None = None) -> None:
        with self._lock:
            if identifier is None:
                self._failures.clear()
            else:
                self._failures.pop(self._key(identifier), None)
