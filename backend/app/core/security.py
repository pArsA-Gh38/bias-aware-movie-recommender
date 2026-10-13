"""Password hashing and fixed-algorithm bearer token primitives."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from pwdlib import PasswordHash

JWT_ALGORITHM = "HS256"


class InvalidAccessTokenError(ValueError):
    """Raised when an access token fails the local authentication contract."""


class PasswordManager:
    """Argon2 password operations backed by pwdlib recommended settings."""

    def __init__(self, password_hash: PasswordHash | None = None) -> None:
        self._password_hash = password_hash or PasswordHash.recommended()
        self._dummy_hash = self._password_hash.hash("dummy-password-not-an-account")

    def hash(self, password: str) -> str:
        return self._password_hash.hash(password)

    def verify(self, password: str, password_hash: str) -> bool:
        return self._password_hash.verify(password, password_hash)

    def verify_dummy(self, password: str) -> None:
        self._password_hash.verify(password, self._dummy_hash)


class TokenManager:
    """Issue and validate HS256 JWT access tokens."""

    def __init__(
        self,
        *,
        secret_key: str,
        lifetime_minutes: int = 30,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not secret_key.strip():
            raise ValueError("authentication secret must not be empty")
        if not 1 <= lifetime_minutes <= 1_440:
            raise ValueError("access token lifetime must be between 1 and 1440 minutes")
        self._secret_key = secret_key
        self._lifetime = timedelta(minutes=lifetime_minutes)
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def expires_in_seconds(self) -> int:
        return int(self._lifetime.total_seconds())

    def create(self, user_id: UUID) -> str:
        issued_at = self._clock()
        claims = {
            "sub": str(user_id),
            "iat": issued_at,
            "exp": issued_at + self._lifetime,
        }
        return jwt.encode(claims, self._secret_key, algorithm=JWT_ALGORITHM)

    def decode_subject(self, token: str) -> UUID:
        try:
            claims = jwt.decode(
                token,
                self._secret_key,
                algorithms=[JWT_ALGORITHM],
                options={"require": ["sub", "iat", "exp"]},
            )
            subject = claims["sub"]
            if not isinstance(subject, str):
                raise InvalidAccessTokenError("invalid token subject")
            return UUID(subject)
        except (jwt.PyJWTError, KeyError, TypeError, ValueError) as exc:
            raise InvalidAccessTokenError("invalid access token") from exc
