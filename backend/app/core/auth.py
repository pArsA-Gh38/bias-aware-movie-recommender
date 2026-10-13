"""Authentication use cases independent of HTTP and SQLAlchemy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.core.security import InvalidAccessTokenError, PasswordManager, TokenManager


@dataclass(frozen=True, slots=True)
class AuthUser:
    id: UUID
    email: str
    password_hash: str
    created_at: datetime
    is_active: bool


class DuplicateEmailError(ValueError):
    """Raised when the normalized email is already persisted."""


class AuthenticationFailedError(ValueError):
    """Generic credential or bearer authentication failure."""


class AuthRepositoryUnavailableError(RuntimeError):
    """Raised when persistence cannot serve an authentication request."""


class UserStore(Protocol):
    def create_user(self, *, email: str, password_hash: str) -> AuthUser: ...

    def get_by_email(self, email: str) -> AuthUser | None: ...

    def get_by_id(self, user_id: UUID) -> AuthUser | None: ...


@dataclass(frozen=True, slots=True)
class AccessToken:
    value: str
    expires_in: int


class AuthService:
    """Coordinate password/token primitives with a user persistence boundary."""

    def __init__(
        self,
        *,
        users: UserStore,
        passwords: PasswordManager,
        tokens: TokenManager,
    ) -> None:
        self._users = users
        self._passwords = passwords
        self._tokens = tokens

    def register(self, *, email: str, password: str) -> AuthUser:
        if self._users.get_by_email(email) is not None:
            raise DuplicateEmailError
        password_hash = self._passwords.hash(password)
        return self._users.create_user(email=email, password_hash=password_hash)

    def login(self, *, email: str, password: str) -> AccessToken:
        user = self._users.get_by_email(email)
        if user is None:
            self._passwords.verify_dummy(password)
            raise AuthenticationFailedError
        if (
            not self._passwords.verify(password, user.password_hash)
            or not user.is_active
        ):
            raise AuthenticationFailedError
        return AccessToken(
            value=self._tokens.create(user.id),
            expires_in=self._tokens.expires_in_seconds,
        )

    def current_user(self, token: str) -> AuthUser:
        try:
            user_id = self._tokens.decode_subject(token)
        except InvalidAccessTokenError as exc:
            raise AuthenticationFailedError from exc
        user = self._users.get_by_id(user_id)
        if user is None or not user.is_active:
            raise AuthenticationFailedError
        return user
