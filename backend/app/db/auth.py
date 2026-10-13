"""SQLAlchemy persistence adapter for local authentication."""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.auth import (
    AuthRepositoryUnavailableError,
    AuthUser,
    DuplicateEmailError,
)
from app.models import User


class SQLAlchemyUserStore:
    """Production user store with explicit transaction ownership."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    @staticmethod
    def _to_auth_user(user: User) -> AuthUser:
        return AuthUser(
            id=user.id,
            email=user.email,
            password_hash=user.password_hash,
            created_at=user.created_at,
            is_active=user.is_active,
        )

    def create_user(self, *, email: str, password_hash: str) -> AuthUser:
        with self._session_factory() as session:
            user = User(email=email, password_hash=password_hash, is_active=True)
            session.add(user)
            try:
                session.commit()
                session.refresh(user)
            except IntegrityError as exc:
                session.rollback()
                raise DuplicateEmailError from exc
            except SQLAlchemyError as exc:
                session.rollback()
                raise AuthRepositoryUnavailableError from exc
            return self._to_auth_user(user)

    def get_by_email(self, email: str) -> AuthUser | None:
        return self._find(select(User).where(User.email == email))

    def get_by_id(self, user_id: UUID) -> AuthUser | None:
        return self._find(select(User).where(User.id == user_id))

    def _find(self, statement: object) -> AuthUser | None:
        try:
            with self._session_factory() as session:
                user = session.scalar(statement)
                return None if user is None else self._to_auth_user(user)
        except SQLAlchemyError as exc:
            raise AuthRepositoryUnavailableError from exc
