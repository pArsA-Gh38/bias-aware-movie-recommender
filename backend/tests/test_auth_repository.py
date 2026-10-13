"""Focused integration coverage for the production SQLAlchemy auth adapter."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.auth import DuplicateEmailError
from app.db.auth import SQLAlchemyUserStore
from app.models import User


class TrackingSession(Session):
    commits = 0
    rollbacks = 0

    def commit(self) -> None:
        super().commit()
        TrackingSession.commits += 1

    def rollback(self) -> None:
        TrackingSession.rollbacks += 1
        super().rollback()


def test_sqlalchemy_store_create_lookup_and_integrity_rollback() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    User.__table__.create(engine)
    TrackingSession.commits = 0
    TrackingSession.rollbacks = 0
    factory = sessionmaker(
        bind=engine,
        class_=TrackingSession,
        autoflush=False,
        expire_on_commit=False,
    )
    store = SQLAlchemyUserStore(factory)

    created = store.create_user(
        email="user@example.com", password_hash="$argon2$not-plaintext"
    )

    assert store.get_by_email("user@example.com") == created
    assert store.get_by_id(created.id) == created
    assert TrackingSession.commits == 1

    try:
        store.create_user(email="user@example.com", password_hash="different")
    except DuplicateEmailError:
        pass
    else:
        raise AssertionError("duplicate normalized email must fail")

    assert TrackingSession.rollbacks == 1
    second = store.create_user(email="second@example.com", password_hash="hashed")
    assert second.email == "second@example.com"
    assert TrackingSession.commits == 2
    engine.dispose()
