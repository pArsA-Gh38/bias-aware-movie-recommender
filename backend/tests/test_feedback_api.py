"""Behavioral coverage for API-006 Slice A."""

from datetime import UTC, datetime
from typing import Self
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.api.auth import AUTHENTICATION_FAILED
from app.api.feedback import EVENT_NOT_FOUND, FEEDBACK_UNAVAILABLE, MOVIE_NOT_IN_EVENT
from app.core.auth import AuthService, AuthUser, DuplicateEmailError
from app.core.config import Settings
from app.core.feedback import (
    FeedbackEventNotFoundError,
    FeedbackRelationshipConflictError,
    FeedbackService,
    FeedbackUnavailableError,
)
from app.core.security import PasswordManager, TokenManager
from app.db.feedback import SQLAlchemyFeedbackRepository
from app.main import create_app
from app.schemas.feedback import FeedbackAction

SECRET = "feedback-test-secret-with-sufficient-entropy"


class MemoryUserStore:
    def __init__(self, user: AuthUser) -> None:
        self.user = user

    def create_user(self, *, email: str, password_hash: str) -> AuthUser:
        raise DuplicateEmailError

    def get_by_email(self, email: str) -> AuthUser | None:
        return self.user if email == self.user.email else None

    def get_by_id(self, user_id: UUID) -> AuthUser | None:
        return self.user if user_id == self.user.id else None


class FakeFeedbackRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, UUID, int, FeedbackAction]] = []
        self.failure: Exception | None = None

    def record(
        self, *, user_id: UUID, event_id: UUID, movie_id: int, action: FeedbackAction
    ) -> None:
        if self.failure:
            raise self.failure
        self.calls.append((user_id, event_id, movie_id, action))


def build_client() -> tuple[TestClient, AuthUser, str, FakeFeedbackRepository]:
    user = AuthUser(
        id=uuid4(),
        email="feedback@example.com",
        password_hash="unused",
        created_at=datetime.now(UTC),
        is_active=True,
    )
    tokens = TokenManager(secret_key=SECRET)
    auth = AuthService(
        users=MemoryUserStore(user),
        passwords=PasswordManager(),
        tokens=tokens,
    )
    repository = FakeFeedbackRepository()
    client = TestClient(
        create_app(
            settings=Settings(),
            auth_service=auth,
            feedback_service=FeedbackService(repository),
        )
    )
    return client, user, tokens.create(user.id), repository


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def payload(*, action: str = "click") -> dict[str, object]:
    return {"event_id": str(uuid4()), "movie_id": 1193, "action": action}


def test_feedback_requires_authentication() -> None:
    client, _, _, _ = build_client()
    response = client.post("/feedback", json=payload())
    assert response.status_code == 401
    assert response.json() == {"detail": AUTHENTICATION_FAILED}


@pytest.mark.parametrize("action", ["click", "like", "dislike"])
def test_valid_feedback_uses_authenticated_user(action: str) -> None:
    client, user, token, repository = build_client()
    request = payload(action=action)
    response = client.post("/feedback", json=request, headers=bearer(token))
    assert response.status_code == 201
    assert response.json() == {"status": "recorded"}
    assert repository.calls == [
        (user.id, UUID(str(request["event_id"])), 1193, FeedbackAction(action))
    ]


@pytest.mark.parametrize(
    "request_body",
    [
        {"event_id": "not-a-uuid", "movie_id": 1, "action": "click"},
        {"event_id": str(uuid4()), "movie_id": "not-an-int", "action": "click"},
        {"event_id": str(uuid4()), "movie_id": True, "action": "click"},
        {"event_id": str(uuid4()), "movie_id": 1, "action": "share"},
        {"event_id": str(uuid4()), "movie_id": 1},
        {
            "event_id": str(uuid4()),
            "movie_id": 1,
            "action": "click",
            "user_id": str(uuid4()),
        },
    ],
)
def test_invalid_feedback_is_422(request_body: dict[str, object]) -> None:
    client, _, token, repository = build_client()
    response = client.post("/feedback", json=request_body, headers=bearer(token))
    assert response.status_code == 422
    assert repository.calls == []


@pytest.mark.parametrize(
    ("failure", "status_code", "detail"),
    [
        (FeedbackEventNotFoundError(), 404, EVENT_NOT_FOUND),
        (FeedbackRelationshipConflictError(), 409, MOVIE_NOT_IN_EVENT),
        (FeedbackUnavailableError("raw database secret"), 503, FEEDBACK_UNAVAILABLE),
    ],
)
def test_domain_failures_have_stable_public_errors(
    failure: Exception, status_code: int, detail: str
) -> None:
    client, _, token, repository = build_client()
    repository.failure = failure
    response = client.post("/feedback", json=payload(), headers=bearer(token))
    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert "secret" not in response.text


def test_openapi_documents_feedback_contract() -> None:
    client, _, _, _ = build_client()
    operation = client.app.openapi()["paths"]["/feedback"]["post"]
    assert operation["security"] == [{"HTTPBearer": []}]
    assert {"201", "401", "404", "409", "422", "503"} <= set(operation["responses"])
    action_schema = client.app.openapi()["components"]["schemas"]["FeedbackAction"]
    assert action_schema["enum"] == ["click", "like", "dislike"]


class FailingSession:
    def __init__(self, *, failure_at: str, rollback_fails: bool) -> None:
        self.failure_at = failure_at
        self.rollback_fails = rollback_fails
        self.scalar_calls = 0
        self.rollback_calls = 0
        self.commit_calls = 0
        self.add_calls = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def scalar(self, statement: object) -> UUID:
        self.scalar_calls += 1
        if self.failure_at == "query":
            raise SQLAlchemyError("raw query message")
        return uuid4()

    def add(self, value: object) -> None:
        self.add_calls += 1

    def commit(self) -> None:
        self.commit_calls += 1
        if self.failure_at == "commit":
            raise SQLAlchemyError("raw commit message")

    def rollback(self) -> None:
        self.rollback_calls += 1
        if self.rollback_fails:
            raise SQLAlchemyError("raw rollback message")


@pytest.mark.parametrize("failure_at", ["query", "commit"])
@pytest.mark.parametrize("rollback_fails", [False, True])
def test_repository_preserves_database_failure_when_rollback_fails(
    failure_at: str,
    rollback_fails: bool,
) -> None:
    session = FailingSession(
        failure_at=failure_at,
        rollback_fails=rollback_fails,
    )
    repository = SQLAlchemyFeedbackRepository(lambda: session)  # type: ignore[arg-type]
    with pytest.raises(
        FeedbackUnavailableError,
        match="feedback transaction failed",
    ) as raised:
        repository.record(
            user_id=uuid4(),
            event_id=uuid4(),
            movie_id=1,
            action=FeedbackAction.CLICK,
        )
    assert isinstance(raised.value.__cause__, SQLAlchemyError)
    assert str(raised.value.__cause__) == f"raw {failure_at} message"
    assert session.rollback_calls == 1
    assert session.scalar_calls == (2 if failure_at == "commit" else 1)
    assert session.commit_calls == (1 if failure_at == "commit" else 0)
    assert session.add_calls == (1 if failure_at == "commit" else 0)


@pytest.mark.parametrize("failure_at", ["query", "commit"])
@pytest.mark.parametrize("rollback_fails", [False, True])
def test_database_and_rollback_messages_are_sanitized_by_api(
    failure_at: str,
    rollback_fails: bool,
) -> None:
    session = FailingSession(
        failure_at=failure_at,
        rollback_fails=rollback_fails,
    )
    repository = SQLAlchemyFeedbackRepository(lambda: session)  # type: ignore[arg-type]
    user = AuthUser(
        id=uuid4(),
        email="failure@example.com",
        password_hash="unused",
        created_at=datetime.now(UTC),
        is_active=True,
    )
    tokens = TokenManager(secret_key=SECRET)
    auth = AuthService(
        users=MemoryUserStore(user),
        passwords=PasswordManager(),
        tokens=tokens,
    )
    client = TestClient(
        create_app(
            settings=Settings(),
            auth_service=auth,
            feedback_service=FeedbackService(repository),
        )
    )

    response = client.post(
        "/feedback",
        json=payload(),
        headers=bearer(tokens.create(user.id)),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": FEEDBACK_UNAVAILABLE}
    assert "raw query message" not in response.text
    assert "raw commit message" not in response.text
    assert "raw rollback message" not in response.text
    assert session.rollback_calls == 1
    assert session.scalar_calls == (2 if failure_at == "commit" else 1)
    assert session.commit_calls == (1 if failure_at == "commit" else 0)


class DomainErrorSession:
    def __init__(self, *, domain_error: str, rollback_fails: bool) -> None:
        self.domain_error = domain_error
        self.rollback_fails = rollback_fails
        self.scalar_calls = 0
        self.rollback_calls = 0
        self.commit_calls = 0
        self.add_calls = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def scalar(self, statement: object) -> UUID | None:
        self.scalar_calls += 1
        if self.domain_error == "not_found" or self.scalar_calls == 2:
            return None
        return uuid4()

    def add(self, value: object) -> None:
        self.add_calls += 1

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1
        if self.rollback_fails:
            raise SQLAlchemyError("raw domain rollback message")


@pytest.mark.parametrize(
    ("domain_error", "expected_error"),
    [
        ("not_found", FeedbackEventNotFoundError),
        ("conflict", FeedbackRelationshipConflictError),
    ],
)
@pytest.mark.parametrize("rollback_fails", [False, True])
def test_domain_error_rollback_translation(
    domain_error: str,
    expected_error: type[Exception],
    rollback_fails: bool,
) -> None:
    session = DomainErrorSession(
        domain_error=domain_error,
        rollback_fails=rollback_fails,
    )
    repository = SQLAlchemyFeedbackRepository(lambda: session)  # type: ignore[arg-type]
    translated_error = FeedbackUnavailableError if rollback_fails else expected_error

    with pytest.raises(translated_error) as raised:
        repository.record(
            user_id=uuid4(),
            event_id=uuid4(),
            movie_id=1,
            action=FeedbackAction.CLICK,
        )

    if rollback_fails:
        assert isinstance(raised.value.__cause__, SQLAlchemyError)
        assert str(raised.value.__cause__) == "raw domain rollback message"
    assert session.rollback_calls == 1
    assert session.scalar_calls == (1 if domain_error == "not_found" else 2)
    assert session.add_calls == 0
    assert session.commit_calls == 0


@pytest.mark.parametrize("domain_error", ["not_found", "conflict"])
def test_domain_rollback_failure_is_sanitized_by_api(domain_error: str) -> None:
    session = DomainErrorSession(domain_error=domain_error, rollback_fails=True)
    repository = SQLAlchemyFeedbackRepository(lambda: session)  # type: ignore[arg-type]
    user = AuthUser(
        id=uuid4(),
        email="domain-failure@example.com",
        password_hash="unused",
        created_at=datetime.now(UTC),
        is_active=True,
    )
    tokens = TokenManager(secret_key=SECRET)
    auth = AuthService(
        users=MemoryUserStore(user),
        passwords=PasswordManager(),
        tokens=tokens,
    )
    client = TestClient(
        create_app(
            settings=Settings(),
            auth_service=auth,
            feedback_service=FeedbackService(repository),
        )
    )

    response = client.post(
        "/feedback",
        json=payload(),
        headers=bearer(tokens.create(user.id)),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": FEEDBACK_UNAVAILABLE}
    assert "raw domain rollback message" not in response.text
    assert session.rollback_calls == 1
    assert session.scalar_calls == (1 if domain_error == "not_found" else 2)
    assert session.add_calls == 0
    assert session.commit_calls == 0
