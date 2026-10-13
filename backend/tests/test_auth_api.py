"""HTTP and service integration coverage for API-002."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi.testclient import TestClient

from app.api.auth import AUTHENTICATION_FAILED, AUTHENTICATION_UNAVAILABLE
from app.core.auth import (
    AuthRepositoryUnavailableError,
    AuthService,
    AuthUser,
    DuplicateEmailError,
)
from app.core.config import Settings
from app.core.security import JWT_ALGORITHM, PasswordManager, TokenManager
from app.main import create_app

SECRET = "api-test-secret-with-sufficient-entropy"
VALID_PASSWORD = "password-123"


class MemoryUserStore:
    def __init__(self) -> None:
        self.users: dict[UUID, AuthUser] = {}
        self.race_duplicate = False

    def create_user(self, *, email: str, password_hash: str) -> AuthUser:
        if self.race_duplicate:
            raise DuplicateEmailError
        user = AuthUser(
            id=uuid4(),
            email=email,
            password_hash=password_hash,
            created_at=datetime.now(UTC),
            is_active=True,
        )
        self.users[user.id] = user
        return user

    def get_by_email(self, email: str) -> AuthUser | None:
        return next((user for user in self.users.values() if user.email == email), None)

    def get_by_id(self, user_id: UUID) -> AuthUser | None:
        return self.users.get(user_id)


class UnavailableUserStore(MemoryUserStore):
    def get_by_email(self, email: str) -> AuthUser | None:
        raise AuthRepositoryUnavailableError

    def get_by_id(self, user_id: UUID) -> AuthUser | None:
        raise AuthRepositoryUnavailableError


class RecordingPasswordManager:
    def __init__(self) -> None:
        self.delegate = PasswordManager()
        self.dummy_verifications = 0

    def hash(self, password: str) -> str:
        return self.delegate.hash(password)

    def verify(self, password: str, password_hash: str) -> bool:
        return self.delegate.verify(password, password_hash)

    def verify_dummy(self, password: str) -> None:
        self.dummy_verifications += 1
        self.delegate.verify_dummy(password)


def build_client(
    store: MemoryUserStore | None = None,
    passwords: RecordingPasswordManager | PasswordManager | None = None,
) -> tuple[TestClient, MemoryUserStore, PasswordManager | RecordingPasswordManager]:
    resolved_store = store or MemoryUserStore()
    resolved_passwords = passwords or PasswordManager()
    service = AuthService(
        users=resolved_store,
        passwords=resolved_passwords,
        tokens=TokenManager(secret_key=SECRET),
    )
    return (
        TestClient(create_app(settings=Settings(), auth_service=service)),
        resolved_store,
        resolved_passwords,
    )


def persisted_user(store: MemoryUserStore, email: str) -> AuthUser:
    user = store.get_by_email(email)
    assert user is not None
    return user


def register(client: TestClient, email: str = "user@example.com") -> None:
    response = client.post(
        "/auth/register", json={"email": email, "password": VALID_PASSWORD}
    )
    assert response.status_code == 201


def test_register_normalizes_email_hashes_password_and_hides_hash() -> None:
    client, store, passwords = build_client()

    response = client.post(
        "/auth/register",
        json={"email": "  User@Example.COM ", "password": VALID_PASSWORD},
    )

    assert response.status_code == 201
    assert response.json()["email"] == "user@example.com"
    assert "password_hash" not in response.json()
    user = persisted_user(store, "user@example.com")
    assert user.password_hash != VALID_PASSWORD
    assert passwords.verify(VALID_PASSWORD, user.password_hash) is True


def test_duplicate_email_and_race_duplicate_return_stable_conflict() -> None:
    client, store, _ = build_client()
    register(client, "user@example.com")

    duplicate = client.post(
        "/auth/register",
        json={"email": " USER@example.com ", "password": VALID_PASSWORD},
    )
    store.race_duplicate = True
    race = client.post(
        "/auth/register",
        json={"email": "race@example.com", "password": VALID_PASSWORD},
    )

    assert duplicate.status_code == race.status_code == 409
    assert duplicate.json() == race.json() == {"detail": "Email is already registered"}


def test_login_success_and_valid_me_return_only_non_secret_user_fields() -> None:
    client, _, _ = build_client()
    register(client)

    login = client.post(
        "/auth/login", json={"email": " USER@EXAMPLE.COM ", "password": VALID_PASSWORD}
    )
    me = client.get(
        "/auth/me", headers={"Authorization": f"Bearer {login.json()['access_token']}"}
    )

    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"
    assert login.json()["expires_in"] == 1_800
    assert me.status_code == 200
    assert set(me.json()) == {"id", "email", "created_at", "is_active"}
    assert me.json()["email"] == "user@example.com"


def test_unknown_wrong_password_and_inactive_user_share_generic_401() -> None:
    client, store, _ = build_client()
    register(client)
    user = persisted_user(store, "user@example.com")

    unknown = client.post(
        "/auth/login", json={"email": "unknown@example.com", "password": VALID_PASSWORD}
    )
    wrong = client.post(
        "/auth/login", json={"email": user.email, "password": "wrong-pass-123"}
    )
    store.users[user.id] = AuthUser(
        id=user.id,
        email=user.email,
        password_hash=user.password_hash,
        created_at=user.created_at,
        is_active=False,
    )
    inactive = client.post(
        "/auth/login", json={"email": user.email, "password": VALID_PASSWORD}
    )

    expected = {"detail": AUTHENTICATION_FAILED}
    for response in (unknown, wrong, inactive):
        assert response.status_code == 401
        assert response.json() == expected
        assert response.headers["www-authenticate"] == "Bearer"


def test_unknown_user_executes_dummy_password_verification() -> None:
    passwords = RecordingPasswordManager()
    client, _, _ = build_client(passwords=passwords)

    response = client.post(
        "/auth/login", json={"email": "unknown@example.com", "password": VALID_PASSWORD}
    )

    assert response.status_code == 401
    assert passwords.dummy_verifications == 1


def test_me_rejects_missing_malformed_invalid_signature_and_expired_tokens() -> None:
    client, _, _ = build_client()
    now = datetime.now(UTC)
    invalid_signature = jwt.encode(
        {"sub": str(uuid4()), "iat": now, "exp": now + timedelta(minutes=5)},
        "wrong-api-test-secret-with-sufficient-entropy",
        algorithm=JWT_ALGORITHM,
    )
    expired = jwt.encode(
        {
            "sub": str(uuid4()),
            "iat": now - timedelta(minutes=2),
            "exp": now - timedelta(minutes=1),
        },
        SECRET,
        algorithm=JWT_ALGORITHM,
    )

    responses = [
        client.get("/auth/me"),
        client.get("/auth/me", headers={"Authorization": "Basic abc"}),
        client.get("/auth/me", headers={"Authorization": "Bearer not-a-jwt"}),
        client.get(
            "/auth/me", headers={"Authorization": f"Bearer {invalid_signature}"}
        ),
        client.get("/auth/me", headers={"Authorization": f"Bearer {expired}"}),
    ]

    assert all(response.status_code == 401 for response in responses)
    assert all(
        response.json() == {"detail": AUTHENTICATION_FAILED} for response in responses
    )


def test_me_rejects_invalid_subject_unknown_user_and_inactive_user() -> None:
    client, store, _ = build_client()
    register(client)
    user = persisted_user(store, "user@example.com")
    now = datetime.now(UTC)
    invalid_subject = jwt.encode(
        {"sub": "not-a-uuid", "iat": now, "exp": now + timedelta(minutes=5)},
        SECRET,
        algorithm=JWT_ALGORITHM,
    )
    unknown = TokenManager(secret_key=SECRET).create(uuid4())
    inactive = TokenManager(secret_key=SECRET).create(user.id)
    store.users[user.id] = AuthUser(
        id=user.id,
        email=user.email,
        password_hash=user.password_hash,
        created_at=user.created_at,
        is_active=False,
    )

    for token in (invalid_subject, unknown, inactive):
        response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 401
        assert response.json() == {"detail": AUTHENTICATION_FAILED}


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": VALID_PASSWORD},
        {"email": "user@example.com", "password": "short"},
        {"email": "user@example.com", "password": "x" * 129},
    ],
)
def test_malformed_email_and_password_return_422(payload: dict[str, str]) -> None:
    client, _, _ = build_client()

    register_response = client.post("/auth/register", json=payload)
    login_response = client.post("/auth/login", json=payload)

    assert register_response.status_code == 422
    assert login_response.status_code == 422
    if payload["password"] != VALID_PASSWORD:
        assert payload["password"] not in register_response.text
        assert payload["password"] not in login_response.text


def test_missing_auth_runtime_configuration_is_503_but_liveness_is_healthy() -> None:
    client = TestClient(create_app(settings=Settings()))

    register_response = client.post(
        "/auth/register",
        json={"email": "user@example.com", "password": VALID_PASSWORD},
    )
    login_response = client.post(
        "/auth/login",
        json={"email": "user@example.com", "password": VALID_PASSWORD},
    )
    me_response = client.get(
        "/auth/me", headers={"Authorization": "Bearer syntactically-present"}
    )

    for response in (register_response, login_response, me_response):
        assert response.status_code == 503
        assert response.json() == {"detail": AUTHENTICATION_UNAVAILABLE}
    assert client.get("/health/live").json() == {"status": "alive"}


def test_database_unavailability_returns_503() -> None:
    client, _, _ = build_client(store=UnavailableUserStore())

    response = client.post(
        "/auth/login", json={"email": "user@example.com", "password": VALID_PASSWORD}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": AUTHENTICATION_UNAVAILABLE}


def test_invalid_expiration_environment_fails_auth_closed_without_breaking_liveness() -> (
    None
):
    settings = Settings.from_environment(
        {
            "MOVIE_RECOMMENDER_DATABASE_URL": "postgresql+psycopg://localhost/test",
            "MOVIE_RECOMMENDER_AUTH_SECRET_KEY": SECRET,
            "MOVIE_RECOMMENDER_AUTH_ACCESS_TOKEN_EXPIRE_MINUTES": "0",
        }
    )
    client = TestClient(create_app(settings=settings))

    assert client.get("/health/live").status_code == 200
    assert (
        client.post(
            "/auth/login",
            json={"email": "user@example.com", "password": VALID_PASSWORD},
        ).status_code
        == 503
    )
