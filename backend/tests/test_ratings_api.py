from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.core.auth import AuthService, AuthUser, DuplicateEmailError
from app.core.config import Settings
from app.core.ratings import RatingService
from app.core.security import PasswordManager, TokenManager
from app.main import create_app

SECRET = "rating-test-secret-with-sufficient-entropy"


@dataclass
class MemoryRating:
    movie_id: int
    rating: int
    created_at: datetime
    updated_at: datetime


class MemoryUserStore:
    def __init__(self, user: AuthUser):
        self.users = {user.id: user}

    def create_user(self, *, email: str, password_hash: str):
        raise DuplicateEmailError

    def get_by_email(self, email: str):
        return next(
            (user for user in self.users.values() if user.email == email),
            None,
        )

    def get_by_id(self, user_id: UUID):
        return self.users.get(user_id)


class MemoryRatingRepository:
    def __init__(self):
        self.items = {}

    def upsert(
        self,
        *,
        user_id: UUID,
        movie_id: int,
        rating_value: int,
    ):
        key = (user_id, movie_id)

        now = datetime.now(UTC)

        if key in self.items:
            item = self.items[key]
            item.rating = rating_value
            item.updated_at = now
            return item

        item = MemoryRating(
            movie_id=movie_id,
            rating=rating_value,
            created_at=now,
            updated_at=now,
        )

        self.items[key] = item
        return item

    def count_positive(
        self,
        *,
        user_id: UUID,
    ):
        return sum(
            1
            for (uid, _), item in self.items.items()
            if uid == user_id and item.rating >= 4
        )

    def list_for_user(
        self,
        *,
        user_id: UUID,
    ):
        return [item for (uid, _), item in self.items.items() if uid == user_id]


def build_client():

    user = AuthUser(
        id=uuid4(),
        email="test@example.com",
        password_hash="unused",
        created_at=datetime.now(UTC),
        is_active=True,
    )

    users = MemoryUserStore(user)

    auth = AuthService(
        users=users,
        passwords=PasswordManager(),
        tokens=TokenManager(secret_key=SECRET),
    )

    ratings = RatingService(MemoryRatingRepository())

    client = TestClient(
        create_app(
            settings=Settings(),
            auth_service=auth,
            rating_service=ratings,
        )
    )

    token = TokenManager(secret_key=SECRET).create(user.id)

    return client, {"Authorization": f"Bearer {token}"}


def test_put_rating_creates_rating():

    client, headers = build_client()

    response = client.put(
        "/ratings/1",
        json={"rating": 5},
        headers=headers,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["movie_id"] == 1
    assert body["rating"] == 5
    assert body["positive_count"] == 1


def test_put_rating_updates_existing_rating():

    client, headers = build_client()

    client.put(
        "/ratings/1",
        json={"rating": 5},
        headers=headers,
    )

    response = client.put(
        "/ratings/1",
        json={"rating": 2},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["positive_count"] == 0


def test_get_my_ratings():

    client, headers = build_client()

    client.put(
        "/ratings/1",
        json={"rating": 4},
        headers=headers,
    )

    response = client.get(
        "/ratings/me",
        headers=headers,
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["items"]) == 1
    assert body["items"][0]["movie_id"] == 1


def test_rating_requires_auth():

    client, _ = build_client()

    response = client.put(
        "/ratings/1",
        json={"rating": 5},
    )

    assert response.status_code == 401
