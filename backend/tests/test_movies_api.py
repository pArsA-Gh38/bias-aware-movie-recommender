"""Focused API-003 Slice 1 movie catalog coverage."""

from __future__ import annotations

import builtins
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi.testclient import TestClient

from app.api.auth import AUTHENTICATION_FAILED
from app.api.movies import CATALOG_UNAVAILABLE
from app.core.auth import AuthService, AuthUser, DuplicateEmailError
from app.core.config import Settings
from app.core.movies import (
    CatalogMovie,
    CatalogRepositoryUnavailableError,
    MovieCatalogService,
)
from app.core.security import JWT_ALGORITHM, PasswordManager, TokenManager
from app.main import create_app

SECRET = "movie-catalog-test-secret-with-sufficient-entropy"


class MemoryUserStore:
    def __init__(self, user: AuthUser) -> None:
        self.users = {user.id: user}

    def create_user(self, *, email: str, password_hash: str) -> AuthUser:
        raise DuplicateEmailError

    def get_by_email(self, email: str) -> AuthUser | None:
        return next((user for user in self.users.values() if user.email == email), None)

    def get_by_id(self, user_id: UUID) -> AuthUser | None:
        return self.users.get(user_id)


class MemoryMovieRepository:
    def __init__(self, movies: list[tuple[CatalogMovie, bool]]) -> None:
        self.movies = movies

    def list_active_after(
        self,
        *,
        after_movie_id: int | None,
        limit: int,
        title_query: str | None = None,
        genre: str | None = None,
    ) -> tuple[CatalogMovie, ...]:
        return tuple(
            movie
            for movie, is_active in sorted(
                self.movies, key=lambda stored: stored[0].movie_id
            )
            if is_active
            and (after_movie_id is None or movie.movie_id > after_movie_id)
            and (title_query is None or title_query.lower() in movie.title.lower())
            and (
                genre is None
                or any(stored.lower() == genre.lower() for stored in movie.genres)
            )
        )[:limit]

    def list_all_active(self) -> tuple[CatalogMovie, ...]:
        return tuple(
            movie
            for movie, is_active in sorted(
                self.movies, key=lambda stored: stored[0].movie_id
            )
            if is_active
        )


class UnavailableMovieRepository(MemoryMovieRepository):
    def list_active_after(
        self,
        *,
        after_movie_id: int | None,
        limit: int,
        title_query: str | None = None,
        genre: str | None = None,
    ) -> tuple[CatalogMovie, ...]:
        raise CatalogRepositoryUnavailableError

    def list_all_active(self) -> tuple[CatalogMovie, ...]:
        raise CatalogRepositoryUnavailableError


def build_client(
    movies: list[tuple[CatalogMovie, bool]] | None = None,
) -> tuple[TestClient, MemoryUserStore, AuthUser, str]:
    user = AuthUser(
        id=uuid4(),
        email="viewer@example.com",
        password_hash="unused",
        created_at=datetime.now(UTC),
        is_active=True,
    )
    users = MemoryUserStore(user)
    tokens = TokenManager(secret_key=SECRET)
    auth_service = AuthService(
        users=users,
        passwords=PasswordManager(),
        tokens=tokens,
    )
    movie_service = MovieCatalogService(MemoryMovieRepository(movies or []))
    client = TestClient(
        create_app(
            settings=Settings(),
            auth_service=auth_service,
            movie_catalog_service=movie_service,
        )
    )
    return client, users, user, tokens.create(user.id)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def assert_unauthorized(response: object) -> None:
    assert response.status_code == 401
    assert response.json() == {"detail": AUTHENTICATION_FAILED}
    assert response.headers["www-authenticate"] == "Bearer"


def test_movies_rejects_missing_token() -> None:
    client, _, _, _ = build_client()

    assert_unauthorized(client.get("/movies"))


def test_movies_rejects_invalid_and_expired_tokens() -> None:
    client, _, _, _ = build_client()
    now = datetime.now(UTC)
    invalid_signature = jwt.encode(
        {"sub": str(uuid4()), "iat": now, "exp": now + timedelta(minutes=5)},
        "wrong-movie-catalog-test-secret-with-sufficient-entropy",
        algorithm=JWT_ALGORITHM,
    )
    invalid_subject = jwt.encode(
        {"sub": "not-a-uuid", "iat": now, "exp": now + timedelta(minutes=5)},
        SECRET,
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

    assert_unauthorized(client.get("/movies", headers={"Authorization": "Basic abc"}))
    assert_unauthorized(client.get("/movies", headers=bearer("not-a-jwt")))
    assert_unauthorized(client.get("/movies", headers=bearer(invalid_signature)))
    assert_unauthorized(client.get("/movies", headers=bearer(invalid_subject)))
    assert_unauthorized(client.get("/movies", headers=bearer(expired)))


def test_movies_rejects_unknown_and_inactive_users_like_api_002() -> None:
    client, users, user, token = build_client()
    unknown_token = TokenManager(secret_key=SECRET).create(uuid4())

    assert_unauthorized(client.get("/movies", headers=bearer(unknown_token)))

    users.users[user.id] = AuthUser(
        id=user.id,
        email=user.email,
        password_hash=user.password_hash,
        created_at=user.created_at,
        is_active=False,
    )
    assert_unauthorized(client.get("/movies", headers=bearer(token)))


def test_authenticated_movies_match_schema_active_filter_and_order() -> None:
    movies = [
        (CatalogMovie(9, "Later", None, ()), True),
        (CatalogMovie(2, "Inactive", 1999, ("Drama",)), False),
        (CatalogMovie(5, "Earlier", 1975, ("Drama", "Comedy")), True),
    ]
    client, _, _, token = build_client(movies)

    response = client.get("/movies", headers=bearer(token))

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "movie_id": 5,
                "title": "Earlier",
                "release_year": 1975,
                "genres": ["Drama", "Comedy"],
            },
            {
                "movie_id": 9,
                "title": "Later",
                "release_year": None,
                "genres": [],
            },
        ],
        "next_cursor": None,
    }
    assert set(response.json()["items"][0]) == {
        "movie_id",
        "title",
        "release_year",
        "genres",
    }


def test_movies_cursor_is_deterministic_and_empty_catalog_is_stable() -> None:
    movies = [
        (CatalogMovie(8, "Eight", 2008, ("Drama",)), True),
        (CatalogMovie(3, "Three", 2003, ("Comedy",)), True),
        (CatalogMovie(20, "Twenty", 2020, ()), True),
    ]
    client, _, _, token = build_client(movies)

    first = client.get("/movies?limit=2", headers=bearer(token))
    final = client.get("/movies?limit=2&cursor=8", headers=bearer(token))
    empty_client, _, _, empty_token = build_client()
    empty = empty_client.get("/movies", headers=bearer(empty_token))

    assert [item["movie_id"] for item in first.json()["items"]] == [3, 8]
    assert first.json()["next_cursor"] == "8"
    assert [item["movie_id"] for item in final.json()["items"]] == [20]
    assert final.json()["next_cursor"] is None
    assert empty.json() == {"items": [], "next_cursor": None}


@pytest.mark.parametrize(
    "query", ["cursor=0", "cursor=-1", "cursor=abc", "limit=0", "limit=101"]
)
def test_movies_rejects_invalid_cursor_and_limit(query: str) -> None:
    client, _, _, token = build_client()

    assert client.get(f"/movies?{query}", headers=bearer(token)).status_code == 422


def test_title_search_is_trimmed_case_insensitive_literal_and_ordered() -> None:
    movies = [
        (CatalogMovie(9, "A 100%_Real Story", 2009, ("Drama",)), True),
        (CatalogMovie(3, "Another REAL Story", 2003, ("Drama",)), True),
        (CatalogMovie(5, "A 100xxReal Story", 2005, ("Drama",)), True),
    ]
    client, _, _, token = build_client(movies)

    case_insensitive = client.get("/movies?q= real ", headers=bearer(token))
    literal = client.get("/movies?q=100%25_", headers=bearer(token))
    no_match = client.get("/movies?q=missing", headers=bearer(token))

    assert [item["movie_id"] for item in case_insensitive.json()["items"]] == [3, 5, 9]
    assert [item["movie_id"] for item in literal.json()["items"]] == [9]
    assert no_match.json() == {"items": [], "next_cursor": None}


def test_genre_filter_is_trimmed_case_insensitive_and_exact_element() -> None:
    movies = [
        (CatalogMovie(1, "Drama", 2001, ("Drama",)), True),
        (CatalogMovie(2, "Dramatic", 2002, ("Dramatic",)), True),
        (CatalogMovie(3, "Mixed", 2003, ("Comedy", "DRAMA")), True),
    ]
    client, _, _, token = build_client(movies)

    response = client.get("/movies?genre= drama ", headers=bearer(token))

    assert [item["movie_id"] for item in response.json()["items"]] == [1, 3]
    assert response.json()["items"][0]["genres"] == ["Drama"]


def test_combined_title_and_genre_filters_use_intersection_before_pagination() -> None:
    movies = [
        (CatalogMovie(2, "Space One", 2002, ("Drama",)), True),
        (CatalogMovie(4, "Space Two", 2004, ("Comedy",)), True),
        (CatalogMovie(7, "Deep Space", 2007, ("Drama",)), True),
        (CatalogMovie(9, "Other", 2009, ("Drama",)), True),
    ]
    client, _, _, token = build_client(movies)

    first = client.get("/movies?q=SPACE&genre=drama&limit=1", headers=bearer(token))
    final = client.get(
        "/movies?q=SPACE&genre=drama&limit=1&cursor=2", headers=bearer(token)
    )

    assert [item["movie_id"] for item in first.json()["items"]] == [2]
    assert first.json()["next_cursor"] == "2"
    assert [item["movie_id"] for item in final.json()["items"]] == [7]
    assert final.json()["next_cursor"] is None


@pytest.mark.parametrize(
    "query",
    [
        "q=%20%20",
        f"q={'x' * 201}",
        "genre=%20%20",
        f"genre={'x' * 51}",
        "onboarding=true&q=space",
        "onboarding=true&genre=drama",
        "onboarding=true&cursor=1",
    ],
)
def test_movie_filter_and_onboarding_validation(query: str) -> None:
    client, _, _, token = build_client()

    response = client.get(f"/movies?{query}", headers=bearer(token))

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_onboarding_is_deterministic_round_robin_catalog_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    movies = [
        (CatalogMovie(5, "Genreless", 2005, ()), True),
        (CatalogMovie(3, "Drama", 2003, ("Drama",)), True),
        (CatalogMovie(1, "Shared", 2001, ("Drama", "Comedy")), True),
        (CatalogMovie(4, "Action", 2004, ("Action",)), True),
        (CatalogMovie(2, "Comedy", 2002, ("Comedy",)), True),
        (CatalogMovie(6, "Inactive", 2006, ("Action",)), False),
    ]
    client, _, _, token = build_client(movies)

    def fail_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("onboarding must not read files or artifacts")

    monkeypatch.setattr(builtins, "open", fail_open)
    first = client.get("/movies?onboarding=true&limit=5", headers=bearer(token))
    second = client.get("/movies?onboarding=true&limit=5", headers=bearer(token))

    assert first.status_code == 200
    assert first.content == second.content
    assert [item["movie_id"] for item in first.json()["items"]] == [4, 1, 3, 2, 5]
    assert first.json()["next_cursor"] is None


def test_catalog_failure_is_stable_503_and_liveness_is_unchanged() -> None:
    _, users, user, token = build_client()
    auth_service = AuthService(
        users=users,
        passwords=PasswordManager(),
        tokens=TokenManager(secret_key=SECRET),
    )
    unavailable_client = TestClient(
        create_app(
            settings=Settings(),
            auth_service=auth_service,
            movie_catalog_service=MovieCatalogService(UnavailableMovieRepository([])),
        )
    )

    response = unavailable_client.get("/movies", headers=bearer(token))

    assert response.status_code == 503
    assert response.json() == {"detail": CATALOG_UNAVAILABLE}
    assert unavailable_client.get("/health/live").status_code == 200
    assert unavailable_client.get("/health/live").json() == {"status": "alive"}
    assert user.id in users.users
