"""Focused API-005 recommendation flow coverage."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar, Self
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from recommender_core import (
    FrozenServingConfiguration,
    RuntimeInferenceError,
    ServingRecommendation,
)
from sqlalchemy.exc import SQLAlchemyError

from app.api.auth import AUTHENTICATION_FAILED
from app.api.recommendations import COLD_START, RECOMMENDATION_UNAVAILABLE
from app.core.auth import AuthService, AuthUser, DuplicateEmailError
from app.core.config import Settings
from app.core.recommendations import (
    PersistedRecommendation,
    ProfileRating,
    RecommendationContext,
    RecommendationItemMetadata,
    RecommendationPersistenceError,
    RecommendationService,
    RecommendationUnavailableError,
    RecommendationWrite,
)
from app.core.security import PasswordManager, TokenManager
from app.db.recommendations import SQLAlchemyRecommendationRepository
from app.main import create_app

SECRET = "recommendation-test-secret-with-sufficient-entropy"


class MemoryUserStore:
    def __init__(self, user: AuthUser) -> None:
        self.user = user

    def create_user(self, *, email: str, password_hash: str) -> AuthUser:
        raise DuplicateEmailError

    def get_by_email(self, email: str) -> AuthUser | None:
        return self.user if email == self.user.email else None

    def get_by_id(self, user_id: UUID) -> AuthUser | None:
        return self.user if user_id == self.user.id else None


class FakeBundle:
    serving_default = "A0_L1"
    candidate_pool_size = 100
    configuration_registry: ClassVar[dict[str, FrozenServingConfiguration]] = {
        "A0_L1": FrozenServingConfiguration("A0_L1", "base", 1.0, None),
        "A1_L07": FrozenServingConfiguration("A1_L07", "diversity_only", 0.7, 1.0),
        "A1_L06": FrozenServingConfiguration("A1_L06", "diversity_only", 0.6, 1.0),
    }

    def __init__(self) -> None:
        self.positive_ids: tuple[int, ...] | None = None
        self.excluded_ids: tuple[int, ...] | None = None
        self.configuration_key: str | None = None
        self.fail = False

    def catalog_item_ids(self) -> tuple[int, ...]:
        return tuple(range(1, 31))

    def recommend_new_user(
        self,
        positive_ids: tuple[int, ...],
        configuration_key: str,
        *,
        excluded_movie_ids: tuple[int, ...],
    ) -> tuple[ServingRecommendation, ...]:
        if self.fail:
            raise RuntimeInferenceError("failed")
        self.positive_ids = positive_ids
        self.excluded_ids = excluded_movie_ids
        self.configuration_key = configuration_key
        configuration = self.configuration_registry[configuration_key]
        available = [
            movie_id
            for movie_id in self.catalog_item_ids()
            if movie_id not in excluded_movie_ids
        ][:10]
        return tuple(
            ServingRecommendation(
                rank,
                movie_id,
                rank,
                configuration.lambda_ / rank,
            )
            for rank, movie_id in enumerate(available, start=1)
        )

    def item_metadata(self, movie_id: int) -> SimpleNamespace:
        return SimpleNamespace(tail_group="head" if movie_id < 20 else "long_tail")


class FakeRecommendationRepository:
    def __init__(self, context: RecommendationContext) -> None:
        self.context = context
        self.write: RecommendationWrite | None = None
        self.event_id = uuid4()
        self.fail_context = False
        self.fail_persist = False
        self.metadata_failure: str | None = None
        self.durable = False
        self.load_context_calls = 0

    def load_context(self, *, user_id: UUID) -> RecommendationContext:
        self.load_context_calls += 1
        if self.fail_context:
            raise RecommendationUnavailableError("registry unavailable")
        return self.context

    def persist(
        self,
        write: RecommendationWrite,
        *,
        validate: Callable[[PersistedRecommendation], None],
    ) -> PersistedRecommendation:
        if self.fail_persist:
            raise RecommendationPersistenceError("rolled back")
        self.write = write
        metadata = tuple(
            RecommendationItemMetadata(
                movie_id=item.movie_id,
                title=f"Authoritative Movie {item.movie_id}",
                genres=("Drama", "Synthetic"),
            )
            for item in write.items
        )
        if self.metadata_failure == "missing":
            metadata = metadata[:-1]
        elif self.metadata_failure == "duplicate":
            metadata = (*metadata[:-1], metadata[0])
        elif self.metadata_failure == "unexpected":
            metadata = (
                *metadata[:-1],
                RecommendationItemMetadata(999_999, "Unexpected", ("Drama",)),
            )
        elif self.metadata_failure == "reordered":
            metadata = tuple(reversed(metadata))
        pending = PersistedRecommendation(
            event_id=self.event_id,
            metadata=metadata,
        )
        validate(pending)
        self.durable = True
        return pending


class FakeRuntime:
    def __init__(self, bundle: FakeBundle) -> None:
        self.bundle: FakeBundle | None = bundle

    def recommend_new_user(
        self,
        positive_movie_ids: tuple[int, ...],
        configuration_key: str,
        *,
        excluded_movie_ids: tuple[int, ...],
    ) -> tuple[ServingRecommendation, ...]:
        if self.bundle is None:
            raise RuntimeInferenceError("artifact unavailable")
        return self.bundle.recommend_new_user(
            positive_movie_ids,
            configuration_key,
            excluded_movie_ids=excluded_movie_ids,
        )


def build_client(
    ratings: tuple[ProfileRating, ...],
) -> tuple[
    TestClient,
    AuthUser,
    str,
    FakeBundle,
    FakeRecommendationRepository,
]:
    user = AuthUser(
        id=uuid4(),
        email="recommender@example.com",
        password_hash="unused",
        created_at=datetime.now(UTC),
        is_active=True,
    )
    tokens = TokenManager(secret_key=SECRET)
    auth_service = AuthService(
        users=MemoryUserStore(user),
        passwords=PasswordManager(),
        tokens=tokens,
    )
    bundle = FakeBundle()
    runtime = FakeRuntime(bundle)
    bundle.runtime = runtime
    repository = FakeRecommendationRepository(
        RecommendationContext(model_version_id=uuid4(), ratings=ratings)
    )
    service = RecommendationService(
        repository,
        runtime,
    )
    client = TestClient(
        create_app(
            settings=Settings(),
            auth_service=auth_service,
            recommendation_service=service,
        )
    )
    return client, user, tokens.create(user.id), bundle, repository


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_service_rejects_empty_configuration_key_before_side_effects() -> None:
    ratings = tuple(ProfileRating(movie_id=value, rating=5) for value in range(1, 6))
    bundle = FakeBundle()
    repository = FakeRecommendationRepository(
        RecommendationContext(model_version_id=uuid4(), ratings=ratings)
    )
    service = RecommendationService(repository, FakeRuntime(bundle))

    with pytest.raises(
        RecommendationUnavailableError,
        match="^frozen serving configuration is unavailable$",
    ):
        service.recommend(user_id=uuid4(), configuration_key="")

    assert repository.load_context_calls == 0
    assert bundle.configuration_key is None
    assert repository.write is None


@pytest.mark.parametrize("explicit_none", [False, True], ids=["omitted", "none"])
def test_service_defaults_omitted_or_none_configuration_to_a0_l1(
    explicit_none: bool,
) -> None:
    ratings = tuple(ProfileRating(movie_id=value, rating=5) for value in range(1, 6))
    bundle = FakeBundle()
    repository = FakeRecommendationRepository(
        RecommendationContext(model_version_id=uuid4(), ratings=ratings)
    )
    service = RecommendationService(repository, FakeRuntime(bundle))

    if explicit_none:
        result = service.recommend(user_id=uuid4(), configuration_key=None)
    else:
        result = service.recommend(user_id=uuid4())

    assert result.configuration_key == "A0_L1"
    assert bundle.configuration_key == "A0_L1"
    assert repository.write is not None
    assert repository.write.lambda_ == 1.0
    assert repository.write.beta == 0.5


def test_recommendations_require_api_002_authentication() -> None:
    client, _, _, bundle, repository = build_client(())

    response = client.post("/recommendations")

    assert response.status_code == 401
    assert response.json() == {"detail": AUTHENTICATION_FAILED}
    assert response.headers["www-authenticate"] == "Bearer"
    assert repository.load_context_calls == 0
    assert bundle.configuration_key is None


def test_openapi_documents_cold_start_and_unavailable_responses() -> None:
    client, _, _, _, _ = build_client(())

    responses = client.app.openapi()["paths"]["/recommendations"]["post"]["responses"]

    assert {"200", "409", "503"} <= set(responses)
    assert responses["409"]["description"] == COLD_START
    item_schema = client.app.openapi()["components"]["schemas"][
        "RecommendationItemResponse"
    ]
    assert {"movie_id", "title", "genres", "rank", "score"} <= set(
        item_schema["required"]
    )


def test_recommendation_uses_positive_profile_excludes_all_rated_and_persists() -> None:
    ratings = tuple(
        ProfileRating(movie_id=movie_id, rating=rating)
        for movie_id, rating in [(1, 5), (2, 4), (3, 5), (4, 4), (5, 5), (6, 2)]
    )
    client, user, token, bundle, repository = build_client(ratings)

    response = client.post("/recommendations", headers=bearer(token))

    assert response.status_code == 200
    assert bundle.positive_ids == (1, 2, 3, 4, 5)
    assert set(bundle.excluded_ids or ()) == {1, 2, 3, 4, 5, 6}
    assert repository.write is not None
    assert repository.write.user_id == user.id
    assert repository.write.model_version_id == repository.context.model_version_id
    assert repository.write.strategy == "als_fold_in"
    assert repository.write.lambda_ == 1.0
    assert repository.write.beta == 0.5
    assert repository.write.candidate_count == 100
    assert len(repository.write.items) == 10
    assert response.json()["event_id"] == str(repository.event_id)
    assert response.json()["model_version_id"] == str(
        repository.context.model_version_id
    )
    assert response.json()["strategy"] == "als_fold_in"
    assert response.json()["configuration_key"] == "A0_L1"
    assert response.json()["mode"] == "base"
    assert response.json()["lambda"] == 1.0
    assert "lambda_" not in response.json()
    returned_ids = {item["movie_id"] for item in response.json()["items"]}
    assert not returned_ids.intersection({1, 2, 3, 4, 5, 6})
    assert response.json()["items"][0]["title"] == "Authoritative Movie 7"
    assert response.json()["items"][0]["genres"] == ["Drama", "Synthetic"]
    assert [item["rank"] for item in response.json()["items"]] == list(range(1, 11))


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unexpected", "reordered"])
def test_service_rejects_invalid_metadata_before_persistence_is_durable(
    failure: str,
) -> None:
    ratings = tuple(ProfileRating(movie_id=value, rating=5) for value in range(1, 6))
    bundle = FakeBundle()
    repository = FakeRecommendationRepository(
        RecommendationContext(model_version_id=uuid4(), ratings=ratings)
    )
    repository.metadata_failure = failure
    with pytest.raises(RecommendationUnavailableError, match="identity mismatch"):
        RecommendationService(repository, FakeRuntime(bundle)).recommend(
            user_id=uuid4()
        )
    assert repository.durable is False


@pytest.mark.parametrize("json_body", [None, {}], ids=["bodyless", "empty-object"])
def test_configuration_selection_defaults_to_a0_l1(
    json_body: dict[str, object] | None,
) -> None:
    ratings = tuple(ProfileRating(movie_id=value, rating=5) for value in range(1, 6))
    client, _, token, bundle, repository = build_client(ratings)

    if json_body is None:
        response = client.post("/recommendations", headers=bearer(token))
    else:
        response = client.post(
            "/recommendations", headers=bearer(token), json=json_body
        )

    assert response.status_code == 200
    assert bundle.configuration_key == "A0_L1"
    assert repository.write is not None
    assert repository.write.lambda_ == 1.0
    assert repository.write.beta == 0.5
    assert response.json()["configuration_key"] == "A0_L1"
    assert response.json()["mode"] == "base"
    assert response.json()["lambda"] == 1.0


@pytest.mark.parametrize(
    ("configuration_key", "expected_mode", "expected_lambda", "expected_beta"),
    [
        ("A0_L1", "base", 1.0, 0.5),
        ("A1_L07", "diversity_only", 0.7, 1.0),
        ("A1_L06", "diversity_only", 0.6, 1.0),
    ],
)
def test_configuration_selection_applies_reports_and_persists_frozen_choice(
    configuration_key: str,
    expected_mode: str,
    expected_lambda: float,
    expected_beta: float,
) -> None:
    ratings = tuple(ProfileRating(movie_id=value, rating=5) for value in range(1, 6))
    client, _, token, bundle, repository = build_client(ratings)

    response = client.post(
        "/recommendations",
        headers=bearer(token),
        json={"configuration_key": configuration_key},
    )

    assert response.status_code == 200
    assert bundle.configuration_key == configuration_key
    assert repository.write is not None
    assert repository.write.strategy == response.json()["strategy"]
    assert repository.write.lambda_ == expected_lambda
    assert repository.write.beta == expected_beta
    assert response.json()["configuration_key"] == configuration_key
    assert response.json()["mode"] == expected_mode
    assert response.json()["lambda"] == expected_lambda
    assert "lambda_" not in response.json()
    assert response.json()["items"][0]["score"] == expected_lambda


@pytest.mark.parametrize(
    "json_body",
    [
        {"configuration_key": "A2_L05"},
        {"lambda": 0.7},
        {"lambda_": 0.7},
        {"mode": "diversity_only"},
        {"beta": 1.0},
        {"unknown": "value"},
        {"configuration_key": "A1_L07", "mode": "base", "lambda": 1.0},
    ],
)
def test_configuration_selection_rejects_unsupported_or_spoofed_input(
    json_body: dict[str, object],
) -> None:
    ratings = tuple(ProfileRating(movie_id=value, rating=5) for value in range(1, 6))
    client, _, token, bundle, repository = build_client(ratings)

    response = client.post("/recommendations", headers=bearer(token), json=json_body)

    assert response.status_code == 422
    assert bundle.configuration_key is None
    assert repository.load_context_calls == 0
    assert repository.write is None


def test_cold_start_counts_only_positive_artifact_catalog_ratings() -> None:
    ratings = tuple(
        ProfileRating(movie_id=movie_id, rating=5) for movie_id in (1, 2, 3, 4, 999_999)
    )
    client, _, token, _, repository = build_client(ratings)

    response = client.post("/recommendations", headers=bearer(token))

    assert response.status_code == 409
    assert response.json() == {"detail": COLD_START}
    assert repository.write is None


@pytest.mark.parametrize("failure", ["artifact", "core", "registry", "transaction"])
def test_dependency_and_transaction_failures_are_stable_503(failure: str) -> None:
    ratings = tuple(ProfileRating(movie_id=value, rating=5) for value in range(1, 6))
    client, _, token, bundle, repository = build_client(ratings)
    if failure == "artifact":
        bundle.runtime.bundle = None
    elif failure == "core":
        bundle.fail = True
    elif failure == "registry":
        repository.fail_context = True
    else:
        repository.fail_persist = True

    response = client.post("/recommendations", headers=bearer(token))

    assert response.status_code == 503
    assert response.json() == {"detail": RECOMMENDATION_UNAVAILABLE}


class FailingSession:
    def __init__(self) -> None:
        self.rolled_back = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def add(self, value: object) -> None:
        return None

    def execute(self, statement: object) -> SimpleNamespace:
        return SimpleNamespace(all=list)

    def add_all(self, values: list[object]) -> None:
        return None

    def commit(self) -> None:
        raise SQLAlchemyError("commit failed")

    def rollback(self) -> None:
        self.rolled_back = True


def test_sqlalchemy_transaction_failure_rolls_back() -> None:
    session = FailingSession()
    repository = SQLAlchemyRecommendationRepository(lambda: session)
    write = RecommendationWrite(
        user_id=uuid4(),
        model_version_id=uuid4(),
        strategy="als_fold_in",
        lambda_=1.0,
        beta=0.5,
        k=1,
        candidate_count=100,
        latency_ms=1.0,
        items=(),
    )

    with pytest.raises(RecommendationPersistenceError):
        repository.persist(write, validate=lambda _pending: None)

    assert session.rolled_back is True
