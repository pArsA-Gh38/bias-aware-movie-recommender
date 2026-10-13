"""Authenticated recommendation orchestration over the public runtime contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol
from uuid import UUID

from recommender_core import (
    DEFAULT_CONFIGURATION_KEY,
    ArtifactError,
    FoldInError,
    RuntimeInferenceError,
    ServingBundleV2,
    ServingRecommendation,
    effective_beta,
)


class RecommendationUnavailableError(RuntimeError):
    """A required recommendation dependency is unavailable."""


class RecommendationPersistenceError(RecommendationUnavailableError):
    """Recommendation persistence failed and was rolled back."""


class ColdStartError(RuntimeError):
    """The user has too few positive ART-002 catalog ratings."""


@dataclass(frozen=True, slots=True)
class ProfileRating:
    movie_id: int
    rating: int


@dataclass(frozen=True, slots=True)
class RecommendationContext:
    model_version_id: UUID
    ratings: tuple[ProfileRating, ...]


@dataclass(frozen=True, slots=True)
class RecommendationItemWrite:
    movie_id: int
    rank: int
    base_score: float
    final_score: float
    popularity_group: str


@dataclass(frozen=True, slots=True)
class RecommendationItemMetadata:
    movie_id: int
    title: str
    genres: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PersistedRecommendation:
    event_id: UUID
    metadata: tuple[RecommendationItemMetadata, ...]


@dataclass(frozen=True, slots=True)
class RecommendationItemResult:
    movie_id: int
    title: str
    genres: tuple[str, ...]
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class RecommendationWrite:
    user_id: UUID
    model_version_id: UUID
    strategy: str
    lambda_: float
    beta: float
    k: int
    candidate_count: int
    latency_ms: float
    items: tuple[RecommendationItemWrite, ...]


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    event_id: UUID
    model_version_id: UUID
    strategy: str
    configuration_key: str
    mode: str
    lambda_: float
    items: tuple[RecommendationItemResult, ...]


class RecommendationRuntime(Protocol):
    bundle: ServingBundleV2 | None

    def recommend_new_user(
        self,
        positive_movie_ids: tuple[int, ...],
        configuration_key: str,
        *,
        excluded_movie_ids: tuple[int, ...],
    ) -> tuple[ServingRecommendation, ...]: ...


class RecommendationRepository(Protocol):
    def load_context(self, *, user_id: UUID) -> RecommendationContext: ...

    def persist(
        self,
        write: RecommendationWrite,
        *,
        validate: Callable[[PersistedRecommendation], None],
    ) -> PersistedRecommendation: ...


class RecommendationService:
    """Build a fold-in profile, run ART-002, then atomically persist its output."""

    def __init__(
        self,
        repository: RecommendationRepository,
        runtime: RecommendationRuntime,
    ) -> None:
        self._repository = repository
        self._runtime = runtime

    def recommend(
        self,
        *,
        user_id: UUID,
        configuration_key: str | None = None,
    ) -> RecommendationResult:
        bundle = self._runtime.bundle
        if bundle is None:
            raise RecommendationUnavailableError("ART-002 is not loaded")

        selected_key = (
            DEFAULT_CONFIGURATION_KEY
            if configuration_key is None
            else configuration_key
        )
        try:
            configuration = bundle.configuration_registry[selected_key]
        except KeyError as exc:
            raise RecommendationUnavailableError(
                "frozen serving configuration is unavailable"
            ) from exc
        if configuration.key != selected_key:
            raise RecommendationUnavailableError(
                "frozen serving configuration identity mismatch"
            )

        context = self._repository.load_context(user_id=user_id)
        catalog_ids = set(bundle.catalog_item_ids())
        rated_catalog_ids = {
            rating.movie_id
            for rating in context.ratings
            if rating.movie_id in catalog_ids
        }
        positive_ids = tuple(
            rating.movie_id
            for rating in context.ratings
            if rating.rating >= 4 and rating.movie_id in catalog_ids
        )
        if len(positive_ids) < 5:
            raise ColdStartError(
                "At least five positive ART-002 catalog ratings are required"
            )

        started = perf_counter()
        try:
            recommendations = self._runtime.recommend_new_user(
                positive_ids,
                configuration.key,
                excluded_movie_ids=tuple(rated_catalog_ids),
            )
        except (ArtifactError, FoldInError, RuntimeInferenceError, KeyError) as exc:
            raise RecommendationUnavailableError(
                "recommendation runtime failed"
            ) from exc
        latency_ms = (perf_counter() - started) * 1000.0

        items = tuple(
            RecommendationItemWrite(
                movie_id=row.movie_id,
                rank=row.rank,
                # The public serving contract exposes its effective serving score.
                # For the frozen base configuration, it is persisted in both score
                # columns because no alternate reranked score is produced.
                base_score=row.score,
                final_score=row.score,
                popularity_group=bundle.item_metadata(row.movie_id).tail_group,
            )
            for row in recommendations
        )
        write = RecommendationWrite(
            user_id=user_id,
            model_version_id=context.model_version_id,
            strategy="als_fold_in",
            lambda_=configuration.lambda_,
            beta=effective_beta(configuration),
            k=len(items),
            candidate_count=bundle.candidate_pool_size,
            latency_ms=latency_ms,
            items=items,
        )

        def validate_metadata(pending: PersistedRecommendation) -> None:
            expected_ids = tuple(item.movie_id for item in items)
            metadata_ids = tuple(item.movie_id for item in pending.metadata)
            if metadata_ids != expected_ids or len(set(metadata_ids)) != len(
                metadata_ids
            ):
                raise RecommendationUnavailableError(
                    "persisted recommendation metadata identity mismatch"
                )

        persisted = self._repository.persist(write, validate=validate_metadata)
        return RecommendationResult(
            event_id=persisted.event_id,
            model_version_id=context.model_version_id,
            strategy=write.strategy,
            configuration_key=configuration.key,
            mode=configuration.mode,
            lambda_=configuration.lambda_,
            items=tuple(
                RecommendationItemResult(
                    movie_id=item.movie_id,
                    title=metadata.title,
                    genres=metadata.genres,
                    rank=item.rank,
                    score=item.final_score,
                )
                for item, metadata in zip(items, persisted.metadata, strict=True)
            ),
        )
