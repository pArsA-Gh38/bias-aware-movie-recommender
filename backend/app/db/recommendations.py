"""SQLAlchemy persistence boundary for recommendation requests."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.recommendations import (
    PersistedRecommendation,
    ProfileRating,
    RecommendationContext,
    RecommendationItemMetadata,
    RecommendationPersistenceError,
    RecommendationUnavailableError,
    RecommendationWrite,
)
from app.models import (
    ModelVersion,
    Movie,
    Rating,
    RecommendationEvent,
    RecommendationItem,
)


class SQLAlchemyRecommendationRepository:
    """Read serving context and atomically write one event with all ranked items."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def load_context(self, *, user_id: UUID) -> RecommendationContext:
        try:
            with self._session_factory() as session:
                model_version_id = session.scalar(
                    select(ModelVersion.id).where(ModelVersion.status == "active")
                )
                if model_version_id is None:
                    raise RecommendationUnavailableError(
                        "active model version is unavailable"
                    )
                ratings = tuple(
                    ProfileRating(movie_id=row.movie_id, rating=row.rating)
                    for row in session.scalars(
                        select(Rating)
                        .where(Rating.user_id == user_id)
                        .order_by(Rating.movie_id.asc())
                    ).all()
                )
                return RecommendationContext(
                    model_version_id=model_version_id,
                    ratings=ratings,
                )
        except RecommendationUnavailableError:
            raise
        except SQLAlchemyError as exc:
            raise RecommendationUnavailableError(
                "recommendation context is unavailable"
            ) from exc

    def persist(
        self,
        write: RecommendationWrite,
        *,
        validate: Callable[[PersistedRecommendation], None],
    ) -> PersistedRecommendation:
        event_id = uuid4()
        with self._session_factory() as session:
            try:
                movie_ids = tuple(item.movie_id for item in write.items)
                rows = session.execute(
                    select(Movie.id, Movie.title, Movie.genres).where(
                        Movie.id.in_(movie_ids), Movie.is_active.is_(True)
                    )
                ).all()
                metadata_by_id = {
                    row.id: RecommendationItemMetadata(
                        movie_id=row.id,
                        title=row.title,
                        genres=tuple(row.genres),
                    )
                    for row in rows
                }
                if len(metadata_by_id) != len(movie_ids) or set(metadata_by_id) != set(
                    movie_ids
                ):
                    session.rollback()
                    raise RecommendationUnavailableError(
                        "authoritative recommendation metadata is unavailable"
                    )
                session.add(
                    RecommendationEvent(
                        id=event_id,
                        user_id=write.user_id,
                        model_version_id=write.model_version_id,
                        strategy=write.strategy,
                        lambda_=Decimal(str(write.lambda_)),
                        beta=Decimal(str(write.beta)),
                        k=write.k,
                        candidate_count=write.candidate_count,
                        latency_ms=write.latency_ms,
                    )
                )
                session.add_all(
                    [
                        RecommendationItem(
                            event_id=event_id,
                            movie_id=item.movie_id,
                            rank=item.rank,
                            base_score=item.base_score,
                            final_score=item.final_score,
                            popularity_group=item.popularity_group,
                        )
                        for item in write.items
                    ]
                )
                pending = PersistedRecommendation(
                    event_id=event_id,
                    metadata=tuple(
                        metadata_by_id[item.movie_id] for item in write.items
                    ),
                )
                validate(pending)
                session.commit()
            except RecommendationUnavailableError:
                session.rollback()
                raise
            except SQLAlchemyError as exc:
                session.rollback()
                raise RecommendationPersistenceError(
                    "recommendation transaction failed"
                ) from exc
        return pending
