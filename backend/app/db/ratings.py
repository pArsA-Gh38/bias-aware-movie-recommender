from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.ratings import RatingRepositoryUnavailableError
from app.models import Movie, Rating


class SQLAlchemyRatingRepository:
    """SQLAlchemy persistence adapter for ratings."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
    ) -> None:
        self._session_factory = session_factory

    def upsert(
        self,
        *,
        user_id: UUID,
        movie_id: int,
        rating_value: int,
    ) -> Rating:
        try:
            with self._session_factory() as session:
                movie = session.get(Movie, movie_id)

                if movie is None or not movie.is_active:
                    raise ValueError("movie_not_found")

                rating = session.scalar(
                    select(Rating).where(
                        Rating.user_id == user_id,
                        Rating.movie_id == movie_id,
                    )
                )

                if rating is None:
                    rating = Rating(
                        user_id=user_id,
                        movie_id=movie_id,
                        rating=rating_value,
                    )
                    session.add(rating)
                else:
                    rating.rating = rating_value

                session.commit()
                session.refresh(rating)

                return rating

        except ValueError:
            raise

        except SQLAlchemyError as exc:
            raise RatingRepositoryUnavailableError from exc

    def count_positive(
        self,
        *,
        user_id: UUID,
    ) -> int:
        try:
            with self._session_factory() as session:
                result = session.scalar(
                    select(func.count(Rating.id)).where(
                        Rating.user_id == user_id,
                        Rating.rating >= 4,
                    )
                )

                return result or 0

        except SQLAlchemyError as exc:
            raise RatingRepositoryUnavailableError from exc

    def list_for_user(
        self,
        *,
        user_id: UUID,
    ) -> list[Rating]:
        try:
            with self._session_factory() as session:
                return list(
                    session.scalars(
                        select(Rating)
                        .where(Rating.user_id == user_id)
                        .order_by(Rating.updated_at.desc())
                    ).all()
                )

        except SQLAlchemyError as exc:
            raise RatingRepositoryUnavailableError from exc
