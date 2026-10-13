from __future__ import annotations

from typing import Protocol
from uuid import UUID


class RatingRepositoryUnavailableError(RuntimeError):
    """Raised when rating persistence is unavailable."""


class RatingRepository(Protocol):
    def upsert(
        self,
        *,
        user_id: UUID,
        movie_id: int,
        rating_value: int,
    ): ...

    def count_positive(
        self,
        *,
        user_id: UUID,
    ) -> int: ...

    def list_for_user(
        self,
        *,
        user_id: UUID,
    ): ...


class RatingService:
    def __init__(
        self,
        repository: RatingRepository,
    ) -> None:
        self.repository = repository

    def rate_movie(
        self,
        *,
        user_id: UUID,
        movie_id: int,
        rating: int,
    ):

        rating_row = self.repository.upsert(
            user_id=user_id,
            movie_id=movie_id,
            rating_value=rating,
        )

        positive_count = self.repository.count_positive(
            user_id=user_id,
        )

        return rating_row, positive_count

    def user_ratings(
        self,
        *,
        user_id: UUID,
    ):
        return self.repository.list_for_user(
            user_id=user_id,
        )
