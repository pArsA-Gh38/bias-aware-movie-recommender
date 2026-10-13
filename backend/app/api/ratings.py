from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.auth import CurrentUserDependency
from app.core.auth import AuthUser
from app.core.ratings import (
    RatingRepositoryUnavailableError,
    RatingService,
)
from app.schemas.ratings import (
    RatingRequest,
    RatingResponse,
    UserRatingItem,
    UserRatingsResponse,
)

RATING_UNAVAILABLE = "Rating service unavailable"
MOVIE_NOT_FOUND = "Movie not found"


def create_ratings_router(
    rating_service: RatingService | None,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:

    router = APIRouter(tags=["ratings"])

    @router.put(
        "/ratings/{movie_id}",
        response_model=RatingResponse,
    )
    def rate_movie(
        movie_id: int,
        payload: RatingRequest,
        user: Annotated[
            AuthUser,
            Depends(current_user_dependency),
        ],
    ) -> RatingResponse:

        if rating_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=RATING_UNAVAILABLE,
            )

        try:
            rating, positive_count = rating_service.rate_movie(
                user_id=user.id,
                movie_id=movie_id,
                rating=payload.rating,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=MOVIE_NOT_FOUND,
            ) from exc

        except RatingRepositoryUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=RATING_UNAVAILABLE,
            ) from exc

        return RatingResponse(
            movie_id=rating.movie_id,
            rating=rating.rating,
            positive_count=positive_count,
            updated_at=rating.updated_at,
        )

    @router.get(
        "/ratings/me",
        response_model=UserRatingsResponse,
    )
    def my_ratings(
        user: Annotated[
            AuthUser,
            Depends(current_user_dependency),
        ],
    ) -> UserRatingsResponse:

        if rating_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=RATING_UNAVAILABLE,
            )

        try:
            ratings = rating_service.user_ratings(
                user_id=user.id,
            )

        except RatingRepositoryUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=RATING_UNAVAILABLE,
            ) from exc

        return UserRatingsResponse(
            items=[
                UserRatingItem(
                    movie_id=item.movie_id,
                    rating=item.rating,
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                )
                for item in ratings
            ]
        )

    return router
