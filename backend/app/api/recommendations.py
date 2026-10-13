"""Authenticated MVP recommendation endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.auth import CurrentUserDependency
from app.core.auth import AuthUser
from app.core.recommendations import (
    ColdStartError,
    RecommendationService,
    RecommendationUnavailableError,
)
from app.schemas.recommendations import (
    RecommendationItemResponse,
    RecommendationRequest,
    RecommendationResponse,
)

RECOMMENDATION_UNAVAILABLE = "Recommendation service unavailable"
COLD_START = "At least five positive ART-002 catalog ratings are required"


def create_recommendations_router(
    recommendation_service: RecommendationService | None,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    router = APIRouter(tags=["recommendations"])

    @router.post(
        "/recommendations",
        response_model=RecommendationResponse,
        responses={
            status.HTTP_409_CONFLICT: {"description": COLD_START},
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": RECOMMENDATION_UNAVAILABLE
            },
        },
    )
    def recommend(
        user: Annotated[AuthUser, Depends(current_user_dependency)],
        request: RecommendationRequest | None = None,
    ) -> RecommendationResponse:
        if recommendation_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=RECOMMENDATION_UNAVAILABLE,
            )
        try:
            result = recommendation_service.recommend(
                user_id=user.id,
                configuration_key=(
                    request.configuration_key if request is not None else None
                ),
            )
        except ColdStartError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=COLD_START,
            ) from exc
        except RecommendationUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=RECOMMENDATION_UNAVAILABLE,
            ) from exc
        return RecommendationResponse(
            event_id=result.event_id,
            model_version_id=result.model_version_id,
            strategy=result.strategy,
            configuration_key=result.configuration_key,
            mode=result.mode,
            lambda_=result.lambda_,
            items=[
                RecommendationItemResponse(
                    movie_id=item.movie_id,
                    title=item.title,
                    genres=list(item.genres),
                    rank=item.rank,
                    score=item.score,
                )
                for item in result.items
            ],
        )

    return router
