"""Authenticated feedback endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.auth import CurrentUserDependency
from app.core.auth import AuthUser
from app.core.feedback import (
    FeedbackEventNotFoundError,
    FeedbackRelationshipConflictError,
    FeedbackService,
    FeedbackUnavailableError,
)
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

EVENT_NOT_FOUND = "Recommendation event not found"
MOVIE_NOT_IN_EVENT = "Movie is not part of the recommendation event"
FEEDBACK_UNAVAILABLE = "Feedback service unavailable"


def create_feedback_router(
    feedback_service: FeedbackService | None,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    router = APIRouter(tags=["feedback"])

    @router.post(
        "/feedback",
        response_model=FeedbackResponse,
        status_code=status.HTTP_201_CREATED,
        responses={
            status.HTTP_401_UNAUTHORIZED: {"description": "Authentication required"},
            status.HTTP_404_NOT_FOUND: {"description": EVENT_NOT_FOUND},
            status.HTTP_409_CONFLICT: {"description": MOVIE_NOT_IN_EVENT},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Invalid feedback request"
            },
            status.HTTP_503_SERVICE_UNAVAILABLE: {"description": FEEDBACK_UNAVAILABLE},
        },
    )
    def record_feedback(
        payload: FeedbackRequest,
        user: Annotated[AuthUser, Depends(current_user_dependency)],
    ) -> FeedbackResponse:
        if feedback_service is None:
            raise HTTPException(503, FEEDBACK_UNAVAILABLE)
        try:
            feedback_service.record(
                user_id=user.id,
                event_id=payload.event_id,
                movie_id=payload.movie_id,
                action=payload.action,
            )
        except FeedbackEventNotFoundError as exc:
            raise HTTPException(404, EVENT_NOT_FOUND) from exc
        except FeedbackRelationshipConflictError as exc:
            raise HTTPException(409, MOVIE_NOT_IN_EVENT) from exc
        except FeedbackUnavailableError as exc:
            raise HTTPException(503, FEEDBACK_UNAVAILABLE) from exc
        return FeedbackResponse(status="recorded")

    return router
