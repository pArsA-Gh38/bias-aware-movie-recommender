"""Authenticated read-only EXP-003 summary endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.auth import CurrentUserDependency
from app.core.auth import AuthUser
from app.core.experiments import (
    ExperimentSummaryProvider,
)
from app.schemas.experiments import ExperimentSummaryResponse

EXPERIMENT_SUMMARY_UNAVAILABLE = "Experiment summary unavailable"


def create_experiments_router(
    provider: ExperimentSummaryProvider | None,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    router = APIRouter(tags=["experiments"])

    @router.get(
        "/experiments/summary",
        response_model=ExperimentSummaryResponse,
        responses={
            status.HTTP_401_UNAUTHORIZED: {"description": "Authentication required"},
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": EXPERIMENT_SUMMARY_UNAVAILABLE
            },
        },
    )
    def experiment_summary(
        _user: Annotated[AuthUser, Depends(current_user_dependency)],
    ) -> ExperimentSummaryResponse:
        if provider is None:
            raise HTTPException(503, EXPERIMENT_SUMMARY_UNAVAILABLE)
        try:
            return provider.get_summary()
        except Exception as exc:
            # The provider is an injected boundary; no implementation detail is public.
            raise HTTPException(503, EXPERIMENT_SUMMARY_UNAVAILABLE) from exc

    return router
