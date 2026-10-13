"""Unauthenticated process and dependency health routes."""

from fastapi import APIRouter, Response, status

from app.core.readiness import ReadinessProvider
from app.schemas.health import (
    HealthStatus,
    LivenessResponse,
    ReadinessChecks,
    ReadinessResponse,
)


def create_health_router(readiness_provider: ReadinessProvider) -> APIRouter:
    """Build health routes around an explicitly supplied readiness provider."""
    router = APIRouter(prefix="/health", tags=["health"])

    @router.get("/live", response_model=LivenessResponse)
    def live() -> LivenessResponse:
        return LivenessResponse(status=HealthStatus.ALIVE)

    @router.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
    )
    def ready(response: Response) -> ReadinessResponse:
        state = readiness_provider.get_state()
        if not state.is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status=HealthStatus.READY if state.is_ready else HealthStatus.NOT_READY,
            checks=ReadinessChecks(
                database_reachable=state.database_reachable,
                artifact_loaded=state.artifact_loaded,
                artifact_schema_compatible=state.artifact_schema_compatible,
            ),
        )

    return router
