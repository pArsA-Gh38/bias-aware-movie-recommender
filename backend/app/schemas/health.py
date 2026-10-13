"""Stable response schemas for health endpoints."""

from enum import Enum

from pydantic import BaseModel


class HealthStatus(str, Enum):
    ALIVE = "alive"
    READY = "ready"
    NOT_READY = "not_ready"


class LivenessResponse(BaseModel):
    status: HealthStatus


class ReadinessChecks(BaseModel):
    database_reachable: bool
    artifact_loaded: bool
    artifact_schema_compatible: bool | None


class ReadinessResponse(BaseModel):
    status: HealthStatus
    checks: ReadinessChecks
