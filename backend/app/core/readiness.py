"""Injectable readiness state for dependencies owned by later Roadmap tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.recommendation.artifact_loader import ArtifactLoader


@dataclass(frozen=True, slots=True)
class ReadinessState:
    """Current state of the dependencies required to serve product traffic."""

    database_reachable: bool
    artifact_loaded: bool
    artifact_schema_compatible: bool | None

    @property
    def is_ready(self) -> bool:
        return (
            self.database_reachable
            and self.artifact_loaded
            and self.artifact_schema_compatible is True
        )

    @classmethod
    def not_configured(cls) -> ReadinessState:
        """Fail closed until DB and artifact integrations supply real state."""
        return cls(
            database_reachable=False,
            artifact_loaded=False,
            artifact_schema_compatible=None,
        )


class ReadinessProvider(Protocol):
    """Boundary implemented by future DB and artifact integration work."""

    def get_state(self) -> ReadinessState:
        """Return a side-effect-free snapshot of current dependency state."""
        ...


@dataclass(frozen=True, slots=True)
class StaticReadinessProvider:
    """Immutable provider used for fail-closed defaults and deterministic tests."""

    state: ReadinessState

    def get_state(self) -> ReadinessState:
        return self.state


@dataclass(frozen=True, slots=True)
class RuntimeReadinessProvider:
    """Probe configured runtime dependencies when readiness is requested."""

    database_engine: Engine | None
    artifact_loader: ArtifactLoader | None

    def get_state(self) -> ReadinessState:
        database_reachable = False
        if self.database_engine is not None:
            try:
                with self.database_engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                database_reachable = True
            except SQLAlchemyError:
                database_reachable = False

        loader = self.artifact_loader
        return ReadinessState(
            database_reachable=database_reachable,
            artifact_loaded=loader.is_loaded if loader is not None else False,
            artifact_schema_compatible=(
                loader.schema_compatible if loader is not None else None
            ),
        )
