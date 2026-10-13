"""Feedback service and persistence protocol."""

from typing import Protocol
from uuid import UUID

from app.schemas.feedback import FeedbackAction


class FeedbackEventNotFoundError(RuntimeError):
    """The authenticated user does not own the requested event."""


class FeedbackRelationshipConflictError(RuntimeError):
    """The requested movie is not part of the owned event."""


class FeedbackUnavailableError(RuntimeError):
    """Feedback persistence failed."""


class FeedbackRepository(Protocol):
    def record(
        self,
        *,
        user_id: UUID,
        event_id: UUID,
        movie_id: int,
        action: FeedbackAction,
    ) -> None: ...


class FeedbackService:
    def __init__(self, repository: FeedbackRepository) -> None:
        self._repository = repository

    def record(
        self,
        *,
        user_id: UUID,
        event_id: UUID,
        movie_id: int,
        action: FeedbackAction,
    ) -> None:
        self._repository.record(
            user_id=user_id,
            event_id=event_id,
            movie_id=movie_id,
            action=action,
        )
