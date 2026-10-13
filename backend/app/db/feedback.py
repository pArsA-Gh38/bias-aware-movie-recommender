"""SQLAlchemy feedback persistence boundary."""

from collections.abc import Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.feedback import (
    FeedbackEventNotFoundError,
    FeedbackRelationshipConflictError,
    FeedbackUnavailableError,
)
from app.models import Feedback, RecommendationEvent, RecommendationItem
from app.schemas.feedback import FeedbackAction


class SQLAlchemyFeedbackRepository:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def record(
        self,
        *,
        user_id: UUID,
        event_id: UUID,
        movie_id: int,
        action: FeedbackAction,
    ) -> None:
        with self._session_factory() as session:
            try:
                owned_event_id = session.scalar(
                    select(RecommendationEvent.id).where(
                        RecommendationEvent.id == event_id,
                        RecommendationEvent.user_id == user_id,
                    )
                )
                if owned_event_id is None:
                    raise FeedbackEventNotFoundError

                item_id = session.scalar(
                    select(RecommendationItem.id).where(
                        RecommendationItem.event_id == event_id,
                        RecommendationItem.movie_id == movie_id,
                    )
                )
                if item_id is None:
                    raise FeedbackRelationshipConflictError

                session.add(
                    Feedback(
                        user_id=user_id,
                        event_id=event_id,
                        movie_id=movie_id,
                        action=action.value,
                    )
                )
                session.commit()
            except (
                FeedbackEventNotFoundError,
                FeedbackRelationshipConflictError,
            ):
                try:
                    session.rollback()
                except SQLAlchemyError as exc:
                    raise FeedbackUnavailableError(
                        "feedback transaction failed"
                    ) from exc
                raise
            except SQLAlchemyError as exc:
                try:
                    session.rollback()
                except SQLAlchemyError:
                    pass
                raise FeedbackUnavailableError("feedback transaction failed") from exc
