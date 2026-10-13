"""Public schemas for authenticated feedback recording."""

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class FeedbackAction(StrEnum):
    CLICK = "click"
    LIKE = "like"
    DISLIKE = "dislike"


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: UUID
    movie_id: int = Field(strict=True)
    action: FeedbackAction


class FeedbackResponse(BaseModel):
    status: Literal["recorded"]
