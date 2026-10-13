"""Response schemas for persisted MVP recommendations."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from recommender_core import ACCEPTED_CONFIGURATION_KEYS, DEFAULT_CONFIGURATION_KEY

AcceptedConfigurationKey = Literal[*ACCEPTED_CONFIGURATION_KEYS]


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configuration_key: AcceptedConfigurationKey = DEFAULT_CONFIGURATION_KEY


class RecommendationItemResponse(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    rank: int
    score: float


class RecommendationResponse(BaseModel):
    event_id: UUID
    model_version_id: UUID
    strategy: str
    configuration_key: str
    mode: str
    lambda_: float = Field(serialization_alias="lambda")
    items: list[RecommendationItemResponse]
