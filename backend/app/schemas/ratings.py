from datetime import datetime

from pydantic import BaseModel, Field


class RatingRequest(BaseModel):
    rating: int = Field(ge=1, le=5)


class RatingResponse(BaseModel):
    movie_id: int
    rating: int
    positive_count: int
    updated_at: datetime


class UserRatingItem(BaseModel):
    movie_id: int
    rating: int
    created_at: datetime
    updated_at: datetime


class UserRatingsResponse(BaseModel):
    items: list[UserRatingItem]
