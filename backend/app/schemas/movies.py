"""Response schemas for the authenticated movie catalog."""

from pydantic import BaseModel


class MovieResponse(BaseModel):
    movie_id: int
    title: str
    release_year: int | None
    genres: list[str]


class MovieListResponse(BaseModel):
    items: list[MovieResponse]
    next_cursor: str | None
