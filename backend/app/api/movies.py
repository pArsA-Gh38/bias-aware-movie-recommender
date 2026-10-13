"""HTTP route for the authenticated movie catalog."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import AfterValidator

from app.api.auth import CurrentUserDependency
from app.core.auth import AuthUser
from app.core.movies import (
    CatalogRepositoryUnavailableError,
    MovieCatalogService,
)
from app.schemas.movies import MovieListResponse, MovieResponse

CATALOG_UNAVAILABLE = "Catalog service unavailable"


def _trimmed_filter(*, name: str, maximum: int):
    def validate(value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        if not 1 <= len(trimmed) <= maximum:
            raise ValueError(f"{name} must contain between 1 and {maximum} characters")
        return trimmed

    return validate


TitleQuery = Annotated[
    str | None,
    Query(),
    AfterValidator(_trimmed_filter(name="q", maximum=200)),
]
GenreQuery = Annotated[
    str | None,
    Query(),
    AfterValidator(_trimmed_filter(name="genre", maximum=50)),
]


def create_movies_router(
    movie_catalog_service: MovieCatalogService | None,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    router = APIRouter(tags=["movies"])

    @router.get("/movies", response_model=MovieListResponse)
    def list_movies(
        _current_user: Annotated[AuthUser, Depends(current_user_dependency)],
        q: TitleQuery = None,
        genre: GenreQuery = None,
        cursor: Annotated[str | None, Query(pattern=r"^[1-9][0-9]*$")] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        onboarding: bool = False,
    ) -> MovieListResponse:
        if onboarding and any(value is not None for value in (q, genre, cursor)):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=[
                    {
                        "type": "value_error",
                        "loc": ["query", "onboarding"],
                        "msg": "Value error, onboarding cannot be combined with q, genre, or cursor",
                        "input": True,
                    }
                ],
            )
        if movie_catalog_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=CATALOG_UNAVAILABLE,
            )
        try:
            page = movie_catalog_service.list_movies(
                cursor=int(cursor) if cursor is not None else None,
                limit=limit,
                title_query=q,
                genre=genre,
                onboarding=onboarding,
            )
        except CatalogRepositoryUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=CATALOG_UNAVAILABLE,
            ) from exc
        return MovieListResponse(
            items=[
                MovieResponse(
                    movie_id=movie.movie_id,
                    title=movie.title,
                    release_year=movie.release_year,
                    genres=list(movie.genres),
                )
                for movie in page.items
            ],
            next_cursor=page.next_cursor,
        )

    return router
