"""SQLAlchemy persistence adapter for the movie catalog."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from sqlalchemy import exists, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.movies import CatalogMovie, CatalogRepositoryUnavailableError
from app.models import Movie


class SQLAlchemyMovieCatalogRepository:
    """Read active catalog movies in canonical MovieLens ID order."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory
        bind = getattr(session_factory, "kw", {}).get("bind")
        self._sqlite = bind is not None and bind.dialect.name == "sqlite"

    def list_active_after(
        self,
        *,
        after_movie_id: int | None,
        limit: int,
        title_query: str | None = None,
        genre: str | None = None,
    ) -> Sequence[CatalogMovie]:
        statement = select(Movie).where(Movie.is_active.is_(True))
        if after_movie_id is not None:
            statement = statement.where(Movie.id > after_movie_id)
        if title_query is not None:
            escaped_query = (
                title_query.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            statement = statement.where(
                Movie.title.ilike(f"%{escaped_query}%", escape="\\")
            )
        if genre is not None:
            if self._sqlite:
                genre_values = (
                    func.json_each(Movie.genres)
                    .table_valued("key", "value")
                    .alias("genre_values")
                )
                genre_column = genre_values.c.value
            else:
                genre_values = (
                    func.unnest(Movie.genres)
                    .table_valued("genre")
                    .render_derived(name="genre_values")
                )
                genre_column = genre_values.c.genre
            statement = statement.where(
                exists(
                    select(1)
                    .select_from(genre_values)
                    .where(func.lower(genre_column) == genre.lower())
                )
            )
        statement = statement.order_by(Movie.id.asc()).limit(limit)
        return self._execute(statement)

    def list_all_active(self) -> Sequence[CatalogMovie]:
        statement = (
            select(Movie).where(Movie.is_active.is_(True)).order_by(Movie.id.asc())
        )
        return self._execute(statement)

    def _execute(self, statement: object) -> Sequence[CatalogMovie]:
        try:
            with self._session_factory() as session:
                movies = session.scalars(statement).all()
        except SQLAlchemyError as exc:
            raise CatalogRepositoryUnavailableError from exc
        return tuple(
            CatalogMovie(
                movie_id=movie.id,
                title=movie.title,
                release_year=movie.release_year,
                genres=tuple(movie.genres),
            )
            for movie in movies
        )
