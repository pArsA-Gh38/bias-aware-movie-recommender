"""SQL query contract coverage for API-003 movie filters."""

from __future__ import annotations

from typing import Self

from sqlalchemy.dialects import postgresql

from app.db.movies import SQLAlchemyMovieCatalogRepository


class EmptyScalarResult:
    def all(self) -> list[object]:
        return []


class CapturingSession:
    def __init__(self) -> None:
        self.statement: object | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def scalars(self, statement: object) -> EmptyScalarResult:
        self.statement = statement
        return EmptyScalarResult()


def test_postgresql_filters_escape_like_literals_and_unnest_exact_genre() -> None:
    session = CapturingSession()
    repository = SQLAlchemyMovieCatalogRepository(lambda: session)

    repository.list_active_after(
        after_movie_id=10,
        limit=51,
        title_query="100%_\\",
        genre="dRaMa",
    )

    assert session.statement is not None
    compiled = session.statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "movies.title ILIKE" in sql
    assert "ESCAPE '\\\\'" in sql
    assert "FROM unnest(movies.genres) AS genre_values(genre)" in sql
    assert "lower(genre_values.genre)" in sql
    assert "%100\\%\\_\\\\%" in compiled.params.values()
    assert "drama" in compiled.params.values()
