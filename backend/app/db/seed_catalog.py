"""Validate and transactionally seed the canonical MovieLens movie catalog."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import create_database_engine
from app.models.database import Movie

CANONICAL_MOVIE_COUNT = 3_883
REQUIRED_COLUMNS = ("movie_id", "title", "genres")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_INPUT_PATH = REPOSITORY_ROOT / "data" / "processed" / "movies.parquet"
TERMINAL_YEAR_PATTERN = re.compile(r"\(([0-9]{4})\)$")
MAX_POSTGRESQL_INTEGER = 2_147_483_647


class CatalogImportError(RuntimeError):
    """Base class for safe catalog-import failures."""


class CatalogInputError(CatalogImportError):
    """The source catalog is missing or violates the DB-002 contract."""


class CatalogVerificationError(CatalogImportError):
    """The persisted catalog does not exactly match the validated source."""


@dataclass(frozen=True, slots=True)
class CatalogMovie:
    """One fully validated row ready for deterministic persistence."""

    id: int
    title: str
    release_year: int | None
    genres: tuple[str, ...]
    is_active: bool = True

    def database_values(self) -> dict[str, object]:
        """Return values owned by the canonical seed."""
        return {
            "id": self.id,
            "title": self.title,
            "release_year": self.release_year,
            "genres": list(self.genres),
            "is_active": self.is_active,
        }


@dataclass(frozen=True, slots=True)
class SeedResult:
    """Concise evidence returned after a verified transaction commits."""

    source_count: int
    source_unique_ids: int
    database_count: int


def extract_release_year(title: str) -> int | None:
    """Extract only an unambiguous terminal ``(YYYY)`` without changing title."""
    match = TERMINAL_YEAR_PATTERN.search(title)
    return int(match.group(1)) if match else None


def split_genres(value: str) -> tuple[str, ...]:
    """Split a MovieLens genre string while preserving token text and order."""
    tokens = tuple(value.split("|"))
    if any(not token.strip() for token in tokens):
        raise CatalogInputError("genre strings must not contain empty tokens")
    return tokens


def validate_catalog_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_count: int = CANONICAL_MOVIE_COUNT,
) -> tuple[CatalogMovie, ...]:
    """Validate and deterministically transform source rows without DB access."""
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")

    source_rows = list(rows)
    if len(source_rows) != expected_count:
        raise CatalogInputError(
            f"unexpected movie row count: {len(source_rows)} != {expected_count}"
        )

    movies: list[CatalogMovie] = []
    seen_ids: set[int] = set()
    for row_number, row in enumerate(source_rows, start=1):
        missing = set(REQUIRED_COLUMNS).difference(row)
        if missing:
            raise CatalogInputError(
                f"row {row_number} is missing columns: {sorted(missing)}"
            )

        movie_id = row["movie_id"]
        if type(movie_id) is not int:
            raise CatalogInputError(
                f"row {row_number} movie_id must be an integer without coercion"
            )
        if movie_id <= 0 or movie_id > MAX_POSTGRESQL_INTEGER:
            raise CatalogInputError(
                f"row {row_number} movie_id is outside the positive INTEGER range"
            )
        if movie_id in seen_ids:
            raise CatalogInputError(f"duplicate movie_id: {movie_id}")
        seen_ids.add(movie_id)

        title = row["title"]
        if not isinstance(title, str) or not title.strip():
            raise CatalogInputError(f"row {row_number} title must be non-empty text")

        genre_text = row["genres"]
        if not isinstance(genre_text, str) or not genre_text.strip():
            raise CatalogInputError(f"row {row_number} genres must be non-empty text")

        movies.append(
            CatalogMovie(
                id=movie_id,
                title=title,
                release_year=extract_release_year(title),
                genres=split_genres(genre_text),
            )
        )

    return tuple(sorted(movies, key=lambda movie: movie.id))


def read_catalog(
    input_path: Path,
    *,
    expected_count: int = CANONICAL_MOVIE_COUNT,
) -> tuple[CatalogMovie, ...]:
    """Read only the accepted catalog columns and validate them fail-closed."""
    path = input_path.expanduser().resolve()
    if not path.is_file():
        raise CatalogInputError(
            "canonical DATA-002 catalog is missing: "
            f"{path}. Restore data/processed/movies.parquet before seeding."
        )

    try:
        schema = pq.read_schema(path)
    except (OSError, pa.ArrowException) as error:
        raise CatalogInputError(
            f"cannot read catalog schema from {path}: {error}"
        ) from error

    missing_columns = set(REQUIRED_COLUMNS).difference(schema.names)
    if missing_columns:
        raise CatalogInputError(
            f"catalog is missing required columns: {sorted(missing_columns)}"
        )

    movie_id_type = schema.field("movie_id").type
    if not pa.types.is_integer(movie_id_type):
        raise CatalogInputError(
            f"movie_id must use an integer Parquet type, found {movie_id_type}"
        )
    for column in ("title", "genres"):
        column_type = schema.field(column).type
        if not (
            pa.types.is_string(column_type) or pa.types.is_large_string(column_type)
        ):
            raise CatalogInputError(
                f"{column} must use a string Parquet type, found {column_type}"
            )

    try:
        table = pq.read_table(path, columns=list(REQUIRED_COLUMNS))
    except (OSError, pa.ArrowException) as error:
        raise CatalogInputError(
            f"cannot read catalog rows from {path}: {error}"
        ) from error

    return validate_catalog_rows(table.to_pylist(), expected_count=expected_count)


def _verification_error(expected_ids: set[int], actual_ids: set[int]) -> str:
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    return (
        "persisted movie identity set does not match the source: "
        f"missing_count={len(missing)} missing_sample={missing[:10]} "
        f"unexpected_count={len(unexpected)} unexpected_sample={unexpected[:10]}"
    )


def seed_catalog(engine: Engine, movies: Sequence[CatalogMovie]) -> SeedResult:
    """Upsert and verify the complete catalog in one PostgreSQL transaction."""
    if engine.dialect.name != "postgresql":
        raise CatalogImportError("DB-002 catalog import requires PostgreSQL")
    if not movies:
        raise CatalogInputError("validated catalog must not be empty")

    expected_by_id = {movie.id: movie for movie in movies}
    if len(expected_by_id) != len(movies):
        raise CatalogInputError("validated catalog contains duplicate movie IDs")

    movies_table = Movie.__table__
    values = [movie.database_values() for movie in movies]
    statement = insert(movies_table).values(values)
    statement = statement.on_conflict_do_update(
        index_elements=[movies_table.c.id],
        set_={
            "title": statement.excluded.title,
            "release_year": statement.excluded.release_year,
            "genres": statement.excluded.genres,
            "is_active": statement.excluded.is_active,
        },
    )

    with engine.begin() as connection:
        connection.execute(statement)
        persisted = connection.execute(
            select(
                movies_table.c.id,
                movies_table.c.title,
                movies_table.c.release_year,
                movies_table.c.genres,
                movies_table.c.is_active,
            ).order_by(movies_table.c.id)
        ).all()

        actual_ids = {row.id for row in persisted}
        expected_ids = set(expected_by_id)
        if actual_ids != expected_ids:
            raise CatalogVerificationError(
                _verification_error(expected_ids, actual_ids)
            )

        for row in persisted:
            expected = expected_by_id[row.id]
            actual_metadata = (
                row.title,
                row.release_year,
                tuple(row.genres),
                row.is_active,
            )
            expected_metadata = (
                expected.title,
                expected.release_year,
                expected.genres,
                expected.is_active,
            )
            if actual_metadata != expected_metadata:
                raise CatalogVerificationError(
                    f"persisted metadata mismatch for movie_id={row.id}"
                )

    return SeedResult(
        source_count=len(movies),
        source_unique_ids=len(expected_by_id),
        database_count=len(persisted),
    )


def seed_catalog_from_path(
    engine: Engine,
    input_path: Path,
    *,
    expected_count: int = CANONICAL_MOVIE_COUNT,
) -> SeedResult:
    """Validate the complete input before starting the write transaction."""
    movies = read_catalog(input_path, expected_count=expected_count)
    return seed_catalog(engine, movies)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately small canonical command interface."""
    parser = argparse.ArgumentParser(
        description="Seed the canonical MovieLens 1M catalog into PostgreSQL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=CANONICAL_INPUT_PATH,
        help=f"catalog Parquet path (default: {CANONICAL_INPUT_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canonical production-count seed command."""
    args = parse_args(argv)
    engine: Engine | None = None
    try:
        movies = read_catalog(args.input)
        engine = create_database_engine()
        result = seed_catalog(engine, movies)
    except (CatalogImportError, RuntimeError, SQLAlchemyError) as error:
        print(f"Catalog seed failed: {error}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()

    print(f"source_path={args.input.expanduser().resolve()}")
    print(f"source_row_count={result.source_count}")
    print(f"source_unique_movie_ids={result.source_unique_ids}")
    print(f"database_movie_count={result.database_count}")
    print("identity_set_verified=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
