"""Movie catalog use cases independent of HTTP and SQLAlchemy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CatalogMovie:
    movie_id: int
    title: str
    release_year: int | None
    genres: tuple[str, ...]


class CatalogRepositoryUnavailableError(RuntimeError):
    """Raised when persistence cannot serve the movie catalog."""


class MovieCatalogRepository(Protocol):
    def list_active_after(
        self,
        *,
        after_movie_id: int | None,
        limit: int,
        title_query: str | None = None,
        genre: str | None = None,
    ) -> Sequence[CatalogMovie]: ...

    def list_all_active(self) -> Sequence[CatalogMovie]: ...


@dataclass(frozen=True, slots=True)
class MovieCatalogPage:
    items: tuple[CatalogMovie, ...]
    next_cursor: str | None


class MovieCatalogService:
    """Coordinate deterministic cursor pagination over active movies."""

    def __init__(self, repository: MovieCatalogRepository) -> None:
        self._repository = repository

    def list_movies(
        self,
        *,
        cursor: int | None = None,
        limit: int = 50,
        title_query: str | None = None,
        genre: str | None = None,
        onboarding: bool = False,
    ) -> MovieCatalogPage:
        if onboarding:
            return self._onboarding_page(limit=limit)
        rows = tuple(
            self._repository.list_active_after(
                after_movie_id=cursor,
                limit=limit + 1,
                title_query=title_query,
                genre=genre,
            )
        )
        items = rows[:limit]
        next_cursor = str(items[-1].movie_id) if len(rows) > limit else None
        return MovieCatalogPage(items=items, next_cursor=next_cursor)

    def _onboarding_page(self, *, limit: int) -> MovieCatalogPage:
        movies = tuple(
            sorted(self._repository.list_all_active(), key=lambda movie: movie.movie_id)
        )
        buckets: dict[str, list[CatalogMovie]] = {}
        for movie in movies:
            for genre in movie.genres:
                if genre:
                    buckets.setdefault(genre, []).append(movie)

        ordered_buckets = [
            buckets[label]
            for label in sorted(buckets, key=lambda label: (label.lower(), label))
        ]
        positions = [0] * len(ordered_buckets)
        selected: list[CatalogMovie] = []
        selected_ids: set[int] = set()

        while len(selected) < limit:
            added_in_round = False
            for bucket_index, bucket in enumerate(ordered_buckets):
                position = positions[bucket_index]
                while (
                    position < len(bucket) and bucket[position].movie_id in selected_ids
                ):
                    position += 1
                positions[bucket_index] = position + 1
                if position >= len(bucket):
                    continue
                movie = bucket[position]
                selected.append(movie)
                selected_ids.add(movie.movie_id)
                added_in_round = True
                if len(selected) == limit:
                    break
            if not added_in_round:
                break

        if len(selected) < limit:
            for movie in movies:
                if movie.movie_id in selected_ids:
                    continue
                selected.append(movie)
                selected_ids.add(movie.movie_id)
                if len(selected) == limit:
                    break

        return MovieCatalogPage(items=tuple(selected), next_cursor=None)
