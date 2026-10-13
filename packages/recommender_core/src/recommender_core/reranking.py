"""Pure deterministic greedy reranking over one authoritative candidate pool.

The caller supplies ordered candidate IDs, raw relevance and fit-popularity
novelty features, and genre metadata.  This module performs no I/O, model
activity, candidate generation, metric evaluation, or target-label access.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike

from recommender_core.normalization import normalize_reranking_features


@dataclass(frozen=True, slots=True)
class RerankedItem:
    """Immutable selection-time evidence for one reranked candidate."""

    movie_id: int
    rank: int
    base_rank: int
    raw_relevance: float
    normalized_relevance: float
    raw_novelty: float
    normalized_novelty: float
    genre_diversity: float
    beyond_score: float
    final_score: float


@dataclass(frozen=True, slots=True)
class _Candidate:
    movie_id: int
    base_rank: int
    raw_relevance: float
    normalized_relevance: float
    raw_novelty: float
    normalized_novelty: float
    genres: frozenset[str]


def _validate_positive_integer(value: Integral, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer scalar, not a boolean")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _validate_unit_interval(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not a boolean")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be inside [0, 1]")
    return result


def _validate_candidate_ids(
    candidate_ids: Sequence[Integral] | np.ndarray,
) -> tuple[int, ...]:
    if isinstance(candidate_ids, np.ndarray):
        if candidate_ids.ndim != 1:
            raise ValueError("candidate_ids must be one-dimensional")
        supplied = tuple(candidate_ids)
    elif isinstance(candidate_ids, (str, bytes)) or not isinstance(
        candidate_ids, Sequence
    ):
        raise TypeError("candidate_ids must be an ordered sequence")
    else:
        supplied = tuple(candidate_ids)

    validated: list[int] = []
    seen: set[int] = set()
    for movie_id in supplied:
        if isinstance(movie_id, bool) or not isinstance(movie_id, Integral):
            raise TypeError("candidate IDs must be non-negative integer scalars")
        integer_id = int(movie_id)
        if integer_id < 0:
            raise ValueError("candidate IDs must be non-negative")
        if integer_id in seen:
            raise ValueError("candidate IDs must be unique")
        seen.add(integer_id)
        validated.append(integer_id)
    return tuple(validated)


def _validate_item_genres(
    item_genres: Mapping[Integral, Collection[str]],
) -> dict[int, frozenset[str]]:
    if not isinstance(item_genres, Mapping):
        raise TypeError("item_genres must be a mapping")

    validated: dict[int, frozenset[str]] = {}
    for movie_id, genres in item_genres.items():
        if isinstance(movie_id, bool) or not isinstance(movie_id, Integral):
            raise TypeError("item_genres keys must be non-negative integer scalars")
        integer_id = int(movie_id)
        if integer_id < 0:
            raise ValueError("item_genres keys must be non-negative")
        if isinstance(genres, (str, bytes)) or not isinstance(genres, Collection):
            raise TypeError("item_genres values must be genre-label collections")

        labels: set[str] = set()
        for label in genres:
            if not isinstance(label, str):
                raise TypeError("genre labels must be strings")
            if label == "":
                raise ValueError("genre labels must not be empty strings")
            if label in labels:
                raise ValueError("genre collections must not contain duplicate labels")
            labels.add(label)
        validated[integer_id] = frozenset(labels)
    return validated


def _jaccard_similarity(
    left: frozenset[str],
    right: frozenset[str],
) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _genre_diversity(
    candidate: _Candidate,
    selected: Sequence[_Candidate],
) -> float:
    if not selected:
        return 1.0
    similarities = (
        _jaccard_similarity(candidate.genres, item.genres) for item in selected
    )
    return 1.0 - max(similarities)


def _selection_key(
    candidate: _Candidate,
    final_score: float,
) -> tuple[float, int, int]:
    return (-final_score, candidate.base_rank, candidate.movie_id)


def _require_unit_score(value: float, *, name: str) -> float:
    if not np.isfinite(value):
        raise ArithmeticError(f"{name} became non-finite")
    if value < 0.0 or value > 1.0:
        raise ArithmeticError(f"{name} escaped [0, 1]")
    return value


def greedy_rerank(
    candidate_ids: Sequence[Integral] | np.ndarray,
    relevance: ArrayLike,
    novelty: ArrayLike,
    item_genres: Mapping[Integral, Collection[str]],
    *,
    k: Integral,
    lambda_: Real,
    beta: Real,
) -> tuple[RerankedItem, ...]:
    """Select up to ``k`` unique candidates with the frozen greedy score.

    Input position defines one-based ``base_rank``.  Relevance and novelty are
    normalized independently by the accepted RERANK-001 implementation.
    Diversity is recomputed against the complete selected prefix at every
    iteration.  Exact ties use lower base rank and then lower movie ID.
    """

    requested_k = _validate_positive_integer(k, name="k")
    relevance_weight = _validate_unit_interval(lambda_, name="lambda_")
    diversity_weight = _validate_unit_interval(beta, name="beta")
    validated_ids = _validate_candidate_ids(candidate_ids)

    raw_relevance = np.asarray(relevance, dtype=np.float64)
    raw_novelty = np.asarray(novelty, dtype=np.float64)
    normalized = normalize_reranking_features(raw_relevance, raw_novelty)
    if len(validated_ids) != normalized.relevance.size:
        raise ValueError(
            "candidate_ids, relevance, and novelty must have equal lengths"
        )

    genres_by_id = _validate_item_genres(item_genres)
    missing_genres = [
        movie_id for movie_id in validated_ids if movie_id not in genres_by_id
    ]
    if missing_genres:
        raise ValueError(
            f"every candidate must be present in item_genres; missing={missing_genres}"
        )

    candidates = tuple(
        _Candidate(
            movie_id=movie_id,
            base_rank=index + 1,
            raw_relevance=float(raw_relevance[index]),
            normalized_relevance=float(normalized.relevance[index]),
            raw_novelty=float(raw_novelty[index]),
            normalized_novelty=float(normalized.novelty[index]),
            genres=genres_by_id[movie_id],
        )
        for index, movie_id in enumerate(validated_ids)
    )
    selected_candidates: list[_Candidate] = []
    selected_results: list[RerankedItem] = []
    remaining = list(candidates)

    while remaining and len(selected_results) < min(requested_k, len(candidates)):
        scored: list[tuple[_Candidate, float, float, float]] = []
        for candidate in remaining:
            diversity = _require_unit_score(
                _genre_diversity(candidate, selected_candidates),
                name="genre_diversity",
            )
            beyond_score = _require_unit_score(
                diversity_weight * diversity
                + (1.0 - diversity_weight) * candidate.normalized_novelty,
                name="beyond_score",
            )
            final_score = _require_unit_score(
                relevance_weight * candidate.normalized_relevance
                + (1.0 - relevance_weight) * beyond_score,
                name="final_score",
            )
            scored.append((candidate, diversity, beyond_score, final_score))

        candidate, diversity, beyond_score, final_score = min(
            scored,
            key=lambda row: _selection_key(row[0], row[3]),
        )
        selected_candidates.append(candidate)
        selected_results.append(
            RerankedItem(
                movie_id=candidate.movie_id,
                rank=len(selected_results) + 1,
                base_rank=candidate.base_rank,
                raw_relevance=candidate.raw_relevance,
                normalized_relevance=candidate.normalized_relevance,
                raw_novelty=candidate.raw_novelty,
                normalized_novelty=candidate.normalized_novelty,
                genre_diversity=diversity,
                beyond_score=beyond_score,
                final_score=final_score,
            )
        )
        remaining.remove(candidate)

    return tuple(selected_results)
