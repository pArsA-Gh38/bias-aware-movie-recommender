"""Deterministic factor scoring and ranking over one complete fit catalog."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np


class RuntimeInferenceError(ValueError):
    """Base error for deterministic runtime inference contract failures."""


class InsufficientCatalogItems(RuntimeInferenceError):
    """Raised when exclusions/non-finite scores leave fewer than requested K."""


@dataclass(frozen=True, slots=True)
class FactorRecommendation:
    rank: int
    movie_id: int
    score: float


def _positive_integer(value: Integral, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer scalar")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _movie_ids(values: Sequence[Integral]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("item_ids must be an ordered sequence")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("item_ids must contain non-negative integer scalars")
        movie_id = int(value)
        if movie_id < 0:
            raise ValueError("item_ids must be non-negative")
        result.append(movie_id)
    if not result:
        raise ValueError("item_ids must not be empty")
    if len(result) != len(set(result)):
        raise ValueError("item_ids must not contain duplicates")
    return tuple(result)


def _excluded_indices(values: Collection[Integral], catalog_size: int) -> set[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise TypeError("excluded_item_indices must be a collection")
    result: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("excluded item indices must be integer scalars")
        index = int(value)
        if index < 0 or index >= catalog_size:
            raise ValueError("excluded item index is outside catalog bounds")
        result.add(index)
    return result


def rank_factor_items(
    *,
    user_factor: np.ndarray,
    item_factors: np.ndarray,
    item_ids: Sequence[Integral],
    excluded_item_indices: Collection[Integral],
    k: Integral,
) -> tuple[FactorRecommendation, ...]:
    """Score all catalog items, exclude seen/profile indices, then return top-K."""

    requested_k = _positive_integer(k, name="recommendation_k")
    catalog = _movie_ids(item_ids)
    factors = np.asarray(item_factors)
    user = np.asarray(user_factor)
    if factors.ndim != 2 or factors.shape[0] != len(catalog):
        raise ValueError("item_factors must align with item_ids")
    if user.ndim != 1 or user.shape[0] != factors.shape[1]:
        raise ValueError("user_factor must align with item factor dimension")
    excluded = _excluded_indices(excluded_item_indices, len(catalog))
    scores = np.asarray(factors @ user, dtype=np.float32)
    candidates = [
        (movie_id, float(scores[index]))
        for index, movie_id in enumerate(catalog)
        if index not in excluded and np.isfinite(scores[index])
    ]
    candidates.sort(key=lambda pair: (-pair[1], pair[0]))
    if len(candidates) < requested_k:
        raise InsufficientCatalogItems(
            "complete fit catalog cannot provide requested recommendation count"
        )
    return tuple(
        FactorRecommendation(rank=rank, movie_id=movie_id, score=score)
        for rank, (movie_id, score) in enumerate(candidates[:requested_k], start=1)
    )
