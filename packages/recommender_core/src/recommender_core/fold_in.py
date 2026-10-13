"""Non-mutating implicit ALS fold-in for genuinely new users.

This module is deliberately limited to input validation, construction of one
confidence CSR row, one ``recalculate_user`` call, and deterministic
full-catalog ranking.  Data loading, base-model fitting, persistence, fallback
ranking, and report writing belong to callers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.sparse import csr_matrix

from recommender_core.scoring import InsufficientCatalogItems, rank_factor_items


class FoldInError(ValueError):
    """Base class for explicit new-user profile contract failures."""


class EmptyPositiveInteractions(FoldInError):
    """Raised when no positive interaction is supplied."""


class InsufficientPositiveInteractions(FoldInError):
    """Raised when the profile belongs to the popularity-fallback strategy."""

    strategy = "fallback_popularity"
    fold_in_eligible = False


class DuplicatePositiveInteraction(FoldInError):
    """Raised when an external movie ID occurs more than once."""


class UnknownProfileItem(FoldInError):
    """Raised when a supplied movie is absent from the fit catalog."""


class InvalidProfileItem(TypeError):
    """Raised when a supplied profile value is not an integer-like movie ID."""


@dataclass(frozen=True)
class ALSFoldInConfig:
    """Validated confidence, threshold, and result-size contract."""

    minimum_positive_interactions: int
    recommendation_k: int
    confidence_alpha: float
    observed_confidence: float
    implicit_library_alpha: float

    def __post_init__(self) -> None:
        for name in ("minimum_positive_interactions", "recommendation_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer scalar")
            if int(value) <= 0:
                raise ValueError(f"{name} must be greater than zero")
            object.__setattr__(self, name, int(value))
        for name in (
            "confidence_alpha",
            "observed_confidence",
            "implicit_library_alpha",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real scalar")
            numeric = float(value)
            if not np.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
            object.__setattr__(self, name, numeric)
        if self.minimum_positive_interactions != 5:
            raise ValueError("minimum_positive_interactions must equal 5")
        if self.implicit_library_alpha != 1.0:
            raise ValueError(
                "implicit_library_alpha must equal 1.0 because observed "
                "confidence is stored in the profile row"
            )
        expected = 1.0 + self.confidence_alpha
        if self.observed_confidence != expected:
            raise ValueError("observed_confidence must equal 1 + confidence_alpha")


@dataclass(frozen=True)
class FoldInEligibility:
    """Cold-start routing decision without implementing fallback ranking."""

    strategy: str
    fold_in_eligible: bool
    positive_interaction_count: int
    minimum_required_positives: int


@dataclass(frozen=True)
class FoldInRecommendation:
    movie_id: int
    score: float
    rank: int


@dataclass(frozen=True)
class FoldInResult:
    strategy: str
    fold_in_eligible: bool
    profile_movie_ids: tuple[int, ...]
    user_factor: np.ndarray
    recommendations: tuple[FoldInRecommendation, ...]
    profile_csr_shape: tuple[int, int]
    profile_csr_nnz: int


def _positive_movie_ids(
    values: Sequence[Integral],
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("positive_movie_ids must be an ordered sequence")
    validated: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise InvalidProfileItem(
                "positive movie IDs must be non-negative integer scalars"
            )
        movie_id = int(value)
        if movie_id < 0:
            raise InvalidProfileItem("positive movie IDs must be non-negative")
        validated.append(movie_id)
    if not validated:
        raise EmptyPositiveInteractions("positive_movie_ids must not be empty")
    if len(validated) != len(set(validated)):
        raise DuplicatePositiveInteraction(
            "positive_movie_ids must not contain duplicates"
        )
    return tuple(sorted(validated))


def decide_fold_in_strategy(
    positive_movie_ids: Sequence[Integral],
    config: ALSFoldInConfig,
) -> FoldInEligibility:
    """Return the frozen routing decision; no fallback recommendations are made."""

    if not isinstance(config, ALSFoldInConfig):
        raise TypeError("config must be an ALSFoldInConfig")
    profile = _positive_movie_ids(positive_movie_ids)
    eligible = len(profile) >= config.minimum_positive_interactions
    return FoldInEligibility(
        strategy="als_fold_in" if eligible else "fallback_popularity",
        fold_in_eligible=eligible,
        positive_interaction_count=len(profile),
        minimum_required_positives=config.minimum_positive_interactions,
    )


def _validated_item_mapping(
    item_id_to_index: Mapping[Integral, Integral],
    catalog_size: int,
) -> dict[int, int]:
    if not isinstance(item_id_to_index, Mapping):
        raise TypeError("item_id_to_index must be a mapping")
    if isinstance(catalog_size, bool) or not isinstance(catalog_size, Integral):
        raise TypeError("catalog_size must be an integer scalar")
    size = int(catalog_size)
    if size <= 0:
        raise ValueError("catalog_size must be greater than zero")
    if len(item_id_to_index) != size:
        raise ValueError("item mapping size must equal catalog_size")
    result: dict[int, int] = {}
    indices: set[int] = set()
    for raw_movie_id, raw_index in item_id_to_index.items():
        if (
            isinstance(raw_movie_id, bool)
            or not isinstance(raw_movie_id, Integral)
            or int(raw_movie_id) < 0
        ):
            raise TypeError("item mapping keys must be non-negative integers")
        if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
            raise TypeError("item mapping indices must be integers")
        movie_id = int(raw_movie_id)
        index = int(raw_index)
        if index < 0 or index >= size:
            raise ValueError("item mapping index is outside catalog bounds")
        if movie_id in result or index in indices:
            raise ValueError("item mapping IDs and indices must be unique")
        result[movie_id] = index
        indices.add(index)
    if indices != set(range(size)):
        raise ValueError("item mapping indices must be contiguous")
    return result


def build_fold_in_confidence_row(
    positive_movie_ids: Sequence[Integral],
    item_id_to_index: Mapping[Integral, Integral],
    catalog_size: int,
    config: ALSFoldInConfig,
) -> csr_matrix:
    """Allocate one canonical CSR row containing observed confidence 11."""

    if not isinstance(config, ALSFoldInConfig):
        raise TypeError("config must be an ALSFoldInConfig")
    profile = _positive_movie_ids(positive_movie_ids)
    if len(profile) < config.minimum_positive_interactions:
        raise InsufficientPositiveInteractions(
            f"ALS fold-in requires at least {config.minimum_positive_interactions} "
            "unique positives; strategy=fallback_popularity"
        )
    mapping = _validated_item_mapping(item_id_to_index, catalog_size)
    unknown = tuple(movie_id for movie_id in profile if movie_id not in mapping)
    if unknown:
        raise UnknownProfileItem(
            f"profile items are outside the fitted catalog: {list(unknown)}"
        )
    column_indices = np.asarray(
        sorted(mapping[movie_id] for movie_id in profile), dtype=np.int32
    )
    data = np.full(len(column_indices), config.observed_confidence, dtype=np.float32)
    row = csr_matrix(
        (
            data.copy(),
            column_indices.copy(),
            np.asarray([0, len(column_indices)], dtype=np.int32),
        ),
        shape=(1, int(catalog_size)),
        dtype=np.float32,
        copy=True,
    )
    row.sort_indices()
    if not row.has_canonical_format or row.nnz != len(profile):
        raise RuntimeError("profile CSR construction violated canonical shape")
    return row


def _validated_item_ids(item_ids: Sequence[Integral]) -> tuple[int, ...]:
    if isinstance(item_ids, (str, bytes)) or not isinstance(item_ids, Sequence):
        raise TypeError("item_ids must be an ordered sequence")
    result: list[int] = []
    for value in item_ids:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("item_ids must contain non-negative integer scalars")
        item_id = int(value)
        if item_id < 0:
            raise ValueError("item_ids must be non-negative")
        result.append(item_id)
    if not result:
        raise ValueError("item_ids must not be empty")
    if len(result) != len(set(result)):
        raise ValueError("item_ids must not contain duplicates")
    return tuple(result)


def _model_item_factors(model: object, item_count: int) -> np.ndarray:
    if not callable(getattr(model, "recalculate_user", None)):
        raise TypeError("model must expose callable recalculate_user")
    alpha = getattr(model, "alpha", None)
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("model alpha must be a real scalar")
    if float(alpha) != 1.0:
        raise ValueError("model alpha must equal implicit_library_alpha 1.0")
    item_factors = np.asarray(getattr(model, "item_factors", None))
    if item_factors.ndim != 2 or item_factors.shape[0] != item_count:
        raise ValueError("model item factors must align with item_ids")
    if item_factors.shape[1] <= 0:
        raise ValueError("model item factor dimension must be positive")
    if item_factors.dtype != np.dtype("float32"):
        raise ValueError("model item factors must use float32")
    if not np.isfinite(item_factors).all():
        raise ValueError("model item factors must be finite")
    return item_factors


def fold_in_new_user(
    model: object,
    positive_movie_ids: Sequence[Integral],
    item_ids: Sequence[Integral],
    config: ALSFoldInConfig,
) -> FoldInResult:
    """Calculate one transient user factor and deterministic full-catalog top-K."""

    if not isinstance(config, ALSFoldInConfig):
        raise TypeError("config must be an ALSFoldInConfig")
    supplied_profile_snapshot = tuple(positive_movie_ids)
    profile = _positive_movie_ids(positive_movie_ids)
    decision = decide_fold_in_strategy(profile, config)
    if not decision.fold_in_eligible:
        raise InsufficientPositiveInteractions(
            f"ALS fold-in requires at least {config.minimum_positive_interactions} "
            "unique positives; strategy=fallback_popularity"
        )
    catalog = _validated_item_ids(item_ids)
    mapping = {movie_id: index for index, movie_id in enumerate(catalog)}
    item_factors = _model_item_factors(model, len(catalog))
    row = build_fold_in_confidence_row(profile, mapping, len(catalog), config)
    user_factor = np.asarray(model.recalculate_user(0, row))
    expected_shape = (item_factors.shape[1],)
    if user_factor.shape != expected_shape:
        raise ValueError(f"recalculated user factor must have shape {expected_shape}")
    if user_factor.dtype != np.dtype("float32"):
        raise ValueError("recalculated user factor must use float32")
    if not np.isfinite(user_factor).all():
        raise ValueError("recalculated user factor must be finite")
    excluded_indices = {mapping[movie_id] for movie_id in profile}
    try:
        ranked = rank_factor_items(
            user_factor=user_factor,
            item_factors=item_factors,
            item_ids=catalog,
            excluded_item_indices=excluded_indices,
            k=config.recommendation_k,
        )
    except InsufficientCatalogItems as error:
        raise FoldInError(
            "complete fit catalog cannot provide the requested recommendation count"
        ) from error
    recommendations = tuple(
        FoldInRecommendation(
            movie_id=row.movie_id,
            score=row.score,
            rank=row.rank,
        )
        for row in ranked
    )
    if tuple(positive_movie_ids) != supplied_profile_snapshot:
        raise RuntimeError("positive_movie_ids was mutated")
    immutable_factor = np.array(user_factor, dtype=np.float32, copy=True)
    immutable_factor.setflags(write=False)
    return FoldInResult(
        strategy="als_fold_in",
        fold_in_eligible=True,
        profile_movie_ids=profile,
        user_factor=immutable_factor,
        recommendations=recommendations,
        profile_csr_shape=row.shape,
        profile_csr_nnz=row.nnz,
    )
