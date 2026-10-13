"""Pure deterministic user-bootstrap statistics for evaluation reporting.

This module has no repository paths, model access, target data, recommendation
generation, or publication behavior.  Callers materialize one index matrix and
reuse it for both members of every paired comparison.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class MeanInterval:
    observed_mean: float
    sample_std: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class DeltaInterval:
    observed_delta: float
    ci_low: float
    ci_high: float


def _positive_integer(value: Integral, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer scalar, not a boolean")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _seed(value: Integral) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("seed must be an integer scalar, not a boolean")
    result = int(value)
    if result < 0:
        raise ValueError("seed must be non-negative")
    return result


def _confidence_level(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("confidence_level must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("confidence_level must be finite and inside (0, 1)")
    return result


def metric_vector(values: ArrayLike, *, name: str = "values") -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True)


def generate_bootstrap_indices(
    n_observations: Integral,
    n_resamples: Integral,
    seed: Integral,
) -> NDArray[np.int64]:
    """Return C-contiguous little-endian int64 user-resampling indices."""

    size = _positive_integer(n_observations, name="n_observations")
    resamples = _positive_integer(n_resamples, name="n_resamples")
    generator = np.random.default_rng(_seed(seed))
    indices = generator.integers(
        0,
        size,
        size=(resamples, size),
        dtype=np.int64,
    )
    return np.asarray(indices, dtype="<i8", order="C")


def validate_bootstrap_indices(
    indices: ArrayLike,
    *,
    n_observations: Integral,
    n_resamples: Integral | None = None,
) -> NDArray[np.int64]:
    size = _positive_integer(n_observations, name="n_observations")
    array = np.asarray(indices)
    if array.ndim != 2:
        raise ValueError("bootstrap indices must be two-dimensional")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError("bootstrap indices must have an integer dtype")
    if array.shape[1] != size:
        raise ValueError("bootstrap index sample width changed")
    if n_resamples is not None and array.shape[0] != _positive_integer(
        n_resamples,
        name="n_resamples",
    ):
        raise ValueError("bootstrap resample count changed")
    if array.size == 0 or np.any(array < 0) or np.any(array >= size):
        raise ValueError("bootstrap index is out of range")
    return np.asarray(array, dtype="<i8", order="C")


def bootstrap_index_sha256(indices: ArrayLike) -> str:
    array = np.asarray(indices)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise TypeError("index hashing requires a two-dimensional integer matrix")
    canonical = np.asarray(array, dtype="<i8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def sample_mean_std(values: ArrayLike, *, ddof: Integral = 1) -> tuple[float, float]:
    array = metric_vector(values)
    degrees = _positive_integer(ddof, name="ddof")
    if array.size <= degrees:
        raise ValueError("sample standard deviation needs more observations than ddof")
    mean = math.fsum(float(value) for value in array) / array.size
    sample_std = float(np.std(array, ddof=degrees))
    if not math.isfinite(sample_std):
        raise ArithmeticError("sample standard deviation became non-finite")
    return mean, sample_std


def percentile_interval(
    distribution: ArrayLike,
    *,
    confidence_level: float = 0.95,
    method: str = "linear",
) -> tuple[float, float]:
    values = metric_vector(distribution, name="distribution")
    confidence = _confidence_level(confidence_level)
    if method != "linear":
        raise ValueError("quantile method must be linear")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha], method=method)
    result = (float(low), float(high))
    if not all(math.isfinite(value) for value in result) or result[0] > result[1]:
        raise ArithmeticError("bootstrap interval is invalid")
    return result


def bootstrap_mean_distribution(
    values: ArrayLike,
    indices: ArrayLike,
) -> NDArray[np.float64]:
    array = metric_vector(values)
    sampled = validate_bootstrap_indices(indices, n_observations=array.size)
    distribution = np.mean(array[sampled], axis=1, dtype=np.float64)
    if not np.isfinite(distribution).all():
        raise ArithmeticError("bootstrap mean distribution became non-finite")
    return distribution


def bootstrap_mean_ci(
    values: ArrayLike,
    indices: ArrayLike,
    *,
    confidence_level: float = 0.95,
    method: str = "linear",
    ddof: Integral = 1,
) -> MeanInterval:
    array = metric_vector(values)
    observed, sample_std = sample_mean_std(array, ddof=ddof)
    distribution = bootstrap_mean_distribution(array, indices)
    low, high = percentile_interval(
        distribution,
        confidence_level=confidence_level,
        method=method,
    )
    return MeanInterval(observed, sample_std, low, high)


def paired_mean_delta_ci(
    base_values: ArrayLike,
    comparison_values: ArrayLike,
    indices: ArrayLike,
    *,
    confidence_level: float = 0.95,
    method: str = "linear",
) -> DeltaInterval:
    base = metric_vector(base_values, name="base_values")
    comparison = metric_vector(comparison_values, name="comparison_values")
    if base.shape != comparison.shape:
        raise ValueError("paired metric vectors must have equal lengths")
    sampled = validate_bootstrap_indices(indices, n_observations=base.size)
    observed = (
        math.fsum(
            float(right - left) for left, right in zip(base, comparison, strict=True)
        )
        / base.size
    )
    distribution = np.mean(comparison[sampled], axis=1) - np.mean(base[sampled], axis=1)
    low, high = percentile_interval(
        distribution,
        confidence_level=confidence_level,
        method=method,
    )
    return DeltaInterval(observed, low, high)


def align_paired_by_user(
    base_user_ids: Sequence[Integral],
    base_values: ArrayLike,
    comparison_user_ids: Sequence[Integral],
    comparison_values: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate exact user equality and align both vectors by external ID."""

    base = metric_vector(base_values, name="base_values")
    comparison = metric_vector(comparison_values, name="comparison_values")
    left_ids = _validated_user_ids(base_user_ids, expected_length=base.size)
    right_ids = _validated_user_ids(
        comparison_user_ids,
        expected_length=comparison.size,
    )
    if set(left_ids) != set(right_ids):
        raise ValueError("paired configurations have different user ID sets")
    left = dict(zip(left_ids, base, strict=True))
    right = dict(zip(right_ids, comparison, strict=True))
    ordered = np.asarray(sorted(left), dtype=np.int64)
    return (
        ordered,
        np.asarray([left[int(user_id)] for user_id in ordered], dtype=np.float64),
        np.asarray([right[int(user_id)] for user_id in ordered], dtype=np.float64),
    )


def _validated_user_ids(
    values: Sequence[Integral],
    *,
    expected_length: int,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("user IDs must be an ordered sequence")
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("user IDs must be integer scalars")
        user_id = int(value)
        if user_id < 0:
            raise ValueError("user IDs must be non-negative")
        if user_id in seen:
            raise ValueError("user IDs must be unique")
        seen.add(user_id)
        result.append(user_id)
    if len(result) != expected_length:
        raise ValueError("user ID and metric-vector lengths differ")
    return tuple(result)


def recommendation_matrix(values: ArrayLike) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("recommendations must be a non-empty two-dimensional matrix")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError("recommendation IDs must have an integer dtype")
    result = np.asarray(array, dtype=np.int64, order="C")
    if np.any(result < 0):
        raise ValueError("recommendation IDs must be non-negative")
    if any(len({int(item) for item in row}) != result.shape[1] for row in result):
        raise ValueError("each recommendation list must contain unique items")
    return result


def catalog_coverage(recommendations: ArrayLike, *, catalog_size: Integral) -> float:
    matrix = recommendation_matrix(recommendations)
    denominator = _positive_integer(catalog_size, name="catalog_size")
    unique_count = np.unique(matrix).size
    if unique_count > denominator:
        raise ValueError("recommendations exceed the fixed catalog universe")
    return unique_count / denominator


def exposure_gini(
    recommendations: ArrayLike,
    catalog_item_ids: Sequence[Integral],
) -> float:
    matrix = recommendation_matrix(recommendations)
    catalog = _validated_catalog(catalog_item_ids)
    positions = {movie_id: index for index, movie_id in enumerate(catalog)}
    try:
        internal = np.fromiter(
            (positions[int(movie_id)] for movie_id in matrix.ravel(order="C")),
            dtype=np.int64,
            count=matrix.size,
        )
    except KeyError as error:
        raise ValueError("recommendation item is absent from fixed catalog") from error
    return _exposure_gini_internal(internal, catalog_size=len(catalog))


def _exposure_gini_internal(
    internal_item_ids: NDArray[np.int64],
    *,
    catalog_size: int,
) -> float:
    counts = np.bincount(internal_item_ids, minlength=catalog_size)
    total = int(counts.sum())
    if total == 0:
        return 0.0
    ordered = np.sort(counts)
    size = catalog_size
    numerator = sum(
        (2 * rank - size - 1) * int(exposure)
        for rank, exposure in enumerate(ordered, start=1)
    )
    return numerator / (size * total)


def _validated_catalog(values: Sequence[Integral]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("catalog_item_ids must be an ordered sequence")
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("catalog item IDs must be integer scalars")
        movie_id = int(value)
        if movie_id < 0 or movie_id in seen:
            raise ValueError("catalog item IDs must be unique and non-negative")
        result.append(movie_id)
        seen.add(movie_id)
    if not result:
        raise ValueError("catalog_item_ids must not be empty")
    return tuple(result)


def bootstrap_global_distributions(
    recommendations: ArrayLike,
    indices: ArrayLike,
    *,
    catalog_item_ids: Sequence[Integral],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    matrix = recommendation_matrix(recommendations)
    sampled = validate_bootstrap_indices(indices, n_observations=matrix.shape[0])
    catalog = _validated_catalog(catalog_item_ids)
    catalog_size = len(catalog)
    positions = {movie_id: index for index, movie_id in enumerate(catalog)}
    try:
        internal = np.fromiter(
            (positions[int(movie_id)] for movie_id in matrix.ravel(order="C")),
            dtype=np.int64,
            count=matrix.size,
        ).reshape(matrix.shape)
    except KeyError as error:
        raise ValueError("recommendation item is absent from fixed catalog") from error
    coverage = np.empty(sampled.shape[0], dtype=np.float64)
    gini = np.empty(sampled.shape[0], dtype=np.float64)
    for replicate, row_indices in enumerate(sampled):
        sample = internal[row_indices]
        coverage[replicate] = np.unique(sample).size / catalog_size
        gini[replicate] = _exposure_gini_internal(
            sample.ravel(order="C"),
            catalog_size=catalog_size,
        )
    if not np.isfinite(coverage).all() or not np.isfinite(gini).all():
        raise ArithmeticError("global bootstrap distribution became non-finite")
    return coverage, gini
