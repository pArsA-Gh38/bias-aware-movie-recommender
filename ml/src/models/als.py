"""Minimal deterministic CPU-only implicit ALS model helpers for MODEL-004."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, isspmatrix_csr
from threadpoolctl import threadpool_limits


def _integer(value: Integral, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer scalar")
    result = int(value)
    if (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _positive_real(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _ids(values: Sequence[Integral], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be an ordered sequence")
    result = tuple(_integer(value, name=f"{name} ID") for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicate IDs")
    return result


@dataclass(frozen=True)
class ALSConfig:
    """One validated CPU configuration; MODEL-004 does not tune it."""

    factors: int
    regularization: float
    iterations: int
    confidence_alpha: float
    implicit_library_alpha: float
    random_state: int
    dtype: str
    num_threads: int
    use_native: bool = True
    use_cg: bool = True
    calculate_training_loss: bool = False
    use_gpu: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "factors", _integer(self.factors, name="factors", positive=True)
        )
        object.__setattr__(
            self,
            "regularization",
            _positive_real(self.regularization, name="regularization"),
        )
        object.__setattr__(
            self,
            "iterations",
            _integer(self.iterations, name="iterations", positive=True),
        )
        object.__setattr__(
            self,
            "confidence_alpha",
            _positive_real(self.confidence_alpha, name="confidence_alpha"),
        )
        object.__setattr__(
            self,
            "implicit_library_alpha",
            _positive_real(
                self.implicit_library_alpha,
                name="implicit_library_alpha",
            ),
        )
        if self.implicit_library_alpha != 1.0:
            raise ValueError(
                "implicit_library_alpha must equal 1.0 because confidence is "
                "derived before fit"
            )
        object.__setattr__(
            self,
            "random_state",
            _integer(self.random_state, name="random_state"),
        )
        if self.dtype != "float32":
            raise ValueError("dtype must equal float32 for MODEL-004")
        if (
            not isinstance(self.num_threads, Integral)
            or isinstance(self.num_threads, bool)
            or int(self.num_threads) != 1
        ):
            raise ValueError("num_threads must equal 1 for MODEL-004")
        object.__setattr__(self, "num_threads", 1)
        for name in (
            "use_native",
            "use_cg",
            "calculate_training_loss",
            "use_gpu",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if not self.use_native or not self.use_cg:
            raise ValueError("use_native and use_cg must be true for MODEL-004")
        if self.calculate_training_loss:
            raise ValueError("calculate_training_loss must be false for MODEL-004")
        if self.use_gpu:
            raise ValueError("GPU is forbidden for MODEL-004")


@dataclass(frozen=True)
class ALSModel:
    """External ID mappings and fitted factors aligned by row."""

    user_ids: tuple[int, ...]
    item_ids: tuple[int, ...]
    user_factors: np.ndarray
    item_factors: np.ndarray
    config: ALSConfig


def _validate_binary_matrix(
    user_item_matrix: csr_matrix,
    *,
    user_count: int,
    item_count: int,
) -> None:
    if not isspmatrix_csr(user_item_matrix):
        raise TypeError("user_item_matrix must be a SciPy CSR matrix")
    if user_item_matrix.shape != (user_count, item_count):
        raise ValueError("user_item_matrix shape must match user/item mappings")
    if user_item_matrix.nnz == 0:
        raise ValueError("user_item_matrix must not be empty")
    if not user_item_matrix.has_canonical_format:
        raise ValueError("user_item_matrix must use canonical CSR without duplicates")
    if not np.isfinite(user_item_matrix.data).all():
        raise ValueError("user_item_matrix values must be finite")
    if not np.equal(user_item_matrix.data, 1.0).all():
        raise ValueError(
            "user_item_matrix must contain binary value 1.0; confidence is derived once"
        )


def _validate_model(model: ALSModel) -> dict[int, int]:
    if not isinstance(model, ALSModel):
        raise TypeError("model must be an ALSModel")
    users = _ids(model.user_ids, name="model user_ids")
    items = _ids(model.item_ids, name="model item_ids")
    if not isinstance(model.config, ALSConfig):
        raise TypeError("model config must be an ALSConfig")
    expected_user_shape = (len(users), model.config.factors)
    expected_item_shape = (len(items), model.config.factors)
    if model.user_factors.shape != expected_user_shape:
        raise ValueError("user factor shape does not match mapping/config")
    if model.item_factors.shape != expected_item_shape:
        raise ValueError("item factor shape does not match mapping/config")
    expected_dtype = np.dtype(model.config.dtype)
    if model.user_factors.dtype != expected_dtype:
        raise ValueError("user factor dtype does not match config")
    if model.item_factors.dtype != expected_dtype:
        raise ValueError("item factor dtype does not match config")
    if not np.isfinite(model.user_factors).all():
        raise ValueError("user factors must be finite")
    if not np.isfinite(model.item_factors).all():
        raise ValueError("item factors must be finite")
    return {user_id: index for index, user_id in enumerate(users)}


def fit_als(
    user_item_matrix: csr_matrix,
    user_ids: Sequence[Integral],
    item_ids: Sequence[Integral],
    config: ALSConfig,
    *,
    als_factory: Callable[..., Any] = AlternatingLeastSquares,
) -> ALSModel:
    """Fit exactly one CPU ALS model on a binary user-item CSR matrix."""

    users = _ids(user_ids, name="user_ids")
    items = _ids(item_ids, name="item_ids")
    if not isinstance(config, ALSConfig):
        raise TypeError("config must be an ALSConfig")
    _validate_binary_matrix(
        user_item_matrix,
        user_count=len(users),
        item_count=len(items),
    )
    confidence_matrix = user_item_matrix.astype(np.float32, copy=True)
    confidence_matrix.data = 1.0 + config.confidence_alpha * confidence_matrix.data
    confidence_matrix.sort_indices()
    with threadpool_limits(limits=1, user_api="blas"):
        estimator = als_factory(
            factors=config.factors,
            regularization=config.regularization,
            alpha=config.implicit_library_alpha,
            dtype=np.dtype(config.dtype).type,
            use_native=config.use_native,
            use_cg=config.use_cg,
            iterations=config.iterations,
            calculate_training_loss=config.calculate_training_loss,
            num_threads=config.num_threads,
            random_state=config.random_state,
        )
        estimator.fit(confidence_matrix, show_progress=False)
    user_factors = np.asarray(estimator.user_factors)
    item_factors = np.asarray(estimator.item_factors)
    model = ALSModel(
        user_ids=users,
        item_ids=items,
        user_factors=user_factors.copy(),
        item_factors=item_factors.copy(),
        config=config,
    )
    _validate_model(model)
    return model


def score_als_candidates(model: ALSModel, user_id: Integral) -> np.ndarray:
    """Return exact full-catalog dot-product scores in item mapping order."""

    user_indices = _validate_model(model)
    validated_user = _integer(user_id, name="user_id")
    if validated_user not in user_indices:
        raise ValueError(f"unknown fitted user ID: {validated_user}")
    user_factor = model.user_factors[user_indices[validated_user]]
    return np.asarray(model.item_factors @ user_factor, dtype=model.item_factors.dtype)


def recommend_als(
    model: ALSModel,
    user_id: Integral,
    previously_rated_items: Collection[Integral],
    k: int,
) -> tuple[tuple[int, float], ...]:
    """Rank all unseen finite scores by score descending then external item ID."""

    validated_k = _integer(k, name="k", positive=True)
    if isinstance(previously_rated_items, (str, bytes)) or not isinstance(
        previously_rated_items, Collection
    ):
        raise TypeError("previously_rated_items must be a collection")
    seen: set[int] = set()
    for value in previously_rated_items:
        item_id = _integer(value, name="previously_rated_items ID")
        if item_id in seen:
            raise ValueError("previously_rated_items must not contain duplicates")
        seen.add(item_id)

    scores = score_als_candidates(model, user_id)
    candidates = [
        (item_id, float(scores[index]))
        for index, item_id in enumerate(model.item_ids)
        if item_id not in seen and np.isfinite(scores[index])
    ]
    candidates.sort(key=lambda pair: (-pair[1], pair[0]))
    return tuple(candidates[:validated_k])
