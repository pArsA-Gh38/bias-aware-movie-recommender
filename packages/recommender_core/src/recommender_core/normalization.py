"""Public candidate-local normalization for fixed reranking features.

Callers supply already-computed relevance and fit-popularity novelty vectors.
This module has no data, artifact, target-label, candidate-generation, or fitted
scaler access.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class NormalizedRerankingFeatures:
    """Independently normalized features in authoritative candidate order."""

    relevance: np.ndarray
    novelty: np.ndarray


def min_max_normalize_candidate_feature(values: ArrayLike) -> np.ndarray:
    """Normalize one user's candidate feature vector to ``[0, 1]``.

    Bounds are local to this one-dimensional vector. Empty input returns an
    empty float64 vector, while every finite constant vector (including a
    singleton) returns exact zeros. Non-finite values and non-vector inputs are
    rejected. The returned array owns independent memory and input order is
    preserved.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("candidate feature values must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("candidate feature values must all be finite")

    if array.size == 0:
        return np.empty(0, dtype=np.float64)

    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum == minimum:
        return np.zeros(array.shape, dtype=np.float64)

    span = maximum - minimum
    if np.isfinite(span):
        normalized = (array - minimum) / span
    else:
        # Opposite-sign finite float64 extremes can overflow max - min. Scaling
        # both bounds first is algebraically equivalent and keeps the result
        # finite without changing the exact-equality constant policy.
        scale = max(abs(minimum), abs(maximum))
        scaled = array / scale
        scaled_minimum = minimum / scale
        scaled_maximum = maximum / scale
        normalized = (scaled - scaled_minimum) / (scaled_maximum - scaled_minimum)

    return np.array(normalized, dtype=np.float64, copy=True)


def normalize_reranking_features(
    relevance: ArrayLike,
    novelty: ArrayLike,
) -> NormalizedRerankingFeatures:
    """Normalize aligned relevance and novelty vectors independently."""

    normalized_relevance = min_max_normalize_candidate_feature(relevance)
    normalized_novelty = min_max_normalize_candidate_feature(novelty)
    if normalized_relevance.shape != normalized_novelty.shape:
        raise ValueError("relevance and novelty must have equal lengths")
    return NormalizedRerankingFeatures(
        relevance=normalized_relevance,
        novelty=normalized_novelty,
    )
