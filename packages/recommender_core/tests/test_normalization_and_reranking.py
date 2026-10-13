from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from recommender_core import (
    greedy_rerank,
    min_max_normalize_candidate_feature,
    normalize_reranking_features,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], []),
        ([3.5], [0.0]),
        ([7.0, 7.0, 7.0], [0.0, 0.0, 0.0]),
        ([-3.0, -1.0, 5.0], [0.0, 0.25, 1.0]),
    ],
)
def test_candidate_local_normalization_edges(
    values: list[float], expected: list[float]
) -> None:
    actual = min_max_normalize_candidate_feature(values)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


def test_opposite_float64_extremes_remain_finite() -> None:
    maximum = np.finfo(np.float64).max
    np.testing.assert_array_equal(
        min_max_normalize_candidate_feature([-maximum, 0.0, maximum]),
        [0.0, 0.5, 1.0],
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_normalization_rejects_nonfinite_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        min_max_normalize_candidate_feature([0.0, invalid])


def test_paired_normalization_is_independent_and_frozen() -> None:
    result = normalize_reranking_features([10.0, 20.0, 30.0], [90.0, 10.0, 50.0])
    np.testing.assert_array_equal(result.relevance, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(result.novelty, [1.0, 0.0, 0.5])
    with pytest.raises(FrozenInstanceError):
        result.relevance = np.zeros(3)  # type: ignore[misc]


def _rerank(*, lambda_: float = 0.5, k: int = 3):
    return greedy_rerank(
        [10, 20, 30],
        [30.0, 20.0, 10.0],
        [1.0, 3.0, 2.0],
        {10: ["Action"], 20: ["Action"], 30: ["Drama"]},
        k=k,
        lambda_=lambda_,
        beta=0.5,
    )


def test_greedy_score_components_and_jaccard_diversity_are_exact() -> None:
    rows = _rerank()
    assert [row.movie_id for row in rows] == [10, 20, 30]
    assert [
        (
            row.normalized_relevance,
            row.normalized_novelty,
            row.genre_diversity,
            row.beyond_score,
            row.final_score,
        )
        for row in rows
    ] == [
        (1.0, 0.0, 1.0, 0.5, 0.75),
        (0.5, 1.0, 0.0, 0.5, 0.5),
        (0.0, 0.5, 1.0, 0.75, 0.375),
    ]


def test_lambda_one_reproduces_authoritative_base_prefix() -> None:
    rows = greedy_rerank(
        [30, 20, 10],
        [5.0, 5.0, 4.0],
        [0.0, 100.0, 50.0],
        {30: ["A"], 20: ["B"], 10: ["C"]},
        k=3,
        lambda_=1.0,
        beta=0.5,
    )
    assert [row.movie_id for row in rows] == [30, 20, 10]
    assert [row.base_rank for row in rows] == [1, 2, 3]


def test_short_pool_is_unique_immutable_and_deterministic() -> None:
    first = _rerank(k=10)
    second = _rerank(k=10)
    assert first == second
    assert len(first) == len({row.movie_id for row in first}) == 3
    with pytest.raises(FrozenInstanceError):
        first[0].rank = 2  # type: ignore[misc]
