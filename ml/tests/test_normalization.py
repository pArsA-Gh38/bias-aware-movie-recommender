from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from src.reranking.normalization import (
    min_max_normalize_candidate_feature,
    normalize_reranking_features,
)


def test_known_min_max_fixture() -> None:
    actual = min_max_normalize_candidate_feature([2.0, 4.0, 6.0])
    np.testing.assert_array_equal(actual, [0.0, 0.5, 1.0])


def test_negative_and_mixed_scores_follow_reference_formula() -> None:
    values = np.array([-3.0, -1.0, 5.0])
    expected = (values - (-3.0)) / (5.0 - (-3.0))
    np.testing.assert_array_equal(min_max_normalize_candidate_feature(values), expected)


def test_constant_vector_maps_exactly_to_zeros() -> None:
    actual = min_max_normalize_candidate_feature([7.0, 7.0, 7.0])
    np.testing.assert_array_equal(actual, np.zeros(3, dtype=np.float64))


def test_singleton_maps_to_zero() -> None:
    np.testing.assert_array_equal(
        min_max_normalize_candidate_feature([3.5]),
        np.array([0.0], dtype=np.float64),
    )


def test_empty_input_returns_empty_floating_vector() -> None:
    actual = min_max_normalize_candidate_feature([])
    assert actual.shape == (0,)
    assert actual.dtype == np.float64


def test_repeated_nonconstant_values() -> None:
    actual = min_max_normalize_candidate_feature([2.0, 2.0, 4.0, 6.0])
    np.testing.assert_array_equal(actual, [0.0, 0.0, 0.5, 1.0])


def test_candidate_order_is_preserved() -> None:
    actual = min_max_normalize_candidate_feature([6.0, 2.0, 4.0])
    np.testing.assert_array_equal(actual, [1.0, 0.0, 0.5])


def test_paired_features_preserve_candidate_alignment() -> None:
    candidate_ids = np.array([101, 202, 303])
    actual = normalize_reranking_features(
        relevance=[10.0, 20.0, 30.0],
        novelty=[90.0, 10.0, 50.0],
    )
    np.testing.assert_array_equal(candidate_ids, [101, 202, 303])
    np.testing.assert_array_equal(actual.relevance, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(actual.novelty, [1.0, 0.0, 0.5])
    assert actual.relevance.shape == actual.novelty.shape == candidate_ids.shape


def test_paired_features_use_independent_bounds() -> None:
    actual = normalize_reranking_features(
        relevance=[-1000.0, 0.0, 1000.0],
        novelty=[8.0, 9.0, 10.0],
    )
    np.testing.assert_array_equal(actual.relevance, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(actual.novelty, [0.0, 0.5, 1.0])


@pytest.mark.parametrize(
    ("relevance", "novelty", "expected_relevance", "expected_novelty"),
    [
        ([4.0, 4.0, 4.0], [2.0, 4.0, 6.0], [0.0] * 3, [0.0, 0.5, 1.0]),
        ([2.0, 4.0, 6.0], [4.0, 4.0, 4.0], [0.0, 0.5, 1.0], [0.0] * 3),
    ],
)
def test_only_constant_paired_feature_maps_to_zeros(
    relevance: list[float],
    novelty: list[float],
    expected_relevance: list[float],
    expected_novelty: list[float],
) -> None:
    actual = normalize_reranking_features(relevance, novelty)
    np.testing.assert_array_equal(actual.relevance, expected_relevance)
    np.testing.assert_array_equal(actual.novelty, expected_novelty)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_are_rejected(invalid: float) -> None:
    with pytest.raises(ValueError, match="must all be finite"):
        min_max_normalize_candidate_feature([0.0, invalid, 1.0])


@pytest.mark.parametrize("invalid", [3.5, [[1.0, 2.0], [3.0, 4.0]]])
def test_non_vector_inputs_are_rejected(invalid: object) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        min_max_normalize_candidate_feature(invalid)  # type: ignore[arg-type]


def test_paired_feature_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        normalize_reranking_features([1.0, 2.0], [1.0])


def test_input_arrays_are_byte_identical_after_normalization() -> None:
    values = np.array([-3.0, -1.0, 5.0], dtype=np.float64)
    before = values.tobytes()
    min_max_normalize_candidate_feature(values)
    assert values.tobytes() == before


def test_output_does_not_share_writable_memory_with_input() -> None:
    values = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    actual = min_max_normalize_candidate_feature(values)
    assert not np.shares_memory(values, actual)
    actual[0] = 0.75
    np.testing.assert_array_equal(values, [2.0, 4.0, 6.0])


def test_repeated_calls_are_byte_identical() -> None:
    values = np.array([-3.0, -1.0, 5.0])
    first = min_max_normalize_candidate_feature(values)
    second = min_max_normalize_candidate_feature(values)
    assert first.dtype == second.dtype == np.float64
    assert first.tobytes() == second.tobytes()


def test_calls_are_stateless_across_different_candidate_pools() -> None:
    first_before = min_max_normalize_candidate_feature([2.0, 4.0, 6.0])
    min_max_normalize_candidate_feature([-1e300, 0.0, 1e300])
    first_after = min_max_normalize_candidate_feature([2.0, 4.0, 6.0])
    np.testing.assert_array_equal(first_after, first_before)


def test_two_users_candidate_bounds_are_independent() -> None:
    first_user = min_max_normalize_candidate_feature([10.0, 20.0, 30.0])
    second_user = min_max_normalize_candidate_feature([-1e200, 0.0, 1e200])
    np.testing.assert_array_equal(first_user, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(second_user, [0.0, 0.5, 1.0])


def test_near_constant_unequal_values_are_not_erased() -> None:
    lower = 1.0
    upper = np.nextafter(lower, 2.0)
    assert upper != lower
    np.testing.assert_array_equal(
        min_max_normalize_candidate_feature([lower, upper]),
        [0.0, 1.0],
    )


def test_opposite_sign_float64_extremes_remain_finite_and_bounded() -> None:
    maximum = np.finfo(np.float64).max
    actual = min_max_normalize_candidate_feature([-maximum, 0.0, maximum])
    np.testing.assert_array_equal(actual, [0.0, 0.5, 1.0])
    assert np.isfinite(actual).all()


def test_integer_input_returns_float64() -> None:
    actual = min_max_normalize_candidate_feature([2, 4, 6])
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, [0.0, 0.5, 1.0])


def test_paired_result_container_is_frozen() -> None:
    actual = normalize_reranking_features([1.0, 2.0], [4.0, 3.0])
    with pytest.raises(FrozenInstanceError):
        actual.relevance = np.zeros(2)  # type: ignore[misc]


def test_normalization_module_has_only_pure_numeric_imports() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "../packages/recommender_core/src/recommender_core/normalization.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported_roots == {"__future__", "dataclasses", "numpy"}
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "open" not in called_names
