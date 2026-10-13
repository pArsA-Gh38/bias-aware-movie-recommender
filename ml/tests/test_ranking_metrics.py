import math

import numpy as np
import pytest

from src.evaluation.ranking_metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

TOLERANCE = {"rel": 1e-12, "abs": 1e-12}


def _metrics(
    recommendations: object,
    relevant_items: object,
    k: object,
) -> tuple[float, float, float, float]:
    return (
        precision_at_k(recommendations, relevant_items, k),  # type: ignore[arg-type]
        recall_at_k(recommendations, relevant_items, k),  # type: ignore[arg-type]
        hit_at_k(recommendations, relevant_items, k),  # type: ignore[arg-type]
        ndcg_at_k(recommendations, relevant_items, k),  # type: ignore[arg-type]
    )


def _assert_metrics(
    actual: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
) -> None:
    assert actual == pytest.approx(expected, **TOLERANCE)


def test_manual_fixture_two_hits_at_ranks_one_and_three() -> None:
    actual = _metrics([10, 20, 30, 40], {10, 30}, 4)

    _assert_metrics(
        actual,
        (0.5, 1.0, 1.0, 0.9197207891481876),
    )


def test_short_list_keeps_k_as_precision_denominator() -> None:
    actual = _metrics([99, 10], {10}, 10)

    _assert_metrics(
        actual,
        (0.1, 1.0, 1.0, 0.6309297535714575),
    )


def test_idcg_is_independent_of_actual_recommendation_length() -> None:
    actual = _metrics([10], {10, 20}, 10)

    _assert_metrics(
        actual,
        (0.1, 0.5, 1.0, 0.6131471927654584),
    )


def test_manual_fixture_one_hit_at_rank_ten() -> None:
    actual = _metrics(list(range(1, 11)), {10}, 10)

    _assert_metrics(
        actual,
        (0.1, 1.0, 1.0, 0.2890648263178879),
    )


def test_perfect_ranking() -> None:
    actual = _metrics([10, 20], {10, 20}, 2)

    _assert_metrics(actual, (1.0, 1.0, 1.0, 1.0))


def test_no_hit_ranking() -> None:
    actual = _metrics([10, 20, 30], {40, 50}, 3)

    _assert_metrics(actual, (0.0, 0.0, 0.0, 0.0))


def test_one_hit_at_rank_one() -> None:
    actual = _metrics([10, 20, 30], {10}, 3)

    _assert_metrics(actual, (1 / 3, 1.0, 1.0, 1.0))


def test_relevant_item_below_k_is_ignored() -> None:
    actual = _metrics([1, 2, 3, 4, 5, 6], {6}, 5)

    _assert_metrics(actual, (0.0, 0.0, 0.0, 0.0))


def test_long_list_uses_only_first_k_ranks() -> None:
    actual = _metrics([10, 20, 30, 40, 50], {10, 40, 50}, 3)

    _assert_metrics(actual, (1 / 3, 1 / 3, 1.0, 0.46927872602275644))


def test_ndcg_is_rank_sensitive() -> None:
    rank_one = _metrics([10, 20, 30], {10}, 3)
    rank_three = _metrics([20, 30, 10], {10}, 3)

    assert rank_one[:3] == rank_three[:3]
    assert rank_one[3] == pytest.approx(1.0, **TOLERANCE)
    assert rank_three[3] == pytest.approx(0.5, **TOLERANCE)
    assert rank_one[3] > rank_three[3]


@pytest.mark.parametrize(
    ("recommendations", "relevant_items", "k"),
    [
        ([1, 2, 3], {1, 2}, 3),
        ([3, 2, 1], {1, 2}, 2),
        ([], {1}, 10),
        ([1], set(), 10),
        ([], set(), 10),
    ],
)
def test_all_outputs_are_finite_and_bounded(
    recommendations: list[int],
    relevant_items: set[int],
    k: int,
) -> None:
    values = _metrics(recommendations, relevant_items, k)

    assert all(math.isfinite(value) for value in values)
    assert all(0.0 <= value <= 1.0 for value in values)


@pytest.mark.parametrize(
    ("recommendations", "relevant_items"),
    [
        ([], {10}),
        ([10], set()),
        ([], set()),
    ],
)
def test_empty_inputs_return_zero_metrics(
    recommendations: list[int],
    relevant_items: set[int],
) -> None:
    _assert_metrics(
        _metrics(recommendations, relevant_items, 10),
        (0.0, 0.0, 0.0, 0.0),
    )


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_non_positive_k_is_rejected(invalid_k: int) -> None:
    for metric in (precision_at_k, recall_at_k, hit_at_k, ndcg_at_k):
        with pytest.raises(ValueError, match="greater than zero"):
            metric([1], {1}, invalid_k)


@pytest.mark.parametrize("invalid_k", [True, 10.0, "10"])
def test_non_integral_k_is_rejected(invalid_k: object) -> None:
    for metric in (precision_at_k, recall_at_k, hit_at_k, ndcg_at_k):
        with pytest.raises(TypeError, match="integer scalar"):
            metric([1], {1}, invalid_k)  # type: ignore[arg-type]


def test_duplicate_recommendation_inside_top_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="recommendations.*duplicate"):
        precision_at_k([10, 20, 10], {10}, 3)


def test_duplicate_recommendation_below_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="recommendations.*duplicate"):
        ndcg_at_k([1, 2, 3, 4, 5, 1], {1}, 5)


def test_duplicate_relevant_sequence_is_rejected() -> None:
    with pytest.raises(ValueError, match="relevant_items.*duplicate"):
        recall_at_k([10, 20], [10, 10], 2)


@pytest.mark.parametrize(
    ("invalid_id", "error_type", "message"),
    [
        (-1, ValueError, "non-negative"),
        (1.5, TypeError, "integer scalar"),
        ("1", TypeError, "integer scalar"),
        (True, TypeError, "integer scalar"),
        (None, TypeError, "integer scalar"),
        ([1], TypeError, "integer scalar"),
    ],
)
def test_invalid_recommendation_item_ids_are_rejected(
    invalid_id: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        precision_at_k([invalid_id], {1}, 1)  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("invalid_id", "error_type", "message"),
    [
        (-1, ValueError, "non-negative"),
        (1.5, TypeError, "integer scalar"),
        ("1", TypeError, "integer scalar"),
        (True, TypeError, "integer scalar"),
        (None, TypeError, "integer scalar"),
        ([1], TypeError, "integer scalar"),
    ],
)
def test_invalid_relevant_item_ids_are_rejected(
    invalid_id: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        recall_at_k([1], [invalid_id], 1)  # type: ignore[list-item]


def test_numpy_integral_scalars_are_accepted() -> None:
    values = _metrics(
        [np.int32(10), np.int64(20)],
        {np.int64(10)},
        np.int64(2),
    )

    _assert_metrics(values, (0.5, 1.0, 1.0, 1.0))


def test_recommendation_input_must_be_an_ordered_sequence() -> None:
    with pytest.raises(TypeError, match="ordered sequence"):
        precision_at_k({1, 2}, {1}, 2)  # type: ignore[arg-type]


def test_relevant_input_must_be_a_collection() -> None:
    with pytest.raises(TypeError, match="collection"):
        precision_at_k([1, 2], None, 2)  # type: ignore[arg-type]


def test_repeated_calls_are_exactly_deterministic() -> None:
    recommendations = [9, 10, 11, 12]
    relevant_items = {10, 12}

    first = _metrics(recommendations, relevant_items, 4)
    for _ in range(20):
        assert _metrics(recommendations, relevant_items, 4) == first


def test_inputs_are_not_mutated() -> None:
    recommendations = [9, 10, 11, 12]
    relevant_items = [10, 12]
    original_recommendations = recommendations.copy()
    original_relevant_items = relevant_items.copy()

    _metrics(recommendations, relevant_items, 4)

    assert recommendations == original_recommendations
    assert relevant_items == original_relevant_items


def test_authoritative_recommendation_order_is_preserved() -> None:
    no_hit = _metrics([20, 10], {10}, 1)
    hit = _metrics([10, 20], {10}, 1)

    _assert_metrics(no_hit, (0.0, 0.0, 0.0, 0.0))
    _assert_metrics(hit, (1.0, 1.0, 1.0, 1.0))


def test_moving_same_hit_from_rank_one_to_rank_ten_only_lowers_ndcg() -> None:
    rank_one = _metrics([10, 1, 2, 3, 4, 5, 6, 7, 8, 9], {10}, 10)
    rank_ten = _metrics([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], {10}, 10)

    assert rank_one[:3] == rank_ten[:3] == (0.1, 1.0, 1.0)
    assert rank_one[3] > rank_ten[3]


def test_adding_non_relevant_item_below_k_leaves_metrics_unchanged() -> None:
    base = list(range(1, 11))

    assert _metrics(base, {1, 5}, 10) == _metrics(
        [*base, 11],
        {1, 5},
        10,
    )


def test_adding_relevant_item_below_k_leaves_metrics_unchanged() -> None:
    base = list(range(1, 11))

    assert _metrics(base, {1, 11}, 10) == _metrics(
        [*base, 11],
        {1, 11},
        10,
    )


def test_adding_hit_inside_top_k_does_not_decrease_metrics() -> None:
    before = _metrics([1, 2, 3, 4], {1, 9}, 4)
    after = _metrics([1, 2, 9, 4], {1, 9}, 4)

    assert all(new >= old for old, new in zip(before, after, strict=True))
