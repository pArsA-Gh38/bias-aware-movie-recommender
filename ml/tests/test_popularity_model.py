from collections import OrderedDict

import numpy as np
import pytest

from src.models.popularity import (
    build_popularity_ranking,
    recommend_popular_items,
)


def test_known_global_ranking_fixture() -> None:
    counts = {40: 1, 30: 3, 20: 5, 10: 5}
    assert build_popularity_ranking(counts) == (10, 20, 30, 40)


def test_equal_count_tie_uses_movie_id_ascending() -> None:
    assert build_popularity_ranking({9: 2, 2: 2, 5: 2}) == (2, 5, 9)


@pytest.mark.parametrize(
    "counts",
    [
        OrderedDict([(40, 1), (30, 3), (20, 5), (10, 5)]),
        OrderedDict([(10, 5), (40, 1), (20, 5), (30, 3)]),
        OrderedDict([(30, 3), (10, 5), (40, 1), (20, 5)]),
    ],
)
def test_mapping_insertion_order_does_not_change_ranking(counts) -> None:
    assert build_popularity_ranking(counts) == (10, 20, 30, 40)


def test_all_catalog_items_appear_exactly_once() -> None:
    counts = {8: 3, 4: 1, 6: 2}
    ranking = build_popularity_ranking(counts)
    assert len(ranking) == len(counts)
    assert set(ranking) == set(counts)


def test_empty_count_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_popularity_ranking({})


@pytest.mark.parametrize("count", [0, -1])
def test_non_positive_count_is_rejected(count) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        build_popularity_ranking({1: count})


@pytest.mark.parametrize("count", [1.0, "1", True, None])
def test_malformed_count_is_rejected(count) -> None:
    with pytest.raises(TypeError, match="positive integer"):
        build_popularity_ranking({1: count})


@pytest.mark.parametrize("item_id", [-1, 1.0, "1", True, None])
def test_invalid_mapping_key_is_rejected(item_id) -> None:
    error = ValueError if item_id == -1 else TypeError
    with pytest.raises(error, match="item IDs"):
        build_popularity_ranking({item_id: 1})


def test_numpy_integer_ids_and_counts_are_accepted() -> None:
    ranking = build_popularity_ranking(
        {np.int32(2): np.int64(1), np.int64(1): np.int32(2)}
    )
    assert ranking == (np.int64(1), np.int32(2))


def test_count_mapping_is_not_mutated() -> None:
    counts = OrderedDict([(2, 1), (1, 2)])
    before = list(counts.items())
    build_popularity_ranking(counts)
    assert list(counts.items()) == before


def test_ranking_repeated_calls_are_exactly_deterministic() -> None:
    counts = {3: 2, 1: 2, 2: 4}
    assert build_popularity_ranking(counts) == build_popularity_ranking(counts)


def test_seen_filtering_fixture() -> None:
    assert recommend_popular_items([10, 20, 30, 40], {10, 30}, 2) == (
        20,
        40,
    )


def test_filtering_occurs_before_truncation() -> None:
    assert recommend_popular_items([1, 2, 3, 4, 5], {1, 2}, 2) == (3, 4)


def test_all_seen_returns_empty() -> None:
    assert recommend_popular_items([1, 2, 3], {1, 2, 3}, 10) == ()


def test_k_larger_than_remaining_unseen_catalog_returns_short_list() -> None:
    assert recommend_popular_items([1, 2, 3], {1}, 10) == (2, 3)


def test_valid_out_of_catalog_seen_id_is_harmless() -> None:
    assert recommend_popular_items([1, 2, 3], {999}, 2) == (1, 2)


@pytest.mark.parametrize("item_id", [-1, 1.5, "1", True, None])
def test_invalid_seen_id_is_rejected(item_id) -> None:
    error = ValueError if item_id == -1 else TypeError
    with pytest.raises(error, match="item IDs"):
        recommend_popular_items([1, 2], {item_id}, 1)


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_is_rejected(k) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        recommend_popular_items([1], set(), k)


@pytest.mark.parametrize("k", [True, 10.0, "10", None])
def test_malformed_k_is_rejected(k) -> None:
    with pytest.raises(TypeError, match="integer scalar"):
        recommend_popular_items([1], set(), k)


def test_recommendation_order_is_preserved() -> None:
    assert recommend_popular_items([8, 2, 9, 4], {2}, 3) == (8, 9, 4)


def test_recommendations_do_not_contain_duplicates() -> None:
    result = recommend_popular_items([4, 3, 2, 1], {3}, 4)
    assert len(result) == len(set(result))


def test_duplicate_popularity_ranking_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        recommend_popular_items([1, 2, 1], set(), 2)


@pytest.mark.parametrize("item_id", [-1, 1.5, "1", True, None])
def test_malformed_ranking_id_is_rejected(item_id) -> None:
    error = ValueError if item_id == -1 else TypeError
    with pytest.raises(error, match="item IDs"):
        recommend_popular_items([item_id], set(), 1)


def test_duplicate_seen_sequence_is_rejected_consistently() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        recommend_popular_items([1, 2, 3], [9, 9], 2)


def test_recommendation_inputs_are_not_mutated() -> None:
    ranking = [1, 2, 3, 4]
    seen = [1, 3]
    ranking_before = ranking.copy()
    seen_before = seen.copy()
    recommend_popular_items(ranking, seen, 2)
    assert ranking == ranking_before
    assert seen == seen_before


def test_same_user_input_always_produces_identical_output() -> None:
    ranking = [5, 4, 3, 2, 1]
    seen = {4, 2}
    assert recommend_popular_items(ranking, seen, 3) == (
        recommend_popular_items(ranking, seen, 3)
    )
