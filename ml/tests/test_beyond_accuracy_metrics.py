import ast
import math
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.beyond_accuracy_metrics import (
    catalog_coverage_at_k,
    intra_list_diversity_at_k,
    novelty_at_k,
)

TOLERANCE = {"rel": 1e-12, "abs": 1e-12}
SOURCE_PATH = (
    Path(__file__).parents[1] / "src" / "evaluation" / "beyond_accuracy_metrics.py"
)


def test_catalog_coverage_known_fixture_equals_four_fifths() -> None:
    catalog = [1, 2, 3, 4, 5]
    recommendation_lists = [[1, 2, 4], [2, 3, 4], [5]]

    actual = catalog_coverage_at_k(recommendation_lists, catalog, 2)

    assert actual == pytest.approx(0.8, **TOLERANCE)


def test_catalog_coverage_all_catalog_items_covered() -> None:
    actual = catalog_coverage_at_k([[1, 2], [3, 4]], [1, 2, 3, 4], 2)

    assert actual == pytest.approx(1.0, **TOLERANCE)


def test_catalog_coverage_no_recommendation_lists_returns_zero() -> None:
    assert catalog_coverage_at_k([], [1, 2, 3], 10) == 0.0


def test_catalog_coverage_all_recommendation_lists_empty_returns_zero() -> None:
    assert catalog_coverage_at_k([[], [], []], [1, 2, 3], 10) == 0.0


def test_catalog_coverage_k_smaller_than_list_length() -> None:
    actual = catalog_coverage_at_k([[1, 2, 3, 4]], [1, 2, 3, 4], 2)

    assert actual == pytest.approx(0.5, **TOLERANCE)


def test_catalog_coverage_k_larger_than_list_length() -> None:
    actual = catalog_coverage_at_k([[1, 2]], [1, 2, 3, 4], 10)

    assert actual == pytest.approx(0.5, **TOLERANCE)


def test_catalog_coverage_valid_item_below_k_is_ignored() -> None:
    without_tail = catalog_coverage_at_k([[1, 2]], [1, 2, 3], 2)
    with_tail = catalog_coverage_at_k([[1, 2, 3]], [1, 2, 3], 2)

    assert with_tail == without_tail == pytest.approx(2 / 3, **TOLERANCE)


def test_catalog_coverage_overlap_across_users_counted_once() -> None:
    actual = catalog_coverage_at_k([[1, 2], [2, 3], [2]], [1, 2, 3, 4], 2)

    assert actual == pytest.approx(0.75, **TOLERANCE)


def test_catalog_coverage_same_item_across_users_is_accepted() -> None:
    assert catalog_coverage_at_k([[1], [1], [1]], [1, 2], 1) == 0.5


@pytest.mark.parametrize(
    "recommendation_lists",
    [
        [[1, 1]],
        [[1, 2, 3, 1]],
    ],
)
def test_catalog_coverage_duplicate_inside_user_list_is_rejected(
    recommendation_lists: list[list[int]],
) -> None:
    with pytest.raises(ValueError, match="recommendations.*duplicate"):
        catalog_coverage_at_k(recommendation_lists, [1, 2, 3], 2)


def test_catalog_coverage_unknown_top_k_item_is_rejected() -> None:
    with pytest.raises(ValueError, match="absent from catalog_items"):
        catalog_coverage_at_k([[1, 99]], [1, 2, 3], 2)


def test_catalog_coverage_empty_catalog_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        catalog_coverage_at_k([], [], 10)


def test_catalog_coverage_duplicate_catalog_sequence_is_rejected() -> None:
    with pytest.raises(ValueError, match="catalog_items.*duplicate"):
        catalog_coverage_at_k([[1]], [1, 2, 1], 1)


@pytest.mark.parametrize(
    ("invalid_id", "error_type"),
    [
        (-1, ValueError),
        (1.5, TypeError),
        ("1", TypeError),
        (True, TypeError),
        (None, TypeError),
        ([1], TypeError),
    ],
)
def test_catalog_coverage_invalid_catalog_ids_are_rejected(
    invalid_id: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="catalog_items.*non-negative"):
        catalog_coverage_at_k([], [invalid_id], 1)  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("invalid_id", "error_type"),
    [
        (-1, ValueError),
        (1.5, TypeError),
        ("1", TypeError),
        (True, TypeError),
        (None, TypeError),
        ([1], TypeError),
    ],
)
def test_catalog_coverage_invalid_recommendation_ids_are_rejected(
    invalid_id: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="recommendations.*non-negative"):
        catalog_coverage_at_k(  # type: ignore[list-item]
            [[invalid_id]],
            [1, 2],
            1,
        )


@pytest.mark.parametrize(
    ("invalid_k", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (10.0, TypeError),
        ("10", TypeError),
        (None, TypeError),
    ],
)
def test_catalog_coverage_invalid_k_is_rejected(
    invalid_k: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        catalog_coverage_at_k([[1]], [1], invalid_k)  # type: ignore[arg-type]


def test_catalog_coverage_inputs_are_not_mutated() -> None:
    recommendation_lists = [[1, 2], [2, 3]]
    catalog = [1, 2, 3, 4]
    original_lists = [row.copy() for row in recommendation_lists]
    original_catalog = catalog.copy()

    catalog_coverage_at_k(recommendation_lists, catalog, 2)

    assert recommendation_lists == original_lists
    assert catalog == original_catalog


def test_catalog_coverage_repeated_calls_are_deterministic_and_bounded() -> None:
    arguments = ([[1, 2], [2, 3]], [1, 2, 3, 4], 2)
    first = catalog_coverage_at_k(*arguments)

    assert all(catalog_coverage_at_k(*arguments) == first for _ in range(20))
    assert 0.0 <= first <= 1.0


def test_novelty_smoothed_known_fixture() -> None:
    counts = {1: 3, 2: 1, 3: 0}

    actual = novelty_at_k([1, 3], counts, 2)

    assert actual == pytest.approx(1.8073549220576042, **TOLERANCE)


def test_novelty_uniform_popularity_fixture() -> None:
    counts = {1: 1, 2: 1, 3: 1}

    actual = novelty_at_k([1, 2, 3], counts, 3)

    assert actual == pytest.approx(1.584962500721156, **TOLERANCE)


def test_novelty_rarer_item_has_higher_self_information() -> None:
    counts = {1: 100, 2: 10, 3: 0}

    popular = novelty_at_k([1], counts, 1)
    rare = novelty_at_k([3], counts, 1)

    assert rare > popular


def test_novelty_empty_recommendation_list_returns_zero() -> None:
    assert novelty_at_k([], {1: 3}, 10) == 0.0


def test_novelty_short_list_averages_returned_items() -> None:
    counts = {1: 3, 2: 1, 3: 0}

    assert novelty_at_k([1, 3], counts, 10) == pytest.approx(
        1.8073549220576042,
        **TOLERANCE,
    )


def test_novelty_only_first_k_contributes() -> None:
    counts = {1: 3, 2: 1, 3: 0}
    expected = 1.3073549220576042

    without_tail = novelty_at_k([1, 2], counts, 2)
    with_tail = novelty_at_k([1, 2, 3], counts, 2)

    assert without_tail == pytest.approx(expected, **TOLERANCE)
    assert with_tail == pytest.approx(expected, **TOLERANCE)


def test_novelty_zero_count_is_finite() -> None:
    actual = novelty_at_k([2], {1: 4, 2: 0}, 1)

    assert math.isfinite(actual)
    assert actual == pytest.approx(2.584962500721156, **TOLERANCE)


def test_novelty_unknown_recommended_item_is_rejected() -> None:
    with pytest.raises(ValueError, match="absent from item_positive_counts"):
        novelty_at_k([99], {1: 3}, 1)


def test_novelty_unknown_item_below_k_is_also_rejected() -> None:
    with pytest.raises(ValueError, match="absent from item_positive_counts"):
        novelty_at_k([1, 99], {1: 3}, 1)


def test_novelty_empty_count_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        novelty_at_k([], {}, 10)


@pytest.mark.parametrize(
    ("invalid_count", "error_type"),
    [
        (-1, ValueError),
        (1.5, TypeError),
        ("1", TypeError),
        (True, TypeError),
        (None, TypeError),
    ],
)
def test_novelty_invalid_counts_are_rejected(
    invalid_count: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="values must be non-negative"):
        novelty_at_k([1], {1: invalid_count}, 1)  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("invalid_id", "error_type"),
    [
        (-1, ValueError),
        (1.5, TypeError),
        ("1", TypeError),
        (True, TypeError),
        (None, TypeError),
    ],
)
def test_novelty_invalid_mapping_key_is_rejected(
    invalid_id: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(
        error_type,
        match="item_positive_counts.*non-negative",
    ):
        novelty_at_k([], {invalid_id: 1}, 1)  # type: ignore[dict-item]


def test_novelty_duplicate_recommendation_is_rejected() -> None:
    with pytest.raises(ValueError, match="recommendations.*duplicate"):
        novelty_at_k([1, 2, 1], {1: 3, 2: 1}, 2)


@pytest.mark.parametrize(
    ("invalid_id", "error_type"),
    [
        (-1, ValueError),
        (1.5, TypeError),
        ("1", TypeError),
        (True, TypeError),
        (None, TypeError),
        ([1], TypeError),
    ],
)
def test_novelty_invalid_recommendation_id_is_rejected(
    invalid_id: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="recommendations.*non-negative"):
        novelty_at_k(  # type: ignore[list-item]
            [invalid_id],
            {1: 3},
            1,
        )


@pytest.mark.parametrize(
    ("invalid_k", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (10.0, TypeError),
        ("10", TypeError),
        (None, TypeError),
    ],
)
def test_novelty_invalid_k_is_rejected(
    invalid_k: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        novelty_at_k([1], {1: 3}, invalid_k)  # type: ignore[arg-type]


def test_novelty_repeated_calls_are_deterministic() -> None:
    arguments = ([1, 3], {1: 3, 2: 1, 3: 0}, 2)
    first = novelty_at_k(*arguments)

    assert all(novelty_at_k(*arguments) == first for _ in range(20))


def test_novelty_inputs_are_not_mutated_and_result_is_valid() -> None:
    recommendations = [1, 3]
    counts = {1: 3, 2: 1, 3: 0}
    original_recommendations = recommendations.copy()
    original_counts = counts.copy()

    actual = novelty_at_k(recommendations, counts, 2)

    assert recommendations == original_recommendations
    assert counts == original_counts
    assert math.isfinite(actual)
    assert actual >= 0.0


def test_ild_mixed_known_fixture() -> None:
    item_genres = {
        1: {"Action", "Drama"},
        2: {"Action"},
        3: {"Comedy"},
    }

    actual = intra_list_diversity_at_k([1, 2, 3], item_genres, 3)

    assert actual == pytest.approx(0.8333333333333334, **TOLERANCE)


def test_ild_identical_genres_equals_zero() -> None:
    actual = intra_list_diversity_at_k(
        [1, 2],
        {1: {"Drama"}, 2: {"Drama"}},
        2,
    )

    assert actual == pytest.approx(0.0, **TOLERANCE)


def test_ild_disjoint_genres_equals_one() -> None:
    actual = intra_list_diversity_at_k(
        [1, 2],
        {1: {"Drama"}, 2: {"Comedy"}},
        2,
    )

    assert actual == pytest.approx(1.0, **TOLERANCE)


def test_ild_both_empty_genre_sets_equal_zero() -> None:
    actual = intra_list_diversity_at_k([1, 2], {1: set(), 2: set()}, 2)

    assert actual == pytest.approx(0.0, **TOLERANCE)


def test_ild_one_empty_genre_set_equals_one() -> None:
    actual = intra_list_diversity_at_k(
        [1, 2],
        {1: set(), 2: {"Drama"}},
        2,
    )

    assert actual == pytest.approx(1.0, **TOLERANCE)


@pytest.mark.parametrize("recommendations", [[], [1]])
def test_ild_fewer_than_two_recommendations_equal_zero(
    recommendations: list[int],
) -> None:
    assert (
        intra_list_diversity_at_k(
            recommendations,
            {1: {"Drama"}},
            10,
        )
        == 0.0
    )


def test_ild_only_first_k_contributes() -> None:
    item_genres = {
        1: {"Drama"},
        2: {"Drama"},
        3: {"Comedy"},
    }

    without_tail = intra_list_diversity_at_k([1, 2], item_genres, 2)
    with_tail = intra_list_diversity_at_k([1, 2, 3], item_genres, 2)

    assert without_tail == with_tail == pytest.approx(0.0, **TOLERANCE)


def test_ild_pair_average_uses_three_unordered_pairs_without_self_pairs() -> None:
    item_genres = {
        1: {"A"},
        2: {"A", "B"},
        3: {"A", "B", "C"},
    }
    expected = 0.5

    actual = intra_list_diversity_at_k([1, 2, 3], item_genres, 3)

    assert actual == pytest.approx(expected, **TOLERANCE)


def test_ild_genre_order_does_not_affect_output() -> None:
    first = {1: ["Action", "Drama"], 2: ["Comedy", "Drama"]}
    second = {1: ["Drama", "Action"], 2: ["Drama", "Comedy"]}

    assert intra_list_diversity_at_k(
        [1, 2],
        first,
        2,
    ) == intra_list_diversity_at_k([1, 2], second, 2)


def test_ild_unknown_item_is_rejected() -> None:
    with pytest.raises(ValueError, match="absent from item_genres"):
        intra_list_diversity_at_k([99], {1: {"Drama"}}, 1)


@pytest.mark.parametrize(
    "recommendations",
    [
        [1, 1],
        [1, 2, 3, 1],
    ],
)
def test_ild_duplicate_recommendation_is_rejected(
    recommendations: list[int],
) -> None:
    genres = {1: {"A"}, 2: {"B"}, 3: {"C"}}
    with pytest.raises(ValueError, match="recommendations.*duplicate"):
        intra_list_diversity_at_k(recommendations, genres, 2)


def test_ild_duplicate_genre_sequence_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate labels"):
        intra_list_diversity_at_k(
            [1],
            {1: ["Drama", "Drama"]},
            1,
        )


def test_ild_empty_string_genre_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty strings"):
        intra_list_diversity_at_k([1], {1: [""]}, 1)


@pytest.mark.parametrize("invalid_label", [1, True, None])
def test_ild_non_string_genre_label_is_rejected(
    invalid_label: object,
) -> None:
    with pytest.raises(TypeError, match="genre labels must be strings"):
        intra_list_diversity_at_k(  # type: ignore[list-item]
            [1],
            {1: [invalid_label]},
            1,
        )


def test_ild_bare_string_genre_collection_is_rejected() -> None:
    with pytest.raises(TypeError, match="genre-label collections"):
        intra_list_diversity_at_k(
            [1],
            {1: "Drama"},  # type: ignore[dict-item]
            1,
        )


@pytest.mark.parametrize(
    ("invalid_id", "error_type"),
    [
        (-1, ValueError),
        (1.5, TypeError),
        ("1", TypeError),
        (True, TypeError),
        (None, TypeError),
        ([1], TypeError),
    ],
)
def test_ild_invalid_recommendation_id_is_rejected(
    invalid_id: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="recommendations.*non-negative"):
        intra_list_diversity_at_k(  # type: ignore[list-item]
            [invalid_id],
            {1: {"Drama"}},
            1,
        )


@pytest.mark.parametrize(
    ("invalid_id", "error_type"),
    [
        (-1, ValueError),
        (1.5, TypeError),
        ("1", TypeError),
        (True, TypeError),
        (None, TypeError),
    ],
)
def test_ild_invalid_mapping_item_id_is_rejected(
    invalid_id: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="item_genres.*non-negative"):
        intra_list_diversity_at_k(  # type: ignore[dict-item]
            [],
            {invalid_id: {"Drama"}},
            1,
        )


@pytest.mark.parametrize(
    ("invalid_k", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (10.0, TypeError),
        ("10", TypeError),
        (None, TypeError),
    ],
)
def test_ild_invalid_k_is_rejected(
    invalid_k: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        intra_list_diversity_at_k(  # type: ignore[arg-type]
            [1],
            {1: {"Drama"}},
            invalid_k,
        )


def test_ild_repeated_calls_are_deterministic_and_bounded() -> None:
    arguments = (
        [1, 2, 3],
        {1: {"A", "B"}, 2: {"A"}, 3: {"C"}},
        3,
    )
    first = intra_list_diversity_at_k(*arguments)

    assert all(intra_list_diversity_at_k(*arguments) == first for _ in range(20))
    assert 0.0 <= first <= 1.0


def test_ild_inputs_are_not_mutated() -> None:
    recommendations = [1, 2]
    item_genres = {1: ["Action", "Drama"], 2: ["Comedy"]}
    original_recommendations = recommendations.copy()
    original_genres = {
        item_id: genres.copy() for item_id, genres in item_genres.items()
    }

    intra_list_diversity_at_k(recommendations, item_genres, 2)

    assert recommendations == original_recommendations
    assert item_genres == original_genres


def test_numpy_integral_scalars_are_accepted_across_metrics() -> None:
    coverage = catalog_coverage_at_k(
        [[np.int32(1), np.int64(2)]],
        [np.int64(1), np.int32(2)],
        np.int64(2),
    )
    novelty = novelty_at_k(
        [np.int64(1)],
        {np.int32(1): np.int64(0)},
        np.int32(1),
    )
    diversity = intra_list_diversity_at_k(
        [np.int64(1), np.int32(2)],
        {np.int32(1): {"A"}, np.int64(2): {"B"}},
        np.int64(2),
    )

    assert coverage == 1.0
    assert novelty == 0.0
    assert diversity == 1.0


def test_recommendations_must_be_ordered_sequences() -> None:
    with pytest.raises(TypeError, match="ordered sequence"):
        novelty_at_k({1, 2}, {1: 1, 2: 1}, 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ordered sequence"):
        intra_list_diversity_at_k(  # type: ignore[arg-type]
            {1, 2},
            {1: {"A"}, 2: {"B"}},
            2,
        )


def test_metric_module_has_no_file_or_dataframe_dependencies() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "pandas" not in imported_roots
    assert "pathlib" not in imported_roots
    assert "open" not in called_names
    assert "test.parquet" not in source
    assert "validation.parquet" not in source


def test_metric_module_has_no_score_sorting_models_or_later_metrics() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    public_functions = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }

    assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    assert "sorted(" not in source
    assert ".sort(" not in source
    assert public_functions == {
        "catalog_coverage_at_k",
        "novelty_at_k",
        "intra_list_diversity_at_k",
    }
