import ast
import math
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.exposure_metrics import (
    assign_popularity_groups,
    exposure_gini_at_k,
    long_tail_exposure_at_k,
)

TOLERANCE = {"rel": 1e-12, "abs": 1e-12}
SOURCE_PATH = Path(__file__).parents[1] / "src" / "evaluation" / "exposure_metrics.py"
KNOWN_COUNTS = {
    1: 100,
    2: 90,
    3: 80,
    4: 70,
    5: 60,
    6: 50,
    7: 40,
    8: 30,
    9: 20,
    10: 10,
}
KNOWN_GROUPS = {
    1: "head",
    2: "head",
    3: "mid_tail",
    4: "mid_tail",
    5: "mid_tail",
    6: "long_tail",
    7: "long_tail",
    8: "long_tail",
    9: "long_tail",
    10: "long_tail",
}


def test_popularity_groups_known_ten_item_assignment() -> None:
    assert assign_popularity_groups(KNOWN_COUNTS) == KNOWN_GROUPS


def test_popularity_groups_equal_count_ties_use_ascending_item_id() -> None:
    counts = {
        100: 100,
        90: 100,
        80: 100,
        70: 100,
        60: 100,
        50: 100,
        40: 100,
        30: 100,
        20: 100,
        10: 100,
    }
    expected = {
        10: "head",
        20: "head",
        30: "mid_tail",
        40: "mid_tail",
        50: "mid_tail",
        60: "long_tail",
        70: "long_tail",
        80: "long_tail",
        90: "long_tail",
        100: "long_tail",
    }

    assert assign_popularity_groups(counts) == expected


def test_popularity_group_sizes_follow_exact_item_count_boundaries() -> None:
    groups = assign_popularity_groups(KNOWN_COUNTS)

    assert list(groups.values()).count("head") == 2
    assert list(groups.values()).count("mid_tail") == 3
    assert list(groups.values()).count("long_tail") == 5


def test_popularity_groups_cover_catalog_and_are_pairwise_disjoint() -> None:
    groups = assign_popularity_groups(KNOWN_COUNTS)
    head = {item_id for item_id, group in groups.items() if group == "head"}
    mid_tail = {item_id for item_id, group in groups.items() if group == "mid_tail"}
    long_tail = {item_id for item_id, group in groups.items() if group == "long_tail"}

    assert head | mid_tail | long_tail == set(KNOWN_COUNTS)
    assert head.isdisjoint(mid_tail)
    assert head.isdisjoint(long_tail)
    assert mid_tail.isdisjoint(long_tail)
    assert len(groups) == len(KNOWN_COUNTS)


def test_popularity_boundaries_use_item_counts_not_interaction_mass() -> None:
    counts = {
        1: 1_000_000,
        2: 9,
        3: 8,
        4: 7,
        5: 6,
        6: 5,
        7: 4,
        8: 3,
        9: 2,
        10: 1,
    }

    groups = assign_popularity_groups(counts)

    assert {item for item, group in groups.items() if group == "head"} == {
        1,
        2,
    }
    assert {item for item, group in groups.items() if group == "mid_tail"} == {
        3,
        4,
        5,
    }


@pytest.mark.parametrize(
    ("catalog_size", "expected_sizes"),
    [
        (1, (0, 0, 1)),
        (2, (0, 1, 1)),
        (3, (0, 1, 2)),
        (4, (0, 2, 2)),
        (5, (1, 1, 3)),
        (6, (1, 2, 3)),
    ],
)
def test_popularity_small_catalogs_follow_exact_floor_boundaries(
    catalog_size: int,
    expected_sizes: tuple[int, int, int],
) -> None:
    counts = {item_id: catalog_size - item_id for item_id in range(catalog_size)}

    groups = assign_popularity_groups(counts)
    actual_sizes = (
        list(groups.values()).count("head"),
        list(groups.values()).count("mid_tail"),
        list(groups.values()).count("long_tail"),
    )

    assert actual_sizes == expected_sizes


def test_popularity_group_assignment_does_not_mutate_counts() -> None:
    counts = KNOWN_COUNTS.copy()
    original = counts.copy()

    assign_popularity_groups(counts)

    assert counts == original


def test_popularity_group_assignment_is_exactly_deterministic() -> None:
    first = assign_popularity_groups(KNOWN_COUNTS)

    assert all(assign_popularity_groups(KNOWN_COUNTS) == first for _ in range(20))


def test_popularity_empty_count_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        assign_popularity_groups({})


def test_popularity_count_input_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        assign_popularity_groups([(1, 2)])  # type: ignore[arg-type]


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
def test_popularity_invalid_counts_are_rejected(
    invalid_count: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="values must be non-negative"):
        assign_popularity_groups({1: invalid_count})  # type: ignore[dict-item]


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
def test_popularity_invalid_item_ids_are_rejected(
    invalid_id: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(
        error_type,
        match="item_positive_counts.*non-negative",
    ):
        assign_popularity_groups({invalid_id: 1})  # type: ignore[dict-item]


def test_popularity_numpy_integral_ids_and_counts_are_accepted() -> None:
    groups = assign_popularity_groups(
        {
            np.int64(2): np.int32(1),
            np.int32(1): np.int64(2),
        }
    )

    assert groups == {np.int32(1): "mid_tail", np.int64(2): "long_tail"}


def test_long_tail_exposure_known_short_list_fixture_equals_one_half() -> None:
    recommendation_lists = [
        [1, 6, 7, 3],
        [10, 2],
        [4, 5, 8],
    ]

    actual = long_tail_exposure_at_k(recommendation_lists, KNOWN_GROUPS, 3)

    assert actual == pytest.approx(0.5, **TOLERANCE)


def test_long_tail_denominator_is_eight_actual_displayed_slots_not_nine() -> None:
    recommendation_lists = [
        [1, 6, 7, 3],
        [10, 2],
        [4, 5, 8],
    ]

    actual = long_tail_exposure_at_k(recommendation_lists, KNOWN_GROUPS, 3)

    assert actual == pytest.approx(4 / 8, **TOLERANCE)
    assert actual != pytest.approx(4 / 9, **TOLERANCE)


def test_long_tail_no_long_tail_slots_equals_zero() -> None:
    assert long_tail_exposure_at_k([[1, 3], [2, 4]], KNOWN_GROUPS, 2) == 0.0


def test_long_tail_all_displayed_slots_are_long_tail_equals_one() -> None:
    assert long_tail_exposure_at_k([[6, 7], [8]], KNOWN_GROUPS, 10) == 1.0


def test_long_tail_no_recommendation_lists_equals_zero() -> None:
    assert long_tail_exposure_at_k([], KNOWN_GROUPS, 10) == 0.0


def test_long_tail_all_user_lists_empty_equals_zero() -> None:
    assert long_tail_exposure_at_k([[], [], []], KNOWN_GROUPS, 10) == 0.0


def test_long_tail_k_smaller_than_list_length_uses_top_k_only() -> None:
    actual = long_tail_exposure_at_k([[1, 6, 7, 8]], KNOWN_GROUPS, 2)

    assert actual == pytest.approx(0.5, **TOLERANCE)


def test_long_tail_k_larger_than_list_length_uses_available_slots() -> None:
    actual = long_tail_exposure_at_k([[1, 6]], KNOWN_GROUPS, 10)

    assert actual == pytest.approx(0.5, **TOLERANCE)


def test_long_tail_item_below_k_does_not_contribute() -> None:
    without_tail = long_tail_exposure_at_k([[1, 3]], KNOWN_GROUPS, 2)
    with_tail = long_tail_exposure_at_k([[1, 3, 6]], KNOWN_GROUPS, 2)

    assert with_tail == without_tail == 0.0


def test_long_tail_same_item_across_users_counts_repeated_exposures() -> None:
    actual = long_tail_exposure_at_k([[6], [6], [1]], KNOWN_GROUPS, 1)

    assert actual == pytest.approx(2 / 3, **TOLERANCE)


@pytest.mark.parametrize(
    "recommendation_lists",
    [
        [[6, 6]],
        [[1, 2, 3, 1]],
    ],
)
def test_long_tail_duplicate_within_complete_list_is_rejected(
    recommendation_lists: list[list[int]],
) -> None:
    with pytest.raises(ValueError, match="recommendations.*duplicate"):
        long_tail_exposure_at_k(recommendation_lists, KNOWN_GROUPS, 2)


def test_long_tail_unknown_recommended_item_is_rejected() -> None:
    with pytest.raises(ValueError, match="absent from popularity_groups"):
        long_tail_exposure_at_k([[99]], KNOWN_GROUPS, 1)


def test_long_tail_unknown_item_below_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="absent from popularity_groups"):
        long_tail_exposure_at_k([[1, 99]], KNOWN_GROUPS, 1)


def test_long_tail_empty_group_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        long_tail_exposure_at_k([], {}, 10)


def test_long_tail_group_input_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        long_tail_exposure_at_k([], [(1, "head")], 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_label", ["tail", "", "Head"])
def test_long_tail_invalid_string_group_labels_are_rejected(
    invalid_label: str,
) -> None:
    with pytest.raises(ValueError, match="head, mid_tail, or long_tail"):
        long_tail_exposure_at_k([], {1: invalid_label}, 1)


@pytest.mark.parametrize("invalid_label", [1, True, None])
def test_long_tail_non_string_group_labels_are_rejected(
    invalid_label: object,
) -> None:
    with pytest.raises(TypeError, match="labels must be strings"):
        long_tail_exposure_at_k([], {1: invalid_label}, 1)  # type: ignore[dict-item]


def test_long_tail_missing_group_mapping_entry_is_rejected() -> None:
    with pytest.raises(ValueError, match="absent from popularity_groups"):
        long_tail_exposure_at_k([[2]], {1: "head"}, 1)


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
def test_long_tail_invalid_recommendation_ids_are_rejected(
    invalid_id: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="recommendations.*non-negative"):
        long_tail_exposure_at_k(  # type: ignore[list-item]
            [[invalid_id]],
            KNOWN_GROUPS,
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
def test_long_tail_invalid_mapping_item_ids_are_rejected(
    invalid_id: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="popularity_groups.*non-negative"):
        long_tail_exposure_at_k(  # type: ignore[dict-item]
            [],
            {invalid_id: "head"},
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
def test_long_tail_invalid_k_is_rejected(
    invalid_k: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        long_tail_exposure_at_k(  # type: ignore[arg-type]
            [[1]],
            KNOWN_GROUPS,
            invalid_k,
        )


def test_long_tail_recommendation_lists_must_be_a_collection() -> None:
    recommendation_lists = (row for row in [[1]])

    with pytest.raises(TypeError, match="must be a collection"):
        long_tail_exposure_at_k(  # type: ignore[arg-type]
            recommendation_lists,
            KNOWN_GROUPS,
            1,
        )


def test_long_tail_each_user_list_must_be_an_ordered_sequence() -> None:
    with pytest.raises(TypeError, match="ordered sequence"):
        long_tail_exposure_at_k([{1, 6}], KNOWN_GROUPS, 2)  # type: ignore[list-item]


def test_long_tail_numpy_integrals_are_accepted() -> None:
    actual = long_tail_exposure_at_k(
        [[np.int64(1), np.int32(6)]],
        {np.int32(1): "head", np.int64(6): "long_tail"},
        np.int64(2),
    )

    assert actual == pytest.approx(0.5, **TOLERANCE)


def test_long_tail_inputs_are_not_mutated() -> None:
    recommendation_lists = [[1, 6], [3]]
    groups = KNOWN_GROUPS.copy()
    original_lists = [row.copy() for row in recommendation_lists]
    original_groups = groups.copy()

    long_tail_exposure_at_k(recommendation_lists, groups, 2)

    assert recommendation_lists == original_lists
    assert groups == original_groups


def test_long_tail_repeated_calls_are_deterministic_and_bounded() -> None:
    arguments = ([[1, 6], [7]], KNOWN_GROUPS, 2)
    first = long_tail_exposure_at_k(*arguments)

    assert all(long_tail_exposure_at_k(*arguments) == first for _ in range(20))
    assert 0.0 <= first <= 1.0


def test_exposure_gini_equal_exposure_fixture_equals_zero() -> None:
    actual = exposure_gini_at_k([[1], [2], [3], [4]], [1, 2, 3, 4], 1)

    assert actual == pytest.approx(0.0, **TOLERANCE)


def test_exposure_gini_maximum_concentration_fixture_equals_three_quarters() -> None:
    actual = exposure_gini_at_k([[1], [1], [1], [1]], [1, 2, 3, 4], 1)

    assert actual == pytest.approx(0.75, **TOLERANCE)
    assert actual != pytest.approx(1.0, **TOLERANCE)


def test_exposure_gini_mixed_fixture_including_zero_equals_three_eighths() -> None:
    recommendation_lists = [[4, 3], [4, 2]]

    actual = exposure_gini_at_k(recommendation_lists, [1, 2, 3, 4], 2)

    assert actual == pytest.approx(0.375, **TOLERANCE)


def test_exposure_gini_zero_exposure_catalog_items_are_included() -> None:
    actual = exposure_gini_at_k([[4, 3], [4, 2]], [1, 2, 3, 4], 2)
    positive_only_vector_gini = 1 / 6

    assert actual == pytest.approx(0.375, **TOLERANCE)
    assert actual != pytest.approx(positive_only_vector_gini, **TOLERANCE)


def test_exposure_gini_no_recommendation_lists_equals_zero() -> None:
    assert exposure_gini_at_k([], [1, 2, 3, 4], 10) == 0.0


def test_exposure_gini_all_user_lists_empty_equals_zero() -> None:
    assert exposure_gini_at_k([[], [], []], [1, 2, 3, 4], 10) == 0.0


def test_exposure_gini_one_item_catalog_always_equals_zero() -> None:
    assert exposure_gini_at_k([[1], [1], [1]], [1], 1) == 0.0


def test_exposure_gini_k_smaller_than_list_length_uses_top_k_only() -> None:
    top_only = exposure_gini_at_k([[1, 2]], [1, 2, 3], 1)
    with_tail = exposure_gini_at_k([[1, 2, 3]], [1, 2, 3], 1)

    assert with_tail == top_only == pytest.approx(2 / 3, **TOLERANCE)


def test_exposure_gini_k_larger_than_list_uses_available_slots() -> None:
    actual = exposure_gini_at_k([[1, 2]], [1, 2, 3, 4], 10)

    assert actual == pytest.approx(0.5, **TOLERANCE)


def test_exposure_gini_item_below_k_does_not_contribute() -> None:
    without_tail = exposure_gini_at_k([[1]], [1, 2, 3], 1)
    with_tail = exposure_gini_at_k([[1, 2]], [1, 2, 3], 1)

    assert with_tail == without_tail == pytest.approx(2 / 3, **TOLERANCE)


def test_exposure_gini_same_item_across_users_increments_exposure() -> None:
    actual = exposure_gini_at_k([[1], [1], [2]], [1, 2, 3], 1)

    assert actual == pytest.approx(4 / 9, **TOLERANCE)


@pytest.mark.parametrize(
    "recommendation_lists",
    [
        [[1, 1]],
        [[1, 2, 3, 1]],
    ],
)
def test_exposure_gini_duplicate_within_complete_list_is_rejected(
    recommendation_lists: list[list[int]],
) -> None:
    with pytest.raises(ValueError, match="recommendations.*duplicate"):
        exposure_gini_at_k(recommendation_lists, [1, 2, 3], 2)


def test_exposure_gini_unknown_recommended_item_is_rejected() -> None:
    with pytest.raises(ValueError, match="absent from catalog_items"):
        exposure_gini_at_k([[99]], [1, 2, 3], 1)


def test_exposure_gini_unknown_item_below_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="absent from catalog_items"):
        exposure_gini_at_k([[1, 99]], [1, 2, 3], 1)


def test_exposure_gini_empty_catalog_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        exposure_gini_at_k([], [], 10)


def test_exposure_gini_catalog_input_must_be_a_collection() -> None:
    catalog = (item for item in [1, 2])

    with pytest.raises(TypeError, match="must be a collection"):
        exposure_gini_at_k([], catalog, 1)  # type: ignore[arg-type]


def test_exposure_gini_duplicate_catalog_sequence_is_rejected() -> None:
    with pytest.raises(ValueError, match="catalog_items.*duplicate"):
        exposure_gini_at_k([], [1, 2, 1], 1)


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
def test_exposure_gini_invalid_catalog_ids_are_rejected(
    invalid_id: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="catalog_items.*non-negative"):
        exposure_gini_at_k([], [invalid_id], 1)  # type: ignore[list-item]


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
def test_exposure_gini_invalid_recommendation_ids_are_rejected(
    invalid_id: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="recommendations.*non-negative"):
        exposure_gini_at_k(  # type: ignore[list-item]
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
def test_exposure_gini_invalid_k_is_rejected(
    invalid_k: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        exposure_gini_at_k([[1]], [1], invalid_k)  # type: ignore[arg-type]


def test_exposure_gini_recommendation_lists_must_be_a_collection() -> None:
    recommendation_lists = (row for row in [[1]])

    with pytest.raises(TypeError, match="must be a collection"):
        exposure_gini_at_k(  # type: ignore[arg-type]
            recommendation_lists,
            [1],
            1,
        )


def test_exposure_gini_each_user_list_must_be_an_ordered_sequence() -> None:
    with pytest.raises(TypeError, match="ordered sequence"):
        exposure_gini_at_k([{1, 2}], [1, 2], 2)  # type: ignore[list-item]


def test_exposure_gini_numpy_integrals_are_accepted() -> None:
    actual = exposure_gini_at_k(
        [[np.int64(1)], [np.int32(2)]],
        [np.int32(1), np.int64(2)],
        np.int64(1),
    )

    assert actual == 0.0


def test_exposure_gini_inputs_are_not_mutated() -> None:
    recommendation_lists = [[1, 2], [2, 3]]
    catalog = [1, 2, 3, 4]
    original_lists = [row.copy() for row in recommendation_lists]
    original_catalog = catalog.copy()

    exposure_gini_at_k(recommendation_lists, catalog, 2)

    assert recommendation_lists == original_lists
    assert catalog == original_catalog


def test_exposure_gini_repeated_calls_are_deterministic_finite_and_bounded() -> None:
    arguments = ([[4, 3], [4, 2]], [1, 2, 3, 4], 2)
    first = exposure_gini_at_k(*arguments)

    assert all(exposure_gini_at_k(*arguments) == first for _ in range(20))
    assert math.isfinite(first)
    assert 0.0 <= first <= 1.0


def test_metric_module_has_no_file_dataframe_or_artifact_dependencies() -> None:
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
    assert "validation.parquet" not in source
    assert "test.parquet" not in source


def test_metric_module_has_only_requested_public_functions_and_no_classes() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    public_functions = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }

    assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    assert public_functions == {
        "assign_popularity_groups",
        "long_tail_exposure_at_k",
        "exposure_gini_at_k",
    }


def test_metric_module_has_no_scores_rank_weighting_or_later_work() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8").lower()
    forbidden_phrases = {
        "bootstrap",
        "head_exposure",
        "average_recommendation_popularity",
        "itemknn",
        "als",
        "rerank",
        "lambda",
        "model class",
        "score sorting",
    }

    assert all(phrase not in source for phrase in forbidden_phrases)
