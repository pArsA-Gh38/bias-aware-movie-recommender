from __future__ import annotations

import ast
import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import numpy as np
import pytest
import recommender_core.reranking as reranker_module

from src.reranking.greedy_reranker import RerankedItem, greedy_rerank

TOLERANCE = {"rel": 1e-15, "abs": 1e-15}
SOURCE_PATH = (
    Path(__file__).parents[2]
    / "packages/recommender_core/src/recommender_core/reranking.py"
)


def _simple_rerank(*, k: int = 3) -> tuple[RerankedItem, ...]:
    return greedy_rerank(
        [30, 20, 10],
        [3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0],
        {30: ["Action"], 20: ["Comedy"], 10: ["Drama"]},
        k=k,
        lambda_=0.5,
        beta=0.5,
    )


def _canonical_bytes(items: tuple[RerankedItem, ...]) -> bytes:
    return (
        json.dumps(
            [asdict(item) for item in items],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def test_empty_candidate_pool_returns_empty_tuple() -> None:
    assert greedy_rerank([], [], [], {}, k=10, lambda_=0.8, beta=0.5) == ()


def test_candidate_pool_shorter_than_k_returns_every_candidate_once() -> None:
    actual = _simple_rerank(k=10)

    assert len(actual) == 3
    assert {item.movie_id for item in actual} == {10, 20, 30}


def test_candidate_pool_at_least_k_returns_exactly_k() -> None:
    assert len(_simple_rerank(k=2)) == 2


def test_output_has_no_duplicate_movie_ids() -> None:
    actual = _simple_rerank()
    ids = [item.movie_id for item in actual]

    assert len(ids) == len(set(ids))


def test_output_is_a_subset_of_the_authoritative_pool() -> None:
    pool = {10, 20, 30}
    assert all(item.movie_id in pool for item in _simple_rerank(k=2))


def test_output_ranks_are_contiguous_and_one_based() -> None:
    actual = _simple_rerank()
    assert [item.rank for item in actual] == [1, 2, 3]


def test_base_ranks_preserve_original_candidate_positions() -> None:
    original = [30, 20, 10]
    actual = _simple_rerank()

    assert {item.movie_id: item.base_rank for item in actual} == {
        movie_id: position for position, movie_id in enumerate(original, start=1)
    }


def test_lambda_one_reproduces_base_prefix_with_ties_before_movie_id() -> None:
    actual = greedy_rerank(
        [30, 20, 10],
        [5.0, 5.0, 4.0],
        [0.0, 100.0, 50.0],
        {30: ["Action"], 20: ["Drama"], 10: ["Comedy"]},
        k=3,
        lambda_=1.0,
        beta=0.5,
    )

    assert [item.movie_id for item in actual] == [30, 20, 10]
    assert [item.base_rank for item in actual] == [1, 2, 3]


def test_first_selection_diversity_and_score_use_empty_prefix_policy() -> None:
    actual = greedy_rerank(
        [10, 20],
        [2.0, 1.0],
        [1.0, 2.0],
        {10: ["Action"], 20: ["Drama"]},
        k=1,
        lambda_=0.5,
        beta=1.0,
    )

    assert actual[0].movie_id == 10
    assert actual[0].genre_diversity == 1.0
    assert actual[0].beyond_score == 1.0
    assert actual[0].final_score == 1.0


def test_diversity_is_recomputed_after_selected_prefix_expands() -> None:
    actual = greedy_rerank(
        [10, 20, 30],
        [3.0, 2.0, 1.0],
        [1.0, 1.0, 1.0],
        {10: ["Action"], 20: ["Action"], 30: ["Drama"]},
        k=3,
        lambda_=0.6,
        beta=1.0,
    )

    assert [item.movie_id for item in actual] == [10, 30, 20]
    assert [item.genre_diversity for item in actual] == [1.0, 1.0, 0.0]


def test_most_similar_selected_item_controls_diversity_and_order() -> None:
    actual = greedy_rerank(
        [10, 20, 30, 40],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        {
            10: ["Action"],
            20: ["Drama"],
            30: ["Action"],
            40: ["Action", "Drama"],
        },
        k=4,
        lambda_=0.0,
        beta=1.0,
    )

    assert [item.movie_id for item in actual] == [10, 20, 40, 30]
    assert [item.genre_diversity for item in actual] == [1.0, 1.0, 0.5, 0.0]


def test_identical_genres_receive_zero_diversity() -> None:
    actual = greedy_rerank(
        [1, 2],
        [2.0, 1.0],
        [1.0, 1.0],
        {1: ["Action"], 2: ["Action"]},
        k=2,
        lambda_=1.0,
        beta=1.0,
    )

    assert actual[1].genre_diversity == 0.0


def test_disjoint_genres_receive_unit_diversity() -> None:
    actual = greedy_rerank(
        [1, 2],
        [2.0, 1.0],
        [1.0, 1.0],
        {1: ["Action"], 2: ["Drama"]},
        k=2,
        lambda_=1.0,
        beta=1.0,
    )

    assert actual[1].genre_diversity == 1.0


def test_partial_overlap_uses_known_jaccard_distance() -> None:
    actual = greedy_rerank(
        [1, 2],
        [2.0, 1.0],
        [1.0, 1.0],
        {
            1: ["Action", "Drama"],
            2: ["Action", "Comedy"],
        },
        k=2,
        lambda_=1.0,
        beta=1.0,
    )

    assert actual[1].genre_diversity == pytest.approx(2 / 3, **TOLERANCE)


def test_empty_genre_semantics_match_frozen_ild_contract() -> None:
    actual = greedy_rerank(
        [1, 2, 3],
        [3.0, 2.0, 1.0],
        [1.0, 1.0, 1.0],
        {1: [], 2: [], 3: ["Action"]},
        k=3,
        lambda_=0.0,
        beta=1.0,
    )

    assert [item.movie_id for item in actual] == [1, 3, 2]
    assert [item.genre_diversity for item in actual] == [1.0, 1.0, 0.0]


def test_beta_one_makes_beyond_score_diversity_only() -> None:
    actual = greedy_rerank(
        [1, 2],
        [2.0, 1.0],
        [1.0, 2.0],
        {1: ["Action"], 2: ["Action"]},
        k=2,
        lambda_=1.0,
        beta=1.0,
    )

    assert all(item.beyond_score == item.genre_diversity for item in actual)


def test_beta_zero_makes_beyond_score_novelty_only() -> None:
    actual = greedy_rerank(
        [1, 2],
        [2.0, 1.0],
        [1.0, 3.0],
        {1: ["Action"], 2: ["Action"]},
        k=2,
        lambda_=1.0,
        beta=0.0,
    )

    assert all(item.beyond_score == item.normalized_novelty for item in actual)


def test_lambda_zero_selects_by_beyond_term_then_frozen_ties() -> None:
    actual = greedy_rerank(
        [10, 20],
        [2.0, 1.0],
        [1.0, 2.0],
        {10: ["Action"], 20: ["Action"]},
        k=2,
        lambda_=0.0,
        beta=0.0,
    )

    assert [item.movie_id for item in actual] == [20, 10]
    assert all(item.final_score == item.beyond_score for item in actual)


def test_constant_relevance_is_zero_and_selection_remains_deterministic() -> None:
    actual = greedy_rerank(
        [30, 20, 10],
        [5.0, 5.0, 5.0],
        [1.0, 3.0, 2.0],
        {30: ["A"], 20: ["B"], 10: ["C"]},
        k=3,
        lambda_=0.5,
        beta=0.0,
    )

    assert all(item.normalized_relevance == 0.0 for item in actual)
    assert [item.movie_id for item in actual] == [20, 10, 30]


def test_constant_novelty_is_zero_and_selection_remains_deterministic() -> None:
    actual = greedy_rerank(
        [30, 20, 10],
        [3.0, 2.0, 1.0],
        [5.0, 5.0, 5.0],
        {30: ["A"], 20: ["B"], 10: ["C"]},
        k=3,
        lambda_=0.5,
        beta=0.0,
    )

    assert all(item.normalized_novelty == 0.0 for item in actual)
    assert [item.movie_id for item in actual] == [30, 20, 10]


def test_both_constant_features_still_use_diversity_and_tie_breaks() -> None:
    actual = greedy_rerank(
        [30, 20, 10],
        [5.0, 5.0, 5.0],
        [7.0, 7.0, 7.0],
        {30: ["Action"], 20: ["Action"], 10: ["Drama"]},
        k=3,
        lambda_=0.0,
        beta=1.0,
    )

    assert [item.movie_id for item in actual] == [30, 10, 20]


def test_hand_calculated_score_components_and_order_are_exact() -> None:
    actual = greedy_rerank(
        [10, 20, 30],
        [30.0, 20.0, 10.0],
        [1.0, 3.0, 2.0],
        {10: ["Action"], 20: ["Action"], 30: ["Drama"]},
        k=3,
        lambda_=0.5,
        beta=0.5,
    )

    assert [item.movie_id for item in actual] == [10, 20, 30]
    expected = [
        (1.0, 0.0, 1.0, 0.5, 0.75),
        (0.5, 1.0, 0.0, 0.5, 0.5),
        (0.0, 0.5, 1.0, 0.75, 0.375),
    ]
    for item, components in zip(actual, expected, strict=True):
        observed = (
            item.normalized_relevance,
            item.normalized_novelty,
            item.genre_diversity,
            item.beyond_score,
            item.final_score,
        )
        assert observed == pytest.approx(components, **TOLERANCE)


@pytest.mark.parametrize(
    ("candidate_ids", "error_type"),
    [
        ([1, 1], ValueError),
        ([-1], ValueError),
        ([True], TypeError),
        ([1.5], TypeError),
        (["1"], TypeError),
    ],
)
def test_invalid_candidate_ids_are_rejected(
    candidate_ids: list[object], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        greedy_rerank(
            candidate_ids,  # type: ignore[arg-type]
            [1.0] * len(candidate_ids),
            [1.0] * len(candidate_ids),
            {1: ["Action"]},
            k=1,
            lambda_=0.5,
            beta=0.5,
        )


def test_unordered_candidate_collection_is_rejected() -> None:
    with pytest.raises(TypeError, match="ordered sequence"):
        greedy_rerank(
            {1, 2},  # type: ignore[arg-type]
            [2.0, 1.0],
            [1.0, 2.0],
            {1: ["A"], 2: ["B"]},
            k=1,
            lambda_=0.5,
            beta=0.5,
        )


@pytest.mark.parametrize(
    ("candidate_ids", "relevance", "novelty"),
    [
        ([1, 2], [1.0], [1.0]),
        ([1], [1.0, 2.0], [1.0, 2.0]),
        ([1, 2], [1.0, 2.0], [1.0]),
    ],
)
def test_candidate_feature_length_mismatches_are_rejected(
    candidate_ids: list[int],
    relevance: list[float],
    novelty: list[float],
) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        greedy_rerank(
            candidate_ids,
            relevance,
            novelty,
            {1: ["A"], 2: ["B"]},
            k=1,
            lambda_=0.5,
            beta=0.5,
        )


def test_missing_candidate_genre_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="every candidate"):
        greedy_rerank(
            [1, 2],
            [2.0, 1.0],
            [1.0, 2.0],
            {1: ["Action"]},
            k=1,
            lambda_=0.5,
            beta=0.5,
        )


@pytest.mark.parametrize(
    ("genres", "error_type"),
    [
        ("Action", TypeError),
        (b"Action", TypeError),
        (["Action", 3], TypeError),
        (["Action", ""], ValueError),
        (["Action", "Action"], ValueError),
    ],
)
def test_malformed_genre_collections_are_rejected(
    genres: object, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        greedy_rerank(
            [1],
            [1.0],
            [1.0],
            {1: genres},  # type: ignore[dict-item]
            k=1,
            lambda_=0.5,
            beta=0.5,
        )


def test_empty_genre_collection_is_valid() -> None:
    assert (
        greedy_rerank([1], [1.0], [1.0], {1: []}, k=1, lambda_=0.5, beta=0.5)[
            0
        ].genre_diversity
        == 1.0
    )


def test_extra_valid_genre_metadata_is_accepted() -> None:
    assert (
        greedy_rerank(
            [1],
            [1.0],
            [1.0],
            {1: ["Action"], 999: ["Drama"]},
            k=1,
            lambda_=0.5,
            beta=0.5,
        )[0].movie_id
        == 1
    )


@pytest.mark.parametrize(
    ("invalid_k", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.0, TypeError),
        ("1", TypeError),
    ],
)
def test_invalid_k_is_rejected(invalid_k: object, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        greedy_rerank(
            [],
            [],
            [],
            {},
            k=invalid_k,  # type: ignore[arg-type]
            lambda_=0.5,
            beta=0.5,
        )


@pytest.mark.parametrize(
    ("name", "invalid", "error_type"),
    [
        ("lambda_", True, TypeError),
        ("lambda_", "0.5", TypeError),
        ("lambda_", -0.1, ValueError),
        ("lambda_", 1.1, ValueError),
        ("lambda_", np.nan, ValueError),
        ("lambda_", np.inf, ValueError),
        ("lambda_", -np.inf, ValueError),
        ("beta", True, TypeError),
        ("beta", "0.5", TypeError),
        ("beta", -0.1, ValueError),
        ("beta", 1.1, ValueError),
        ("beta", np.nan, ValueError),
        ("beta", np.inf, ValueError),
        ("beta", -np.inf, ValueError),
    ],
)
def test_invalid_unit_interval_parameters_are_rejected(
    name: str, invalid: object, error_type: type[Exception]
) -> None:
    parameters = {"lambda_": 0.5, "beta": 0.5}
    parameters[name] = invalid
    with pytest.raises(error_type):
        greedy_rerank(
            [],
            [],
            [],
            {},
            k=1,
            lambda_=parameters["lambda_"],  # type: ignore[arg-type]
            beta=parameters["beta"],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("feature_name", "invalid", "message"),
    [
        ("relevance", [0.0, np.nan], "finite"),
        ("relevance", [0.0, np.inf], "finite"),
        ("novelty", [0.0, -np.inf], "finite"),
        ("relevance", 1.0, "one-dimensional"),
        ("novelty", 1.0, "one-dimensional"),
        ("novelty", [[1.0], [2.0]], "one-dimensional"),
    ],
)
def test_invalid_features_are_rejected_by_normalizer_contract(
    feature_name: str, invalid: object, message: str
) -> None:
    features: dict[str, object] = {
        "relevance": [2.0, 1.0],
        "novelty": [1.0, 2.0],
    }
    features[feature_name] = invalid
    with pytest.raises(ValueError, match=message):
        greedy_rerank(
            [1, 2],
            features["relevance"],  # type: ignore[arg-type]
            features["novelty"],  # type: ignore[arg-type]
            {1: ["A"], 2: ["B"]},
            k=1,
            lambda_=0.5,
            beta=0.5,
        )


def test_inputs_remain_unchanged() -> None:
    candidate_ids = [30, 20, 10]
    relevance = np.array([3.0, 2.0, 1.0])
    novelty = np.array([1.0, 2.0, 3.0])
    item_genres = {30: ["A"], 20: ["B"], 10: ["C"]}
    ids_before = candidate_ids.copy()
    relevance_before = relevance.tobytes()
    novelty_before = novelty.tobytes()
    genres_before = {key: value.copy() for key, value in item_genres.items()}

    greedy_rerank(
        candidate_ids,
        relevance,
        novelty,
        item_genres,
        k=3,
        lambda_=0.5,
        beta=0.5,
    )

    assert candidate_ids == ids_before
    assert relevance.tobytes() == relevance_before
    assert novelty.tobytes() == novelty_before
    assert item_genres == genres_before


def test_repeated_calls_have_byte_identical_canonical_output() -> None:
    first = _canonical_bytes(_simple_rerank())
    second = _canonical_bytes(_simple_rerank())

    assert first == second


def test_calls_are_stateless_across_different_candidate_pools() -> None:
    expected = _simple_rerank()
    greedy_rerank([99], [1.0], [1.0], {99: []}, k=1, lambda_=0.0, beta=1.0)

    assert _simple_rerank() == expected


def test_result_record_is_frozen() -> None:
    item = _simple_rerank(k=1)[0]
    with pytest.raises(FrozenInstanceError):
        item.rank = 2  # type: ignore[misc]


def test_all_output_components_are_finite_and_bounded() -> None:
    for item in _simple_rerank():
        assert all(
            np.isfinite(value)
            for value in (
                item.raw_relevance,
                item.normalized_relevance,
                item.raw_novelty,
                item.normalized_novelty,
                item.genre_diversity,
                item.beyond_score,
                item.final_score,
            )
        )
        assert all(
            0.0 <= value <= 1.0
            for value in (
                item.normalized_relevance,
                item.normalized_novelty,
                item.genre_diversity,
                item.beyond_score,
                item.final_score,
            )
        )


def test_numpy_integer_candidate_ids_are_accepted() -> None:
    actual = greedy_rerank(
        np.array([2, 1], dtype=np.int64),
        [2.0, 1.0],
        [1.0, 2.0],
        {2: ["A"], 1: ["B"]},
        k=np.int64(2),
        lambda_=np.float64(1.0),
        beta=np.float64(0.5),
    )

    assert [item.movie_id for item in actual] == [2, 1]


def test_accepted_paired_normalizer_is_called_once(monkeypatch) -> None:
    calls = 0
    accepted = reranker_module.normalize_reranking_features

    def counted(relevance: object, novelty: object):
        nonlocal calls
        calls += 1
        return accepted(relevance, novelty)

    monkeypatch.setattr(reranker_module, "normalize_reranking_features", counted)
    _simple_rerank()

    assert calls == 1


def test_core_scope_is_pure_numeric_and_target_blind() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
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
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert imported_roots == {
        "__future__",
        "collections",
        "dataclasses",
        "numbers",
        "numpy",
        "recommender_core",
    }
    assert not {"open", "eval", "fit", "partial_fit_users"} & called_names
    source = SOURCE_PATH.read_text(encoding="utf-8").lower()
    assert "test.parquet" not in source
    assert "validation.parquet" not in source
    assert "target_label" not in source
