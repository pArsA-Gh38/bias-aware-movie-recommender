"""Focused correctness, invariant, and scope tests for sparse ItemKNN."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_csr

from src.models.itemknn import (
    ItemKNNModel,
    build_binary_user_item_matrix,
    fit_itemknn,
    recommend_itemknn,
    score_itemknn_candidates,
)

ITEMKNN_SOURCE = Path(__file__).resolve().parents[1] / "src/models/itemknn.py"
SMOKE_SOURCE = Path(__file__).resolve().parents[1] / "scripts/build_itemknn_smoke.py"


@pytest.fixture
def known_interactions() -> tuple[tuple[int, int], ...]:
    return (
        (1, 10),
        (1, 20),
        (2, 10),
        (2, 20),
        (2, 30),
        (3, 10),
        (3, 30),
        (4, 40),
    )


@pytest.fixture
def known_matrix(
    known_interactions: tuple[tuple[int, int], ...],
) -> csr_matrix:
    return build_binary_user_item_matrix(
        known_interactions,
        (1, 2, 3, 4),
        (10, 20, 30, 40),
    )


@pytest.fixture
def known_model(known_matrix: csr_matrix) -> ItemKNNModel:
    return fit_itemknn(known_matrix, (10, 20, 30, 40), 10)


def handcrafted_model(
    edges: tuple[tuple[int, int, float], ...],
    *,
    item_ids: tuple[int, ...] = (10, 20, 30, 40),
    n_neighbors: int = 4,
) -> ItemKNNModel:
    index = {item_id: position for position, item_id in enumerate(item_ids)}
    rows = [index[source] for source, _, _ in edges]
    columns = [index[candidate] for _, candidate, _ in edges]
    values = [score for _, _, score in edges]
    graph = csr_matrix(
        (values, (rows, columns)),
        shape=(len(item_ids), len(item_ids)),
        dtype=np.float64,
    )
    graph.sort_indices()
    return ItemKNNModel(item_ids, graph, n_neighbors)


def neighbor_mapping(
    model: ItemKNNModel,
    source_item: int,
) -> dict[int, float]:
    source_index = model.item_ids.index(source_item)
    start = model.similarities.indptr[source_index]
    end = model.similarities.indptr[source_index + 1]
    return {
        model.item_ids[int(index)]: float(score)
        for index, score in zip(
            model.similarities.indices[start:end],
            model.similarities.data[start:end],
            strict=True,
        )
    }


def test_binary_matrix_has_exact_shape_and_nnz(
    known_matrix: csr_matrix,
) -> None:
    assert known_matrix.shape == (4, 4)
    assert known_matrix.nnz == 8


def test_binary_matrix_is_csr_with_only_one_values(
    known_matrix: csr_matrix,
) -> None:
    assert isspmatrix_csr(known_matrix)
    assert np.equal(known_matrix.data, 1.0).all()


def test_binary_matrix_respects_mapping_alignment() -> None:
    matrix = build_binary_user_item_matrix(
        ((20, 200), (10, 100)),
        (20, 10),
        (100, 200),
    )
    assert matrix.toarray().tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_binary_matrix_rejects_duplicate_interactions() -> None:
    with pytest.raises(ValueError, match="duplicate pairs"):
        build_binary_user_item_matrix(
            ((1, 10), (1, 10)),
            (1,),
            (10,),
        )


@pytest.mark.parametrize(
    ("interactions", "match"),
    [
        (((2, 10),), "user ID is absent"),
        (((1, 20),), "item ID is absent"),
    ],
)
def test_binary_matrix_rejects_unknown_mapping_ids(
    interactions: tuple[tuple[int, int], ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_binary_user_item_matrix(interactions, (1,), (10,))


@pytest.mark.parametrize(
    ("user_ids", "item_ids", "error"),
    [
        ((True,), (10,), TypeError),
        ((1.5,), (10,), TypeError),
        ((-1,), (10,), ValueError),
        ((1,), (False,), TypeError),
        ((1,), (-10,), ValueError),
    ],
)
def test_binary_matrix_rejects_invalid_mapping_ids(
    user_ids: tuple[object, ...],
    item_ids: tuple[object, ...],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        build_binary_user_item_matrix((), user_ids, item_ids)


def test_binary_matrix_rejects_duplicate_mapping_ids() -> None:
    with pytest.raises(ValueError, match="duplicate IDs"):
        build_binary_user_item_matrix((), (1, 1), (10,))


def test_binary_matrix_is_deterministic_and_does_not_mutate_inputs() -> None:
    interactions = [(2, 20), (1, 10)]
    users = [1, 2]
    items = [10, 20]
    originals = (list(interactions), list(users), list(items))
    first = build_binary_user_item_matrix(interactions, users, items)
    second = build_binary_user_item_matrix(
        reversed(interactions),
        users,
        items,
    )
    assert (first != second).nnz == 0
    assert interactions == originals[0]
    assert users == originals[1]
    assert items == originals[2]


def test_known_cosines_are_exact_and_symmetric_before_pruning(
    known_model: ItemKNNModel,
) -> None:
    expected = 0.8164965809277261
    neighbors_10 = neighbor_mapping(known_model, 10)
    neighbors_20 = neighbor_mapping(known_model, 20)
    neighbors_30 = neighbor_mapping(known_model, 30)
    assert neighbors_10[20] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert neighbors_10[30] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert neighbors_20[10] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert neighbors_30[10] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert neighbors_20[30] == pytest.approx(0.5, rel=1e-12, abs=1e-12)
    assert neighbors_30[20] == pytest.approx(0.5, rel=1e-12, abs=1e-12)


def test_mandatory_k_one_neighbor_fixture(
    known_matrix: csr_matrix,
) -> None:
    model = fit_itemknn(known_matrix, (10, 20, 30, 40), 1)
    assert neighbor_mapping(model, 10) == {
        20: pytest.approx(0.8164965809277261, rel=1e-12, abs=1e-12)
    }
    assert tuple(neighbor_mapping(model, 20)) == (10,)
    assert tuple(neighbor_mapping(model, 30)) == (10,)
    assert neighbor_mapping(model, 40) == {}


def test_similarity_graph_has_zero_diagonal_and_isolated_item(
    known_model: ItemKNNModel,
) -> None:
    assert np.count_nonzero(known_model.similarities.diagonal()) == 0
    assert neighbor_mapping(known_model, 40) == {}


def test_similarity_graph_is_valid_sparse_csr(
    known_model: ItemKNNModel,
) -> None:
    graph = known_model.similarities
    assert isspmatrix_csr(graph)
    assert graph.nnz == 6
    assert np.isfinite(graph.data).all()
    assert np.all(graph.data > 0.0)
    assert np.all(graph.data <= 1.0)
    assert not np.equal(graph.data, 0.0).any()


@pytest.mark.parametrize("n_neighbors", [1, 2, 20])
def test_similarity_row_never_exceeds_neighbor_count(
    known_matrix: csr_matrix,
    n_neighbors: int,
) -> None:
    model = fit_itemknn(
        known_matrix,
        (10, 20, 30, 40),
        n_neighbors,
    )
    assert np.diff(model.similarities.indptr).max() <= n_neighbors


def test_large_k_retains_all_available_neighbors_without_self(
    known_matrix: csr_matrix,
) -> None:
    model = fit_itemknn(known_matrix, (10, 20, 30, 40), 100)
    assert model.similarities.nnz == 6
    assert np.count_nonzero(model.similarities.diagonal()) == 0


@pytest.mark.parametrize("n_neighbors", [0, -1])
def test_fit_rejects_nonpositive_neighbor_count(
    known_matrix: csr_matrix,
    n_neighbors: int,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        fit_itemknn(known_matrix, (10, 20, 30, 40), n_neighbors)


@pytest.mark.parametrize("n_neighbors", [True, 1.5])
def test_fit_rejects_noninteger_neighbor_count(
    known_matrix: csr_matrix,
    n_neighbors: object,
) -> None:
    with pytest.raises(TypeError, match="integer scalar"):
        fit_itemknn(known_matrix, (10, 20, 30, 40), n_neighbors)


def test_fit_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="at least one interaction"):
        fit_itemknn(csr_matrix((2, 2), dtype=np.float64), (10, 20), 1)


def test_fit_allows_zero_norm_item_as_empty_neighbor_row() -> None:
    matrix = csr_matrix([[1.0, 0.0], [1.0, 0.0]])
    model = fit_itemknn(matrix, (10, 20), 5)
    assert model.similarities.nnz == 0
    assert neighbor_mapping(model, 20) == {}


def test_fit_rejects_non_csr_input() -> None:
    matrix = coo_matrix(([1.0], ([0], [0])), shape=(1, 1))
    with pytest.raises(TypeError, match="CSR"):
        fit_itemknn(matrix, (10,), 1)


@pytest.mark.parametrize(
    "matrix",
    [
        csr_matrix([[2.0]]),
        csr_matrix([[0.5]]),
        csr_matrix([[np.nan]]),
    ],
)
def test_fit_rejects_nonbinary_or_nonfinite_values(
    matrix: csr_matrix,
) -> None:
    with pytest.raises(ValueError):
        fit_itemknn(matrix, (10,), 1)


def test_fit_rejects_item_shape_mismatch(known_matrix: csr_matrix) -> None:
    with pytest.raises(ValueError, match="column count"):
        fit_itemknn(known_matrix, (10, 20), 1)


def test_fit_does_not_mutate_input_matrix(known_matrix: csr_matrix) -> None:
    before_data = known_matrix.data.copy()
    before_indices = known_matrix.indices.copy()
    before_indptr = known_matrix.indptr.copy()
    fit_itemknn(known_matrix, (10, 20, 30, 40), 2)
    assert np.array_equal(known_matrix.data, before_data)
    assert np.array_equal(known_matrix.indices, before_indices)
    assert np.array_equal(known_matrix.indptr, before_indptr)


def test_repeated_fit_is_byte_deterministic(known_matrix: csr_matrix) -> None:
    first = fit_itemknn(known_matrix, (10, 20, 30, 40), 2)
    second = fit_itemknn(known_matrix, (10, 20, 30, 40), 2)
    assert first.item_ids == second.item_ids
    assert first.similarities.data.tobytes() == second.similarities.data.tobytes()
    assert first.similarities.indices.tobytes() == second.similarities.indices.tobytes()
    assert first.similarities.indptr.tobytes() == second.similarities.indptr.tobytes()


def test_fit_is_independent_of_item_mapping_insertion_order(
    known_interactions: tuple[tuple[int, int], ...],
) -> None:
    canonical = build_binary_user_item_matrix(
        known_interactions,
        (1, 2, 3, 4),
        (10, 20, 30, 40),
    )
    permuted_ids = (30, 10, 40, 20)
    permuted = build_binary_user_item_matrix(
        known_interactions,
        (1, 2, 3, 4),
        permuted_ids,
    )
    first = fit_itemknn(canonical, (10, 20, 30, 40), 2)
    second = fit_itemknn(permuted, permuted_ids, 2)
    assert first.item_ids == second.item_ids
    assert (first.similarities != second.similarities).nnz == 0


def test_single_profile_scores_and_recommendations(
    known_model: ItemKNNModel,
) -> None:
    scores = score_itemknn_candidates(known_model, (20,))
    by_id = dict(zip(known_model.item_ids, scores, strict=True))
    assert by_id[10] == pytest.approx(0.8164965809277261, rel=1e-12, abs=1e-12)
    assert by_id[30] == pytest.approx(0.5, rel=1e-12, abs=1e-12)
    assert by_id[40] == 0.0
    recommendations = recommend_itemknn(known_model, (20,), (20,), 10)
    assert tuple(item_id for item_id, _ in recommendations) == (10, 30)


def test_candidate_scores_use_sum_not_max() -> None:
    model = handcrafted_model(
        (
            (10, 30, 0.4),
            (20, 30, 0.4),
            (10, 40, 0.7),
        )
    )
    result = recommend_itemknn(model, (10, 20), (10, 20), 2)
    assert result == (
        (30, pytest.approx(0.8, rel=1e-12, abs=1e-12)),
        (40, pytest.approx(0.7, rel=1e-12, abs=1e-12)),
    )


def test_seen_item_filter_removes_higher_scored_candidate() -> None:
    model = handcrafted_model(((10, 20, 0.9), (10, 30, 0.8)))
    result = recommend_itemknn(model, (10,), (10, 20), 10)
    assert result == ((30, pytest.approx(0.8)),)


@pytest.mark.parametrize("rated_item", [20, 30, 40])
def test_seen_filter_is_rating_agnostic_collection_behavior(
    rated_item: int,
) -> None:
    model = handcrafted_model(((10, 20, 0.9), (10, 30, 0.8), (10, 40, 0.7)))
    result = recommend_itemknn(model, (10,), (10, rated_item), 10)
    assert rated_item not in {item_id for item_id, _ in result}


def test_filtering_occurs_before_top_k() -> None:
    model = handcrafted_model(((10, 20, 0.9), (10, 30, 0.8), (10, 40, 0.7)))
    result = recommend_itemknn(model, (10,), (10, 20), 2)
    assert tuple(item_id for item_id, _ in result) == (30, 40)


def test_equal_score_candidate_tie_uses_movie_id_ascending() -> None:
    model = handcrafted_model(((10, 30, 0.5), (10, 20, 0.5)))
    result = recommend_itemknn(model, (10,), (10,), 2)
    assert tuple(item_id for item_id, _ in result) == (20, 30)


def test_no_positive_score_candidate_returns_empty_without_fill() -> None:
    model = handcrafted_model(())
    assert recommend_itemknn(model, (10,), (10,), 10) == ()


def test_duplicate_profile_is_rejected() -> None:
    model = handcrafted_model(((10, 20, 0.5),))
    with pytest.raises(ValueError, match="duplicate"):
        recommend_itemknn(model, (10, 10), (10,), 2)


def test_unknown_profile_item_is_rejected() -> None:
    model = handcrafted_model(((10, 20, 0.5),))
    with pytest.raises(ValueError, match="absent from fit catalog"):
        score_itemknn_candidates(model, (999,))


def test_valid_out_of_catalog_seen_item_is_harmless() -> None:
    model = handcrafted_model(((10, 20, 0.5),))
    assert recommend_itemknn(model, (10,), (10, 999), 2) == ((20, pytest.approx(0.5)),)


@pytest.mark.parametrize("seen_item", [True, -1, 1.5])
def test_invalid_seen_item_is_rejected(seen_item: object) -> None:
    model = handcrafted_model(((10, 20, 0.5),))
    with pytest.raises((TypeError, ValueError)):
        recommend_itemknn(model, (10,), (10, seen_item), 2)


def test_empty_profile_returns_empty_scores_and_recommendations() -> None:
    model = handcrafted_model(((10, 20, 0.5),))
    assert np.equal(score_itemknn_candidates(model, ()), 0.0).all()
    assert recommend_itemknn(model, (), (), 10) == ()


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_recommendation_rejects_invalid_k(k: object) -> None:
    model = handcrafted_model(((10, 20, 0.5),))
    with pytest.raises((TypeError, ValueError)):
        recommend_itemknn(model, (10,), (10,), k)


def test_recommendation_output_invariants_and_determinism() -> None:
    model = handcrafted_model(((10, 20, 0.5), (10, 30, 0.8), (10, 40, 0.5)))
    profile = [10]
    rated = [10]
    first = recommend_itemknn(model, profile, rated, 3)
    second = recommend_itemknn(model, profile, rated, 3)
    ids = [item_id for item_id, _ in first]
    scores = [score for _, score in first]
    assert first == second
    assert len(ids) == len(set(ids))
    assert scores == sorted(scores, reverse=True)
    assert ids == [30, 20, 40]
    assert profile == [10]
    assert rated == [10]


def test_production_source_has_no_complete_matrix_densification() -> None:
    source = ITEMKNN_SOURCE.read_text(encoding="utf-8")
    forbidden = (".toarray(", ".todense(", "np.asarray(similarity")
    assert not any(token in source for token in forbidden)


def test_pure_model_has_no_file_or_experiment_dependencies() -> None:
    tree = ast.parse(ITEMKNN_SOURCE.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "pandas" not in imported_roots
    assert "random" not in imported_roots
    source = ITEMKNN_SOURCE.read_text(encoding="utf-8").lower()
    for forbidden in (
        "validation.parquet",
        "test.parquet",
        "alternatingleastsquares",
        "popularity fallback",
        "ndcg_at_k",
        "precision_at_k",
    ):
        assert forbidden not in source


def test_smoke_source_does_not_load_targets_or_tune() -> None:
    source = SMOKE_SOURCE.read_text(encoding="utf-8").lower()
    for forbidden in (
        "validation.parquet",
        "test.parquet",
        "alternatingleastsquares",
        "random.",
        "ndcg_at_k",
        "precision_at_k",
    ):
        assert forbidden not in source
    tree = ast.parse(source)
    assert all(
        not (
            isinstance(node, (ast.List, ast.Tuple, ast.Set))
            and [getattr(item, "value", None) for item in node.elts] == [50, 100, 200]
        )
        for node in ast.walk(tree)
    )
