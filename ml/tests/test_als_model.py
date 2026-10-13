from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix

from src.models.als import (
    ALSConfig,
    ALSModel,
    fit_als,
    recommend_als,
    score_als_candidates,
)


def config(**changes: object) -> ALSConfig:
    values = {
        "factors": 2,
        "regularization": 0.01,
        "iterations": 2,
        "confidence_alpha": 20,
        "implicit_library_alpha": 1.0,
        "random_state": 42,
        "dtype": "float32",
        "num_threads": 1,
        "use_native": True,
        "use_cg": True,
        "calculate_training_loss": False,
        "use_gpu": False,
    }
    values.update(changes)
    return ALSConfig(**values)


def binary_matrix() -> csr_matrix:
    return csr_matrix(
        np.asarray(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )


def handcrafted_model() -> ALSModel:
    return ALSModel(
        user_ids=(1,),
        item_ids=(10, 20, 30, 40),
        user_factors=np.asarray([[1.0, 2.0]], dtype=np.float32),
        item_factors=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        config=config(),
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("factors", 0, ValueError),
        ("factors", True, TypeError),
        ("regularization", 0.0, ValueError),
        ("regularization", float("nan"), ValueError),
        ("iterations", 0, ValueError),
        ("confidence_alpha", 0.0, ValueError),
        ("implicit_library_alpha", 0.0, ValueError),
        ("implicit_library_alpha", 20.0, ValueError),
        ("random_state", -1, ValueError),
        ("random_state", True, TypeError),
        ("dtype", "float64", ValueError),
        ("num_threads", 0, ValueError),
        ("num_threads", 2, ValueError),
        ("use_native", False, ValueError),
        ("use_cg", False, ValueError),
        ("calculate_training_loss", True, ValueError),
        ("use_gpu", True, ValueError),
    ],
)
def test_config_rejects_invalid_values(
    field: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        config(**{field: value})


def test_config_retains_exact_values() -> None:
    value = config()
    assert value == ALSConfig(
        factors=2,
        regularization=0.01,
        iterations=2,
        confidence_alpha=20.0,
        implicit_library_alpha=1.0,
        random_state=42,
        dtype="float32",
        num_threads=1,
        use_native=True,
        use_cg=True,
        calculate_training_loss=False,
        use_gpu=False,
    )


def test_fit_requires_csr() -> None:
    with pytest.raises(TypeError, match="CSR"):
        fit_als(np.ones((1, 1)), (1,), (10,), config())  # type: ignore[arg-type]


def test_fit_requires_exact_mapping_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        fit_als(binary_matrix(), (1, 2), (10, 20, 30), config())


def test_fit_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_als(
            csr_matrix((2, 2), dtype=np.float32),
            (1, 2),
            (10, 20),
            config(),
        )


def test_fit_rejects_nonbinary_matrix() -> None:
    matrix = binary_matrix()
    matrix.data[0] = 21.0
    with pytest.raises(ValueError, match="binary value 1.0"):
        fit_als(matrix, (1, 2, 3), (10, 20, 30), config())


def test_fit_rejects_nonfinite_matrix() -> None:
    matrix = binary_matrix()
    matrix.data[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_als(matrix, (1, 2, 3), (10, 20, 30), config())


def test_fit_rejects_duplicate_csr_entry() -> None:
    matrix = coo_matrix(
        (
            np.asarray([1.0, 1.0], dtype=np.float32),
            (np.asarray([0, 0]), np.asarray([0, 0])),
        ),
        shape=(1, 1),
    ).tocsr()
    matrix.has_canonical_format = False
    with pytest.raises(ValueError, match="without duplicates"):
        fit_als(matrix, (1,), (10,), config())


def test_fit_rejects_duplicate_or_invalid_mapping_ids() -> None:
    matrix = binary_matrix()
    with pytest.raises(ValueError, match="duplicate"):
        fit_als(matrix, (1, 1, 3), (10, 20, 30), config())
    with pytest.raises(ValueError, match="non-negative"):
        fit_als(matrix, (1, 2, 3), (10, -20, 30), config())


def test_fit_derives_exact_confidence_and_disables_library_scaling() -> None:
    calls: dict[str, object] = {}
    source = binary_matrix()
    model_config = config()
    assert model_config.confidence_alpha == 20.0

    class SpyALS:
        def __init__(self, **kwargs: object) -> None:
            calls["kwargs"] = kwargs

        def fit(self, matrix: csr_matrix, *, show_progress: bool) -> None:
            calls["matrix"] = matrix.copy()
            calls["show_progress"] = show_progress
            self.user_factors = np.ones((3, 2), dtype=np.float32)
            self.item_factors = np.ones((3, 2), dtype=np.float32)

    fit_als(
        source,
        (1, 2, 3),
        (10, 20, 30),
        model_config,
        als_factory=SpyALS,
    )
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs == {
        "factors": 2,
        "regularization": 0.01,
        "alpha": 1.0,
        "dtype": np.float32,
        "use_native": True,
        "use_cg": True,
        "iterations": 2,
        "calculate_training_loss": False,
        "num_threads": 1,
        "random_state": 42,
    }
    passed = calls["matrix"]
    assert isinstance(passed, csr_matrix)
    assert np.equal(source.data, 1.0).all()
    assert np.equal(passed.data, 21.0).all()
    assert passed is not source
    assert calls["show_progress"] is False


def test_fit_does_not_mutate_input() -> None:
    matrix = binary_matrix()
    before = matrix.copy()
    fit_als(matrix, (1, 2, 3), (10, 20, 30), config())
    assert (matrix != before).nnz == 0
    np.testing.assert_array_equal(matrix.indptr, before.indptr)
    np.testing.assert_array_equal(matrix.indices, before.indices)


def test_actual_fit_shapes_dtype_finiteness_alignment_and_determinism() -> None:
    first = fit_als(binary_matrix(), (3, 5, 7), (30, 20, 10), config())
    second = fit_als(binary_matrix(), (3, 5, 7), (30, 20, 10), config())
    assert first.user_ids == (3, 5, 7)
    assert first.item_ids == (30, 20, 10)
    assert first.config == config()
    assert first.user_factors.shape == (3, 2)
    assert first.item_factors.shape == (3, 2)
    assert first.user_factors.dtype == np.float32
    assert first.item_factors.dtype == np.float32
    assert np.isfinite(first.user_factors).all()
    assert np.isfinite(first.item_factors).all()
    np.testing.assert_array_equal(first.user_factors, second.user_factors)
    np.testing.assert_array_equal(first.item_factors, second.item_factors)


@pytest.mark.parametrize(
    "attribute,value,match",
    [
        ("user_factors", np.ones((2, 2), dtype=np.float32), "user factor shape"),
        ("item_factors", np.ones((2, 2), dtype=np.float32), "item factor shape"),
        (
            "user_factors",
            np.full((3, 2), np.nan, dtype=np.float32),
            "user factors must be finite",
        ),
        (
            "item_factors",
            np.ones((3, 2), dtype=np.float64),
            "item factor dtype",
        ),
    ],
)
def test_fit_rejects_invalid_returned_factors(
    attribute: str, value: np.ndarray, match: str
) -> None:
    class BrokenALS:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, _: csr_matrix, *, show_progress: bool) -> None:
            assert show_progress is False
            self.user_factors = np.ones((3, 2), dtype=np.float32)
            self.item_factors = np.ones((3, 2), dtype=np.float32)
            setattr(self, attribute, value)

    with pytest.raises(ValueError, match=match):
        fit_als(
            binary_matrix(),
            (1, 2, 3),
            (10, 20, 30),
            config(),
            als_factory=BrokenALS,
        )


def test_handcrafted_dot_product_and_negative_score_recommendation() -> None:
    model = handcrafted_model()
    np.testing.assert_array_equal(
        score_als_candidates(model, 1),
        np.asarray([1.0, 2.0, 3.0, -1.0], dtype=np.float32),
    )
    assert recommend_als(model, 1, {20}, 3) == (
        (30, 3.0),
        (10, 1.0),
        (40, -1.0),
    )


def test_external_movie_id_breaks_exact_score_ties() -> None:
    model = ALSModel(
        user_ids=(1,),
        item_ids=(30, 20),
        user_factors=np.asarray([[1.0]], dtype=np.float32),
        item_factors=np.asarray([[0.5], [0.5]], dtype=np.float32),
        config=replace(config(), factors=1),
    )
    assert recommend_als(model, 1, set(), 2) == ((20, 0.5), (30, 0.5))


@pytest.mark.parametrize("rated_movie", [10, 20, 30])
def test_every_supplied_rating_1_2_or_3_seen_item_is_filtered(
    rated_movie: int,
) -> None:
    recommendations = recommend_als(handcrafted_model(), 1, {rated_movie}, 3)
    assert rated_movie not in {item for item, _ in recommendations}


def test_future_validation_rating_is_not_filtered_unless_supplied_as_seen() -> None:
    recommendations = recommend_als(handcrafted_model(), 1, {20}, 3)
    assert 30 in {item for item, _ in recommendations}


def test_duplicate_seen_id_is_rejected_and_out_of_catalog_is_harmless() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        recommend_als(handcrafted_model(), 1, [20, 20], 3)
    assert recommend_als(handcrafted_model(), 1, {999}, 1) == ((30, 3.0),)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k_is_rejected(k: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        recommend_als(handcrafted_model(), 1, set(), k)  # type: ignore[arg-type]


def test_unknown_user_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fitted user"):
        score_als_candidates(handcrafted_model(), 999)


def test_recommendation_is_unique_deterministic_and_inputs_are_not_mutated() -> None:
    model = handcrafted_model()
    seen = [20]
    factors_before = model.item_factors.copy()
    first = recommend_als(model, 1, seen, 3)
    second = recommend_als(model, 1, seen, 3)
    assert first == second
    assert len({item for item, _ in first}) == len(first)
    assert seen == [20]
    np.testing.assert_array_equal(model.item_factors, factors_before)
