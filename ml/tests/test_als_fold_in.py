from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, isspmatrix_csr
from threadpoolctl import threadpool_limits

_BLAS_LIMITER = threadpool_limits(limits=1, user_api="blas")

from src.inference.als_fold_in import (
    ALSFoldInConfig,
    DuplicatePositiveInteraction,
    EmptyPositiveInteractions,
    FoldInError,
    InsufficientPositiveInteractions,
    InvalidProfileItem,
    UnknownProfileItem,
    build_fold_in_confidence_row,
    decide_fold_in_strategy,
    fold_in_new_user,
)


def config(**changes: object) -> ALSFoldInConfig:
    values: dict[str, object] = {
        "minimum_positive_interactions": 5,
        "recommendation_k": 1,
        "confidence_alpha": 10.0,
        "observed_confidence": 11.0,
        "implicit_library_alpha": 1.0,
    }
    values.update(changes)
    return ALSFoldInConfig(**values)  # type: ignore[arg-type]


class SpyModel:
    def __init__(
        self,
        item_factors: np.ndarray,
        user_factor: np.ndarray,
        *,
        alpha: float = 1.0,
    ) -> None:
        self.alpha = alpha
        self.item_factors = np.asarray(item_factors, dtype=np.float32)
        self.user_factors = np.asarray([[0.25, -0.5]], dtype=np.float32)
        self._user_factor = np.asarray(user_factor, dtype=np.float32)
        self.recalculate_user_count = 0
        self.fit_count = 0
        self.partial_fit_user_count = 0
        self.partial_fit_item_count = 0
        self.received_userid: object | None = None
        self.received_row: csr_matrix | None = None

    def recalculate_user(self, userid: object, row: csr_matrix) -> np.ndarray:
        self.recalculate_user_count += 1
        self.received_userid = userid
        self.received_row = row.copy()
        return self._user_factor.copy()

    def fit(self, *_args: object, **_kwargs: object) -> None:
        self.fit_count += 1
        raise AssertionError("full fit is forbidden after profile arrival")

    def partial_fit_users(self, *_args: object, **_kwargs: object) -> None:
        self.partial_fit_user_count += 1
        raise AssertionError("partial user update is forbidden")

    def partial_fit_items(self, *_args: object, **_kwargs: object) -> None:
        self.partial_fit_item_count += 1
        raise AssertionError("partial item update is forbidden")


def six_item_model() -> SpyModel:
    return SpyModel(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [0.5, -0.5],
                [2.0, 0.5],
            ],
            dtype=np.float32,
        ),
        np.asarray([1.0, 0.25], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("minimum_positive_interactions", 4, ValueError),
        ("minimum_positive_interactions", True, TypeError),
        ("recommendation_k", 0, ValueError),
        ("recommendation_k", True, TypeError),
        ("confidence_alpha", 0.0, ValueError),
        ("confidence_alpha", float("nan"), ValueError),
        ("observed_confidence", 10.0, ValueError),
        ("observed_confidence", True, TypeError),
        ("implicit_library_alpha", 10.0, ValueError),
        ("implicit_library_alpha", 0.0, ValueError),
    ],
)
def test_config_rejects_contract_changes(
    field: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        config(**{field: value})


def test_exact_five_positive_success() -> None:
    model = six_item_model()
    result = fold_in_new_user(
        model,
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50, 60],
        config(),
    )
    assert result.strategy == "als_fold_in"
    assert result.fold_in_eligible is True
    assert result.profile_movie_ids == (10, 20, 30, 40, 50)
    assert len(result.recommendations) == 1
    assert result.recommendations[0].movie_id == 60
    assert result.profile_csr_shape == (1, 6)
    assert result.profile_csr_nnz == 5


def test_four_positive_strategy_is_fallback_and_fold_in_is_not_called() -> None:
    model = six_item_model()
    decision = decide_fold_in_strategy([10, 20, 30, 40], config())
    assert decision.strategy == "fallback_popularity"
    assert decision.fold_in_eligible is False
    assert decision.positive_interaction_count == 4
    with pytest.raises(
        InsufficientPositiveInteractions, match="fallback_popularity"
    ) as raised:
        fold_in_new_user(
            model,
            [10, 20, 30, 40],
            [10, 20, 30, 40, 50, 60],
            config(),
        )
    assert raised.value.strategy == "fallback_popularity"
    assert raised.value.fold_in_eligible is False
    assert model.recalculate_user_count == 0


def test_duplicate_is_rejected_before_threshold_can_be_satisfied() -> None:
    model = six_item_model()
    with pytest.raises(DuplicatePositiveInteraction, match="duplicates"):
        fold_in_new_user(
            model,
            [10, 20, 30, 40, 40],
            [10, 20, 30, 40, 50, 60],
            config(),
        )
    assert model.recalculate_user_count == 0


def test_empty_profile_is_rejected() -> None:
    with pytest.raises(EmptyPositiveInteractions, match="empty"):
        decide_fold_in_strategy([], config())


@pytest.mark.parametrize("value", [True, False, 1.5, "10", None])
def test_invalid_profile_item_is_rejected(value: object) -> None:
    profile = [10, 20, 30, 40, value]
    with pytest.raises(InvalidProfileItem):
        fold_in_new_user(
            six_item_model(),
            profile,  # type: ignore[arg-type]
            [10, 20, 30, 40, 50, 60],
            config(),
        )


def test_negative_profile_item_is_rejected() -> None:
    with pytest.raises(InvalidProfileItem, match="non-negative"):
        fold_in_new_user(
            six_item_model(),
            [10, 20, 30, 40, -1],
            [10, 20, 30, 40, 50, 60],
            config(),
        )


def test_unknown_item_is_rejected_without_silent_drop() -> None:
    model = six_item_model()
    with pytest.raises(UnknownProfileItem, match="999"):
        fold_in_new_user(
            model,
            [10, 20, 30, 40, 999],
            [10, 20, 30, 40, 50, 60],
            config(),
        )
    assert model.recalculate_user_count == 0


def test_confidence_row_is_new_canonical_float32_csr_with_eleven() -> None:
    mapping = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5}
    mapping_before = mapping.copy()
    row_one = build_fold_in_confidence_row([50, 10, 40, 20, 30], mapping, 6, config())
    row_two = build_fold_in_confidence_row([10, 20, 30, 40, 50], mapping, 6, config())
    assert isspmatrix_csr(row_one)
    assert row_one is not row_two
    assert row_one.shape == (1, 6)
    assert row_one.nnz == 5
    assert row_one.dtype == np.dtype("float32")
    assert row_one.has_canonical_format
    assert row_one.indices.tolist() == [0, 1, 2, 3, 4]
    assert np.equal(row_one.data, 11.0).all()
    assert (row_one != row_two).nnz == 0
    assert mapping == mapping_before


def test_spy_proves_one_transient_row_and_no_mutating_calls() -> None:
    model = six_item_model()
    fold_in_new_user(
        model,
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50, 60],
        config(),
    )
    assert model.recalculate_user_count == 1
    assert model.received_userid == 0
    assert model.received_row is not None
    assert model.received_row.shape == (1, 6)
    assert model.received_row.nnz == 5
    assert np.equal(model.received_row.data, 11.0).all()
    assert model.alpha == 1.0
    assert model.fit_count == 0
    assert model.partial_fit_user_count == 0
    assert model.partial_fit_item_count == 0


def test_model_library_alpha_must_be_one() -> None:
    model = six_item_model()
    model.alpha = 10.0
    with pytest.raises(ValueError, match="alpha"):
        fold_in_new_user(
            model,
            [10, 20, 30, 40, 50],
            [10, 20, 30, 40, 50, 60],
            config(),
        )
    assert model.recalculate_user_count == 0


def test_input_order_invariance_and_input_immutability() -> None:
    first = [10, 20, 30, 40, 50]
    second = [50, 30, 10, 40, 20]
    first_before = first.copy()
    second_before = second.copy()
    first_model = six_item_model()
    second_model = six_item_model()
    first_result = fold_in_new_user(
        first_model, first, [10, 20, 30, 40, 50, 60], config()
    )
    second_result = fold_in_new_user(
        second_model, second, [10, 20, 30, 40, 50, 60], config()
    )
    assert first == first_before
    assert second == second_before
    assert first_result.profile_movie_ids == second_result.profile_movie_ids
    assert np.array_equal(first_result.user_factor, second_result.user_factor)
    assert first_result.recommendations == second_result.recommendations
    assert first_model.received_row is not None
    assert second_model.received_row is not None
    assert (first_model.received_row != second_model.received_row).nnz == 0


def test_handcrafted_full_catalog_ranking_filters_profile_retains_negative_and_ties() -> (
    None
):
    item_ids = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    item_factors = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [5.0, 0.0],
            [0.5, 0.0],
            [0.5, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    model = SpyModel(item_factors, np.asarray([1.0, 0.0], dtype=np.float32))
    result = fold_in_new_user(
        model,
        [10, 20, 30, 40, 50],
        item_ids,
        config(recommendation_k=4),
    )
    assert [row.movie_id for row in result.recommendations] == [90, 60, 70, 80]
    assert [row.score for row in result.recommendations] == [1.0, 0.5, 0.5, -1.0]
    assert [row.rank for row in result.recommendations] == [1, 2, 3, 4]
    assert len({row.movie_id for row in result.recommendations}) == 4
    assert not set(result.profile_movie_ids).intersection(
        row.movie_id for row in result.recommendations
    )


def test_nonfinite_candidate_is_not_returned_but_finite_negative_is() -> None:
    factors = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    result = fold_in_new_user(
        SpyModel(factors, np.asarray([1.0, 0.0], dtype=np.float32)),
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50, 60, 70],
        config(recommendation_k=2),
    )
    assert [row.score for row in result.recommendations] == [1.0, -1.0]


def test_short_complete_catalog_fails_instead_of_mixing_fallback() -> None:
    model = six_item_model()
    with pytest.raises(FoldInError, match="recommendation count"):
        fold_in_new_user(
            model,
            [10, 20, 30, 40, 50],
            [10, 20, 30, 40, 50, 60],
            config(recommendation_k=2),
        )


def _closed_form_fixture() -> tuple[np.ndarray, list[int], list[int]]:
    factors = np.asarray(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.25],
            [1.0, 1.0, -0.5],
            [-0.5, 0.75, 1.0],
            [0.25, -1.0, 0.5],
            [1.5, 0.25, -0.25],
            [-1.0, 0.5, 0.75],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    return factors, list(range(10, 90, 10)), [10, 20, 30, 40, 50]


def _official_model(item_factors: np.ndarray) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors=item_factors.shape[1],
        regularization=0.01,
        alpha=1.0,
        dtype=np.float32,
        iterations=1,
        num_threads=1,
        random_state=42,
    )
    model.item_factors = item_factors.copy()
    model.user_factors = np.zeros((1, item_factors.shape[1]), dtype=np.float32)
    return model


def _manual_solution(
    factors: np.ndarray,
    profile_indices: list[int],
    *,
    variant: str = "correct",
) -> np.ndarray:
    y = factors.astype(np.float64)
    liked = y[profile_indices]
    confidence = 11.0
    regularization = 0.01
    base = y.T @ y
    adjustment_weight = confidence - 1.0
    rhs_weight = confidence
    if variant == "omit_regularization":
        regularization = 0.0
    elif variant == "confidence_not_minus_one":
        adjustment_weight = confidence
    elif variant == "rhs_confidence_minus_one":
        rhs_weight = confidence - 1.0
    elif variant == "liked_only_gram":
        base = liked.T @ liked
    elif variant != "correct":
        raise AssertionError(f"unknown variant: {variant}")
    matrix = (
        base
        + regularization * np.eye(y.shape[1], dtype=np.float64)
        + adjustment_weight * (liked.T @ liked)
    )
    rhs = rhs_weight * liked.sum(axis=0)
    return np.linalg.solve(matrix, rhs)


def test_installed_fold_in_matches_independent_closed_form() -> None:
    factors, item_ids, profile = _closed_form_fixture()
    model = _official_model(factors)
    result = fold_in_new_user(
        model,
        profile,
        item_ids,
        config(recommendation_k=3),
    )
    expected = _manual_solution(factors, [0, 1, 2, 3, 4])
    assert np.allclose(result.user_factor, expected, rtol=1e-5, atol=1e-5)
    manual_scores = factors.astype(np.float64) @ expected
    expected_ranking = sorted(
        (
            (item_ids[index], float(score))
            for index, score in enumerate(manual_scores)
            if item_ids[index] not in profile
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )[:3]
    assert [row.movie_id for row in result.recommendations] == [
        movie_id for movie_id, _ in expected_ranking
    ]
    assert np.allclose(
        [row.score for row in result.recommendations],
        [score for _, score in expected_ranking],
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "variant",
    [
        "omit_regularization",
        "confidence_not_minus_one",
        "rhs_confidence_minus_one",
        "liked_only_gram",
    ],
)
def test_closed_form_fixture_kills_equation_mutations(variant: str) -> None:
    factors, item_ids, profile = _closed_form_fixture()
    production = fold_in_new_user(
        _official_model(factors),
        profile,
        item_ids,
        config(recommendation_k=3),
    ).user_factor
    mutated = _manual_solution(factors, [0, 1, 2, 3, 4], variant=variant)
    assert not np.allclose(production, mutated, rtol=1e-5, atol=1e-5)


def test_original_model_factors_remain_byte_identical() -> None:
    factors, item_ids, profile = _closed_form_fixture()
    model = _official_model(factors)
    item_before = hashlib.sha256(model.item_factors.tobytes()).hexdigest()
    user_before = hashlib.sha256(model.user_factors.tobytes()).hexdigest()
    item_copy = model.item_factors.copy()
    user_copy = model.user_factors.copy()
    fold_in_new_user(model, profile, item_ids, config(recommendation_k=3))
    assert hashlib.sha256(model.item_factors.tobytes()).hexdigest() == item_before
    assert hashlib.sha256(model.user_factors.tobytes()).hexdigest() == user_before
    assert np.array_equal(model.item_factors, item_copy)
    assert np.array_equal(model.user_factors, user_copy)


def test_returned_user_factor_is_not_writeable() -> None:
    result = fold_in_new_user(
        six_item_model(),
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50, 60],
        config(),
    )
    assert result.user_factor.flags.writeable is False
    with pytest.raises(ValueError):
        result.user_factor[0] = 99.0


def test_inference_source_has_no_file_fit_partial_update_or_fallback_ranking() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "packages/recommender_core/src/recommender_core/fold_in.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "model.fit(",
        "partial_fit_users(",
        "partial_fit_items(",
        "read_parquet(",
        "read_csv(",
        "write_text(",
        "to_csv(",
        "pickle",
    )
    for marker in forbidden:
        assert marker not in source
    assert "fallback_popularity" in source
    assert "recalculate_user(0, row)" in source


def test_dataclass_replace_cannot_lower_threshold() -> None:
    with pytest.raises(ValueError, match="must equal 5"):
        replace(config(), minimum_positive_interactions=4)
