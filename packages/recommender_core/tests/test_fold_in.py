from __future__ import annotations

import numpy as np
import pytest

from recommender_core import (
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


def config(*, k: int = 1) -> ALSFoldInConfig:
    return ALSFoldInConfig(5, k, 10.0, 11.0, 1.0)


class SpyModel:
    def __init__(self) -> None:
        self.alpha = 1.0
        self.item_factors = np.asarray(
            [[0.0, 0.0]] * 5 + [[1.0, 0.0], [0.5, 0.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        self.user_factors = np.asarray([[0.25, -0.5]], dtype=np.float32)
        self.recalculate_user_count = 0
        self.fit_count = 0
        self.partial_fit_user_count = 0
        self.partial_fit_item_count = 0

    def recalculate_user(self, _userid: int, _row: object) -> np.ndarray:
        self.recalculate_user_count += 1
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def fit(self, *_args: object, **_kwargs: object) -> None:
        self.fit_count += 1
        raise AssertionError("fit forbidden")

    def partial_fit_users(self, *_args: object, **_kwargs: object) -> None:
        self.partial_fit_user_count += 1
        raise AssertionError("partial fit forbidden")

    def partial_fit_items(self, *_args: object, **_kwargs: object) -> None:
        self.partial_fit_item_count += 1
        raise AssertionError("partial fit forbidden")


CATALOG = [10, 20, 30, 40, 50, 60, 70, 80]
PROFILE = [10, 20, 30, 40, 50]


def test_exact_five_positive_fold_in_is_non_mutating_and_complete_catalog() -> None:
    model = SpyModel()
    before = (model.item_factors.tobytes(), model.user_factors.tobytes())
    result = fold_in_new_user(model, PROFILE, CATALOG, config(k=3))
    assert [row.movie_id for row in result.recommendations] == [60, 70, 80]
    assert [row.score for row in result.recommendations] == [1.0, 0.5, -1.0]
    assert result.profile_movie_ids == tuple(PROFILE)
    assert result.user_factor.dtype == np.float32
    assert result.user_factor.flags.writeable is False
    assert (model.item_factors.tobytes(), model.user_factors.tobytes()) == before
    assert model.recalculate_user_count == 1
    assert (
        model.fit_count
        == model.partial_fit_user_count
        == model.partial_fit_item_count
        == 0
    )


def test_confidence_row_is_canonical_float32_with_frozen_confidence() -> None:
    row = build_fold_in_confidence_row(
        PROFILE,
        {movie_id: index for index, movie_id in enumerate(CATALOG)},
        len(CATALOG),
        config(),
    )
    assert row.shape == (1, 8)
    assert row.nnz == 5
    assert row.dtype == np.float32
    assert row.has_canonical_format
    assert row.indices.tolist() == [0, 1, 2, 3, 4]
    assert row.data.tolist() == [11.0] * 5


def test_four_positive_profile_routes_to_fallback_without_fold_in() -> None:
    model = SpyModel()
    decision = decide_fold_in_strategy(PROFILE[:4], config())
    assert decision.strategy == "fallback_popularity"
    assert decision.fold_in_eligible is False
    with pytest.raises(InsufficientPositiveInteractions, match="fallback_popularity"):
        fold_in_new_user(model, PROFILE[:4], CATALOG, config())
    assert model.recalculate_user_count == 0


@pytest.mark.parametrize(
    ("profile", "error"),
    [
        ([], EmptyPositiveInteractions),
        ([10, 20, 30, 40, 40], DuplicatePositiveInteraction),
        ([10, 20, 30, 40, True], InvalidProfileItem),
        ([10, 20, 30, 40, 999], UnknownProfileItem),
    ],
)
def test_required_profile_error_taxonomy(
    profile: list[object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        fold_in_new_user(SpyModel(), profile, CATALOG, config())  # type: ignore[arg-type]


def test_short_catalog_uses_fold_in_error_taxonomy() -> None:
    with pytest.raises(FoldInError, match="recommendation count"):
        fold_in_new_user(SpyModel(), PROFILE, CATALOG, config(k=4))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_positive_interactions", 4),
        ("observed_confidence", 10.0),
        ("implicit_library_alpha", 10.0),
    ],
)
def test_frozen_config_semantics_cannot_change(field: str, value: object) -> None:
    values = {
        "minimum_positive_interactions": 5,
        "recommendation_k": 1,
        "confidence_alpha": 10.0,
        "observed_confidence": 11.0,
        "implicit_library_alpha": 1.0,
    }
    values[field] = value
    with pytest.raises(ValueError):
        ALSFoldInConfig(**values)  # type: ignore[arg-type]
