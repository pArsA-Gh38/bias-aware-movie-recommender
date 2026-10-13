from __future__ import annotations

import numpy as np
import pytest

from recommender_core import InsufficientCatalogItems, rank_factor_items


def _rank(*, k: int = 3, excluded: set[int] | None = None):
    return rank_factor_items(
        user_factor=np.asarray([1.0, 0.0], dtype=np.float32),
        item_factors=np.asarray(
            [[9.0, 0.0], [0.5, 0.0], [0.5, 1.0], [-1.0, 0.0], [np.nan, 0.0]],
            dtype=np.float32,
        ),
        item_ids=[10, 60, 50, 80, 100],
        excluded_item_indices={0} if excluded is None else excluded,
        k=k,
    )


def test_scoring_filters_seen_and_nonfinite_but_retains_negative_and_ties() -> None:
    rows = _rank()
    assert [(row.rank, row.movie_id, row.score) for row in rows] == [
        (1, 50, 0.5),
        (2, 60, 0.5),
        (3, 80, -1.0),
    ]


def test_score_ties_use_movie_id_ascending() -> None:
    assert [row.movie_id for row in _rank(k=2)] == [50, 60]


@pytest.mark.parametrize("k", [0, -1])
def test_nonpositive_k_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        _rank(k=k)


@pytest.mark.parametrize("k", [True, 1.0, "1"])
def test_noninteger_k_is_rejected(k: object) -> None:
    with pytest.raises(TypeError, match="integer scalar"):
        _rank(k=k)  # type: ignore[arg-type]


def test_short_catalog_fails_closed() -> None:
    with pytest.raises(InsufficientCatalogItems, match="recommendation count"):
        _rank(k=4)


def test_inputs_are_not_mutated() -> None:
    user = np.asarray([1.0, 0.0], dtype=np.float32)
    items = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    user_before = user.tobytes()
    items_before = items.tobytes()
    rank_factor_items(
        user_factor=user,
        item_factors=items,
        item_ids=[20, 10],
        excluded_item_indices=set(),
        k=2,
    )
    assert user.tobytes() == user_before
    assert items.tobytes() == items_before
