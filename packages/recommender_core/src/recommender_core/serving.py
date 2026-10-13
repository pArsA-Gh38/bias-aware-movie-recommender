"""Additive serving operations over the immutable ART-002 v2 runtime."""

from __future__ import annotations

from collections.abc import Sequence

from recommender_core.artifact import ServingRecommendation
from recommender_core.artifact_v2 import (
    DEFAULT_CONFIGURATION_KEY,
    FrozenServingConfiguration,
    ServingBundleV2,
)
from recommender_core.fold_in import build_fold_in_confidence_row


def effective_beta(configuration: FrozenServingConfiguration) -> float:
    """Return the beta actually used by the frozen serving configuration."""
    return 0.5 if configuration.beta is None else configuration.beta


def recommend_new_user(
    bundle: ServingBundleV2,
    positive_movie_ids: Sequence[int],
    configuration_key: str = DEFAULT_CONFIGURATION_KEY,
    *,
    excluded_movie_ids: Sequence[int] = (),
) -> tuple[ServingRecommendation, ...]:
    """Recommend for a fold-in profile while excluding additional rated movies."""
    configuration = bundle._configuration(configuration_key)
    row = build_fold_in_confidence_row(
        positive_movie_ids,
        bundle._item_index,
        len(bundle._catalog_item_ids),
        bundle._fold_in_config,
    )
    factor = bundle._fold_in_model.recalculate_user(0, row)
    excluded = {
        bundle._item_index[int(movie_id)]
        for movie_id in (*positive_movie_ids, *excluded_movie_ids)
        if int(movie_id) in bundle._item_index
    }
    return bundle._recommend_from_base(
        bundle._base_pool(factor, excluded),
        configuration,
    )
