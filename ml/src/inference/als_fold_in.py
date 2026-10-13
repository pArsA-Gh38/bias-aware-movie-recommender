"""Compatibility exports for fold-in behavior owned by recommender_core."""

from recommender_core.fold_in import (
    ALSFoldInConfig,
    DuplicatePositiveInteraction,
    EmptyPositiveInteractions,
    FoldInEligibility,
    FoldInError,
    FoldInRecommendation,
    FoldInResult,
    InsufficientPositiveInteractions,
    InvalidProfileItem,
    UnknownProfileItem,
    build_fold_in_confidence_row,
    decide_fold_in_strategy,
    fold_in_new_user,
)

__all__ = [
    "ALSFoldInConfig",
    "DuplicatePositiveInteraction",
    "EmptyPositiveInteractions",
    "FoldInEligibility",
    "FoldInError",
    "FoldInRecommendation",
    "FoldInResult",
    "InsufficientPositiveInteractions",
    "InvalidProfileItem",
    "UnknownProfileItem",
    "build_fold_in_confidence_row",
    "decide_fold_in_strategy",
    "fold_in_new_user",
]
