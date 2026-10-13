"""Compatibility exports for normalization owned by recommender_core."""

from recommender_core.normalization import (
    NormalizedRerankingFeatures,
    min_max_normalize_candidate_feature,
    normalize_reranking_features,
)

__all__ = [
    "NormalizedRerankingFeatures",
    "min_max_normalize_candidate_feature",
    "normalize_reranking_features",
]
