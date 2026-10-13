"""Backend-side dependency proof for the shared runtime package."""

from recommender_core import FactorRecommendation, rank_factor_items

__all__ = ["FactorRecommendation", "rank_factor_items"]
