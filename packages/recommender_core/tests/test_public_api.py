from __future__ import annotations

import ast
from pathlib import Path

import recommender_core


def test_public_exports_are_explicit_and_complete() -> None:
    assert set(recommender_core.__all__) == {
        "ACCEPTED_CONFIGURATION_KEYS",
        "BUNDLE_ID_V2",
        "DEFAULT_CONFIGURATION_KEY",
        "FIT_ELIGIBILITY_MINIMUM",
        "ALSFoldInConfig",
        "ArtifactCompatibilityError",
        "ArtifactError",
        "ArtifactIntegrityError",
        "DuplicatePositiveInteraction",
        "EmptyPositiveInteractions",
        "FactorRecommendation",
        "FitEligibility",
        "FoldInEligibility",
        "FoldInError",
        "FoldInRecommendation",
        "FoldInResult",
        "FrozenServingConfiguration",
        "InsufficientCatalogItems",
        "InsufficientPositiveInteractions",
        "InvalidProfileItem",
        "ItemMetadata",
        "NormalizedRerankingFeatures",
        "RerankedItem",
        "RuntimeInferenceError",
        "ServingBundle",
        "ServingBundleV2",
        "ServingRecommendation",
        "UnknownProfileItem",
        "build_fold_in_confidence_row",
        "decide_fold_in_strategy",
        "fold_in_new_user",
        "effective_beta",
        "greedy_rerank",
        "load_serving_bundle",
        "load_serving_bundle_v2",
        "min_max_normalize_candidate_feature",
        "normalize_reranking_features",
        "rank_factor_items",
        "recommend_new_user",
    }


def test_core_production_imports_have_no_ml_backend_or_research_edges() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src/recommender_core"
    forbidden = {"ml", "backend", "src", "pandas", "implicit"}
    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        roots = {
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        roots.update(
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not roots & forbidden
