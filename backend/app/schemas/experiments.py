"""Public schemas for the immutable EXP-003 presentation summary."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


FiniteFloat = Field(allow_inf_nan=False)


class ModelComparisonRow(StrictSchema):
    model: str
    metric: str
    mean: float = FiniteFloat
    ci_low: float = FiniteFloat
    ci_high: float = FiniteFloat
    confidence_level: float = FiniteFloat
    ci_method: str
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_unit: str
    fit_scope: str
    target_scope: str
    eligible_user_count: int
    recommendation_k: int
    catalog_size: int
    evidence_scope: str
    source_role: str


class RerankingRow(StrictSchema):
    configuration_key: str
    mode: str
    lambda_: float = Field(
        alias="lambda", serialization_alias="lambda", allow_inf_nan=False
    )
    beta: float | None = FiniteFloat
    metric: str
    observed_value: float = FiniteFloat
    ci_low: float = FiniteFloat
    ci_high: float = FiniteFloat
    confidence_level: float = FiniteFloat
    ci_method: str
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_unit: str
    fit_scope: str
    target_scope: str
    eligible_user_count: int
    recommendation_k: int
    catalog_size: int
    evidence_scope: str
    source_role: str


class ExposureRow(StrictSchema):
    configuration_key: str
    popularity_group: str
    displayed_slot_count: int
    total_displayed_slot_count: int
    displayed_slot_share: float = FiniteFloat
    share_sum_tolerance: float = FiniteFloat
    fit_scope: str
    target_scope: str
    evidence_scope: str
    source_role: str


class ActivityGroupRow(StrictSchema):
    configuration_key: str
    activity_group: str
    activity_group_user_count: int
    metric: str
    observed_value: float = FiniteFloat
    ci_low: float = FiniteFloat
    ci_high: float = FiniteFloat
    confidence_level: float = FiniteFloat
    ci_method: str
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_unit: str
    fit_scope: str
    target_scope: str
    evidence_scope: str
    source_role: str


class FrozenTestRow(StrictSchema):
    configuration_key: str
    metric: str
    point_estimate: float = FiniteFloat
    confidence_interval_status: str
    evidence_kind: str
    fit_scope: str
    target_scope: str
    eligible_user_count: int
    recommendation_k: int
    evidence_scope: str
    source_role: str


class HypothesisOutcome(StrictSchema):
    hypothesis: str
    scientific_outcome: str
    engineering_acceptance: str
    verdict: str
    evidence_scope: str
    reason: str
    source_path: str


class ValidationSummary(StrictSchema):
    fit_scope: Literal["train"]
    target_scope: Literal["validation"]
    eligible_user_count: Literal[480]
    models: tuple[ModelComparisonRow, ...]
    reranking: tuple[RerankingRow, ...]
    exposure: tuple[ExposureRow, ...]
    activity_groups: tuple[ActivityGroupRow, ...]


class FrozenTestSummary(StrictSchema):
    fit_scope: Literal["train+validation"]
    target_scope: Literal["test"]
    eligible_user_count: Literal[1114]
    results: tuple[FrozenTestRow, ...]


class ExperimentSummaryResponse(StrictSchema):
    protocol_id: Literal["exp003-stored-evidence-publication-v1"]
    schema_version: Literal[1]
    evidence_base_git_sha: Literal["8d3d2e27b81676648b982bb3e0f656c6f8d36888"]
    manifest_sha256: Literal[
        "2015148fba12d453f92fa275e01ee735e5460337fc32fa7cb8fd6e67aa0bb63c"
    ]
    validation: ValidationSummary
    frozen_test: FrozenTestSummary
    hypotheses: tuple[HypothesisOutcome, ...]
