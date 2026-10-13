export type Credentials = { email: string; password: string };

export type User = {
  id: string;
  email: string;
  created_at: string;
  is_active: boolean;
};

export type Movie = {
  movie_id: number;
  title: string;
  release_year: number | null;
  genres: string[];
};

export type MovieListResponse = { items: Movie[]; next_cursor: string | null };

export type UserRating = {
  movie_id: number;
  rating: number;
  created_at: string;
  updated_at: string;
};

export type UserRatingsResponse = { items: UserRating[] };

export type RatingResponse = {
  movie_id: number;
  rating: number;
  positive_count: number;
  updated_at: string;
};

export type RecommendationItem = {
  movie_id: number;
  title: string;
  genres: string[];
  rank: number;
  score: number;
};

export type RecommendationConfigurationKey =
  | "A0_L1"
  | "A1_L07"
  | "A1_L06";

export type RecommendationMode = "base" | "diversity_only";

export type RecommendationRequest = {
  configuration_key: RecommendationConfigurationKey;
};

export type RecommendationResponse = {
  event_id: string;
  model_version_id: string;
  strategy: string;
  configuration_key: RecommendationConfigurationKey;
  mode: RecommendationMode;
  lambda: number;
  items: RecommendationItem[];
};

export type FeedbackAction = "click" | "like" | "dislike";

export type FeedbackRequest = {
  event_id: string;
  movie_id: number;
  action: FeedbackAction;
};

export type FeedbackResponse = { status: "recorded" };

export type ModelComparisonRow = {
  model: string; metric: string; mean: number; ci_low: number; ci_high: number;
  confidence_level: number; ci_method: string; bootstrap_resamples: number;
  bootstrap_seed: number; bootstrap_unit: string; fit_scope: string;
  target_scope: string; eligible_user_count: number; recommendation_k: number;
  catalog_size: number; evidence_scope: string; source_role: string;
};

export type RerankingRow = {
  configuration_key: string; mode: string; lambda: number; beta: number | null;
  metric: string; observed_value: number; ci_low: number; ci_high: number;
  confidence_level: number; ci_method: string; bootstrap_resamples: number;
  bootstrap_seed: number; bootstrap_unit: string; fit_scope: string;
  target_scope: string; eligible_user_count: number; recommendation_k: number;
  catalog_size: number; evidence_scope: string; source_role: string;
};

export type ExposureRow = {
  configuration_key: string; popularity_group: string; displayed_slot_count: number;
  total_displayed_slot_count: number; displayed_slot_share: number;
  share_sum_tolerance: number; fit_scope: string; target_scope: string;
  evidence_scope: string; source_role: string;
};

export type ActivityGroupRow = {
  configuration_key: string; activity_group: string; activity_group_user_count: number;
  metric: string; observed_value: number; ci_low: number; ci_high: number;
  confidence_level: number; ci_method: string; bootstrap_resamples: number;
  bootstrap_seed: number; bootstrap_unit: string; fit_scope: string;
  target_scope: string; evidence_scope: string; source_role: string;
};

export type FrozenTestRow = {
  configuration_key: string; metric: string; point_estimate: number;
  confidence_interval_status: string; evidence_kind: string; fit_scope: string;
  target_scope: string; eligible_user_count: number; recommendation_k: number;
  evidence_scope: string; source_role: string;
};

export type HypothesisOutcome = {
  hypothesis: string; scientific_outcome: string; engineering_acceptance: string;
  verdict: string; evidence_scope: string; reason: string; source_path: string;
};

export type ExperimentSummaryResponse = {
  protocol_id: string;
  schema_version: number;
  evidence_base_git_sha: string;
  manifest_sha256: string;
  validation: {
    fit_scope: string; target_scope: string; eligible_user_count: number;
    models: ModelComparisonRow[]; reranking: RerankingRow[];
    exposure: ExposureRow[]; activity_groups: ActivityGroupRow[];
  };
  frozen_test: {
    fit_scope: string; target_scope: string; eligible_user_count: number;
    results: FrozenTestRow[];
  };
  hypotheses: HypothesisOutcome[];
};
