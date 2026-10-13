import type {
  ActivityGroupRow, ExperimentSummaryResponse, ExposureRow, FrozenTestRow,
  HypothesisOutcome, ModelComparisonRow, RerankingRow,
} from "./types";

const CONFIGURATIONS = [
  "A0_L1", "A1_L1", "A1_L09", "A1_L08", "A1_L07", "A1_L06",
  "A2_L1", "A2_L09", "A2_L08", "A2_L07", "A2_L06",
  "A3_L1", "A3_L09", "A3_L08", "A3_L07", "A3_L06",
] as const;
const METRICS = [
  "precision_at_10", "recall_at_10", "hit_rate_at_10", "ndcg_at_10",
  "catalog_coverage_at_10", "novelty_at_10", "ild_at_10",
  "exposure_gini_at_10", "long_tail_exposure_at_10",
] as const;
const MODELS = ["Popularity", "ItemKNN-50", "ItemKNN-100", "ItemKNN-200", "ALS-selected-seed42"] as const;
const ACTIVITY_GROUPS = ["low", "medium", "high"] as const;
const POPULARITY_GROUPS = ["head", "mid_tail", "long_tail"] as const;
// API-006 publishes this exact reconciliation tolerance from the immutable EXP-003 manifest/CSV.
const EXPOSURE_SHARE_TOLERANCE = 1e-15;
const FROZEN_CONFIGURATIONS = ["A0_L1", "A1_L07", "A1_L06"] as const;
const HYPOTHESES = ["H1", "H2", "H3", "H4"] as const;

type RecordValue = Record<string, unknown>;

function object(value: unknown, label: string): RecordValue {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`Invalid ${label}`);
  return value as RecordValue;
}
function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`Invalid ${label}`);
  return value;
}
function string(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) throw new Error(`Invalid ${label}`);
  return value;
}
function number(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`Invalid ${label}`);
  return value;
}
function integer(value: unknown, label: string): number {
  const parsed = number(value, label);
  if (!Number.isInteger(parsed)) throw new Error(`Invalid ${label}`);
  return parsed;
}
function nullableNumber(value: unknown, label: string): number | null {
  return value === null ? null : number(value, label);
}
function exact(value: unknown, expected: unknown, label: string): void {
  if (value !== expected) throw new Error(`Invalid ${label}`);
}
function fields(row: RecordValue, numeric: readonly string[], integers: readonly string[], strings: readonly string[]) {
  numeric.forEach((key) => number(row[key], key));
  integers.forEach((key) => integer(row[key], key));
  strings.forEach((key) => string(row[key], key));
}
function exactKeys(row: RecordValue, expected: readonly string[], label: string) {
  const actual = Object.keys(row).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) throw new Error(`Invalid ${label} fields`);
}
function uniqueExact(rows: readonly unknown[], expected: readonly string[], identity: (row: RecordValue) => string, label: string) {
  if (rows.length !== expected.length) throw new Error(`Invalid ${label} cardinality`);
  const identities = rows.map((row) => identity(object(row, label)));
  if (new Set(identities).size !== identities.length || identities.some((value, index) => value !== expected[index])) {
    throw new Error(`Invalid ${label} identities`);
  }
}

function validateModel(rowValue: unknown): ModelComparisonRow {
  const row = object(rowValue, "model row");
  exactKeys(row, ["model", "metric", "mean", "ci_low", "ci_high", "confidence_level", "ci_method", "bootstrap_resamples", "bootstrap_seed", "bootstrap_unit", "fit_scope", "target_scope", "eligible_user_count", "recommendation_k", "catalog_size", "evidence_scope", "source_role"], "model row");
  fields(row, ["mean", "ci_low", "ci_high", "confidence_level"], ["bootstrap_resamples", "bootstrap_seed", "eligible_user_count", "recommendation_k", "catalog_size"], ["model", "metric", "ci_method", "bootstrap_unit", "fit_scope", "target_scope", "evidence_scope", "source_role"]);
  exact(row.fit_scope, "train", "model fit scope"); exact(row.target_scope, "validation", "model target scope"); exact(row.eligible_user_count, 480, "model population");
  return row as ModelComparisonRow;
}
function expectedConfiguration(key: string) {
  const mode = key.startsWith("A0") ? "base" : key.startsWith("A1") ? "diversity_only" : key.startsWith("A2") ? "popularity_only" : "combined";
  const lambda = key.endsWith("L09") ? 0.9 : key.endsWith("L08") ? 0.8 : key.endsWith("L07") ? 0.7 : key.endsWith("L06") ? 0.6 : 1;
  const beta = key.startsWith("A0") ? null : key.startsWith("A1") ? 1 : key.startsWith("A2") ? 0 : 0.5;
  return { mode, lambda, beta };
}
function validateReranking(rowValue: unknown): RerankingRow {
  const row = object(rowValue, "reranking row");
  exactKeys(row, ["configuration_key", "mode", "lambda", "beta", "metric", "observed_value", "ci_low", "ci_high", "confidence_level", "ci_method", "bootstrap_resamples", "bootstrap_seed", "bootstrap_unit", "fit_scope", "target_scope", "eligible_user_count", "recommendation_k", "catalog_size", "evidence_scope", "source_role"], "reranking row");
  fields(row, ["lambda", "observed_value", "ci_low", "ci_high", "confidence_level"], ["bootstrap_resamples", "bootstrap_seed", "eligible_user_count", "recommendation_k", "catalog_size"], ["configuration_key", "mode", "metric", "ci_method", "bootstrap_unit", "fit_scope", "target_scope", "evidence_scope", "source_role"]);
  nullableNumber(row.beta, "beta"); exact(row.fit_scope, "train", "reranking fit scope"); exact(row.target_scope, "validation", "reranking target scope"); exact(row.eligible_user_count, 480, "reranking population");
  const expected = expectedConfiguration(row.configuration_key as string);
  exact(row.mode, expected.mode, "mode"); exact(row.lambda, expected.lambda, "lambda"); exact(row.beta, expected.beta, "beta");
  return row as RerankingRow;
}
function validateExposure(rowValue: unknown): ExposureRow {
  const row = object(rowValue, "exposure row");
  exactKeys(row, ["configuration_key", "popularity_group", "displayed_slot_count", "total_displayed_slot_count", "displayed_slot_share", "share_sum_tolerance", "fit_scope", "target_scope", "evidence_scope", "source_role"], "exposure row");
  fields(row, ["displayed_slot_share", "share_sum_tolerance"], ["displayed_slot_count", "total_displayed_slot_count"], ["configuration_key", "popularity_group", "fit_scope", "target_scope", "evidence_scope", "source_role"]);
  exact(row.fit_scope, "train", "exposure fit scope"); exact(row.target_scope, "validation", "exposure target scope");
  if ((row.displayed_slot_count as number) < 0) throw new Error("Invalid exposure count");
  if ((row.total_displayed_slot_count as number) <= 0) throw new Error("Invalid exposure total");
  if ((row.displayed_slot_share as number) < 0 || (row.displayed_slot_share as number) > 1) throw new Error("Invalid exposure share");
  exact(row.share_sum_tolerance, EXPOSURE_SHARE_TOLERANCE, "exposure share tolerance");
  return row as ExposureRow;
}

function validateExposureReconciliation(rows: ExposureRow[]): void {
  for (const configuration of CONFIGURATIONS) {
    const buckets = rows.filter((row) => row.configuration_key === configuration);
    const total = buckets[0].total_displayed_slot_count;
    if (buckets.some((row) => row.total_displayed_slot_count !== total)) throw new Error("Inconsistent exposure totals");
    if (buckets.reduce((sum, row) => sum + row.displayed_slot_count, 0) !== total) throw new Error("Invalid exposure count sum");
    const shareSum = buckets.reduce((sum, row) => sum + row.displayed_slot_share, 0);
    if (Math.abs(shareSum - 1) > EXPOSURE_SHARE_TOLERANCE) throw new Error("Invalid exposure share sum");
    if (buckets.some((row) => Math.abs(row.displayed_slot_count / total - row.displayed_slot_share) > EXPOSURE_SHARE_TOLERANCE)) {
      throw new Error("Exposure count/share mismatch");
    }
  }
}
function validateActivity(rowValue: unknown): ActivityGroupRow {
  const row = object(rowValue, "activity row");
  exactKeys(row, ["configuration_key", "activity_group", "activity_group_user_count", "metric", "observed_value", "ci_low", "ci_high", "confidence_level", "ci_method", "bootstrap_resamples", "bootstrap_seed", "bootstrap_unit", "fit_scope", "target_scope", "evidence_scope", "source_role"], "activity row");
  fields(row, ["observed_value", "ci_low", "ci_high", "confidence_level"], ["activity_group_user_count", "bootstrap_resamples", "bootstrap_seed"], ["configuration_key", "activity_group", "metric", "ci_method", "bootstrap_unit", "fit_scope", "target_scope", "evidence_scope", "source_role"]);
  exact(row.fit_scope, "train", "activity fit scope"); exact(row.target_scope, "validation", "activity target scope");
  const counts: Record<string, number> = { low: 9, medium: 50, high: 421 };
  exact(row.activity_group_user_count, counts[row.activity_group as string], "activity cohort");
  return row as ActivityGroupRow;
}
function validateFrozen(rowValue: unknown): FrozenTestRow {
  const row = object(rowValue, "frozen row");
  exactKeys(row, ["configuration_key", "metric", "point_estimate", "confidence_interval_status", "evidence_kind", "fit_scope", "target_scope", "eligible_user_count", "recommendation_k", "evidence_scope", "source_role"], "frozen row");
  fields(row, ["point_estimate"], ["eligible_user_count", "recommendation_k"], ["configuration_key", "metric", "confidence_interval_status", "evidence_kind", "fit_scope", "target_scope", "evidence_scope", "source_role"]);
  exact(row.fit_scope, "train+validation", "frozen fit scope"); exact(row.target_scope, "test", "frozen target scope"); exact(row.eligible_user_count, 1114, "frozen population");
  return row as FrozenTestRow;
}
function validateHypothesis(rowValue: unknown): HypothesisOutcome {
  const row = object(rowValue, "hypothesis row");
  exactKeys(row, ["hypothesis", "scientific_outcome", "engineering_acceptance", "verdict", "evidence_scope", "reason", "source_path"], "hypothesis row");
  fields(row, [], [], ["hypothesis", "scientific_outcome", "engineering_acceptance", "verdict", "evidence_scope", "reason", "source_path"]);
  const contracts: Record<string, readonly [string, string, string, string]> = {
    H1: ["rejected", "accepted_evidence", "rejected", "validation"],
    H2: ["accepted", "accepted_evidence", "accepted", "validation"],
    H3: ["not_evaluable", "accepted_evidence", "not evaluable", "validation"],
    H4: ["accepted", "PASS", "accepted", "engineering_cpu_latency"],
  };
  const expected = contracts[row.hypothesis as string];
  [row.scientific_outcome, row.engineering_acceptance, row.verdict, row.evidence_scope].forEach((value, index) => exact(value, expected[index], "hypothesis outcome"));
  return row as HypothesisOutcome;
}

export function parseExperimentSummary(value: unknown): ExperimentSummaryResponse {
  const root = object(value, "experiment response");
  exactKeys(root, ["protocol_id", "schema_version", "evidence_base_git_sha", "manifest_sha256", "validation", "frozen_test", "hypotheses"], "experiment response");
  exact(root.protocol_id, "exp003-stored-evidence-publication-v1", "protocol");
  exact(root.schema_version, 1, "schema version");
  exact(root.evidence_base_git_sha, "8d3d2e27b81676648b982bb3e0f656c6f8d36888", "evidence base");
  exact(root.manifest_sha256, "2015148fba12d453f92fa275e01ee735e5460337fc32fa7cb8fd6e67aa0bb63c", "manifest");
  const validation = object(root.validation, "validation");
  exactKeys(validation, ["fit_scope", "target_scope", "eligible_user_count", "models", "reranking", "exposure", "activity_groups"], "validation");
  exact(validation.fit_scope, "train", "validation fit scope"); exact(validation.target_scope, "validation", "validation target scope"); exact(validation.eligible_user_count, 480, "validation population");
  const frozenTest = object(root.frozen_test, "frozen test");
  exactKeys(frozenTest, ["fit_scope", "target_scope", "eligible_user_count", "results"], "frozen test");
  exact(frozenTest.fit_scope, "train+validation", "frozen-test fit scope"); exact(frozenTest.target_scope, "test", "frozen-test target scope"); exact(frozenTest.eligible_user_count, 1114, "frozen-test population");
  const rawGroups = [array(validation.models, "models"), array(validation.reranking, "reranking"), array(validation.exposure, "exposure"), array(validation.activity_groups, "activity groups"), array(frozenTest.results, "frozen results"), array(root.hypotheses, "hypotheses")];
  const empty = rawGroups.every((rows) => rows.length === 0);
  if (!empty && rawGroups.some((rows) => rows.length === 0)) throw new Error("Partial experiment evidence");
  if (empty) return root as ExperimentSummaryResponse;

  const [modelsRaw, rerankingRaw, exposureRaw, activityRaw, frozenRaw, hypothesesRaw] = rawGroups;
  uniqueExact(modelsRaw, MODELS.flatMap((model) => METRICS.map((metric) => `${model}\u0000${metric}`)), (row) => `${row.model}\u0000${row.metric}`, "models");
  uniqueExact(rerankingRaw, CONFIGURATIONS.flatMap((key) => METRICS.map((metric) => `${key}\u0000${metric}`)), (row) => `${row.configuration_key}\u0000${row.metric}`, "reranking");
  uniqueExact(exposureRaw, CONFIGURATIONS.flatMap((key) => POPULARITY_GROUPS.map((group) => `${key}\u0000${group}`)), (row) => `${row.configuration_key}\u0000${row.popularity_group}`, "exposure");
  uniqueExact(activityRaw, CONFIGURATIONS.flatMap((key) => ACTIVITY_GROUPS.flatMap((group) => METRICS.map((metric) => `${key}\u0000${group}\u0000${metric}`))), (row) => `${row.configuration_key}\u0000${row.activity_group}\u0000${row.metric}`, "activity");
  uniqueExact(frozenRaw, FROZEN_CONFIGURATIONS.flatMap((key) => METRICS.map((metric) => `${key}\u0000${metric}`)), (row) => `${row.configuration_key}\u0000${row.metric}`, "frozen results");
  uniqueExact(hypothesesRaw, HYPOTHESES, (row) => String(row.hypothesis), "hypotheses");

  const parsed = root as ExperimentSummaryResponse;
  parsed.validation.models = modelsRaw.map(validateModel);
  parsed.validation.reranking = rerankingRaw.map(validateReranking);
  parsed.validation.exposure = exposureRaw.map(validateExposure);
  validateExposureReconciliation(parsed.validation.exposure);
  parsed.validation.activity_groups = activityRaw.map(validateActivity);
  parsed.frozen_test.results = frozenRaw.map(validateFrozen);
  parsed.hypotheses = hypothesesRaw.map(validateHypothesis);
  return parsed;
}
