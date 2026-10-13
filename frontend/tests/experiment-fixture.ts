import type { ExperimentSummaryResponse } from "@/lib/api/types";

export const metrics = [
  "precision_at_10", "recall_at_10", "hit_rate_at_10", "ndcg_at_10",
  "catalog_coverage_at_10", "novelty_at_10", "ild_at_10",
  "exposure_gini_at_10", "long_tail_exposure_at_10",
] as const;

type Metric = (typeof metrics)[number];
type ExposureValues = readonly [count: number, share: number];

function metricValues(ndcg: number, longTail: number, ild: number, gini: number, coverage: number): Record<Metric, number> {
  return {
    precision_at_10: 0.101, recall_at_10: 0.202, hit_rate_at_10: 0.303,
    ndcg_at_10: ndcg, catalog_coverage_at_10: coverage, novelty_at_10: 4.404,
    ild_at_10: ild, exposure_gini_at_10: gini, long_tail_exposure_at_10: longTail,
  };
}

export const configurationFixtures = [
  { key: "A0_L1", mode: "base", lambda: 1, beta: null, values: metricValues(0.411, 0.021, 0.531, 0.641, 0.751), exposure: [[4519, 0.9414583333333333], [281, 0.058541666666666665], [0, 0]] },
  { key: "A1_L1", mode: "diversity_only", lambda: 1, beta: 1, values: metricValues(0.411, 0.021, 0.531, 0.641, 0.751), exposure: [[4519, 0.9414583333333333], [281, 0.058541666666666665], [0, 0]] },
  { key: "A1_L09", mode: "diversity_only", lambda: 0.9, beta: 1, values: metricValues(0.412, 0.022, 0.542, 0.652, 0.762), exposure: [[4527, 0.943125], [273, 0.056875], [0, 0]] },
  { key: "A1_L08", mode: "diversity_only", lambda: 0.8, beta: 1, values: metricValues(0.413, 0.023, 0.553, 0.663, 0.773), exposure: [[4537, 0.9452083333333333], [263, 0.05479166666666667], [0, 0]] },
  { key: "A1_L07", mode: "diversity_only", lambda: 0.7, beta: 1, values: metricValues(0.414, 0.024, 0.564, 0.674, 0.784), exposure: [[4530, 0.94375], [270, 0.05625], [0, 0]] },
  { key: "A1_L06", mode: "diversity_only", lambda: 0.6, beta: 1, values: metricValues(0.415, 0.025, 0.575, 0.685, 0.795), exposure: [[4512, 0.94], [288, 0.06], [0, 0]] },
  { key: "A2_L1", mode: "popularity_only", lambda: 1, beta: 0, values: metricValues(0.411, 0.021, 0.531, 0.641, 0.751), exposure: [[4519, 0.9414583333333333], [281, 0.058541666666666665], [0, 0]] },
  { key: "A2_L09", mode: "popularity_only", lambda: 0.9, beta: 0, values: metricValues(0.422, 0.032, 0.582, 0.692, 0.802), exposure: [[4445, 0.9260416666666667], [355, 0.07395833333333333], [0, 0]] },
  { key: "A2_L08", mode: "popularity_only", lambda: 0.8, beta: 0, values: metricValues(0.423, 0.033, 0.593, 0.703, 0.813), exposure: [[4337, 0.9035416666666667], [463, 0.09645833333333333], [0, 0]] },
  { key: "A2_L07", mode: "popularity_only", lambda: 0.7, beta: 0, values: metricValues(0.424, 0.034, 0.604, 0.714, 0.824), exposure: [[4151, 0.8647916666666666], [649, 0.13520833333333335], [0, 0]] },
  { key: "A2_L06", mode: "popularity_only", lambda: 0.6, beta: 0, values: metricValues(0.425, 0.035, 0.615, 0.725, 0.835), exposure: [[3830, 0.7979166666666667], [968, 0.20166666666666666], [2, 0.0004166666666666667]] },
  { key: "A3_L1", mode: "combined", lambda: 1, beta: 0.5, values: metricValues(0.411, 0.021, 0.531, 0.641, 0.751), exposure: [[4519, 0.9414583333333333], [281, 0.058541666666666665], [0, 0]] },
  { key: "A3_L09", mode: "combined", lambda: 0.9, beta: 0.5, values: metricValues(0.432, 0.042, 0.622, 0.732, 0.842), exposure: [[4487, 0.9347916666666667], [313, 0.06520833333333333], [0, 0]] },
  { key: "A3_L08", mode: "combined", lambda: 0.8, beta: 0.5, values: metricValues(0.433, 0.043, 0.633, 0.743, 0.853), exposure: [[4445, 0.9260416666666667], [355, 0.07395833333333333], [0, 0]] },
  { key: "A3_L07", mode: "combined", lambda: 0.7, beta: 0.5, values: metricValues(0.434, 0.044, 0.644, 0.754, 0.864), exposure: [[4395, 0.915625], [405, 0.084375], [0, 0]] },
  { key: "A3_L06", mode: "combined", lambda: 0.6, beta: 0.5, values: metricValues(0.435, 0.045, 0.655, 0.765, 0.875), exposure: [[4307, 0.8972916666666667], [493, 0.10270833333333333], [0, 0]] },
] as const satisfies readonly {
  key: string; mode: string; lambda: number; beta: number | null;
  values: Record<Metric, number>;
  exposure: readonly [ExposureValues, ExposureValues, ExposureValues];
}[];

export const configurations = configurationFixtures.map(({ key }) => key);

export function experimentFixture(): ExperimentSummaryResponse {
  const common = {
    confidence_level: 0.95, ci_method: "percentile", bootstrap_resamples: 1000,
    bootstrap_seed: 42, bootstrap_unit: "user", fit_scope: "train",
    target_scope: "validation", eligible_user_count: 480, recommendation_k: 10,
    catalog_size: 3468, evidence_scope: "validation_reranking", source_role: "accepted_ablation_evidence",
  };
  return {
    protocol_id: "exp003-stored-evidence-publication-v1", schema_version: 1,
    evidence_base_git_sha: "8d3d2e27b81676648b982bb3e0f656c6f8d36888",
    manifest_sha256: "2015148fba12d453f92fa275e01ee735e5460337fc32fa7cb8fd6e67aa0bb63c",
    validation: {
      fit_scope: "train", target_scope: "validation", eligible_user_count: 480,
      models: ["Popularity", "ItemKNN-50", "ItemKNN-100", "ItemKNN-200", "ALS-selected-seed42"].flatMap((model, modelIndex) =>
        metrics.map((metric, metricIndex) => ({
          model, metric, mean: 0.1 + modelIndex / 100 + metricIndex / 1000,
          ci_low: 0.08, ci_high: 0.12, ...common,
          evidence_scope: "validation_model_comparison", source_role: "accepted_stored_evidence",
        }))),
      reranking: configurationFixtures.flatMap(({ key, mode, lambda, beta, values }) => metrics.map((metric) => ({
        configuration_key: key, mode, lambda, beta, metric, observed_value: values[metric],
        ci_low: 0.18, ci_high: 0.22, ...common,
      }))),
      exposure: configurationFixtures.flatMap(({ key, exposure }) =>
        (["head", "mid_tail", "long_tail"] as const).map((popularity_group, index) => ({
          configuration_key: key, popularity_group, displayed_slot_count: exposure[index][0],
          total_displayed_slot_count: 4800, displayed_slot_share: exposure[index][1], share_sum_tolerance: 1e-15,
          fit_scope: "train", target_scope: "validation", evidence_scope: "validation_displayed_top10",
          source_role: "committed_recommendation_and_candidate_evidence",
        }))),
      activity_groups: configurationFixtures.flatMap(({ key }) => (["low", "medium", "high"] as const).flatMap((activity_group) => metrics.map((metric) => ({
        configuration_key: key, activity_group,
        activity_group_user_count: activity_group === "low" ? 9 : activity_group === "medium" ? 50 : 421,
        metric, observed_value: 0.2, ci_low: 0.1, ci_high: 0.3,
        confidence_level: 0.95, ci_method: "percentile", bootstrap_resamples: 1000,
        bootstrap_seed: 42, bootstrap_unit: "user", fit_scope: "train", target_scope: "validation",
        evidence_scope: "validation_activity_group", source_role: "accepted_ablation_activity_evidence",
      })))),
    },
    frozen_test: {
      fit_scope: "train+validation", target_scope: "test", eligible_user_count: 1114,
      results: ["A0_L1", "A1_L07", "A1_L06"].flatMap((configuration_key) => metrics.map((metric) => ({
        configuration_key, metric, point_estimate: 0.2, confidence_interval_status: "unavailable_not_calculated",
        evidence_kind: "confirmatory_point_estimate", fit_scope: "train+validation", target_scope: "test",
        eligible_user_count: 1114, recommendation_k: 10, evidence_scope: "frozen_confirmatory_test",
        source_role: "stored_exp002_aggregate_only",
      }))),
    },
    hypotheses: [
      ["H1", "rejected", "accepted_evidence", "rejected", "validation", "No combined configuration satisfied all frozen H1 conditions."],
      ["H2", "accepted", "accepted_evidence", "accepted", "validation", "Qualifying configurations: A1_L07 and A1_L06."],
      ["H3", "not_evaluable", "accepted_evidence", "not evaluable", "validation", "Dependency on H1 was not met."],
      ["H4", "accepted", "PASS", "accepted", "engineering_cpu_latency", "7.43526535 ms <= 300 ms."],
    ].map(([hypothesis, scientific_outcome, engineering_acceptance, verdict, evidence_scope, reason]) => ({
      hypothesis, scientific_outcome, engineering_acceptance, verdict, evidence_scope, reason,
      source_path: hypothesis === "H4" ? "reports/performance/perf002/v3/perf002_h4_benchmark.json" : "reports/experiments/exp001/exp001_decision.json",
    })),
  } as ExperimentSummaryResponse;
}

export function emptyExperimentFixture(): ExperimentSummaryResponse {
  const fixture = experimentFixture();
  fixture.validation.models = [];
  fixture.validation.reranking = [];
  fixture.validation.exposure = [];
  fixture.validation.activity_groups = [];
  fixture.frozen_test.results = [];
  fixture.hypotheses = [];
  return fixture;
}
