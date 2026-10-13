"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useInsertionEffect, useMemo, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api/client";
import type { ExperimentSummaryResponse, ExposureRow, RerankingRow } from "@/lib/api/types";

type ViewState =
  | { kind: "loading" }
  | { kind: "success"; data: ExperimentSummaryResponse }
  | { kind: "empty" }
  | { kind: "unavailable" }
  | { kind: "error" }
  | { kind: "redirecting" };

const CONFIGURATIONS = [
  "A0_L1", "A1_L1", "A1_L09", "A1_L08", "A1_L07", "A1_L06",
  "A2_L1", "A2_L09", "A2_L08", "A2_L07", "A2_L06",
  "A3_L1", "A3_L09", "A3_L08", "A3_L07", "A3_L06",
] as const;
const MODE_STYLE: Record<string, { color: string; dash?: string; shape: "circle" | "square" | "diamond" | "triangle" }> = {
  base: { color: "#555555", shape: "circle" },
  diversity_only: { color: "#0072b2", shape: "square" },
  popularity_only: { color: "#d55e00", dash: "8 4", shape: "diamond" },
  combined: { color: "#009e73", dash: "3 3", shape: "triangle" },
};
const MODE_LABELS: Record<string, string> = {
  base: "Base — circle, solid", diversity_only: "Diversity only — square, solid",
  popularity_only: "Popularity only — diamond, dashed", combined: "Combined — triangle, dotted",
};

type ExperimentRequest = {
  controller: AbortController;
  promise: Promise<ExperimentSummaryResponse>;
};

const experimentConsumers = new Set<symbol>();
let experimentRequest: ExperimentRequest | null = null;

function registerExperimentConsumer(consumer: symbol): () => void {
  experimentConsumers.add(consumer);
  return () => {
    if (!experimentConsumers.delete(consumer) || experimentConsumers.size !== 0) return;
    const abandoned = experimentRequest;
    if (!abandoned || experimentRequest !== abandoned) return;
    experimentRequest = null;
    abandoned.controller.abort("No dashboard consumers remain");
  };
}

function getExperimentSummary(): Promise<ExperimentSummaryResponse> {
  if (experimentRequest) return experimentRequest.promise;
  const controller = new AbortController();
  const request = { controller, promise: api.experimentSummary(controller.signal) };
  experimentRequest = request;
  const clear = () => { if (experimentRequest === request) experimentRequest = null; };
  void request.promise.then(clear, clear);
  return request.promise;
}

function metricRows(rows: RerankingRow[], metric: string): Map<string, RerankingRow> {
  return new Map(rows.filter((row) => row.metric === metric).map((row) => [row.configuration_key, row]));
}
function extent(values: number[]): [number, number] {
  const minimum = Math.min(...values); const maximum = Math.max(...values);
  const padding = maximum === minimum ? Math.max(Math.abs(minimum) * 0.05, 0.01) : (maximum - minimum) * 0.08;
  return [minimum - padding, maximum + padding];
}
function scale(value: number, range: [number, number], start: number, end: number) {
  return start + ((value - range[0]) / (range[1] - range[0])) * (end - start);
}
function format(value: number) { return value.toLocaleString("en-US", { maximumSignificantDigits: 5 }); }

function collisionGroups(rows: RerankingRow[], xValue: (row: RerankingRow) => number, yValue: (row: RerankingRow) => number) {
  const groups = new Map<string, RerankingRow[]>();
  for (const row of rows) {
    const key = `${xValue(row)}\u0000${yValue(row)}`;
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  return groups;
}

function Marker({ x, y, row, collisionIndex = 0, collisionCount = 1 }: { x: number; y: number; row: RerankingRow; collisionIndex?: number; collisionCount?: number }) {
  const style = MODE_STYLE[row.mode];
  const size = 4.5 + collisionIndex * 3;
  const presentation = collisionCount > 1 ? { fill: "none", stroke: style.color, strokeWidth: 2 } : { fill: style.color };
  if (style.shape === "square") return <rect x={x - size} y={y - size} width={size * 2} height={size * 2} {...presentation} />;
  if (style.shape === "diamond") return <path d={`M ${x} ${y - size} L ${x + size} ${y} L ${x} ${y + size} L ${x - size} ${y} Z`} {...presentation} />;
  if (style.shape === "triangle") return <path d={`M ${x} ${y - size} L ${x + size} ${y + size * .8} L ${x - size} ${y + size * .8} Z`} {...presentation} />;
  return <circle cx={x} cy={y} r={size} {...presentation} />;
}

function Axes({ xLabel, yLabel, xRange, yRange }: { xLabel: string; yLabel: string; xRange: [number, number]; yRange: [number, number] }) {
  return <>
    <line x1="58" y1="18" x2="58" y2="238" className="experiment-chart__axis" />
    <line x1="58" y1="238" x2="500" y2="238" className="experiment-chart__axis" />
    <text x="279" y="276" textAnchor="middle">{xLabel}</text>
    <text x="16" y="128" textAnchor="middle" transform="rotate(-90 16 128)">{yLabel}</text>
    <text x="58" y="255" textAnchor="middle">{format(xRange[0])}</text>
    <text x="500" y="255" textAnchor="middle">{format(xRange[1])}</text>
    <text x="50" y="238" textAnchor="end">{format(yRange[0])}</text>
    <text x="50" y="23" textAnchor="end">{format(yRange[1])}</text>
  </>;
}

function Legend() {
  return <ul className="experiment-chart__legend" aria-label="Series legend">
    {Object.entries(MODE_LABELS).map(([mode, label]) => <li key={mode}><span className={`experiment-legend-marker experiment-legend-marker--${MODE_STYLE[mode].shape}`} style={{ "--series": MODE_STYLE[mode].color } as React.CSSProperties} aria-hidden="true" />{label}</li>)}
  </ul>;
}

function ScatterPanel({ rows, yMetric, title, yLabel }: { rows: RerankingRow[]; yMetric: string; title: string; yLabel: string }) {
  const xRows = metricRows(rows, "ndcg_at_10"); const yRows = metricRows(rows, yMetric);
  const points = CONFIGURATIONS.map((key) => ({ x: xRows.get(key)!, y: yRows.get(key)! }));
  const collisions = collisionGroups(points.map(({ x }) => x), (row) => row.observed_value, (row) => yRows.get(row.configuration_key)!.observed_value);
  const xRange = extent(points.map((point) => point.x.observed_value));
  const yRange = extent(points.map((point) => point.y.observed_value));
  const id = title.toLowerCase().replace(/[^a-z]+/g, "-");
  return <article className="experiment-panel">
    <h3>{title}</h3>
    <svg viewBox="0 0 520 290" role="img" aria-labelledby={`${id}-title`} aria-describedby={`${id}-desc`}>
      <title id={`${id}-title`}>{title}</title>
      <desc id={`${id}-desc`}>Validation configurations plotted by API-sourced NDCG@10 and {yLabel}, with concentric outlined markers preserving coincident configurations and marker shapes identifying modes.</desc>
      <Axes xLabel="NDCG@10" yLabel={yLabel} xRange={xRange} yRange={yRange} />
      {points.map(({ x, y }) => {
        const group = collisions.get(`${x.observed_value}\u0000${y.observed_value}`)!;
        const label = `${x.configuration_key}: NDCG@10 ${x.observed_value}; ${yLabel} ${y.observed_value}`;
        return <g key={x.configuration_key} role="graphics-symbol" tabIndex={0} aria-label={label}
          data-point-configuration={x.configuration_key} data-x-metric="ndcg_at_10" data-x-value={x.observed_value}
          data-y-metric={yMetric} data-y-value={y.observed_value}>
        <Marker x={scale(x.observed_value, xRange, 58, 500)} y={scale(y.observed_value, yRange, 238, 18)} row={x} collisionIndex={group.indexOf(x)} collisionCount={group.length} />
        <title>{`${x.configuration_key}: NDCG@10 ${x.observed_value}; ${yLabel} ${y.observed_value}`}</title>
      </g>;
      })}
    </svg>
    <Legend />
    {[...collisions.values()].filter((group) => group.length > 1).map((group) => {
      const first = group[0]; const y = yRows.get(first.configuration_key)!;
      return <p className="experiment-collision" key={`${first.observed_value}-${y.observed_value}`}>Coincident at NDCG@10 {first.observed_value}, {yLabel} {y.observed_value}: {group.map((row) => `${row.configuration_key} (λ=${row.lambda})`).join(", ")}</p>;
    })}
  </article>;
}

function ExposurePanel({ rows }: { rows: ExposureRow[] }) {
  const groups = ["head", "mid_tail", "long_tail"];
  const styles: Record<string, { color: string; pattern: string }> = {
    head: { color: "#CC79A7", pattern: "solid" }, mid_tail: { color: "#56B4E9", pattern: "striped" }, long_tail: { color: "#009E73", pattern: "dotted" },
  };
  return <article className="experiment-panel experiment-panel--wide">
    <h3>Head, mid-tail, and long-tail exposure</h3>
    <svg viewBox="0 0 760 390" role="img" aria-labelledby="exposure-title" aria-describedby="exposure-desc">
      <title id="exposure-title">Head, mid-tail, and long-tail exposure</title>
      <desc id="exposure-desc">Stacked bars show API-sourced displayed-slot shares for each validation configuration. Head is pink solid; mid-tail is blue striped; long-tail is green dotted.</desc>
      <defs>
        <pattern id="mid-stripes" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#56B4E9" /><path d="M0 8L8 0" stroke="#ffffff" strokeWidth="2" /></pattern>
        <pattern id="tail-dots" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#009E73" /><circle cx="2" cy="2" r="1.2" fill="#ffffff" /></pattern>
      </defs>
      {CONFIGURATIONS.map((key, index) => {
        const configRows = groups.map((group) => rows.find((row) => row.configuration_key === key && row.popularity_group === group)!);
        let offset = 0;
        return <g key={key} transform={`translate(112 ${18 + index * 21})`}>
          <text x="-8" y="12" textAnchor="end">{key}</text>
          {configRows.map((row) => {
            const x = offset * 610; const width = row.displayed_slot_share * 610; offset += row.displayed_slot_share;
            return <rect key={row.popularity_group} x={x} y="0" width={width} height="15" fill={row.popularity_group === "mid_tail" ? "url(#mid-stripes)" : row.popularity_group === "long_tail" ? "url(#tail-dots)" : styles[row.popularity_group].color}
              data-configuration={key} data-popularity-group={row.popularity_group} data-color={styles[row.popularity_group].color}
              data-pattern={styles[row.popularity_group].pattern} data-count={row.displayed_slot_count} data-share={row.displayed_slot_share}>
              <title>{`${key} ${row.popularity_group}: ${row.displayed_slot_share} (${row.displayed_slot_count} of ${row.total_displayed_slot_count} slots)`}</title>
            </rect>;
          })}
        </g>;
      })}
    </svg>
    <ul className="experiment-chart__legend" aria-label="Exposure legend">{groups.map((group) => <li key={group} data-testid={`exposure-legend-${group}`} data-color={styles[group].color}><span className={`experiment-swatch experiment-swatch--${styles[group].pattern}`} style={{ "--series": styles[group].color } as React.CSSProperties} aria-hidden="true" />{group.replace("_", "-")} — {styles[group].pattern}</li>)}</ul>
  </article>;
}

function LinePanel({ rows, metric, title, yLabel }: { rows: RerankingRow[]; metric: string; title: string; yLabel: string }) {
  const selected = rows.filter((row) => row.metric === metric);
  const collisions = collisionGroups(selected, (row) => row.lambda, (row) => row.observed_value);
  const xRange: [number, number] = [0.58, 1.02]; const yRange = extent(selected.map((row) => row.observed_value));
  const id = metric.replace(/_/g, "-");
  return <article className="experiment-panel">
    <h3>{title}</h3>
    <svg viewBox="0 0 520 290" role="img" aria-labelledby={`${id}-title`} aria-describedby={`${id}-desc`}>
      <title id={`${id}-title`}>{title}</title>
      <desc id={`${id}-desc`}>API-sourced {yLabel} plotted against lambda. Line styles and point shapes identify each reranking mode.</desc>
      <Axes xLabel="Lambda" yLabel={yLabel} xRange={xRange} yRange={yRange} />
      {Object.keys(MODE_STYLE).map((mode) => {
        const series = selected.filter((row) => row.mode === mode).sort((a, b) => a.lambda - b.lambda);
        const points = series.map((row) => `${scale(row.lambda, xRange, 58, 500)},${scale(row.observed_value, yRange, 238, 18)}`).join(" ");
        return <g key={mode}>
          {series.length > 1 && <polyline points={points} fill="none" stroke={MODE_STYLE[mode].color} strokeWidth="2" strokeDasharray={MODE_STYLE[mode].dash} />}
          {series.map((row) => {
            const group = collisions.get(`${row.lambda}\u0000${row.observed_value}`)!;
            const label = `${row.configuration_key}: lambda ${row.lambda}; ${yLabel} ${row.observed_value}`;
            return <g key={row.configuration_key} role="graphics-symbol" tabIndex={0} aria-label={label}
              data-point-configuration={row.configuration_key} data-x-metric="lambda" data-x-value={row.lambda}
              data-y-metric={metric} data-y-value={row.observed_value}>
              <Marker x={scale(row.lambda, xRange, 58, 500)} y={scale(row.observed_value, yRange, 238, 18)} row={row} collisionIndex={group.indexOf(row)} collisionCount={group.length} />
              <title>{label}</title>
            </g>;
          })}
        </g>;
      })}
    </svg>
    <Legend />
    {[...collisions.values()].filter((group) => group.length > 1).map((group) => {
      const first = group[0];
      return <p className="experiment-collision" key={`${first.lambda}-${first.observed_value}`}>Coincident at lambda {first.lambda}, {yLabel} {first.observed_value}: {group.map((row) => `${row.configuration_key} (λ=${row.lambda})`).join(", ")}</p>;
    })}
  </article>;
}

function EvidenceHeader({ data }: { data: ExperimentSummaryResponse }) {
  const activity = ["low", "medium", "high"].map((group) => data.validation.activity_groups.find((row) => row.activity_group === group)!);
  return <section className="experiment-evidence" aria-labelledby="experiment-evidence-heading">
    <h2 id="experiment-evidence-heading">Evidence context</h2>
    <dl>
      <div><dt>Validation</dt><dd>Fit: {data.validation.fit_scope}; target: {data.validation.target_scope}; eligible users: {data.validation.eligible_user_count}</dd></div>
      <div><dt>Frozen test</dt><dd>Fit: {data.frozen_test.fit_scope}; target: {data.frozen_test.target_scope}; eligible users: {data.frozen_test.eligible_user_count}</dd></div>
      <div><dt>Activity groups</dt><dd>{activity.map((row) => `${row.activity_group[0].toUpperCase()}${row.activity_group.slice(1)}: ${row.activity_group_user_count}`).join("; ")}</dd></div>
      <div><dt>Protocol</dt><dd>{data.protocol_id} · schema {data.schema_version}</dd></div>
      <div><dt>Evidence base Git SHA</dt><dd className="experiment-hash">{data.evidence_base_git_sha}</dd></div>
      <div><dt>Manifest SHA-256</dt><dd className="experiment-hash">{data.manifest_sha256}</dd></div>
    </dl>
    <h3>Accepted outcomes</h3>
    <ul className="experiment-hypotheses">{data.hypotheses.map((row) => <li key={row.hypothesis}><strong>{row.hypothesis}</strong> — scientific outcome: {row.scientific_outcome}; engineering acceptance: {row.engineering_acceptance}; scope: {row.evidence_scope}. {row.reason}</li>)}</ul>
  </section>;
}

function EvidenceTable({ data }: { data: ExperimentSummaryResponse }) {
  const metrics = useMemo(() => new Map(data.validation.reranking.map((row) => [`${row.configuration_key}\u0000${row.metric}`, row])), [data]);
  const exposure = useMemo(() => new Map(data.validation.exposure.map((row) => [`${row.configuration_key}\u0000${row.popularity_group}`, row])), [data]);
  const value = (key: string, metric: string) => metrics.get(`${key}\u0000${metric}`)!.observed_value;
  return <section aria-labelledby="experiment-table-heading">
    <h2 id="experiment-table-heading">Exact configuration evidence</h2>
    <div className="experiment-table-scroll" tabIndex={0} role="region" aria-label="Scrollable experiment evidence table">
      <table>
        <caption>API-sourced validation evidence; numeric cells preserve the values received from API-006.</caption>
        <thead><tr><th scope="col">Configuration</th><th scope="col">Mode</th><th scope="col">Lambda</th><th scope="col">Beta</th><th scope="col">NDCG@10</th><th scope="col">Long-tail Exposure@10</th><th scope="col">ILD@10</th><th scope="col">Exposure Gini@10</th><th scope="col">Catalog Coverage@10</th><th scope="col">Head count</th><th scope="col">Head share</th><th scope="col">Mid-tail count</th><th scope="col">Mid-tail share</th><th scope="col">Long-tail count</th><th scope="col">Long-tail share</th></tr></thead>
        <tbody>{CONFIGURATIONS.map((key) => {
          const identity = metrics.get(`${key}\u0000ndcg_at_10`)!;
          const head = exposure.get(`${key}\u0000head`)!; const mid = exposure.get(`${key}\u0000mid_tail`)!; const tail = exposure.get(`${key}\u0000long_tail`)!;
          return <tr key={key}><th scope="row">{key}</th><td>{identity.mode}</td><td>{String(identity.lambda)}</td><td>{identity.beta === null ? "Not applicable" : String(identity.beta)}</td><td>{String(value(key, "ndcg_at_10"))}</td><td>{String(value(key, "long_tail_exposure_at_10"))}</td><td>{String(value(key, "ild_at_10"))}</td><td>{String(value(key, "exposure_gini_at_10"))}</td><td>{String(value(key, "catalog_coverage_at_10"))}</td><td>{String(head.displayed_slot_count)}</td><td>{String(head.displayed_slot_share)}</td><td>{String(mid.displayed_slot_count)}</td><td>{String(mid.displayed_slot_share)}</td><td>{String(tail.displayed_slot_count)}</td><td>{String(tail.displayed_slot_share)}</td></tr>;
        })}</tbody>
      </table>
    </div>
  </section>;
}

export function ExperimentDashboard() {
  const { replace } = useRouter();
  const [state, setState] = useState<ViewState>({ kind: "loading" });
  const inFlight = useRef(false); const requestId = useRef(0); const consumer = useRef(Symbol("experiment-dashboard"));
  const shouldFocusResults = useRef(false); const errorHeading = useRef<HTMLHeadingElement>(null); const resultsHeading = useRef<HTMLHeadingElement>(null);

  const load = useCallback((userInitiated: boolean) => {
    if (inFlight.current) return;
    inFlight.current = true; shouldFocusResults.current = userInitiated;
    const id = ++requestId.current;
    void getExperimentSummary().then((data) => {
      if (id !== requestId.current) return;
      const empty = data.validation.reranking.length === 0;
      inFlight.current = false;
      setState(empty ? { kind: "empty" } : { kind: "success", data });
    }, (caught: unknown) => {
      if (id !== requestId.current) return;
      inFlight.current = false;
      if (caught instanceof ApiError && caught.status === 401) { setState({ kind: "redirecting" }); replace("/login"); }
      else if (caught instanceof ApiError && (caught.status === 503 || caught.status === 504)) setState({ kind: "unavailable" });
      else setState({ kind: "error" });
    }).finally(() => {
      if (id === requestId.current) inFlight.current = false;
    });
  }, [replace]);

  // Insertion-effect subscriptions are not replayed by React's development-only
  // passive-effect probe, so consumer lifetime remains independent of probe timing.
  useInsertionEffect(() => registerExperimentConsumer(consumer.current), []);
  useEffect(() => {
    void load(false);
    return () => { requestId.current += 1; inFlight.current = false; };
  }, [load]);
  useEffect(() => {
    if (state.kind === "error" || state.kind === "unavailable") errorHeading.current?.focus();
    if (state.kind === "success" && shouldFocusResults.current) resultsHeading.current?.focus();
  }, [state]);

  return <div className="experiment-page">
    <header><p className="experiment-page__eyebrow">Accepted EXP-003 evidence</p><h1>Experiment dashboard</h1><p>Validation trade-offs, exposure, concentration, and reach from the authenticated summary API.</p></header>
    <div className="experiment-page__body" aria-busy={state.kind === "loading"}>
      {state.kind === "loading" && <div className="experiment-state" role="status" aria-live="polite"><span className="experiment-spinner" aria-hidden="true" />Loading experiment evidence…</div>}
      {state.kind === "empty" && <div className="experiment-state"><h2>No experiment evidence available</h2><p>The summary is valid but contains no evidence rows.</p><button type="button" onClick={() => { setState({ kind: "loading" }); void load(true); }}>Try again</button></div>}
      {(state.kind === "error" || state.kind === "unavailable") && <div className="experiment-state experiment-state--error" role="alert"><h2 ref={errorHeading} tabIndex={-1}>{state.kind === "unavailable" ? "Experiment evidence is temporarily unavailable" : "Experiment evidence could not be loaded"}</h2><p>{state.kind === "unavailable" ? "The experiment service is temporarily unavailable. Please try again." : "The response could not be safely displayed. Please try again."}</p><button type="button" onClick={() => { setState({ kind: "loading" }); void load(true); }}>Try again</button></div>}
      {state.kind === "success" && <div className="experiment-results">
        <h2 ref={resultsHeading} tabIndex={-1}>Experiment evidence</h2>
        <EvidenceHeader data={state.data} />
        <section aria-labelledby="tradeoffs-heading"><h2 id="tradeoffs-heading">Validation trade-offs</h2><div className="experiment-grid"><ScatterPanel rows={state.data.validation.reranking} yMetric="long_tail_exposure_at_10" title="NDCG@10 vs Long-tail Exposure@10" yLabel="Long-tail Exposure@10" /><ScatterPanel rows={state.data.validation.reranking} yMetric="ild_at_10" title="NDCG@10 vs ILD@10" yLabel="ILD@10" /></div></section>
        <section aria-labelledby="exposure-heading"><h2 id="exposure-heading">Exposure distribution</h2><ExposurePanel rows={state.data.validation.exposure} /></section>
        <section aria-labelledby="concentration-heading"><h2 id="concentration-heading">Concentration and reach</h2><div className="experiment-grid"><LinePanel rows={state.data.validation.reranking} metric="exposure_gini_at_10" title="Exposure Gini@10 by lambda" yLabel="Exposure Gini@10" /><LinePanel rows={state.data.validation.reranking} metric="catalog_coverage_at_10" title="Catalog Coverage@10 by lambda" yLabel="Catalog Coverage@10" /></div></section>
        <EvidenceTable data={state.data} />
      </div>}
    </div>
  </div>;
}
