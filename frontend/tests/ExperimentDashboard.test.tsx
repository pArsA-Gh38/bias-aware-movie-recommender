import { StrictMode } from "react";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { ExperimentDashboard } from "@/components/ExperimentDashboard";
import { emptyExperimentFixture, experimentFixture } from "./experiment-fixture";

const replace = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace }) }));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((yes) => { resolve = yes; });
  return { promise, resolve };
}

beforeEach(() => { replace.mockReset(); vi.stubGlobal("fetch", vi.fn()); });

it("renders all three sections, five accessible panels, evidence, and the exact-value table", async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json(experimentFixture()));
  render(<ExperimentDashboard />);
  expect(await screen.findByRole("heading", { name: "Experiment evidence" })).toBeInTheDocument();
  for (const name of ["Validation trade-offs", "Exposure distribution", "Concentration and reach"]) expect(screen.getByRole("heading", { name })).toBeInTheDocument();
  for (const name of ["NDCG@10 vs Long-tail Exposure@10", "NDCG@10 vs ILD@10", "Head, mid-tail, and long-tail exposure", "Exposure Gini@10 by lambda", "Catalog Coverage@10 by lambda"]) {
    expect(screen.getByRole("img", { name })).toHaveAccessibleDescription();
  }
  expect(screen.getAllByText("A0_L1").length).toBeGreaterThan(0);
  expect(screen.getAllByText("0.9414583333333333").length).toBe(4);
  expect(screen.getByText("8d3d2e27b81676648b982bb3e0f656c6f8d36888")).toBeInTheDocument();
  expect(screen.getByText(/Low: 9/)).toBeInTheDocument();
  expect(screen.getByText("H4").closest("li")).toHaveTextContent("engineering_cpu_latency");
  expect(screen.getByRole("table", { name: /API-sourced validation evidence/ })).toBeVisible();
});

it("maps every named panel and table column to distinct scientific fields", async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json(experimentFixture()));
  render(<ExperimentDashboard />);
  await screen.findByRole("heading", { name: "Experiment evidence" });

  const longTail = screen.getByRole("img", { name: "NDCG@10 vs Long-tail Exposure@10" });
  const ild = screen.getByRole("img", { name: "NDCG@10 vs ILD@10" });
  const gini = screen.getByRole("img", { name: "Exposure Gini@10 by lambda" });
  const coverage = screen.getByRole("img", { name: "Catalog Coverage@10 by lambda" });
  const point = (panel: HTMLElement, key: string) => panel.querySelector(`[data-point-configuration="${key}"]`)!;
  expect(point(longTail, "A1_L09")).toHaveAttribute("data-x-value", "0.412");
  expect(point(longTail, "A1_L09")).toHaveAttribute("data-y-value", "0.022");
  expect(point(longTail, "A1_L09")).toHaveAttribute("data-y-metric", "long_tail_exposure_at_10");
  expect(point(ild, "A1_L09")).toHaveAttribute("data-y-value", "0.542");
  expect(point(ild, "A1_L09")).toHaveAttribute("data-y-metric", "ild_at_10");
  expect(point(gini, "A1_L09")).toHaveAttribute("data-y-value", "0.652");
  expect(point(gini, "A1_L09")).toHaveAttribute("data-y-metric", "exposure_gini_at_10");
  expect(point(coverage, "A1_L09")).toHaveAttribute("data-y-value", "0.762");
  expect(point(coverage, "A1_L09")).toHaveAttribute("data-y-metric", "catalog_coverage_at_10");

  const row = screen.getByRole("rowheader", { name: "A1_L09" }).closest("tr")!;
  expect(within(row).getAllByRole("cell").map((cell) => cell.textContent)).toEqual([
    "diversity_only", "0.9", "1", "0.412", "0.022", "0.542", "0.652", "0.762",
    "4527", "0.943125", "273", "0.056875", "0", "0",
  ]);
});

it("preserves every coincident configuration at exact coordinates with deterministic labels", async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json(experimentFixture()));
  render(<ExperimentDashboard />);
  const panel = await screen.findByRole("img", { name: "NDCG@10 vs ILD@10" });
  const coincident = ["A0_L1", "A1_L1", "A2_L1", "A3_L1"];
  const points = coincident.map((key) => panel.querySelector(`[data-point-configuration="${key}"]`)!);
  expect(points.every((point) => point.getAttribute("data-x-value") === "0.411" && point.getAttribute("data-y-value") === "0.531")).toBe(true);
  expect(points.map((point) => point.getAttribute("aria-label"))).toEqual([
    "A0_L1: NDCG@10 0.411; ILD@10 0.531",
    "A1_L1: NDCG@10 0.411; ILD@10 0.531",
    "A2_L1: NDCG@10 0.411; ILD@10 0.531",
    "A3_L1: NDCG@10 0.411; ILD@10 0.531",
  ]);
  expect(screen.getByText("Coincident at NDCG@10 0.411, ILD@10 0.531: A0_L1 (λ=1), A1_L1 (λ=1), A2_L1 (λ=1), A3_L1 (λ=1)")).toBeVisible();
});

it("uses the accepted exposure bucket palette, textures, legend, and exact segment data", async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json(experimentFixture()));
  render(<ExperimentDashboard />);
  const panel = await screen.findByRole("img", { name: "Head, mid-tail, and long-tail exposure" });
  const expected = [
    ["head", "#CC79A7", "solid", "4519", "0.9414583333333333"],
    ["mid_tail", "#56B4E9", "striped", "281", "0.058541666666666665"],
    ["long_tail", "#009E73", "dotted", "0", "0"],
  ];
  for (const [group, color, pattern, count, share] of expected) {
    const segment = panel.querySelector(`[data-configuration="A0_L1"][data-popularity-group="${group}"]`)!;
    expect(segment).toHaveAttribute("data-color", color);
    expect(segment).toHaveAttribute("data-pattern", pattern);
    expect(segment).toHaveAttribute("data-count", count);
    expect(segment).toHaveAttribute("data-share", share);
    const legend = screen.getByTestId(`exposure-legend-${group}`);
    expect(legend).toHaveAttribute("data-color", color);
    expect(legend).toHaveTextContent(pattern);
  }
  const orderedSegments = [...panel.querySelectorAll('[data-configuration="A0_L1"]')];
  expect(orderedSegments.map((segment) => segment.getAttribute("data-popularity-group"))).toEqual(["head", "mid_tail", "long_tail"]);
  expect(orderedSegments.map((segment) => segment.getAttribute("x"))).toEqual(["0", "574.2895833333333", "610"]);
  expect(panel).toHaveAccessibleDescription(/head is pink solid; mid-tail is blue striped; long-tail is green dotted/i);
});

it("uses exact @10 metric labels in titles, axes, descriptions, and table headers", async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json(experimentFixture()));
  render(<ExperimentDashboard />);
  await screen.findByRole("heading", { name: "Experiment evidence" });
  for (const label of ["NDCG@10", "Long-tail Exposure@10", "ILD@10", "Exposure Gini@10", "Catalog Coverage@10"]) {
    expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    expect(screen.getByRole("columnheader", { name: label })).toBeInTheDocument();
  }
});

it("shows non-focusing initial loading and a retryable valid empty state", async () => {
  const pending = deferred<Response>();
  vi.mocked(fetch).mockReturnValueOnce(pending.promise).mockResolvedValueOnce(Response.json(emptyExperimentFixture()));
  render(<ExperimentDashboard />);
  expect(screen.getByRole("status")).toHaveTextContent("Loading experiment evidence");
  expect(document.body).toHaveFocus();
  pending.resolve(Response.json(emptyExperimentFixture()));
  expect(await screen.findByRole("heading", { name: "No experiment evidence available" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
});

it.each([
  [Response.json({ protocol_id: "bad" }), "Experiment evidence could not be loaded"],
  [Response.json({ detail: "private" }, { status: 503 }), "Experiment evidence is temporarily unavailable"],
  [Response.json({ detail: "private" }, { status: 504 }), "Experiment evidence is temporarily unavailable"],
  [Response.json({ detail: "private" }, { status: 500 }), "Experiment evidence could not be loaded"],
] as const)("shows a sanitized focused retryable failure", async (response, headingName) => {
  vi.mocked(fetch).mockResolvedValue(response);
  render(<ExperimentDashboard />);
  const heading = await screen.findByRole("heading", { name: headingName });
  await waitFor(() => expect(heading).toHaveFocus());
  expect(screen.getByRole("alert")).not.toHaveTextContent("private");
  expect(screen.getByRole("button", { name: "Try again" })).toBeEnabled();
});

it("redirects 401 without announcing an error", async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json({}, { status: 401 }));
  render(<ExperimentDashboard />);
  await waitFor(() => expect(replace).toHaveBeenCalledWith("/login"));
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("locks synchronous duplicate retries and focuses results after recovery", async () => {
  vi.mocked(fetch).mockResolvedValueOnce(Response.json({}, { status: 500 }));
  render(<ExperimentDashboard />);
  const retry = await screen.findByRole("button", { name: "Try again" });
  const pending = deferred<Response>();
  vi.mocked(fetch).mockReturnValue(pending.promise);
  fireEvent.click(retry); fireEvent.click(retry);
  expect(fetch).toHaveBeenCalledTimes(2);
  pending.resolve(Response.json(experimentFixture()));
  const heading = await screen.findByRole("heading", { name: "Experiment evidence" });
  await waitFor(() => expect(heading).toHaveFocus());
});

it("suppresses obsolete completions and aborts on unmount", async () => {
  const first = deferred<Response>();
  vi.mocked(fetch).mockReturnValueOnce(first.promise).mockResolvedValueOnce(Response.json(experimentFixture()));
  const view = render(<ExperimentDashboard />);
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
  const firstSignal = vi.mocked(fetch).mock.calls[0]?.[1]?.signal as AbortSignal;
  view.unmount();
  expect(firstSignal.aborted).toBe(true);
  first.resolve(Response.json({ protocol_id: "obsolete" }));
  await act(async () => undefined);
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("performs one productive initial request under React Strict Mode", async () => {
  vi.mocked(fetch).mockResolvedValue(Response.json(experimentFixture()));
  render(<StrictMode><ExperimentDashboard /></StrictMode>);
  expect(await screen.findByRole("heading", { name: "Experiment evidence" })).toBeInTheDocument();
  expect(fetch).toHaveBeenCalledTimes(1);
});

it("shares an in-flight initial GET and aborts only after the last consumer unmounts", async () => {
  const pending = deferred<Response>();
  vi.mocked(fetch).mockReturnValue(pending.promise);
  const first = render(<ExperimentDashboard />);
  const second = render(<ExperimentDashboard />);
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
  const signal = vi.mocked(fetch).mock.calls[0]?.[1]?.signal as AbortSignal;
  first.unmount();
  expect(signal.aborted).toBe(false);
  first.unmount();
  expect(signal.aborted).toBe(false);
  second.unmount();
  expect(signal.aborted).toBe(true);
  pending.resolve(Response.json(experimentFixture()));
});

it("starts a fresh GET on immediate remount before the abandoned request settles", async () => {
  const abandoned = deferred<Response>();
  const current = deferred<Response>();
  vi.mocked(fetch).mockReturnValueOnce(abandoned.promise).mockReturnValueOnce(current.promise);

  const first = render(<ExperimentDashboard />);
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
  const abandonedSignal = vi.mocked(fetch).mock.calls[0]?.[1]?.signal as AbortSignal;
  first.unmount();
  expect(abandonedSignal.aborted).toBe(true);

  const second = render(<ExperimentDashboard />);
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  const currentSignal = vi.mocked(fetch).mock.calls[1]?.[1]?.signal as AbortSignal;
  expect(currentSignal.aborted).toBe(false);

  abandoned.resolve(Response.json(experimentFixture()));
  await act(async () => undefined);
  expect(fetch).toHaveBeenCalledTimes(2);
  expect(currentSignal.aborted).toBe(false);
  expect(screen.queryByRole("heading", { name: "Experiment evidence" })).not.toBeInTheDocument();

  const concurrent = render(<ExperimentDashboard />);
  expect(fetch).toHaveBeenCalledTimes(2);
  current.resolve(Response.json(experimentFixture()));
  await waitFor(() => expect(screen.getAllByRole("heading", { name: "Experiment evidence" })).toHaveLength(2));
  second.unmount();
  concurrent.unmount();
});

it("removes completed and failed entries so later loads issue fresh GETs", async () => {
  vi.mocked(fetch)
    .mockResolvedValueOnce(Response.json(experimentFixture()))
    .mockResolvedValueOnce(Response.json({}, { status: 500 }))
    .mockResolvedValueOnce(Response.json(experimentFixture()));
  const first = render(<ExperimentDashboard />);
  await screen.findByRole("heading", { name: "Experiment evidence" });
  first.unmount();
  const second = render(<ExperimentDashboard />);
  const retry = await screen.findByRole("button", { name: "Try again" });
  fireEvent.click(retry);
  const heading = await screen.findByRole("heading", { name: "Experiment evidence" });
  await waitFor(() => expect(heading).toHaveFocus());
  expect(fetch).toHaveBeenCalledTimes(3);
  second.unmount();
});

it("keeps retry keyboard-operable", async () => {
  vi.mocked(fetch).mockResolvedValueOnce(Response.json({}, { status: 500 })).mockResolvedValueOnce(Response.json(experimentFixture()));
  render(<ExperimentDashboard />);
  const retry = await screen.findByRole("button", { name: "Try again" });
  retry.focus();
  await userEvent.keyboard("{Enter}");
  const heading = await screen.findByRole("heading", { name: "Experiment evidence" });
  await waitFor(() => expect(heading).toHaveFocus());
});
