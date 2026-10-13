import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "@/lib/api/client";
import { parseExperimentSummary } from "@/lib/api/experiment-summary";
import { emptyExperimentFixture, experimentFixture } from "./experiment-fixture";

function controlledJsonResponse(status = 200) {
  let body!: ReadableStreamDefaultController<Uint8Array>;
  let requestSignal!: AbortSignal;
  const stream = new ReadableStream<Uint8Array>({ start(controller) { body = controller; } });
  vi.stubGlobal("fetch", vi.fn((_path: string, init?: RequestInit) => {
    requestSignal = init?.signal as AbortSignal;
    requestSignal.addEventListener("abort", () => body.error(new DOMException("Aborted", "AbortError")), { once: true });
    return Promise.resolve(new Response(stream, { status, headers: { "Content-Type": "application/json" } }));
  }));
  return {
    close(value: unknown) {
      body.enqueue(new TextEncoder().encode(JSON.stringify(value)));
      body.close();
    },
    get signal() { return requestSignal; },
  };
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("experiment summary parser", () => {
  it("accepts the complete API contract and preserves exact chart/table fields", () => {
    const parsed = parseExperimentSummary(experimentFixture());
    const row = parsed.validation.reranking.find((item) => item.configuration_key === "A0_L1" && item.metric === "ndcg_at_10")!;
    expect(row).toMatchObject({ mode: "base", lambda: 1, beta: null, observed_value: 0.411 });
    expect(parsed.validation.exposure[0]).toMatchObject({ displayed_slot_count: 4519, displayed_slot_share: 0.9414583333333333 });
  });

  it("accepts only the contract-shaped all-empty response as the empty state", () => {
    expect(parseExperimentSummary(emptyExperimentFixture()).validation.reranking).toEqual([]);
    const partial = emptyExperimentFixture();
    partial.hypotheses = experimentFixture().hypotheses;
    expect(() => parseExperimentSummary(partial)).toThrow();
  });

  it.each([
    ["non-object", null],
    ["missing provenance", (() => { const value = experimentFixture() as unknown as Record<string, unknown>; delete value.manifest_sha256; return value; })()],
    ["non-finite number", (() => { const value = experimentFixture(); value.validation.reranking[0].observed_value = Number.NaN; return value; })()],
    ["missing row", (() => { const value = experimentFixture(); value.validation.exposure.pop(); return value; })()],
    ["unexpected field", (() => { const value = experimentFixture() as unknown as Record<string, unknown>; value.timestamp = "not in API-006"; return value; })()],
  ])("rejects %s", (_name, value) => expect(() => parseExperimentSummary(value)).toThrow());

  it("rejects duplicate scientific identities and inconsistent cohorts", () => {
    const duplicate = experimentFixture();
    duplicate.validation.reranking[1] = { ...duplicate.validation.reranking[0] };
    expect(() => parseExperimentSummary(duplicate)).toThrow();

    const cohort = experimentFixture();
    cohort.validation.activity_groups[0].activity_group_user_count = 10;
    expect(() => parseExperimentSummary(cohort)).toThrow();
  });

  it("rejects an inconsistent hypothesis outcome without deriving a replacement", () => {
    const invalid = experimentFixture();
    invalid.hypotheses[3].engineering_acceptance = "accepted_evidence";
    expect(() => parseExperimentSummary(invalid)).toThrow();
  });

  it("requires nullable beta exactly where the configuration contract defines it", () => {
    const valid = experimentFixture();
    expect(valid.validation.reranking[0].beta).toBeNull();
    const invalid = experimentFixture();
    invalid.validation.reranking[0].beta = 0;
    expect(() => parseExperimentSummary(invalid)).toThrow();
  });

  it.each([
    ["negative count", (value: ReturnType<typeof experimentFixture>) => { value.validation.exposure[0].displayed_slot_count = -1; }],
    ["fractional count", (value: ReturnType<typeof experimentFixture>) => { value.validation.exposure[0].displayed_slot_count = 1.5; }],
    ["negative share", (value: ReturnType<typeof experimentFixture>) => { value.validation.exposure[0].displayed_slot_share = -0.01; }],
    ["share above one", (value: ReturnType<typeof experimentFixture>) => { value.validation.exposure[0].displayed_slot_share = 1.01; }],
    ["inconsistent totals", (value: ReturnType<typeof experimentFixture>) => { value.validation.exposure[1].total_displayed_slot_count = 4799; }],
    ["count sum mismatch", (value: ReturnType<typeof experimentFixture>) => { value.validation.exposure[0].displayed_slot_count -= 1; }],
    ["share sum mismatch", (value: ReturnType<typeof experimentFixture>) => { value.validation.exposure[1].displayed_slot_share -= 0.001; }],
    ["count/share mismatch", (value: ReturnType<typeof experimentFixture>) => { value.validation.exposure[0].displayed_slot_share -= 0.001; value.validation.exposure[1].displayed_slot_share += 0.001; }],
    ["non-authoritative tolerance", (value: ReturnType<typeof experimentFixture>) => { value.validation.exposure[0].share_sum_tolerance = 0.01; }],
  ])("rejects exposure %s", (_name, mutate) => {
    const invalid = experimentFixture();
    mutate(invalid);
    expect(() => parseExperimentSummary(invalid)).toThrow();
  });

  it("accepts authoritative rounded shares and the zero count/share boundary", () => {
    const parsed = parseExperimentSummary(experimentFixture());
    expect(parsed.validation.exposure.slice(0, 3)).toMatchObject([
      { popularity_group: "head", displayed_slot_count: 4519, displayed_slot_share: 0.9414583333333333 },
      { popularity_group: "mid_tail", displayed_slot_count: 281, displayed_slot_share: 0.058541666666666665 },
      { popularity_group: "long_tail", displayed_slot_count: 0, displayed_slot_share: 0 },
    ]);
  });
});

describe("experiment summary client body lifetime", () => {
  it.each([
    ["successful", 200],
    ["error", 422],
  ])("times out a stalled %s JSON body after headers", async (_name, status) => {
    vi.useFakeTimers();
    controlledJsonResponse(status);
    const result = api.experimentSummary();
    const rejected = expect(result).rejects.toMatchObject({ status: 503 });
    await vi.advanceTimersByTimeAsync(15_000);
    await rejected;
  });

  it("keeps caller cancellation active after headers arrive", async () => {
    const response = controlledJsonResponse();
    const caller = new AbortController();
    const result = api.experimentSummary(caller.signal);
    await Promise.resolve();
    caller.abort("left page");
    try { response.close(experimentFixture()); } catch { /* the aborted stream is already closed */ }
    await expect(result).rejects.toMatchObject({ name: "AbortError" });
  });

  it("preserves ordinary success and API error semantics", async () => {
    vi.stubGlobal("fetch", vi.fn()
      .mockResolvedValueOnce(Response.json(experimentFixture()))
      .mockResolvedValueOnce(Response.json({ detail: "Expected detail" }, { status: 422 })));
    await expect(api.experimentSummary()).resolves.toMatchObject({ schema_version: 1 });
    await expect(api.experimentSummary()).rejects.toEqual(new ApiError("Expected detail", 422));
  });

  it("cleans up timeout and caller listeners after the body completes", async () => {
    vi.useFakeTimers();
    const response = controlledJsonResponse();
    const caller = new AbortController();
    const result = api.experimentSummary(caller.signal);
    await Promise.resolve();
    response.close(experimentFixture());
    await expect(result).resolves.toMatchObject({ schema_version: 1 });
    await vi.advanceTimersByTimeAsync(15_000);
    caller.abort("too late");
    expect(response.signal.aborted).toBe(false);
  });
});
