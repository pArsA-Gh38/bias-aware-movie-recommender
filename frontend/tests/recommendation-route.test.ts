import { beforeEach, describe, expect, it, vi } from "vitest";

const values = new Map<string, string>();
const cookieStore = {
  get: vi.fn((name: string) => values.has(name) ? { value: values.get(name)! } : undefined),
};

vi.mock("next/headers", () => ({ cookies: async () => cookieStore }));

import { POST } from "@/app/api/recommendations/route";

function requestWithBody(body?: BodyInit) {
  return new Request("http://localhost/api/recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
}

function requestWithJson(body: unknown) {
  return requestWithBody(JSON.stringify(body));
}

describe("recommendation BFF route", () => {
  beforeEach(() => {
    values.clear();
    cookieStore.get.mockClear();
    vi.stubGlobal("fetch", vi.fn());
  });

  it.each(["A0_L1", "A1_L07", "A1_L06"] as const)(
    "forwards only the accepted %s payload with bearer authentication",
    async (configurationKey) => {
      values.set("movie_session", "session-token");
      vi.mocked(fetch).mockResolvedValue(Response.json({ ok: true }));
      const response = await POST(requestWithJson({ configuration_key: configurationKey }));

      expect(response.status).toBe(200);
      expect(fetch).toHaveBeenCalledTimes(1);
      const [url, init] = vi.mocked(fetch).mock.calls[0];
      expect(url).toBe("http://127.0.0.1:8000/recommendations");
      expect(init?.method).toBe("POST");
      expect(JSON.parse(String(init?.body))).toEqual({ configuration_key: configurationKey });
      expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer session-token");
      expect(new Headers(init?.headers).get("Content-Type")).toBe("application/json");
    },
  );

  it.each([
    ["mode", { configuration_key: "A0_L1", mode: "base" }],
    ["lambda", { configuration_key: "A0_L1", lambda: 1 }],
    ["lambda_", { configuration_key: "A0_L1", lambda_: 1 }],
    ["beta", { configuration_key: "A0_L1", beta: 0.5 }],
    ["unknown", { configuration_key: "A0_L1", arbitrary: true }],
  ])("rejects the forbidden or unknown %s field without an upstream request", async (_name, body) => {
    values.set("movie_session", "session-token");
    const response = await POST(requestWithJson(body));

    expect(response.status).toBe(422);
    expect(await response.json()).toEqual({ detail: "Invalid recommendation configuration" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each([
    ["empty body", requestWithBody()],
    ["malformed JSON", requestWithBody("{")],
    ["JSON null", requestWithJson(null)],
    ["empty object", requestWithJson({})],
    ["empty key", requestWithJson({ configuration_key: "" })],
    ["unsupported key", requestWithJson({ configuration_key: "A2_L05" })],
    ["array body", requestWithJson([])],
    ["string body", requestWithJson("A0_L1")],
    ["number body", requestWithJson(1)],
    ["null key", requestWithJson({ configuration_key: null })],
    ["number key", requestWithJson({ configuration_key: 1 })],
    ["object key", requestWithJson({ configuration_key: {} })],
    ["array key", requestWithJson({ configuration_key: ["A0_L1"] })],
  ])("returns 422 with zero upstream fetches for authenticated invalid input: %s", async (_name, request) => {
    values.set("movie_session", "session-token");
    const response = await POST(request);

    expect(response.status).toBe(422);
    expect(await response.json()).toEqual({ detail: "Invalid recommendation configuration" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("returns the existing 401 response without upstream activity for a valid unauthenticated request", async () => {
    const response = await POST(requestWithJson({ configuration_key: "A0_L1" }));

    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ detail: "Not authenticated" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each([
    ["malformed JSON", requestWithBody("{")],
    ["unsupported key", requestWithJson({ configuration_key: "A2_L05" })],
    ["forbidden field", requestWithJson({ configuration_key: "A0_L1", mode: "base" })],
  ])("checks authentication before parsing or validating %s", async (_name, request) => {
    const response = await POST(request);

    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ detail: "Not authenticated" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("returns the existing sanitized 503 response when the backend is unavailable", async () => {
    values.set("movie_session", "session-token");
    vi.mocked(fetch).mockRejectedValue(new Error("private connection detail"));
    const response = await POST(requestWithJson({ configuration_key: "A0_L1" }));

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ detail: "Backend service unavailable" });
  });

  it("preserves an upstream 504 status through the existing proxy behavior", async () => {
    values.set("movie_session", "session-token");
    vi.mocked(fetch).mockResolvedValue(Response.json(
      { detail: "Recommendation service unavailable" },
      { status: 504 },
    ));
    const response = await POST(requestWithJson({ configuration_key: "A1_L07" }));

    expect(response.status).toBe(504);
    expect(await response.json()).toEqual({ detail: "Recommendation service unavailable" });
  });
});
