import { beforeEach, describe, expect, it, vi } from "vitest";

const values = new Map<string, string>();
vi.mock("next/headers", () => ({ cookies: async () => ({ get: (name: string) => values.has(name) ? { value: values.get(name)! } : undefined }) }));

import { GET } from "@/app/api/experiments/summary/route";

describe("experiment summary BFF route", () => {
  beforeEach(() => { values.clear(); vi.stubGlobal("fetch", vi.fn()); });

  it("returns local 401 with no upstream activity", async () => {
    const response = await GET();
    expect(response.status).toBe(401);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("forwards one authenticated bodyless GET to the exact endpoint", async () => {
    values.set("movie_session", "session-token");
    vi.mocked(fetch).mockResolvedValue(Response.json({ protocol_id: "fixture" }));
    const response = await GET();
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ protocol_id: "fixture" });
    expect(fetch).toHaveBeenCalledTimes(1);
    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe("http://127.0.0.1:8000/experiments/summary");
    expect(init?.method).toBe("GET");
    expect(init?.body).toBeUndefined();
    expect(new URL(String(url)).search).toBe("");
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer session-token");
    expect(init?.cache).toBe("no-store");
  });

  it.each([401, 403, 404, 422, 503, 504])("preserves upstream %s status and body", async (status) => {
    values.set("movie_session", "session-token");
    vi.mocked(fetch).mockResolvedValue(Response.json({ detail: `status-${status}` }, { status }));
    const response = await GET();
    expect(response.status).toBe(status);
    expect(await response.json()).toEqual({ detail: `status-${status}` });
  });

  it("sanitizes network failures as 503", async () => {
    values.set("movie_session", "session-token");
    vi.mocked(fetch).mockRejectedValue(new Error("private host detail"));
    const response = await GET();
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ detail: "Backend service unavailable" });
  });
});
