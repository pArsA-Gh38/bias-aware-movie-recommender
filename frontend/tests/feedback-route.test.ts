import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const values = new Map<string, string>();
const cookieStore = {
  get: vi.fn((name: string) => values.has(name) ? { value: values.get(name)! } : undefined),
};

vi.mock("next/headers", () => ({ cookies: async () => cookieStore }));

import { POST } from "@/app/api/feedback/route";

const EVENT_ID = "7c638ee0-21ca-4c50-a2ac-87f8d7232a89";

function requestWithBody(body?: BodyInit) {
  return new Request("http://localhost/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
}

function requestWithJson(body: unknown) {
  return requestWithBody(JSON.stringify(body));
}

function validPayload(action = "like") {
  return { event_id: EVENT_ID, movie_id: 1193, action };
}

describe("feedback BFF route", () => {
  beforeEach(() => {
    values.clear();
    cookieStore.get.mockClear();
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => vi.restoreAllMocks());

  it.each(["click", "like", "dislike"] as const)(
    "forwards the exact accepted %s payload with bearer authentication",
    async (action) => {
      values.set("movie_session", "session-token");
      vi.mocked(fetch).mockResolvedValue(Response.json({ status: "recorded" }, { status: 201 }));

      const response = await POST(requestWithJson(validPayload(action)));

      expect(response.status).toBe(201);
      expect(await response.json()).toEqual({ status: "recorded" });
      expect(fetch).toHaveBeenCalledTimes(1);
      const [url, init] = vi.mocked(fetch).mock.calls[0];
      expect(url).toBe("http://127.0.0.1:8000/feedback");
      expect(init?.method).toBe("POST");
      expect(JSON.parse(String(init?.body))).toEqual(validPayload(action));
      expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer session-token");
      expect(new Headers(init?.headers).get("Content-Type")).toBe("application/json");
      expect(init?.cache).toBe("no-store");
    },
  );

  it.each([
    ["missing event", { movie_id: 1193, action: "like" }],
    ["missing movie", { event_id: EVENT_ID, action: "like" }],
    ["missing action", { event_id: EVENT_ID, movie_id: 1193 }],
    ["extra field", { ...validPayload(), user_id: "private" }],
    ["invalid UUID", { ...validPayload(), event_id: "not-a-uuid" }],
    ["numeric event", { ...validPayload(), event_id: 1 }],
    ["fractional movie", { ...validPayload(), movie_id: 1.5 }],
    ["string movie", { ...validPayload(), movie_id: "1193" }],
    ["boolean movie", { ...validPayload(), movie_id: true }],
    ["invalid action", validPayload("share")],
    ["null", null],
    ["array", []],
    ["string", "feedback"],
    ["number", 1],
    ["empty object", {}],
  ])("rejects authenticated invalid input locally: %s", async (_name, body) => {
    values.set("movie_session", "session-token");

    const response = await POST(requestWithJson(body));

    expect(response.status).toBe(422);
    expect(await response.json()).toEqual({ detail: "Invalid feedback request" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("rejects malformed JSON locally", async () => {
    values.set("movie_session", "session-token");
    const response = await POST(requestWithBody("{"));
    expect(response.status).toBe(422);
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each([
    ["valid", requestWithJson(validPayload())],
    ["malformed", requestWithBody("{")],
    ["invalid", requestWithJson({ ...validPayload(), action: "share" })],
  ])("checks authentication before parsing or validating %s input", async (_name, request) => {
    const response = await POST(request);
    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ detail: "Not authenticated" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each([201, 401, 404, 409, 422, 503, 504])(
    "preserves upstream status %s and body",
    async (status) => {
      values.set("movie_session", "session-token");
      const body = status === 201 ? { status: "recorded" } : { detail: `status-${status}` };
      vi.mocked(fetch).mockResolvedValue(Response.json(body, { status }));
      const response = await POST(requestWithJson(validPayload()));
      expect(response.status).toBe(status);
      expect(await response.json()).toEqual(body);
    },
  );

  it("sanitizes an upstream network failure", async () => {
    values.set("movie_session", "session-token");
    vi.mocked(fetch).mockRejectedValue(new Error("private connection detail"));
    const response = await POST(requestWithJson(validPayload()));
    expect(response.status).toBe(503);
    const responseText = await response.text();
    expect(JSON.parse(responseText)).toEqual({ detail: "Backend service unavailable" });
    expect(responseText).not.toContain("private");
  });

  it("sanitizes an upstream timeout failure", async () => {
    values.set("movie_session", "session-token");
    vi.spyOn(AbortSignal, "timeout").mockReturnValue(AbortSignal.abort("timeout"));
    vi.mocked(fetch).mockImplementation((_input, init) => {
      expect(init?.signal?.aborted).toBe(true);
      return Promise.reject(new DOMException("private timeout detail", "AbortError"));
    });
    const response = await POST(requestWithJson(validPayload()));
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ detail: "Backend service unavailable" });
  });
});
