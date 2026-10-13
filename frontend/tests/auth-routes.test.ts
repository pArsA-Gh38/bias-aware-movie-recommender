import { beforeEach, describe, expect, it, vi } from "vitest";

const values = new Map<string, string>();
const cookieStore = {
  get: vi.fn((name: string) => values.has(name) ? { value: values.get(name)! } : undefined),
  set: vi.fn((name: string, value: string) => values.set(name, value)),
  delete: vi.fn((name: string) => values.delete(name)),
};

vi.mock("next/headers", () => ({ cookies: async () => cookieStore }));

import { POST as login } from "@/app/api/auth/login/route";
import { POST as logout } from "@/app/api/auth/logout/route";
import { GET as me } from "@/app/api/auth/me/route";
import { POST as register } from "@/app/api/auth/register/route";

const credentials = { email: "person@example.com", password: "password123" };
const user = { id: "u1", email: credentials.email, created_at: "2026-01-01T00:00:00Z", is_active: true };

describe("authentication route handlers", () => {
  beforeEach(() => {
    values.clear();
    vi.stubGlobal("fetch", vi.fn());
  });

  it("stores a successful login token in an HttpOnly cookie and returns the user", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(Response.json({ access_token: "secret-token", token_type: "bearer", expires_in: 1800 }))
      .mockResolvedValueOnce(Response.json(user));
    const response = await login(new Request("http://localhost/api/auth/login", { method: "POST", body: JSON.stringify(credentials) }));
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual(user);
    expect(cookieStore.set).toHaveBeenCalledWith("movie_session", "secret-token", expect.objectContaining({ httpOnly: true, sameSite: "lax", path: "/", maxAge: 1800 }));
    expect(vi.mocked(fetch).mock.calls[1]?.[1]?.headers).toEqual(expect.objectContaining({}));
    expect(new Headers(vi.mocked(fetch).mock.calls[1]?.[1]?.headers).get("Authorization")).toBe("Bearer secret-token");
  });

  it("registers, logs in, and establishes a session", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(Response.json(user, { status: 201 }))
      .mockResolvedValueOnce(Response.json({ access_token: "registered-token", expires_in: 900 }));
    const response = await register(new Request("http://localhost/api/auth/register", { method: "POST", body: JSON.stringify(credentials) }));
    expect(response.status).toBe(201);
    expect(values.get("movie_session")).toBe("registered-token");
  });

  it("returns 503 when the backend request times out", async () => {
    const controller = new AbortController();
    const timeout = vi.spyOn(AbortSignal, "timeout").mockReturnValue(controller.signal);
    vi.mocked(fetch).mockImplementationOnce((_input, init) => new Promise((_, reject) => {
      const rejectTimeout = () => reject(new DOMException("Timed out", "TimeoutError"));
      if (init?.signal?.aborted) rejectTimeout();
      else init?.signal?.addEventListener("abort", rejectTimeout);
    }));

    const pendingResponse = register(new Request("http://localhost/api/auth/register", {
      method: "POST",
      body: JSON.stringify(credentials),
    }));
    controller.abort();
    const response = await pendingResponse;

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ detail: "Backend service unavailable" });
    timeout.mockRestore();
  });

  it("forwards the session as a bearer token for me validation", async () => {
    values.set("movie_session", "session-token");
    vi.mocked(fetch).mockResolvedValueOnce(Response.json(user));
    const response = await me();
    expect(response.status).toBe(200);
    expect(new Headers(vi.mocked(fetch).mock.calls[0]?.[1]?.headers).get("Authorization")).toBe("Bearer session-token");
  });

  it("rejects me validation when the session cookie is absent", async () => {
    const response = await me();
    expect(response.status).toBe(401);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("clears the cookie on logout", async () => {
    values.set("movie_session", "session-token");
    const response = await logout();
    expect(response.status).toBe(200);
    expect(values.has("movie_session")).toBe(false);
    expect(cookieStore.delete).toHaveBeenCalledWith("movie_session");
  });
});
