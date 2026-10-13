import { cookies } from "next/headers";

export const SESSION_COOKIE = "movie_session";
const API_BASE_URL = (process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const DEFAULT_REQUEST_TIMEOUT_MS = 10_000;

function requestTimeoutMs(): number {
  const configured = Number(process.env.BACKEND_REQUEST_TIMEOUT_MS);
  return Number.isInteger(configured) && configured >= 1_000 && configured <= 60_000
    ? configured
    : DEFAULT_REQUEST_TIMEOUT_MS;
}

export async function backendFetch(
  path: string,
  init: RequestInit = {},
  authenticated = false,
): Promise<Response> {
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");

  if (authenticated) {
    const token = (await cookies()).get(SESSION_COOKIE)?.value;
    if (!token) return Response.json({ detail: "Not authenticated" }, { status: 401 });
    headers.set("Authorization", `Bearer ${token}`);
  }

  try {
    const timeout = AbortSignal.timeout(requestTimeoutMs());
    const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout;
    return await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers,
      signal,
      cache: "no-store",
    });
  } catch {
    return Response.json({ detail: "Backend service unavailable" }, { status: 503 });
  }
}

export async function proxyResponse(response: Response): Promise<Response> {
  const contentType = response.headers.get("content-type") ?? "application/json";
  return new Response(await response.text(), {
    status: response.status,
    headers: { "Content-Type": contentType },
  });
}

export async function setSessionCookie(token: string, maxAge: number) {
  (await cookies()).set(SESSION_COOKIE, token, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge,
  });
}

export async function clearSessionCookie() {
  (await cookies()).delete(SESSION_COOKIE);
}
