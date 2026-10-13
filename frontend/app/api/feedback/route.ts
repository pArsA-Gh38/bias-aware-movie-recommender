import { cookies } from "next/headers";
import { backendFetch, proxyResponse, SESSION_COOKIE } from "@/lib/server/backend";
import type { FeedbackAction, FeedbackRequest } from "@/lib/api/types";

const FEEDBACK_ACTIONS: ReadonlySet<string> = new Set(["click", "like", "dislike"]);
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function isFeedbackAction(value: unknown): value is FeedbackAction {
  return typeof value === "string" && FEEDBACK_ACTIONS.has(value);
}

function isFeedbackRequest(value: unknown): value is FeedbackRequest {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const body = value as Record<string, unknown>;
  const keys = Object.keys(body);
  return keys.length === 3
    && keys.every((key) => key === "event_id" || key === "movie_id" || key === "action")
    && typeof body.event_id === "string"
    && UUID_PATTERN.test(body.event_id)
    && Number.isSafeInteger(body.movie_id)
    && isFeedbackAction(body.action);
}

export async function POST(request: Request) {
  if (!(await cookies()).get(SESSION_COOKIE)?.value) {
    return Response.json({ detail: "Not authenticated" }, { status: 401 });
  }

  const body: unknown = await request.json().catch(() => null);
  if (!isFeedbackRequest(body)) {
    return Response.json({ detail: "Invalid feedback request" }, { status: 422 });
  }

  return proxyResponse(await backendFetch("/feedback", {
    method: "POST",
    body: JSON.stringify(body),
  }, true));
}
