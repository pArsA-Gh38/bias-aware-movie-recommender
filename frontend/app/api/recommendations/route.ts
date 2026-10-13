import { cookies } from "next/headers";
import { backendFetch, proxyResponse, SESSION_COOKIE } from "@/lib/server/backend";
import type {
  RecommendationConfigurationKey,
  RecommendationRequest,
} from "@/lib/api/types";

const CONFIGURATION_KEYS: ReadonlySet<string> = new Set([
  "A0_L1",
  "A1_L07",
  "A1_L06",
]);

function isConfigurationKey(value: unknown): value is RecommendationConfigurationKey {
  return typeof value === "string" && CONFIGURATION_KEYS.has(value);
}

export async function POST(request: Request) {
  if (!(await cookies()).get(SESSION_COOKIE)?.value) {
    return Response.json({ detail: "Not authenticated" }, { status: 401 });
  }

  const body: unknown = await request.json().catch(() => null);
  if (
    typeof body !== "object"
    || body === null
    || Array.isArray(body)
    || Object.keys(body).length !== 1
    || !isConfigurationKey((body as Record<string, unknown>).configuration_key)
  ) {
    return Response.json(
      { detail: "Invalid recommendation configuration" },
      { status: 422 },
    );
  }

  const recommendation: RecommendationRequest = {
    configuration_key: (body as RecommendationRequest).configuration_key,
  };
  return proxyResponse(await backendFetch("/recommendations", {
    method: "POST",
    body: JSON.stringify(recommendation),
  }, true));
}
