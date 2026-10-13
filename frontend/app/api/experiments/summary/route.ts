import { backendFetch, proxyResponse } from "@/lib/server/backend";

export async function GET() {
  return proxyResponse(await backendFetch("/experiments/summary", { method: "GET" }, true));
}
