import { backendFetch, proxyResponse } from "@/lib/server/backend";

export async function GET(request: Request) {
  return proxyResponse(await backendFetch(`/movies${new URL(request.url).search}`, {}, true));
}
