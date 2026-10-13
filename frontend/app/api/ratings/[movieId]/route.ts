import { backendFetch, proxyResponse } from "@/lib/server/backend";

export async function PUT(request: Request, { params }: { params: Promise<{ movieId: string }> }) {
  const { movieId } = await params;
  return proxyResponse(
    await backendFetch(
      `/ratings/${encodeURIComponent(movieId)}`,
      { method: "PUT", body: await request.text() },
      true,
    ),
  );
}
