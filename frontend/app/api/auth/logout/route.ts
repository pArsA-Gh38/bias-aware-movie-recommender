import { clearSessionCookie } from "@/lib/server/backend";

export async function POST() {
  await clearSessionCookie();
  return Response.json({ ok: true });
}
