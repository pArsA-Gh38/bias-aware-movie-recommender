import type { Credentials, User } from "@/lib/api/types";
import { backendFetch, proxyResponse, setSessionCookie } from "@/lib/server/backend";

type TokenResponse = { access_token: string; expires_in: number };

export async function POST(request: Request) {
  const credentials = (await request.json()) as Credentials;
  const loginResponse = await backendFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify(credentials),
  });
  if (!loginResponse.ok) return proxyResponse(loginResponse);

  const token = (await loginResponse.json()) as TokenResponse;
  await setSessionCookie(token.access_token, token.expires_in);
  const meResponse = await backendFetch("/auth/me", {}, true);
  if (!meResponse.ok) return proxyResponse(meResponse);
  return Response.json((await meResponse.json()) as User);
}
