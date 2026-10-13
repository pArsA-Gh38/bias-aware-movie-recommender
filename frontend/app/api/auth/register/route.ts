import type { Credentials, User } from "@/lib/api/types";
import { backendFetch, proxyResponse, setSessionCookie } from "@/lib/server/backend";

type TokenResponse = { access_token: string; expires_in: number };

export async function POST(request: Request) {
  const credentials = (await request.json()) as Credentials;
  const registerResponse = await backendFetch("/auth/register", {
    method: "POST",
    body: JSON.stringify(credentials),
  });
  if (!registerResponse.ok) return proxyResponse(registerResponse);
  const user = (await registerResponse.json()) as User;
  const loginResponse = await backendFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify(credentials),
  });
  if (!loginResponse.ok) return proxyResponse(loginResponse);
  const token = (await loginResponse.json()) as TokenResponse;
  await setSessionCookie(token.access_token, token.expires_in);
  return Response.json(user, { status: 201 });
}
