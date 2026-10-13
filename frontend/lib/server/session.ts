import type { User } from "@/lib/api/types";
import { backendFetch } from "./backend";

export async function getSession(): Promise<User | null> {
  const response = await backendFetch("/auth/me", {}, true);
  if (!response.ok) return null;
  return response.json() as Promise<User>;
}
