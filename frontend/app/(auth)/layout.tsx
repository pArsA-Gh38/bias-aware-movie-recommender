import { redirect } from "next/navigation";
import { getSession } from "@/lib/server/session";

export default async function AuthLayout({ children }: { children: React.ReactNode }) {
  if (await getSession()) redirect("/movies");
  return <main className="auth-shell">{children}</main>;
}
