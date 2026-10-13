import { redirect } from "next/navigation";
import { AppHeader } from "@/components/AppHeader";
import { getSession } from "@/lib/server/session";

export default async function ProtectedLayout({ children }: { children: React.ReactNode }) {
  const user = await getSession();
  if (!user) redirect("/login");
  return <><AppHeader email={user.email} /><main className="shell main">{process.env.DEMO_DATASET_LABEL && <p className="muted">{process.env.DEMO_DATASET_LABEL}</p>}{children}</main></>;
}
