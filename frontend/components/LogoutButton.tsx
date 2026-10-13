"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { api } from "@/lib/api/client";

export function LogoutButton() {
  const router = useRouter();
  const [pending, setPending] = useState(false);
  return <button className="button-link" disabled={pending} onClick={async () => {
    setPending(true);
    try { await api.logout(); } finally { router.replace("/login"); router.refresh(); }
  }}>{pending ? "Signing out…" : "Sign out"}</button>;
}
