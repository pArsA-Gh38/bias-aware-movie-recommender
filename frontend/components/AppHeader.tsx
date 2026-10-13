import Link from "next/link";
import { LogoutButton } from "./LogoutButton";

export function AppHeader({ email }: { email: string }) {
  return <header className="header"><div className="shell header-inner">
    <Link className="brand" href="/movies">Movie Recommender</Link>
    <nav className="nav" aria-label="Primary">
      <Link href="/movies">Movies</Link>
      <Link href="/recommendations">Recommendations</Link>
      <Link href="/experiments">Experiments</Link>
      <span className="muted">{email}</span>
      <LogoutButton />
    </nav>
  </div></header>;
}
