"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { api } from "@/lib/api/client";
import { SpectrumStrip } from "./SpectrumStrip";
import styles from "./AuthForm.module.css";

export function AuthForm({ mode }: { mode: "login" | "register" }) {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const isLogin = mode === "login";

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setPending(true);
    const data = new FormData(event.currentTarget);
    try {
      await api[mode]({ email: String(data.get("email")), password: String(data.get("password")) });
      router.replace("/movies");
      router.refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Authentication failed.");
    } finally {
      setPending(false);
    }
  }

  return (
    <div className={styles.loginPage}>
      <div className={styles.vignette} aria-hidden="true" />
      <form className={styles.loginCard} onSubmit={submit}>
        <div className={styles.brand}>
          <SpectrumStrip />
          <p className={styles.eyebrow}>Movie Recommender</p>
        </div>

        <div>
          <h1 className={styles.heading}>
            {isLogin ? "Sign in" : "Create an account"}
          </h1>
          {!isLogin && <span className={styles.headingAccent} aria-hidden="true" />}
          <p className={styles.subheading}>
            {isLogin
              ? "Pick up where you left off and keep discovering films beyond the obvious picks."
              : "Start with a few movie ratings."}
          </p>
        </div>

        {error && (
          <div className={styles.errorBanner} role="alert">
            <span className={styles.errorIcon} aria-hidden="true">!</span>
            <span>{error}</span>
          </div>
        )}

        <div className={styles.fields}>
          <label className={styles.field}>
            Email
            <input
              className={styles.input}
              name="email"
              type="email"
              autoComplete="email"
              placeholder={isLogin ? "you@example.com" : undefined}
              required
            />
          </label>
          <label className={styles.field}>
            Password
            <input
              className={styles.input}
              name="password"
              type="password"
              autoComplete={isLogin ? "current-password" : "new-password"}
              minLength={8}
              maxLength={128}
              placeholder={isLogin ? "••••••••" : undefined}
              required
            />
          </label>
        </div>

        <button
          className={styles.submit}
          type="submit"
          disabled={pending}
          aria-busy={pending}
        >
          {pending ? (isLogin ? "Signing in…" : "Please wait…") : isLogin ? "Sign in" : "Register"}
        </button>
        <span className={styles.srOnly} role="status" aria-live="polite">
          {pending ? (isLogin ? "Signing in, please wait." : "Registering, please wait.") : ""}
        </span>

        <p className={styles.authLinkLine}>
          {isLogin ? "New here?" : "Already registered?"}{" "}
          <Link className={styles.authLink} href={isLogin ? "/register" : "/login"}>
            {isLogin ? "Create an account" : "Sign in"}
          </Link>
        </p>
      </form>
    </div>
  );
}
