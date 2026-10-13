"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ApiError, api } from "@/lib/api/client";
import type { Movie } from "@/lib/api/types";
import { ErrorState, LoadingState } from "./AsyncStates";
import { MovieList } from "./MovieList";
import styles from "./MoviesPageClient.module.css";

export function MoviesPageClient() {
  const router = useRouter();
  const [movies, setMovies] = useState<Movie[]>([]);
  const [ratings, setRatings] = useState<Map<number, number>>(new Map());
  const [positiveCount, setPositiveCount] = useState(0);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [onboarding, setOnboarding] = useState(true);
  const [query, setQuery] = useState("");
  const [genre, setGenre] = useState("");
  const [filters, setFilters] = useState({ query: "", genre: "" });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadMovies = useCallback(async (append = false, cursor?: string) => {
    setLoading(true);
    setError(null);
    const parameters = new URLSearchParams({ limit: "24" });
    if (onboarding) parameters.set("onboarding", "true");
    else {
      if (filters.query) parameters.set("q", filters.query);
      if (filters.genre) parameters.set("genre", filters.genre);
      if (cursor) parameters.set("cursor", cursor);
    }
    try {
      const response = await api.movies(parameters);
      setMovies((current) => append ? [...current, ...response.items] : response.items);
      setNextCursor(response.next_cursor);
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 401) router.replace("/login");
      setError(caught instanceof Error ? caught.message : "Could not load movies.");
    } finally {
      setLoading(false);
    }
  }, [filters, onboarding, router]);

  useEffect(() => {
    api.ratings().then((response) => {
      const next = new Map(response.items.map((item) => [item.movie_id, item.rating]));
      setRatings(next);
      setPositiveCount(response.items.filter((item) => item.rating >= 4).length);
    }).catch(() => undefined);
  }, []);

  useEffect(() => {
    // The request owns the loading/error transitions for every filter change.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void loadMovies();
  }, [loadMovies]);

  const remaining = Math.max(0, 5 - positiveCount);
  const ratingsByMovie = useMemo(() => ratings, [ratings]);

  async function rate(movieId: number, value: number) {
    const previous = ratings.get(movieId);
    setRatings((current) => new Map(current).set(movieId, value));
    try {
      const response = await api.rate(movieId, value);
      setPositiveCount(response.positive_count);
    } catch (caught) {
      setRatings((current) => {
        const rollback = new Map(current);
        if (previous === undefined) rollback.delete(movieId); else rollback.set(movieId, previous);
        return rollback;
      });
      throw caught;
    }
  }

  function resetFilters() {
    setQuery("");
    setGenre("");
    setFilters({ query: "", genre: "" });
  }

  const complete = remaining === 0;
  const filledDots = Math.min(positiveCount, 5);

  return <div className={styles.moviesPage}>
    <div className={styles.container}>
      <header className={styles.pageHeader}>
        <h1>Movies</h1>
        <p>Rate movies from 1 to 5. Ratings of 4 or 5 count toward recommendations.</p>
      </header>

      <section className={styles.progressSection} aria-labelledby="onboarding-progress-heading">
        <h2 id="onboarding-progress-heading" className={styles.srOnly}>Onboarding progress</h2>
        <div className={`${styles.progressPanel} ${complete ? styles.progressComplete : ""}`}>
          <div className={styles.progressDots} aria-hidden="true">
            {Array.from({ length: 5 }, (_, index) => <span
              key={index}
              className={`${styles.progressDot} ${index < filledDots ? styles.progressDotFilled : ""}`}
            />)}
          </div>
          <div className={styles.progressCopy}>
            <p className={styles.progressStatus} aria-live="polite">
              {complete ? "You are ready for recommendations." : `${remaining} more positive ${remaining === 1 ? "rating" : "ratings"} needed`}
            </p>
            {!complete && <p>Give at least five movies a 4 or 5 to unlock personalized recommendations.</p>}
          </div>
          {complete && <Link className={styles.primaryLink} href="/recommendations">View recommendations</Link>}
        </div>
      </section>

      <section className={styles.controls} aria-label="Browse movies">
        <button className={styles.secondaryButton} type="button" onClick={() => setOnboarding((value) => !value)}>
          {onboarding ? "Browse full catalog" : "Show onboarding picks"}
        </button>
        {!onboarding && <form className={styles.filters} aria-label="Search the catalog" onSubmit={(event) => { event.preventDefault(); setFilters({ query: query.trim(), genre: genre.trim() }); }}>
          <label>Title<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search by title" /></label>
          <label>Genre<input value={genre} onChange={(event) => setGenre(event.target.value)} placeholder="e.g. Comedy" /></label>
          <button className={styles.primaryButton} disabled={loading}>Search</button>
        </form>}
      </section>

      <section className={styles.results} aria-label="Movie results">
        {loading && movies.length === 0 ? <LoadingState label="Loading movies" /> : error ? <ErrorState message={error} onRetry={() => void loadMovies()} /> : <>
          <MovieList movies={movies} ratings={ratingsByMovie} onRate={rate} onReset={resetFilters} />
          {nextCursor && !onboarding && <div className={styles.loadMoreRow}><button className={styles.secondaryButton} disabled={loading} onClick={() => void loadMovies(true, nextCursor)}>{loading ? "Loading…" : "Load more"}</button></div>}
        </>}
      </section>
    </div>
  </div>;
}
