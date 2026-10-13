"use client";

import { useState } from "react";
import type { Movie } from "@/lib/api/types";
import { RatingControl } from "./RatingControl";
import styles from "./MovieCard.module.css";

const ACCENT_CLASSES = [styles.amber, styles.teal, styles.violet, styles.rose, styles.blue];

export function MovieCard({
  movie,
  rating,
  onRate,
}: {
  movie: Movie;
  rating?: number;
  onRate: (movieId: number, rating: number) => Promise<void>;
}) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function rate(value: number) {
    setPending(true);
    setError(null);
    try {
      await onRate(movie.movie_id, value);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not save rating.");
    } finally {
      setPending(false);
    }
  }

  return (
    <article className={styles.card}>
      <div className={`${styles.decorativeHeader} ${ACCENT_CLASSES[Math.abs(movie.movie_id) % ACCENT_CLASSES.length]}`} aria-hidden="true"><span>{movie.title.charAt(0).toUpperCase()}</span></div>
      <div className={styles.body}>
        <h2>{movie.title}</h2>
        <p className={styles.metadata}>Movie #{movie.movie_id}{movie.release_year ? ` · ${movie.release_year}` : ""}</p>
        {movie.genres.length > 0 && <ul className={styles.genres} aria-label="Genres">{movie.genres.map((genre) => <li key={genre}>{genre}</li>)}</ul>}
        <div className={styles.ratingSlot}><RatingControl value={rating} pending={pending} onChange={rate} /></div>
        {error && <p className={styles.error} role="alert">{error} Choose a rating to retry.</p>}
      </div>
    </article>
  );
}
