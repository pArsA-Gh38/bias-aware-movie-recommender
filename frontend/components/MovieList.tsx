import type { Movie } from "@/lib/api/types";
import { EmptyState } from "./AsyncStates";
import { MovieCard } from "./MovieCard";
import styles from "./MovieList.module.css";

export function MovieList({
  movies,
  ratings,
  onRate,
  onReset,
}: {
  movies: Movie[];
  ratings: Map<number, number>;
  onRate: (movieId: number, rating: number) => Promise<void>;
  onReset?: () => void;
}) {
  if (movies.length === 0) {
    return <EmptyState message="No movies found." action={onReset && <button className="button" onClick={onReset}>Reset filters</button>} />;
  }
  return <ul className={styles.grid}>{movies.map((movie) => (
    <li key={movie.movie_id}><MovieCard movie={movie} rating={ratings.get(movie.movie_id)} onRate={onRate} /></li>
  ))}</ul>;
}
