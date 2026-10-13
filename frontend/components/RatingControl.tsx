"use client";

import styles from "./RatingControl.module.css";

export function RatingControl({
  value,
  pending,
  onChange,
}: {
  value?: number;
  pending: boolean;
  onChange: (rating: number) => void;
}) {
  return (
    <fieldset className={styles.fieldset} disabled={pending}>
      <legend>{pending ? "Saving rating…" : value ? `Your rating: ${value}` : "Rate this movie"}</legend>
      <div className={styles.buttons}>
        {[1, 2, 3, 4, 5].map((rating) => (
          <button
            type="button"
            key={rating}
            className={value === rating ? styles.selected : undefined}
            aria-label={`Rate ${rating} out of 5`}
            aria-pressed={value === rating}
            onClick={() => onChange(rating)}
          >
            <span>{rating}</span>{value === rating && <span className={styles.check} aria-hidden="true">✓</span>}
          </button>
        ))}
      </div>
    </fieldset>
  );
}
