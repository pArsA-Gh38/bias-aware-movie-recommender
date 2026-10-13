import styles from "./SpectrumStrip.module.css";

const spectrumChips = [
  "var(--chip-amber)",
  "var(--chip-teal)",
  "var(--chip-violet)",
  "var(--chip-rose)",
  "var(--chip-blue)",
] as const;

export function SpectrumStrip() {
  return (
    <div className={styles.strip} aria-hidden="true">
      {spectrumChips.map((color, index) => (
        <span
          className={styles.chip}
          key={color}
          style={{ backgroundColor: color, animationDelay: `${index * 140}ms` }}
        />
      ))}
    </div>
  );
}
