import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.positive_interactions import (
    MINIMUM_POSITIVE_RATING,
    POSITIVE_RATING_VALUES,
    POSITIVE_SORT_COLUMNS,
    build_positive_interactions,
    validate_positive_interactions,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIRECTORY = REPOSITORY_ROOT / "data" / "processed"
RATINGS_PATH = PROCESSED_DIRECTORY / "ratings.parquet"
POSITIVES_PATH = PROCESSED_DIRECTORY / "positives.parquet"
MANIFEST_PATH = PROCESSED_DIRECTORY / "data_manifest.json"
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"


def _relative_path(path: Path) -> str:
    return str(path.relative_to(REPOSITORY_ROOT))


def _calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def _load_manifest() -> dict[str, object]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Data manifest is missing. Run make data first.")

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Data manifest contains invalid JSON: {error}") from error

    if not isinstance(manifest, dict):
        raise TypeError("Data manifest must contain a JSON object")

    return manifest


def _atomic_write_parquet(dataframe: pd.DataFrame) -> None:
    temporary_path = POSITIVES_PATH.with_name(f"{POSITIVES_PATH.name}.tmp")

    try:
        dataframe.to_parquet(
            temporary_path,
            engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
            index=False,
        )
        temporary_path.replace(POSITIVES_PATH)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_manifest(manifest: dict[str, object]) -> None:
    temporary_path = MANIFEST_PATH.with_name(f".{MANIFEST_PATH.name}.tmp")
    serialized = (
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    try:
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(MANIFEST_PATH)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _update_manifest(
    manifest: dict[str, object],
    ratings: pd.DataFrame,
    positives: pd.DataFrame,
    output_sha256: str,
) -> None:
    source_row_count = len(ratings)
    positive_row_count = len(positives)
    rating_value_counts = positives["rating"].value_counts()
    positive_ratio = positive_row_count / source_row_count if source_row_count else 0.0

    manifest["positive_interactions"] = {
        "source_path": _relative_path(RATINGS_PATH),
        "output_path": _relative_path(POSITIVES_PATH),
        "rule": {
            "column": "rating",
            "operator": ">=",
            "threshold": MINIMUM_POSITIVE_RATING,
            "included_values": list(POSITIVE_RATING_VALUES),
        },
        "source_row_count": source_row_count,
        "positive_row_count": positive_row_count,
        "excluded_row_count": source_row_count - positive_row_count,
        "positive_ratio": positive_ratio,
        "rating_value_counts": {
            str(value): int(rating_value_counts.get(value, 0))
            for value in POSITIVE_RATING_VALUES
        },
        "columns": positives.columns.tolist(),
        "dtypes": positives.dtypes.astype(str).to_dict(),
        "sort_order": POSITIVE_SORT_COLUMNS,
        "parquet_engine": PARQUET_ENGINE,
        "compression": PARQUET_COMPRESSION,
        "output_sha256": output_sha256,
    }
    _atomic_write_manifest(manifest)


def main() -> None:
    """Build and validate the deterministic DATA-004 Parquet output."""

    if not RATINGS_PATH.exists():
        raise FileNotFoundError("Parsed ratings are missing. Run make data first.")

    manifest = _load_manifest()
    ratings = pd.read_parquet(RATINGS_PATH)
    positives = build_positive_interactions(ratings)

    _atomic_write_parquet(positives)

    stored_positives = pd.read_parquet(POSITIVES_PATH)
    validate_positive_interactions(stored_positives, ratings)
    output_sha256 = _calculate_sha256(POSITIVES_PATH)
    _update_manifest(
        manifest,
        ratings,
        stored_positives,
        output_sha256,
    )

    print("MovieLens 1M positive interactions generated:")
    print(f"- source rows: {len(ratings)}")
    print(f"- positive rows: {len(stored_positives)}")
    print(f"- excluded rows: {len(ratings) - len(stored_positives)}")
    print(f"- output: {_relative_path(POSITIVES_PATH)}")
    print("- data manifest updated successfully")


if __name__ == "__main__":
    main()
