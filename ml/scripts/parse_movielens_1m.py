import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data.movielens_parser import (
    read_movies,
    read_ratings,
    read_users,
    validate_movies,
    validate_ratings,
    validate_referential_integrity,
    validate_users,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_DIRECTORY = REPOSITORY_ROOT / "data" / "raw" / "ml-1m"
PROCESSED_DIRECTORY = REPOSITORY_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIRECTORY / "data_manifest.json"


def update_parsing_manifest(
    datasets: dict[str, pd.DataFrame],
) -> None:
    """Record parser, schema, and Parquet metadata."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Data manifest does not exist. Run DATA-001 first.")

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Data manifest contains invalid JSON: {error}") from error

    if not isinstance(manifest, dict):
        raise TypeError("Data manifest must contain a JSON object")

    dataset_encodings = {
        "ratings": "ascii",
        "movies": "latin-1",
        "users": "ascii",
    }

    parsed_datasets: dict[str, object] = {}

    for dataset_name, dataframe in datasets.items():
        source_path = (RAW_DIRECTORY / f"{dataset_name}.dat").relative_to(
            REPOSITORY_ROOT
        )

        output_path = (PROCESSED_DIRECTORY / f"{dataset_name}.parquet").relative_to(
            REPOSITORY_ROOT
        )

        parsed_datasets[dataset_name] = {
            "source_path": str(source_path),
            "source_encoding": dataset_encodings[dataset_name],
            "row_count": len(dataframe),
            "columns": dataframe.columns.tolist(),
            "dtypes": dataframe.dtypes.astype(str).to_dict(),
            "output_path": str(output_path),
        }

    now_utc = datetime.now(UTC).isoformat()

    existing_parsing = manifest.get("parsing")

    if isinstance(existing_parsing, dict):
        existing_parsed_at = existing_parsing.get("parsed_at_utc")
    else:
        existing_parsed_at = None

    is_first_parse = not isinstance(existing_parsed_at, str)

    parsed_at_utc = (
        existing_parsed_at if isinstance(existing_parsed_at, str) else now_utc
    )
    manifest["parsing"] = {
        "separator": "::",
        "pandas_engine": "python",
        "output_format": "parquet",
        "parquet_engine": "pyarrow",
        "compression": "snappy",
        "datasets": parsed_datasets,
        "parsed_at_utc": parsed_at_utc,
    }

    if is_first_parse:
        manifest["updated_at_utc"] = parsed_at_utc

    serialized_manifest = (
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    temporary_path = MANIFEST_PATH.with_suffix(".json.tmp")

    try:
        temporary_path.write_text(
            serialized_manifest,
            encoding="utf-8",
        )
        temporary_path.replace(MANIFEST_PATH)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print("Parsing metadata written to data manifest.")


def main() -> None:
    """Parse, validate, and store the MovieLens 1M source files."""

    ratings = read_ratings(RAW_DIRECTORY / "ratings.dat")
    movies = read_movies(RAW_DIRECTORY / "movies.dat")
    users = read_users(RAW_DIRECTORY / "users.dat")

    validate_ratings(ratings)
    validate_movies(movies)
    validate_users(users)
    validate_referential_integrity(ratings, movies, users)

    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)

    datasets = {
        "ratings": ratings,
        "movies": movies,
        "users": users,
    }

    for dataset_name, dataframe in datasets.items():
        output_path = PROCESSED_DIRECTORY / f"{dataset_name}.parquet"

        dataframe.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        relative_path = output_path.relative_to(REPOSITORY_ROOT)

        print(f"{dataset_name}: {len(dataframe)} rows written to {relative_path}")
    update_parsing_manifest(datasets)


if __name__ == "__main__":
    main()
