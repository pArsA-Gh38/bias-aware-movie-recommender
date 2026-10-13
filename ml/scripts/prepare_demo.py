"""Train an independent, item-only local ALS demo; never rerun frozen research."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from recommender_core.local_artifact import load_local_bundle, write_local_bundle
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
GENRES = (
    "Drama",
    "Comedy",
    "Adventure",
    "Mystery",
    "Sci-Fi",
    "Romance",
    "Thriller",
    "Animation",
)


def synthetic() -> tuple[csr_matrix, np.ndarray, list[dict]]:
    rng = np.random.default_rng(42)
    ids = np.arange(1, 161, dtype=np.int64)
    names = (
        "Quiet",
        "Golden",
        "Hidden",
        "Distant",
        "Silver",
        "Midnight",
        "Lost",
        "Bright",
    )
    subjects = (
        "Horizon",
        "Journey",
        "Garden",
        "Signal",
        "Harbor",
        "Promise",
        "Echo",
        "River",
        "Voyage",
        "Dream",
        "Trail",
        "Story",
        "Moon",
        "Letter",
        "World",
        "Sky",
        "Tide",
        "Bridge",
        "Light",
        "Road",
    )
    catalog = [
        {
            "movie_id": int(i),
            "title": f"{names[(i - 1) % 8]} {subjects[(i - 1) // 8]} ({2000 + (i % 24)})",
            "genres": [GENRES[(i - 1) % 8], GENRES[(i + 2) % 8]],
        }
        for i in ids
    ]
    rows, cols = [], []
    for user in range(240):
        preferred = np.arange(user % 8, 160, 8)
        chosen = np.unique(
            np.r_[
                rng.choice(preferred, 12, replace=False),
                rng.choice(160, 8, replace=False),
            ]
        )
        rows.extend([user] * len(chosen))
        cols.extend(chosen.tolist())
    # Every fictional catalog item has at least one training interaction.
    rows.extend((np.arange(160) % 240).tolist())
    cols.extend(range(160))
    matrix = csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(240, 160), dtype=np.float32
    )
    matrix.data[:] = 11.0
    return matrix, ids, catalog


def movielens() -> tuple[csr_matrix, np.ndarray, list[dict]]:
    from scripts.download_movielens_1m import main as download
    from scripts.extract_movielens_1m import extract_movielens
    from src.data.movielens_parser import read_movies, read_ratings

    download()
    extract_movielens()
    raw = ROOT / "data/raw/ml-1m"
    ratings = read_ratings(raw / "ratings.dat").sort_values(
        ["timestamp", "user_id", "movie_id"], kind="stable"
    )
    train = ratings.iloc[: int(len(ratings) * 0.8)]
    positives = train.loc[train["rating"] >= 4]
    ids = np.sort(positives["movie_id"].unique()).astype(np.int64)
    users = np.sort(positives["user_id"].unique())
    user_index = {value: index for index, value in enumerate(users)}
    item_index = {value: index for index, value in enumerate(ids)}
    matrix = csr_matrix(
        (
            np.full(len(positives), 11.0, dtype=np.float32),
            (
                positives["user_id"].map(user_index),
                positives["movie_id"].map(item_index),
            ),
        ),
        shape=(len(users), len(ids)),
    )
    metadata = read_movies(raw / "movies.dat").set_index("movie_id")
    catalog = [
        {
            "movie_id": int(i),
            "title": str(metadata.loc[i, "title"]),
            "genres": str(metadata.loc[i, "genres"]).split("|"),
        }
        for i in ids
    ]
    return matrix, ids, catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("synthetic", "movielens"), default="synthetic"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / ".local" / args.source / "artifact"
    if output.exists():
        load_local_bundle(output)
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["provenance"]["source"] != args.source:
            raise ValueError("existing local artifact belongs to another dataset")
        print(f"Verified existing {args.source} local artifact.")
        return
    matrix, ids, catalog = synthetic() if args.source == "synthetic" else movielens()
    matrix.sort_indices()
    digest = hashlib.sha256()
    for array in (ids, matrix.indptr, matrix.indices, matrix.data):
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    model = AlternatingLeastSquares(
        factors=32,
        regularization=0.01,
        iterations=30,
        alpha=1.0,
        random_state=42,
        num_threads=1,
    )
    model.fit(matrix, show_progress=False)
    counts = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(np.int64)
    write_local_bundle(
        output,
        item_ids=ids,
        item_factors=model.item_factors.astype(np.float32),
        popularity_counts=counts,
        catalog=catalog,
        provenance={
            "source": args.source,
            "training_matrix_sha256": digest.hexdigest(),
            "seed": 42,
            "factors": 32,
            "regularization": 0.01,
            "iterations": 30,
            "confidence_alpha": 10,
            "observed_confidence": 11,
            "implicit_alpha": 1,
            "evaluation_performed": False,
            "fit_scope": "synthetic interactions"
            if args.source == "synthetic"
            else "earliest 80% of globally ordered ratings; positives >=4",
        },
    )
    print(
        f"Independent {args.source} local model ready: {len(ids)} items; no research evaluation performed."
    )


if __name__ == "__main__":
    main()
