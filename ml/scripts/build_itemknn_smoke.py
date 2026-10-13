"""Build the deterministic train-only sparse ItemKNN smoke graph.

The configured neighbor count of 100 is a neutral MODEL-002 implementation
smoke value.  It is not tuned or selected; the later comparison task owns the
three-value validation experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype
from scipy.sparse import csr_matrix

from src.models.itemknn import (
    ItemKNNModel,
    build_binary_user_item_matrix,
    fit_itemknn,
)

TRAIN_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
USER_MAPPING_COLUMNS = ["fit_scope", "user_id", "user_index"]
ITEM_MAPPING_COLUMNS = ["fit_scope", "movie_id", "item_index"]
SOURCE_FILENAMES = {
    "train": "train.parquet",
    "user_mapping": "user_mapping.parquet",
    "item_mapping": "item_mapping.parquet",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_itemknn_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the untuned implementation-smoke config."""

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    required = {
        "model_name": "itemknn",
        "algorithm": "binary_item_cosine",
        "fit_scope": "train",
        "positive_threshold": 4,
        "similarity": "cosine",
        "neighbor_count": 100,
        "neighbor_tie_break": ["similarity_desc", "movie_id_asc"],
        "candidate_score": "sum_neighbor_similarity",
        "candidate_tie_break": ["score_desc", "movie_id_asc"],
        "remove_self_neighbor": True,
        "seen_filter": "all_ratings_at_or_before_train_cutoff",
        "selection_status": "not_tuned",
        "purpose": "MODEL-002 implementation smoke",
        "stochastic": False,
        "seed": None,
    }
    missing = sorted(set(required).difference(config))
    if missing:
        raise ValueError(f"ItemKNN config is missing fields: {missing}")
    for key, expected in required.items():
        if config[key] != expected:
            raise ValueError(f"ItemKNN config has invalid {key}")
    forbidden = {"generated_at", "created_at", "timestamp"}
    if forbidden.intersection(config):
        raise ValueError("ItemKNN config must not contain timestamps")
    return config


def _validate_integer_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    *,
    name: str,
) -> None:
    invalid = {
        column: str(dataframe[column].dtype)
        for column in columns
        if is_bool_dtype(dataframe[column].dtype)
        or not is_integer_dtype(dataframe[column].dtype)
    }
    if invalid:
        raise TypeError(f"{name} integer columns have invalid dtypes: {invalid}")


def _validate_train(train: pd.DataFrame) -> None:
    if not isinstance(train, pd.DataFrame) or train.empty:
        raise ValueError("train must be a non-empty pandas DataFrame")
    if train.columns.tolist() != TRAIN_COLUMNS:
        raise ValueError("train has unexpected columns")
    if train.isna().any().any():
        raise ValueError("train must not contain null values")
    _validate_integer_columns(train, TRAIN_COLUMNS, name="train")
    if not train["rating"].isin([4, 5]).all():
        raise ValueError("train must contain only ratings 4 and 5")
    if not train["user_id"].gt(0).all() or not train["movie_id"].gt(0).all():
        raise ValueError("train IDs must be positive")
    if train.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("train contains duplicate user-item interactions")


def _select_mapping(
    mapping: pd.DataFrame,
    *,
    columns: list[str],
    id_column: str,
    index_column: str,
    expected_ids: pd.Index,
) -> tuple[int, ...]:
    if mapping.columns.tolist() != columns:
        raise ValueError(f"{id_column} mapping has unexpected columns")
    if mapping.isna().any().any():
        raise ValueError(f"{id_column} mapping contains null values")
    _validate_integer_columns(
        mapping,
        [id_column, index_column],
        name=f"{id_column} mapping",
    )
    selected = mapping.loc[mapping["fit_scope"].eq("train")].copy()
    if selected.empty:
        raise ValueError(f"train {id_column} mapping is empty")
    expected_indices = np.arange(len(selected), dtype=np.int64)
    if not np.array_equal(selected[index_column].to_numpy(), expected_indices):
        raise ValueError(f"train {id_column} mapping is not contiguous")
    if selected[id_column].duplicated().any():
        raise ValueError(f"train {id_column} mapping contains duplicate IDs")
    expected = tuple(int(value) for value in expected_ids.sort_values())
    actual = tuple(int(value) for value in selected[id_column])
    if actual != expected:
        raise ValueError(f"train {id_column} mapping does not match train")
    return actual


def load_train_inputs(
    processed_directory: Path,
) -> tuple[pd.DataFrame, tuple[int, ...], tuple[int, ...], dict[str, str]]:
    """Load only train positives and the published train mapping rows."""

    paths = {
        name: Path(processed_directory) / filename
        for name, filename in SOURCE_FILENAMES.items()
    }
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    train = frames["train"]
    _validate_train(train)
    user_ids = _select_mapping(
        frames["user_mapping"],
        columns=USER_MAPPING_COLUMNS,
        id_column="user_id",
        index_column="user_index",
        expected_ids=pd.Index(train["user_id"].unique()),
    )
    item_ids = _select_mapping(
        frames["item_mapping"],
        columns=ITEM_MAPPING_COLUMNS,
        id_column="movie_id",
        index_column="item_index",
        expected_ids=pd.Index(train["movie_id"].unique()),
    )
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    return train, user_ids, item_ids, hashes


def build_smoke_model(
    train: pd.DataFrame,
    user_ids: tuple[int, ...],
    item_ids: tuple[int, ...],
    *,
    n_neighbors: int,
) -> tuple[csr_matrix, ItemKNNModel]:
    """Build the binary matrix and sparse graph from train positives only."""

    interactions = train[["user_id", "movie_id"]].itertuples(
        index=False,
        name=None,
    )
    matrix = build_binary_user_item_matrix(
        interactions,
        user_ids,
        item_ids,
    )
    model = fit_itemknn(matrix, item_ids, n_neighbors)
    return matrix, model


def structural_summary(
    matrix: csr_matrix,
    model: ItemKNNModel,
    *,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Validate and summarize sparse MODEL-002 structural evidence."""

    similarities = model.similarities
    row_counts = np.diff(similarities.indptr)
    diagonal_nnz = int(np.count_nonzero(similarities.diagonal()))
    if matrix.nnz != len(matrix.data) or not np.equal(matrix.data, 1.0).all():
        raise ValueError("binary matrix contains invalid stored values")
    if similarities.shape != (matrix.shape[1], matrix.shape[1]):
        raise ValueError("similarity graph shape does not match item catalog")
    if diagonal_nnz:
        raise ValueError("similarity graph contains self-neighbors")
    if row_counts.size and int(row_counts.max()) > model.n_neighbors:
        raise ValueError("similarity graph row exceeds neighbor limit")
    if not np.isfinite(similarities.data).all():
        raise ValueError("similarity graph contains a non-finite value")
    if np.any(similarities.data <= 0.0) or np.any(similarities.data > 1.0):
        raise ValueError("similarity graph contains an invalid cosine")

    return {
        "model": "itemknn",
        "fit_scope": "train",
        "n_neighbors": model.n_neighbors,
        "selection_status": "not_tuned",
        "purpose": "MODEL-002 implementation smoke",
        "binary_matrix": {
            "shape": list(matrix.shape),
            "nnz": int(matrix.nnz),
        },
        "similarities": {
            "shape": list(similarities.shape),
            "nnz": int(similarities.nnz),
            "diagonal_nnz": diagonal_nnz,
            "max_row_neighbors": (int(row_counts.max()) if row_counts.size else 0),
            "empty_neighbor_rows": int(np.count_nonzero(row_counts == 0)),
            "sparse_data_bytes": int(
                similarities.data.nbytes
                + similarities.indices.nbytes
                + similarities.indptr.nbytes
            ),
            "dense_float64_equivalent_bytes": int(
                similarities.shape[0] * similarities.shape[1] * 8
            ),
        },
        "source_data_hashes": dict(sorted(source_hashes.items())),
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def write_deterministic_sparse_npz(
    matrix: csr_matrix,
    output_path: Path,
) -> None:
    """Write SciPy-compatible NPZ bytes with fixed ZIP metadata."""

    members = {
        "data.npy": _npy_bytes(matrix.data),
        "format.npy": _npy_bytes(np.asarray(matrix.format.encode("ascii"))),
        "indices.npy": _npy_bytes(matrix.indices),
        "indptr.npy": _npy_bytes(matrix.indptr),
        "shape.npy": _npy_bytes(np.asarray(matrix.shape)),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, members[name])
    Path(output_path).write_bytes(buffer.getvalue())


def write_smoke_outputs(
    output_directory: Path,
    model: ItemKNNModel,
    summary: dict[str, Any],
) -> dict[str, str]:
    """Write only explicitly requested temporary smoke outputs."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    similarities_path = directory / "similarities.npz"
    item_ids_path = directory / "item_ids.json"
    manifest_path = directory / "smoke_manifest.json"
    write_deterministic_sparse_npz(model.similarities, similarities_path)
    item_ids_path.write_text(
        json.dumps(model.item_ids, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        path.name: _sha256_file(path)
        for path in (similarities_path, item_ids_path, manifest_path)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the untuned train-only ItemKNN smoke graph"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional new directory for temporary sparse outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    config = load_itemknn_config(
        repository_root / "configs/itemknn_implementation.json"
    )
    train, user_ids, item_ids, hashes = load_train_inputs(
        repository_root / "data/processed"
    )
    matrix, model = build_smoke_model(
        train,
        user_ids,
        item_ids,
        n_neighbors=config["neighbor_count"],
    )
    summary = structural_summary(matrix, model, source_hashes=hashes)
    if args.output_dir is not None:
        summary["output_hashes"] = write_smoke_outputs(
            args.output_dir,
            model,
            summary,
        )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
