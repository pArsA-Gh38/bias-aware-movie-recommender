"""Independent item-only serving format for locally built portfolio demos.

This format carries no source-user histories or factors. It reuses the tested
fold-in/reranking runtime, without claiming the identity of the frozen research
artifact. Checksums provide integrity, not third-party authenticity.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from recommender_core.artifact import (
    ArtifactCompatibilityError,
    ArtifactIntegrityError,
    canonical_json_bytes,
    sha256_file,
)
from recommender_core.artifact_v2 import ServingBundleV2

FORMAT = "local-serving-v1"
PAYLOAD = frozenset(
    {"catalog.json", "item_ids.npy", "item_factors.npy", "popularity_counts.npy"}
)
FILES = PAYLOAD | {"manifest.json"}


def _versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("numpy", "scipy")}


def _identity(files: dict, provenance: dict) -> str:
    return hashlib.sha256(
        canonical_json_bytes({"files": files, "provenance": provenance})
    ).hexdigest()


def _validate(
    ids: np.ndarray, factors: np.ndarray, counts: np.ndarray, catalog: Any
) -> None:
    n = len(ids) if ids.ndim == 1 else 0
    if (
        not 110 <= n <= 100_000
        or ids.dtype != np.int64
        or np.any(ids <= 0)
        or np.any(np.diff(ids) <= 0)
    ):
        raise ArtifactIntegrityError(
            "local item IDs must be sorted, unique positive int64 values"
        )
    if (
        factors.dtype != np.float32
        or factors.shape != (n, 32)
        or not np.isfinite(factors).all()
    ):
        raise ArtifactIntegrityError(
            "local factors must be finite float32 with shape (items, 32)"
        )
    if (
        counts.dtype != np.int64
        or counts.shape != (n,)
        or np.any(counts < 0)
        or np.any(counts > 1_000_000_000)
        or counts.sum() <= 0
    ):
        raise ArtifactIntegrityError("invalid local popularity counts")
    if not isinstance(catalog, list) or len(catalog) != n:
        raise ArtifactIntegrityError("local catalog must align with every item")
    for index, row in enumerate(catalog):
        if not isinstance(row, dict) or set(row) != {"movie_id", "title", "genres"}:
            raise ArtifactIntegrityError("invalid local catalog fields")
        if type(row["movie_id"]) is not int or row["movie_id"] != int(ids[index]):
            raise ArtifactIntegrityError("local catalog ID alignment mismatch")
        if not isinstance(row["title"], str) or not row["title"].strip():
            raise ArtifactIntegrityError("local catalog titles must be nonempty")
        genres = row["genres"]
        if (
            not isinstance(genres, list)
            or not genres
            or any(not isinstance(g, str) or not g.strip() for g in genres)
        ):
            raise ArtifactIntegrityError("local catalog genres must be nonempty")


def load_local_bundle(directory: Path) -> ServingBundleV2:
    path = Path(directory)
    if path.is_symlink() or not path.is_dir():
        raise ArtifactIntegrityError("local artifact must be a real directory")
    entries = list(path.iterdir())
    if {p.name for p in entries} != FILES or any(
        p.is_symlink() or not p.is_file() for p in entries
    ):
        raise ArtifactIntegrityError("local artifact file inventory mismatch")
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
            raise ArtifactIntegrityError("unsupported local artifact format")
        if manifest.get("numerical_versions") != _versions():
            raise ArtifactCompatibilityError(
                "local artifact numerical versions differ from this runtime"
            )
        metadata = {
            name: {
                "sha256": sha256_file(path / name),
                "size_bytes": (path / name).stat().st_size,
            }
            for name in sorted(PAYLOAD)
        }
        provenance = manifest["provenance"]
        if (
            not isinstance(provenance, dict)
            or not isinstance(provenance.get("source"), str)
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(provenance.get("training_matrix_sha256", ""))
            )
        ):
            raise ArtifactIntegrityError("invalid local artifact provenance")
        if manifest.get("files") != metadata or manifest.get(
            "bundle_sha256"
        ) != _identity(metadata, provenance):
            raise ArtifactIntegrityError("local artifact checksum or identity mismatch")
        ids, factors, counts = (
            np.load(path / name, allow_pickle=False)
            for name in ("item_ids.npy", "item_factors.npy", "popularity_counts.npy")
        )
        catalog = json.loads((path / "catalog.json").read_text())
        _validate(ids, factors, counts, catalog)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, (ArtifactIntegrityError, ArtifactCompatibilityError)):
            raise
        raise ArtifactIntegrityError("invalid local artifact payload") from error
    probabilities = (counts.astype(np.float64) + 1) / (counts.sum() + len(counts))
    order = np.lexsort((ids, -counts))
    cumulative = np.cumsum(counts[order]) / counts.sum()
    tails = np.empty(len(ids), dtype=np.int8)
    tails[order] = np.where(cumulative <= 0.8, 0, np.where(cumulative <= 0.95, 1, 2))
    arrays = {
        "item_ids": ids,
        "item_factors": factors,
        "popularity_counts": counts,
        "popularity_probabilities": probabilities,
        "tail_groups": tails,
        # Empty compatibility arrays exist only in memory; source-user state is never saved.
        "user_ids": np.empty(0, dtype=np.int64),
        "user_factors": np.empty((0, 32), dtype=np.float32),
        "fit_positive_counts": np.empty(0, dtype=np.int64),
        "seen_indices": np.empty(0, dtype=np.int32),
        "seen_indptr": np.zeros(1, dtype=np.int64),
    }
    return ServingBundleV2(arrays, [frozenset(row["genres"]) for row in catalog])


def write_local_bundle(
    directory: Path,
    *,
    item_ids: np.ndarray,
    item_factors: np.ndarray,
    popularity_counts: np.ndarray,
    catalog: list[dict],
    provenance: dict,
) -> Path:
    path = Path(directory)
    _validate(item_ids, item_factors, popularity_counts, catalog)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite an existing local artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".local-artifact-", dir=path.parent))
    try:
        for name, values in (
            ("item_ids.npy", item_ids),
            ("item_factors.npy", item_factors),
            ("popularity_counts.npy", popularity_counts),
        ):
            with (stage / name).open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
        (stage / "catalog.json").write_bytes(canonical_json_bytes(catalog))
        files = {
            name: {
                "sha256": sha256_file(stage / name),
                "size_bytes": (stage / name).stat().st_size,
            }
            for name in sorted(PAYLOAD)
        }
        manifest = {
            "format": FORMAT,
            "files": files,
            "provenance": provenance,
            "numerical_versions": _versions(),
            "bundle_sha256": _identity(files, provenance),
        }
        (stage / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        load_local_bundle(stage)
        if path.exists():
            raise FileExistsError("local artifact appeared while building")
        stage.rename(path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return path
