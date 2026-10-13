"""ART-002 self-contained serving bundle creation and fail-closed runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from recommender_core.fold_in import ALSFoldInConfig, build_fold_in_confidence_row
from recommender_core.reranking import RerankedItem, greedy_rerank
from recommender_core.scoring import rank_factor_items

SCHEMA_VERSION = 1
BUNDLE_ID = "art-002-final-als-f32-r0p01-i30-a10-s42-train-validation-v1"
PAYLOAD_FILES = frozenset(
    {
        "genre_indices.npy",
        "genre_indptr.npy",
        "genre_vocabulary.json",
        "golden_outputs.json",
        "golden_requests.json",
        "item_factors.npy",
        "item_ids.npy",
        "popularity_counts.npy",
        "popularity_probabilities.npy",
        "provenance.json",
        "seen_indices.npy",
        "seen_indptr.npy",
        "serving_config.json",
        "tail_groups.npy",
        "user_factors.npy",
        "user_ids.npy",
    }
)
EXPECTED_FILES = frozenset({"CHECKSUMS.sha256", "manifest.json", *PAYLOAD_FILES})
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
TAIL_LABELS = ("head", "mid_tail", "long_tail")
DEPENDENCIES = ("numpy", "scipy")
RUNTIME_SOURCE_POLICY = {
    "artifact_runtime_source": "artifact.py",
    "fold_in_source": "fold_in.py",
    "normalization_source": "normalization.py",
    "reranking_source": "reranking.py",
    "scoring_source": "scoring.py",
}
INVENTORY_SHAPE = {
    "locks": {
        ("ml_lock", "ml/uv.lock"),
        ("runtime_package_lock", "packages/recommender_core/uv.lock"),
    },
    "sources": {
        (
            "artifact_runtime_source",
            "packages/recommender_core/src/recommender_core/artifact.py",
        ),
        ("fold_in_source", "packages/recommender_core/src/recommender_core/fold_in.py"),
        (
            "normalization_source",
            "packages/recommender_core/src/recommender_core/normalization.py",
        ),
        (
            "reranking_source",
            "packages/recommender_core/src/recommender_core/reranking.py",
        ),
        ("scoring_source", "packages/recommender_core/src/recommender_core/scoring.py"),
        ("bundle_builder_source", "ml/scripts/build_art002_bundle.py"),
    },
    "configs": {
        ("art002_bundle_config", "configs/art002_serving_bundle.json"),
        ("fold_in_config", "configs/als_fold_in.json"),
        ("rerank_config", "configs/rerank_lambda_sweep.json"),
        ("provenance_schema", "configs/provenance_record_v1.schema.json"),
        ("runtime_package_policy", "configs/arch001_runtime_package_policy.json"),
    },
    "input_artifacts": {
        ("exp001_decision", "reports/experiments/exp001/exp001_decision.json"),
        (
            "final_fit_evidence",
            "artifacts/final-fit/final-001-als-f32-r0p01-i30-a10-s42-train-validation-v1/execution_evidence.json",
        ),
        (
            "final_fit_manifest",
            "artifacts/final-fit/final-001-als-f32-r0p01-i30-a10-s42-train-validation-v1/manifest.json",
        ),
        (
            "final_fit_provenance",
            "artifacts/final-fit/final-001-als-f32-r0p01-i30-a10-s42-train-validation-v1/provenance.json",
        ),
    },
    "input_data": {("movie_genres", "data/processed/movies.parquet")},
}


class ArtifactError(ValueError):
    """Base ART-002 error."""


class ArtifactIntegrityError(ArtifactError):
    """Raised before or during fail-closed bundle loading."""


class ArtifactCompatibilityError(ArtifactIntegrityError):
    """Raised when the current runtime differs from the frozen runtime."""


@dataclass(frozen=True, slots=True)
class ServingRecommendation:
    rank: int
    movie_id: int
    base_rank: int
    score: float


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _load_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactIntegrityError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"invalid JSON payload: {path.name}") from error
    if not isinstance(value, dict):
        raise ArtifactIntegrityError(f"JSON payload must be an object: {path.name}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ArtifactIntegrityError(f"cannot hash bundle file: {path.name}") from error
    return digest.hexdigest()


def _read_checksums(directory: Path) -> dict[str, str]:
    try:
        content = (directory / "CHECKSUMS.sha256").read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise ArtifactIntegrityError("invalid checksum file") from error
    if not content.endswith("\n") or "\r" in content:
        raise ArtifactIntegrityError("checksum file is not canonical LF text")
    result: dict[str, str] = {}
    order: list[str] = []
    for line in content.splitlines():
        parts = line.split("  ")
        if len(parts) != 2 or not SHA_RE.fullmatch(parts[0]):
            raise ArtifactIntegrityError("invalid checksum line")
        name = parts[1]
        if PurePosixPath(name).name != name or name in result:
            raise ArtifactIntegrityError("unsafe or duplicate checksum path")
        result[name] = parts[0]
        order.append(name)
    expected = EXPECTED_FILES - {"CHECKSUMS.sha256"}
    if set(result) != expected or order != sorted(order):
        raise ArtifactIntegrityError("checksum coverage or order mismatch")
    for name, digest in result.items():
        if sha256_file(directory / name) != digest:
            raise ArtifactIntegrityError(f"checksum mismatch: {name}")
    return result


def _load_npy(path: Path) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError(f"invalid NumPy payload: {path.name}") from error
    if value.dtype.hasobject:
        raise ArtifactIntegrityError(f"object array rejected: {path.name}")
    return value


def _expected_config(config: Mapping[str, Any]) -> None:
    try:
        valid = (
            config["schema_version"] == 1
            and config["task"] == "ART-002"
            and config["bundle_id"] == BUNDLE_ID
            and config["model"]
            == {
                "confidence_alpha": 10.0,
                "factors": 32,
                "family": "implicit_als",
                "implicit_library_alpha": 1.0,
                "minimum_positive_interactions": 5,
                "observed_confidence": 11.0,
                "regularization": 0.01,
            }
            and config["serving"]
            == {
                "beta": 0.5,
                "candidate_pool": 100,
                "genre_diversity": "one_minus_max_jaccard",
                "lambda": 1.0,
                "mode": "base_only_no_qualifying_h1_point",
                "normalization": "candidate_local_minmax",
                "novelty": "negative_log2_fit_probability",
                "output_k": 10,
                "tie_break": ["final_score_desc", "base_rank_asc", "movie_id_asc"],
            }
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ArtifactIntegrityError("frozen serving configuration mismatch")


def _validate_provenance(record: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if record.get("schema_version") != 1 or record.get("consumer") != "ART-002":
        raise ArtifactIntegrityError("provenance schema or consumer mismatch")
    git = record.get("git")
    if not isinstance(git, dict) or not GIT_RE.fullmatch(
        str(git.get("commit_sha", ""))
    ):
        raise ArtifactIntegrityError("invalid provenance Git SHA")
    if git.get("working_tree") != "clean" or config.get("source_git_sha") != git.get(
        "commit_sha"
    ):
        raise ArtifactIntegrityError("provenance Git state mismatch")
    inventories = record.get("inventories")
    if not isinstance(inventories, dict) or set(inventories) != set(INVENTORY_SHAPE):
        raise ArtifactIntegrityError("provenance inventory categories mismatch")
    for kind, expected in INVENTORY_SHAPE.items():
        entries = inventories[kind]
        if not isinstance(entries, list):
            raise ArtifactIntegrityError("provenance inventory must be a list")
        identities = {
            (entry.get("role"), entry.get("path"))
            for entry in entries
            if isinstance(entry, dict)
        }
        if identities != expected or len(entries) != len(expected):
            raise ArtifactIntegrityError(f"provenance {kind} inventory mismatch")
        if any(not SHA_RE.fullmatch(str(entry.get("sha256", ""))) for entry in entries):
            raise ArtifactIntegrityError("invalid provenance SHA-256")
    runtime = record.get("runtime")
    expected_dependencies = [
        {"name": name, "version": importlib.metadata.version(name)}
        for name in DEPENDENCIES
    ]
    if not isinstance(runtime, dict):
        raise ArtifactCompatibilityError("missing runtime compatibility record")
    if runtime.get("python_implementation") != platform.python_implementation():
        raise ArtifactCompatibilityError("Python implementation mismatch")
    if runtime.get("python_version") != platform.python_version():
        raise ArtifactCompatibilityError("Python version mismatch")
    if runtime.get("dependencies") != expected_dependencies:
        raise ArtifactCompatibilityError("dependency version mismatch")
    expectations = config.get("provenance_expectations")
    actual = {
        entry["role"]: entry["sha256"]
        for entries in inventories.values()
        for entry in entries
    }
    if expectations != actual:
        raise ArtifactIntegrityError("provenance/config identity mismatch")


def _validate_runtime_source_provenance(record: Mapping[str, Any]) -> None:
    """Bind recorded runtime roles to the source files executing this loader."""

    try:
        source_entries = record["inventories"]["sources"]
    except (KeyError, TypeError) as error:
        raise ArtifactCompatibilityError(
            "missing runtime source provenance inventory"
        ) from error
    by_role: dict[str, Mapping[str, Any]] = {}
    for entry in source_entries:
        if not isinstance(entry, Mapping):
            raise ArtifactCompatibilityError("invalid runtime source provenance entry")
        role = entry.get("role")
        if role in RUNTIME_SOURCE_POLICY:
            if role in by_role:
                raise ArtifactCompatibilityError(
                    f"ambiguous runtime source provenance role: {role}"
                )
            by_role[role] = entry
    if set(by_role) != set(RUNTIME_SOURCE_POLICY):
        raise ArtifactCompatibilityError(
            "missing expected runtime source provenance identity"
        )

    package_directory = Path(__file__).parent
    try:
        directory_mode = package_directory.lstat().st_mode
        resolved_directory = package_directory.resolve(strict=True)
    except OSError as error:
        raise ArtifactCompatibilityError(
            "installed runtime package directory is unavailable"
        ) from error
    if stat.S_ISLNK(directory_mode) or not stat.S_ISDIR(directory_mode):
        raise ArtifactCompatibilityError(
            "installed runtime package directory must be a real directory"
        )

    for role, filename in RUNTIME_SOURCE_POLICY.items():
        entry = by_role[role]
        if PurePosixPath(str(entry.get("path", ""))).name != filename:
            raise ArtifactCompatibilityError(
                f"runtime source provenance path mismatch: {role}"
            )
        candidate = package_directory / filename
        try:
            mode = candidate.lstat().st_mode
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ArtifactCompatibilityError(
                f"runtime source is missing or unreadable: {role}"
            ) from error
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ArtifactCompatibilityError(
                f"runtime source must be a non-symlink regular file: {role}"
            )
        if resolved.parent != resolved_directory:
            raise ArtifactCompatibilityError(
                f"runtime source escapes installed package: {role}"
            )
        if sha256_file(candidate) != entry.get("sha256"):
            raise ArtifactCompatibilityError(f"runtime source SHA-256 mismatch: {role}")


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    user_ids, item_ids = arrays["user_ids"], arrays["item_ids"]
    user_factors, item_factors = arrays["user_factors"], arrays["item_factors"]
    if user_ids.shape != (5975,) or item_ids.shape != (3489,):
        raise ArtifactIntegrityError("mapping shape mismatch")
    if user_factors.shape != (5975, 32) or item_factors.shape != (3489, 32):
        raise ArtifactIntegrityError("factor shape mismatch")
    for ids in (user_ids, item_ids):
        if (
            ids.dtype != np.dtype("int64")
            or not np.array_equal(ids, np.unique(ids))
            or np.any(ids <= 0)
        ):
            raise ArtifactIntegrityError(
                "IDs must be positive canonical ascending unique int64"
            )
    for factors in (user_factors, item_factors):
        if factors.dtype != np.dtype("float32") or not np.isfinite(factors).all():
            raise ArtifactIntegrityError("factors must be finite float32")
    indptr, indices = arrays["seen_indptr"], arrays["seen_indices"]
    if indptr.dtype != np.dtype("int64") or indices.dtype != np.dtype("int64"):
        raise ArtifactIntegrityError("seen arrays must be int64")
    if (
        len(indptr) != 5976
        or indptr[0] != 0
        or indptr[-1] != len(indices)
        or np.any(np.diff(indptr) < 0)
    ):
        raise ArtifactIntegrityError("seen-history structure mismatch")
    if np.any(indices < 0) or np.any(indices >= 3489):
        raise ArtifactIntegrityError("seen-history mapping mismatch")
    for row in range(5975):
        values = indices[int(indptr[row]) : int(indptr[row + 1])]
        if len(values) and not np.array_equal(values, np.unique(values)):
            raise ArtifactIntegrityError("seen-history rows must be sorted unique")
    counts, probabilities, tails = (
        arrays["popularity_counts"],
        arrays["popularity_probabilities"],
        arrays["tail_groups"],
    )
    if (
        counts.shape != (3489,)
        or counts.dtype != np.dtype("int64")
        or np.any(counts <= 0)
        or int(counts.sum()) != 517752
    ):
        raise ArtifactIntegrityError("invalid fit-scope popularity counts")
    expected_probability = (counts + 1).astype(np.float64) / float((counts + 1).sum())
    if (
        probabilities.shape != (3489,)
        or probabilities.dtype != np.dtype("float64")
        or not np.allclose(probabilities, expected_probability, rtol=0, atol=1e-16)
    ):
        raise ArtifactIntegrityError("invalid fit-scope novelty state")
    popularity_order = np.lexsort((item_ids, -counts))
    expected_tails = np.full(3489, 2, dtype=np.uint8)
    expected_tails[popularity_order[:697]] = 0
    expected_tails[popularity_order[697:1744]] = 1
    if (
        tails.shape != (3489,)
        or tails.dtype != np.dtype("uint8")
        or not np.array_equal(tails, expected_tails)
    ):
        raise ArtifactIntegrityError("invalid fit-scope tail state")
    genre_indptr, genre_indices = arrays["genre_indptr"], arrays["genre_indices"]
    if genre_indptr.dtype != np.dtype("int64") or genre_indices.dtype != np.dtype(
        "int32"
    ):
        raise ArtifactIntegrityError("invalid genre array types")
    if (
        genre_indptr.shape != (3490,)
        or genre_indptr[0] != 0
        or genre_indptr[-1] != len(genre_indices)
        or np.any(np.diff(genre_indptr) <= 0)
    ):
        raise ArtifactIntegrityError("invalid genre representation")


class _FoldInModel:
    alpha = 1.0

    def __init__(self, item_factors: np.ndarray, regularization: float):
        self.item_factors = item_factors
        self.regularization = regularization
        self._gram = np.asarray(item_factors.T @ item_factors, dtype=np.float32)

    def recalculate_user(self, userid: int, user_items: Any) -> np.ndarray:
        del userid
        indices = np.asarray(user_items.indices, dtype=np.int64)
        confidence = np.asarray(user_items.data, dtype=np.float32)
        selected = np.asarray(self.item_factors[indices], dtype=np.float32)
        matrix = self._gram.copy()
        matrix.flat[:: matrix.shape[0] + 1] += np.float32(self.regularization)
        matrix += (selected.T * (confidence - np.float32(1.0))) @ selected
        right = selected.T @ confidence
        return np.asarray(np.linalg.solve(matrix, right), dtype=np.float32)


class ServingBundle:
    """Verified, immutable ART-002 runtime state."""

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        config: Mapping[str, Any],
        genres: tuple[frozenset[str], ...],
    ):
        self._arrays = arrays
        self.config = dict(config)
        self._genres = genres
        self._user_index = {
            int(value): index for index, value in enumerate(arrays["user_ids"])
        }
        self._item_index = {
            int(value): index for index, value in enumerate(arrays["item_ids"])
        }
        model = config["model"]
        self._fold_in_config = ALSFoldInConfig(
            minimum_positive_interactions=model["minimum_positive_interactions"],
            recommendation_k=config["serving"]["output_k"],
            confidence_alpha=model["confidence_alpha"],
            observed_confidence=model["observed_confidence"],
            implicit_library_alpha=model["implicit_library_alpha"],
        )
        self._fold_in_model = _FoldInModel(
            arrays["item_factors"], model["regularization"]
        )

    def _recommend_factor(
        self, factor: np.ndarray, excluded: set[int]
    ) -> tuple[ServingRecommendation, ...]:
        serving = self.config["serving"]
        base = rank_factor_items(
            user_factor=factor,
            item_factors=self._arrays["item_factors"],
            item_ids=self._arrays["item_ids"].tolist(),
            excluded_item_indices=excluded,
            k=serving["candidate_pool"],
        )
        candidate_ids = [row.movie_id for row in base]
        candidate_indices = [self._item_index[movie_id] for movie_id in candidate_ids]
        probabilities = self._arrays["popularity_probabilities"][candidate_indices]
        novelty = [-math.log2(float(value)) for value in probabilities]
        genres = {
            movie_id: self._genres[index]
            for movie_id, index in zip(candidate_ids, candidate_indices, strict=True)
        }
        reranked: tuple[RerankedItem, ...] = greedy_rerank(
            candidate_ids,
            [row.score for row in base],
            novelty,
            genres,
            k=serving["output_k"],
            lambda_=serving["lambda"],
            beta=serving["beta"],
        )
        return tuple(
            ServingRecommendation(
                row.rank, row.movie_id, row.base_rank, row.final_score
            )
            for row in reranked
        )

    def recommend_existing_user(
        self, user_id: int
    ) -> tuple[ServingRecommendation, ...]:
        if (
            isinstance(user_id, bool)
            or not isinstance(user_id, (int, np.integer))
            or int(user_id) not in self._user_index
        ):
            raise KeyError("unknown canonical user ID")
        row = self._user_index[int(user_id)]
        start, end = (
            int(self._arrays["seen_indptr"][row]),
            int(self._arrays["seen_indptr"][row + 1]),
        )
        excluded = {int(value) for value in self._arrays["seen_indices"][start:end]}
        return self._recommend_factor(self._arrays["user_factors"][row], excluded)

    def recommend_new_user(
        self, positive_movie_ids: Sequence[int]
    ) -> tuple[ServingRecommendation, ...]:
        row = build_fold_in_confidence_row(
            positive_movie_ids, self._item_index, 3489, self._fold_in_config
        )
        factor = self._fold_in_model.recalculate_user(0, row)
        excluded = {self._item_index[int(movie_id)] for movie_id in positive_movie_ids}
        return self._recommend_factor(factor, excluded)


def load_serving_bundle(
    directory: Path, *, verify_golden: bool = True
) -> ServingBundle:
    path = Path(directory)
    if path.is_symlink() or not path.is_dir():
        raise ArtifactIntegrityError("bundle path must be a real directory")
    entries = list(path.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ArtifactIntegrityError("bundle entries must be regular files")
    names = {entry.name for entry in entries}
    if names != EXPECTED_FILES:
        raise ArtifactIntegrityError(
            f"exact file set mismatch; missing={sorted(EXPECTED_FILES - names)}, extra={sorted(names - EXPECTED_FILES)}"
        )
    checksums = _read_checksums(path)
    manifest = _load_json(path / "manifest.json")
    config = _load_json(path / "serving_config.json")
    provenance = _load_json(path / "provenance.json")
    requests = _load_json(path / "golden_requests.json")
    outputs = _load_json(path / "golden_outputs.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("bundle_id") != BUNDLE_ID
    ):
        raise ArtifactIntegrityError("unsupported artifact schema or bundle identity")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactIntegrityError("unsupported artifact schema")
    if manifest.get("expected_files") != sorted(EXPECTED_FILES):
        raise ArtifactIntegrityError("manifest exact file inventory mismatch")
    metadata = {
        name: {"sha256": checksums[name], "size_bytes": (path / name).stat().st_size}
        for name in sorted(PAYLOAD_FILES)
    }
    if manifest.get("files") != metadata:
        raise ArtifactIntegrityError("manifest file metadata mismatch")
    identity = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    if manifest.get("bundle_sha256") != identity:
        raise ArtifactIntegrityError("bundle identity mismatch")
    _expected_config(config)
    _validate_provenance(provenance, config)
    _validate_runtime_source_provenance(provenance)
    names_to_stems = {
        "user_factors": "user_factors.npy",
        "item_factors": "item_factors.npy",
        "user_ids": "user_ids.npy",
        "item_ids": "item_ids.npy",
        "seen_indptr": "seen_indptr.npy",
        "seen_indices": "seen_indices.npy",
        "popularity_counts": "popularity_counts.npy",
        "popularity_probabilities": "popularity_probabilities.npy",
        "tail_groups": "tail_groups.npy",
        "genre_indptr": "genre_indptr.npy",
        "genre_indices": "genre_indices.npy",
    }
    arrays = {key: _load_npy(path / name) for key, name in names_to_stems.items()}
    _validate_arrays(arrays)
    vocabulary = _load_json(path / "genre_vocabulary.json").get("genres")
    if (
        not isinstance(vocabulary, list)
        or not vocabulary
        or vocabulary != sorted(set(vocabulary))
        or any(not isinstance(value, str) or not value for value in vocabulary)
    ):
        raise ArtifactIntegrityError("invalid genre vocabulary")
    if len(arrays["genre_indices"]) and (
        np.any(arrays["genre_indices"] < 0)
        or np.any(arrays["genre_indices"] >= len(vocabulary))
    ):
        raise ArtifactIntegrityError("genre index outside vocabulary")
    genres = tuple(
        frozenset(
            vocabulary[int(index)]
            for index in arrays["genre_indices"][
                int(arrays["genre_indptr"][row]) : int(arrays["genre_indptr"][row + 1])
            ]
        )
        for row in range(3489)
    )
    bundle = ServingBundle(arrays, config, genres)
    if verify_golden:
        expected_request_keys = {
            "existing_user",
            "five_positive_new_user",
            "schema_version",
        }
        if set(requests) != expected_request_keys or outputs.get("schema_version") != 1:
            raise ArtifactIntegrityError("golden fixture shape mismatch")
        actual_existing = [
            row.movie_id
            for row in bundle.recommend_existing_user(
                requests["existing_user"]["user_id"]
            )
        ]
        actual_new = [
            row.movie_id
            for row in bundle.recommend_new_user(
                requests["five_positive_new_user"]["positive_movie_ids"]
            )
        ]
        if actual_existing != outputs.get("existing_user", {}).get(
            "movie_ids"
        ) or actual_new != outputs.get("five_positive_new_user", {}).get("movie_ids"):
            raise ArtifactIntegrityError("golden verification mismatch")
    return bundle


def write_bundle_files(
    stage: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    genres: Sequence[Sequence[str]],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    golden_requests: Mapping[str, Any],
) -> None:
    """Write deterministic payloads, derive goldens through runtime, and seal."""
    stage.mkdir(parents=True, exist_ok=False)
    for key, value in arrays.items():
        with (stage / f"{key}.npy").open("wb") as stream:
            np.save(stream, np.asarray(value), allow_pickle=False)
    vocabulary = sorted({label for labels in genres for label in labels})
    lookup = {label: index for index, label in enumerate(vocabulary)}
    indptr = [0]
    indices: list[int] = []
    for labels in genres:
        indices.extend(sorted(lookup[label] for label in labels))
        indptr.append(len(indices))
    with (stage / "genre_indptr.npy").open("wb") as stream:
        np.save(stream, np.asarray(indptr, dtype=np.int64), allow_pickle=False)
    with (stage / "genre_indices.npy").open("wb") as stream:
        np.save(stream, np.asarray(indices, dtype=np.int32), allow_pickle=False)
    (stage / "genre_vocabulary.json").write_bytes(
        canonical_json_bytes({"genres": vocabulary, "schema_version": 1})
    )
    (stage / "serving_config.json").write_bytes(canonical_json_bytes(config))
    (stage / "provenance.json").write_bytes(canonical_json_bytes(provenance))
    (stage / "golden_requests.json").write_bytes(canonical_json_bytes(golden_requests))
    # Temporary valid placeholders let the same load/runtime path generate goldens.
    (stage / "golden_outputs.json").write_bytes(
        canonical_json_bytes({"schema_version": 1})
    )
    _seal(stage)
    bundle = load_serving_bundle(stage, verify_golden=False)
    outputs = {
        "existing_user": {
            "movie_ids": [
                row.movie_id
                for row in bundle.recommend_existing_user(
                    golden_requests["existing_user"]["user_id"]
                )
            ]
        },
        "five_positive_new_user": {
            "movie_ids": [
                row.movie_id
                for row in bundle.recommend_new_user(
                    golden_requests["five_positive_new_user"]["positive_movie_ids"]
                )
            ]
        },
        "schema_version": 1,
    }
    (stage / "golden_outputs.json").write_bytes(canonical_json_bytes(outputs))
    _seal(stage)
    load_serving_bundle(stage, verify_golden=True)


def _seal(stage: Path) -> None:
    for name in ("manifest.json", "CHECKSUMS.sha256"):
        candidate = stage / name
        if candidate.exists():
            candidate.unlink()
    metadata = {
        name: {
            "sha256": sha256_file(stage / name),
            "size_bytes": (stage / name).stat().st_size,
        }
        for name in sorted(PAYLOAD_FILES)
    }
    manifest = {
        "bundle_id": BUNDLE_ID,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(),
        "expected_files": sorted(EXPECTED_FILES),
        "files": metadata,
        "schema_version": SCHEMA_VERSION,
        "source_final_fit_state_digest": "22429b1d5582ed804af9950820838fe8303a77f85d66d573f51a97c89c082804",
    }
    (stage / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    covered = sorted({"manifest.json", *PAYLOAD_FILES})
    lines = [f"{sha256_file(stage / name)}  {name}\n" for name in covered]
    (stage / "CHECKSUMS.sha256").write_text(
        "".join(lines), encoding="ascii", newline="\n"
    )


def publish_bundle(stage: Path, destination: Path) -> None:
    load_serving_bundle(stage)
    if destination.exists():
        raise ArtifactIntegrityError("official ART-002 destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(stage, destination)
