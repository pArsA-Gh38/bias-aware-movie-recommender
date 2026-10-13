"""Versioned ART-002 v2 serving bundle and immutable EXP-002 contract API."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from recommender_core.artifact import (
    ArtifactCompatibilityError,
    ArtifactIntegrityError,
    ServingRecommendation,
    _FoldInModel,
    _load_json,
    _load_npy,
    canonical_json_bytes,
    sha256_file,
)
from recommender_core.fold_in import ALSFoldInConfig, build_fold_in_confidence_row
from recommender_core.reranking import RerankedItem, greedy_rerank
from recommender_core.scoring import FactorRecommendation, rank_factor_items

SCHEMA_VERSION_V2 = 2
BUNDLE_ID_V2 = "art-002-final-als-f32-r0p01-i30-a10-s42-train-validation-v2"
SUPERSEDED_BUNDLE_ID = "art-002-final-als-f32-r0p01-i30-a10-s42-train-validation-v1"
SUPERSEDED_BUNDLE_SHA256 = (
    "0592f5fcae879f569f9def8f55ec072ba4d219b9bd91dc9ef0b6fc26a62a2c8b"
)
FINAL_FIT_EXECUTION_ID = "final-001-als-f32-r0p01-i30-a10-s42-train-validation-v1"
FINAL_FIT_STATE_DIGEST = (
    "22429b1d5582ed804af9950820838fe8303a77f85d66d573f51a97c89c082804"
)
EXP001_DECISION_SHA256 = (
    "66071770ec06bd69a51887f54f6e6adbf30cb0bb77bb2db4086b0d8c2766a3ff"
)
TRAIN_SHA256 = "6e6cec4925c7d7b448c2f7c1949975dac84c2f5d615debd99911a425d3358f16"
VALIDATION_SHA256 = "ed72d96c27b0db19047d2f8699a28be71a96cf16eeee46d921118894e094dd4b"
USER_MAPPING_SHA256 = "df16a116af8345c32d9059047d1dcd2293e624f8b4a4799ea39d5bd6e1839990"
FIT_POSITIVE_SEMANTIC_SHA256 = (
    "c57ae6123662bc2bc1b6e7450f5c772884b4c01dfc6795e97ffed748ae137bd3"
)
FIT_POSITIVE_TOTAL = 517_752
FIT_ELIGIBILITY_MINIMUM = 5
FIT_FINAL_CUTOFF = 977_938_855
POSITIVE_RULE = "rating >= 4"
CANDIDATE_POOL = 100
OUTPUT_K = 10
DEFAULT_CONFIGURATION_KEY = "A0_L1"
ACCEPTED_CONFIGURATION_KEYS = ("A0_L1", "A1_L07", "A1_L06")
TAIL_LABELS = ("head", "mid_tail", "long_tail")
DEPENDENCIES_V2 = ("numpy", "scipy")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class FrozenServingConfiguration:
    """One immutable configuration in the confirmatory allowlist."""

    key: str
    mode: str
    lambda_: float
    beta: float | None


@dataclass(frozen=True, slots=True)
class FitEligibility:
    """Verified fit-side eligibility for one canonical existing user."""

    user_id: int
    fit_positive_count: int
    minimum_required: int
    eligible: bool


@dataclass(frozen=True, slots=True)
class ItemMetadata:
    """Trusted immutable metadata aligned with one fit-catalog item."""

    movie_id: int
    genres: frozenset[str]
    popularity_count: int
    popularity_probability: float
    tail_group: str


_CONFIGURATION_REGISTRY = MappingProxyType(
    {
        "A0_L1": FrozenServingConfiguration("A0_L1", "base", 1.0, None),
        "A1_L07": FrozenServingConfiguration("A1_L07", "diversity_only", 0.7, 1.0),
        "A1_L06": FrozenServingConfiguration("A1_L06", "diversity_only", 0.6, 1.0),
    }
)

ARRAY_FILES_V2 = MappingProxyType(
    {
        "fit_positive_counts": "fit_positive_counts.npy",
        "genre_indices": "genre_indices.npy",
        "genre_indptr": "genre_indptr.npy",
        "item_factors": "item_factors.npy",
        "item_ids": "item_ids.npy",
        "popularity_counts": "popularity_counts.npy",
        "popularity_probabilities": "popularity_probabilities.npy",
        "seen_indices": "seen_indices.npy",
        "seen_indptr": "seen_indptr.npy",
        "tail_groups": "tail_groups.npy",
        "user_factors": "user_factors.npy",
        "user_ids": "user_ids.npy",
    }
)
PAYLOAD_FILES_V2 = frozenset(
    {
        *ARRAY_FILES_V2.values(),
        "genre_vocabulary.json",
        "golden_outputs.json",
        "golden_requests.json",
        "provenance.json",
        "serving_config.json",
    }
)
EXPECTED_FILES_V2 = frozenset({"CHECKSUMS.sha256", "manifest.json", *PAYLOAD_FILES_V2})
BUNDLE_IDENTITY_FILES_V2 = PAYLOAD_FILES_V2 - {"provenance.json"}

RUNTIME_SOURCE_POLICY_V2 = MappingProxyType(
    {
        "artifact_v2_runtime_source": "artifact_v2.py",
        "artifact_v1_compat_source": "artifact.py",
        "fold_in_source": "fold_in.py",
        "normalization_source": "normalization.py",
        "reranking_source": "reranking.py",
        "scoring_source": "scoring.py",
    }
)
INVENTORY_SHAPE_V2 = MappingProxyType(
    {
        "locks": frozenset(
            {
                ("ml_lock", "ml/uv.lock"),
                ("runtime_package_lock", "packages/recommender_core/uv.lock"),
            }
        ),
        "sources": frozenset(
            {
                (
                    "artifact_v2_runtime_source",
                    "packages/recommender_core/src/recommender_core/artifact_v2.py",
                ),
                (
                    "artifact_v1_compat_source",
                    "packages/recommender_core/src/recommender_core/artifact.py",
                ),
                (
                    "bundle_v2_builder_source",
                    "ml/scripts/build_art002_bundle_v2.py",
                ),
                (
                    "fold_in_source",
                    "packages/recommender_core/src/recommender_core/fold_in.py",
                ),
                (
                    "normalization_source",
                    "packages/recommender_core/src/recommender_core/normalization.py",
                ),
                (
                    "reranking_source",
                    "packages/recommender_core/src/recommender_core/reranking.py",
                ),
                (
                    "scoring_source",
                    "packages/recommender_core/src/recommender_core/scoring.py",
                ),
            }
        ),
        "configs": frozenset(
            {
                ("art002_v2_bundle_config", "configs/art002_serving_bundle_v2.json"),
                ("fold_in_config", "configs/als_fold_in.json"),
                ("rerank_config", "configs/rerank_lambda_sweep.json"),
                ("exp001_decision_config", "configs/exp001_decision.json"),
                ("exp001_decision_schema", "configs/exp001_decision.schema.json"),
            }
        ),
        "input_artifacts": frozenset(
            {
                (
                    "exp001_decision",
                    "reports/experiments/exp001/exp001_decision.json",
                ),
                (
                    "final001_execution_state",
                    "artifacts/final-fit/final-001-execution-state.json",
                ),
                (
                    "final_fit_manifest",
                    "artifacts/final-fit/final-001-als-f32-r0p01-i30-a10-s42-train-validation-v1/manifest.json",
                ),
                (
                    "art002_v1_manifest",
                    "artifacts/serving/art-002-final-als-f32-r0p01-i30-a10-s42-train-validation-v1/manifest.json",
                ),
            }
        ),
        "fit_sources": frozenset(
            {
                ("fit_train", "data/processed/train.parquet"),
                ("fit_validation", "data/processed/validation.parquet"),
            }
        ),
    }
)

_SCIENTIFIC_COUNTERS = MappingProxyType(
    {
        "ART002_V1_MODIFIED": False,
        "EXP001_MODIFIED": False,
        "EXP002_IMPLEMENTED": False,
        "EXP002_PERFORMED": False,
        "FIT_RETRIES": 0,
        "MODEL_FITS": 0,
        "PARTIAL_FITS": 0,
        "REAL_TEST_BOOTSTRAP_EXECUTIONS": 0,
        "REAL_TEST_CANDIDATE_GENERATIONS": 0,
        "REAL_TEST_RERANK_EXECUTIONS": 0,
        "RESELECTION_EVENTS": 0,
        "ROADMAP_MODIFIED": False,
        "TEST_DATA_ACCESSES": 0,
        "TEST_DATA_HASHES": 0,
        "TEST_EVALUATIONS": 0,
        "VALIDATION_EVALUATIONS": 0,
        "VALIDATION_RESELECTION": 0,
    }
)


def fit_positive_counts_semantic_sha256(
    user_ids: np.ndarray, counts: np.ndarray
) -> str:
    """Hash the ordered user/count meaning, independent of NPY encoding."""

    users = np.asarray(user_ids)
    values = np.asarray(counts)
    payload = {
        "counts": [int(value) for value in values],
        "dtype": str(values.dtype),
        "positive_rule": POSITIVE_RULE,
        "schema_version": 1,
        "shape": [int(value) for value in values.shape],
        "user_ids": [int(value) for value in users],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _read_checksums_v2(directory: Path) -> dict[str, str]:
    try:
        content = (directory / "CHECKSUMS.sha256").read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise ArtifactIntegrityError("invalid v2 checksum file") from error
    if not content.endswith("\n") or "\r" in content:
        raise ArtifactIntegrityError("v2 checksum file is not canonical LF text")
    result: dict[str, str] = {}
    order: list[str] = []
    for line in content.splitlines():
        parts = line.split("  ")
        if len(parts) != 2 or not SHA_RE.fullmatch(parts[0]):
            raise ArtifactIntegrityError("invalid v2 checksum line")
        name = parts[1]
        if PurePosixPath(name).name != name or name in result:
            raise ArtifactIntegrityError("unsafe or duplicate v2 checksum path")
        result[name] = parts[0]
        order.append(name)
    expected = EXPECTED_FILES_V2 - {"CHECKSUMS.sha256"}
    if set(result) != expected or order != sorted(order):
        raise ArtifactIntegrityError("v2 checksum coverage or order mismatch")
    for name, digest in result.items():
        if sha256_file(directory / name) != digest:
            raise ArtifactIntegrityError(f"v2 checksum mismatch: {name}")
    return result


def _file_metadata(directory: Path, names: frozenset[str]) -> dict[str, Any]:
    return {
        name: {
            "sha256": sha256_file(directory / name),
            "size_bytes": (directory / name).stat().st_size,
        }
        for name in sorted(names)
    }


def _bundle_identity_v2(directory: Path) -> str:
    metadata = _file_metadata(directory, BUNDLE_IDENTITY_FILES_V2)
    return hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()


def _expected_config_v2(config: Mapping[str, Any]) -> None:
    dynamic_keys = {"provenance_expectations", "source_git_sha"}
    static = {key: value for key, value in config.items() if key not in dynamic_keys}
    expected = {
        "bundle_id": BUNDLE_ID_V2,
        "configuration_registry": {
            "A0_L1": {"beta": None, "lambda": 1.0, "mode": "base"},
            "A1_L06": {
                "beta": 1.0,
                "lambda": 0.6,
                "mode": "diversity_only",
            },
            "A1_L07": {
                "beta": 1.0,
                "lambda": 0.7,
                "mode": "diversity_only",
            },
        },
        "fit_positive_contract": {
            "eligibility_minimum": FIT_ELIGIBILITY_MINIMUM,
            "final_cutoff": {
                "column": "timestamp",
                "inclusive_max": FIT_FINAL_CUTOFF,
            },
            "positive_rule": POSITIVE_RULE,
            "semantic_sha256": FIT_POSITIVE_SEMANTIC_SHA256,
            "total": FIT_POSITIVE_TOTAL,
            "train_sha256": TRAIN_SHA256,
            "user_mapping_sha256": USER_MAPPING_SHA256,
            "validation_sha256": VALIDATION_SHA256,
        },
        "model": {
            "confidence_alpha": 10.0,
            "factors": 32,
            "family": "implicit_als",
            "implicit_library_alpha": 1.0,
            "minimum_positive_interactions": FIT_ELIGIBILITY_MINIMUM,
            "observed_confidence": 11.0,
            "regularization": 0.01,
        },
        "schema_version": SCHEMA_VERSION_V2,
        "serving": {
            "candidate_pool": CANDIDATE_POOL,
            "default_configuration_key": DEFAULT_CONFIGURATION_KEY,
            "genre_diversity": "one_minus_max_jaccard",
            "normalization": "candidate_local_minmax",
            "novelty": "negative_log2_fit_probability",
            "output_k": OUTPUT_K,
            "tie_break": [
                "final_score_desc",
                "base_rank_asc",
                "movie_id_asc",
            ],
        },
        "source_final_fit": (
            "artifacts/final-fit/"
            "final-001-als-f32-r0p01-i30-a10-s42-train-validation-v1"
        ),
        "supersedes": {
            "bundle_id": SUPERSEDED_BUNDLE_ID,
            "bundle_sha256": SUPERSEDED_BUNDLE_SHA256,
        },
        "task": "ART-002",
    }
    expectations = config.get("provenance_expectations")
    if (
        static != expected
        or not GIT_RE.fullmatch(str(config.get("source_git_sha", "")))
        or not isinstance(expectations, dict)
        or not expectations
        or any(
            not isinstance(role, str) or not SHA_RE.fullmatch(str(digest))
            for role, digest in expectations.items()
        )
    ):
        raise ArtifactIntegrityError("frozen v2 serving configuration mismatch")


def _validate_provenance_v2(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    bundle_sha256: str,
) -> None:
    if record.get("schema_version") != 2 or record.get("consumer") != "ART-002-v2":
        raise ArtifactIntegrityError("v2 provenance schema or consumer mismatch")
    if record.get("bundle") != {
        "bundle_id": BUNDLE_ID_V2,
        "bundle_sha256": bundle_sha256,
    }:
        raise ArtifactIntegrityError("v2 provenance bundle identity mismatch")
    git = record.get("git")
    if (
        not isinstance(git, dict)
        or not GIT_RE.fullmatch(str(git.get("commit_sha", "")))
        or git.get("working_tree") != "clean"
        or config.get("source_git_sha") != git.get("commit_sha")
    ):
        raise ArtifactIntegrityError("v2 provenance Git state mismatch")
    if record.get("traceability") != {
        "exp001_decision_sha256": EXP001_DECISION_SHA256,
        "final_fit_execution_id": FINAL_FIT_EXECUTION_ID,
        "final_fit_state_digest": FINAL_FIT_STATE_DIGEST,
        "superseded_art002_v1_bundle_id": SUPERSEDED_BUNDLE_ID,
        "superseded_art002_v1_bundle_sha256": SUPERSEDED_BUNDLE_SHA256,
    }:
        raise ArtifactIntegrityError("v2 traceability mismatch")
    if record.get("fit_positive_counts") != {
        "dtype": "int64",
        "final_cutoff": {
            "column": "timestamp",
            "inclusive_max": FIT_FINAL_CUTOFF,
        },
        "positive_rule": POSITIVE_RULE,
        "semantic_sha256": FIT_POSITIVE_SEMANTIC_SHA256,
        "shape": [5975],
        "total": FIT_POSITIVE_TOTAL,
        "train_sha256": TRAIN_SHA256,
        "user_mapping_sha256": USER_MAPPING_SHA256,
        "validation_sha256": VALIDATION_SHA256,
    }:
        raise ArtifactIntegrityError("v2 fit-positive provenance mismatch")
    if record.get("scientific_freeze") != {
        "base_configuration": "A0_L1",
        "candidate_pool": CANDIDATE_POOL,
        "confirmatory_configurations": ["A1_L07", "A1_L06"],
        "h1_operating_point": None,
        "h1_verdict": "rejected",
        "h2_verdict": "accepted",
        "h3_verdict": "not_evaluable",
        "output_k": OUTPUT_K,
    }:
        raise ArtifactIntegrityError("v2 scientific freeze mismatch")
    counters = record.get("scientific_counters")
    if counters != dict(_SCIENTIFIC_COUNTERS):
        raise ArtifactIntegrityError("v2 scientific counter mismatch")
    if record.get("human_test_access_gate") != "NOT_GRANTED":
        raise ArtifactIntegrityError("v2 Human Test Access Gate mismatch")

    inventories = record.get("inventories")
    if not isinstance(inventories, dict) or set(inventories) != set(INVENTORY_SHAPE_V2):
        raise ArtifactIntegrityError("v2 provenance inventory categories mismatch")
    actual_hashes: dict[str, str] = {}
    for kind, expected in INVENTORY_SHAPE_V2.items():
        entries = inventories[kind]
        if not isinstance(entries, list):
            raise ArtifactIntegrityError("v2 provenance inventory must be a list")
        identities = {
            (entry.get("role"), entry.get("path"))
            for entry in entries
            if isinstance(entry, dict)
        }
        if identities != expected or len(entries) != len(expected):
            raise ArtifactIntegrityError(f"v2 provenance {kind} inventory mismatch")
        for entry in entries:
            digest = str(entry.get("sha256", ""))
            if not SHA_RE.fullmatch(digest):
                raise ArtifactIntegrityError("invalid v2 provenance SHA-256")
            actual_hashes[str(entry["role"])] = digest
    if config.get("provenance_expectations") != actual_hashes:
        raise ArtifactIntegrityError("v2 provenance/config identity mismatch")
    if actual_hashes.get("fit_train") != TRAIN_SHA256:
        raise ArtifactIntegrityError("v2 train source hash mismatch")
    if actual_hashes.get("fit_validation") != VALIDATION_SHA256:
        raise ArtifactIntegrityError("v2 validation source hash mismatch")
    if actual_hashes.get("exp001_decision") != EXP001_DECISION_SHA256:
        raise ArtifactIntegrityError("v2 EXP-001 decision hash mismatch")

    runtime = record.get("runtime")
    expected_dependencies = [
        {"name": name, "version": importlib.metadata.version(name)}
        for name in DEPENDENCIES_V2
    ]
    if not isinstance(runtime, dict):
        raise ArtifactCompatibilityError("missing v2 runtime compatibility record")
    if runtime.get("python_implementation") != platform.python_implementation():
        raise ArtifactCompatibilityError("v2 Python implementation mismatch")
    if runtime.get("python_version") != platform.python_version():
        raise ArtifactCompatibilityError("v2 Python version mismatch")
    if runtime.get("dependencies") != expected_dependencies:
        raise ArtifactCompatibilityError("v2 dependency version mismatch")


def _validate_runtime_sources_v2(record: Mapping[str, Any]) -> None:
    try:
        entries = record["inventories"]["sources"]
    except (KeyError, TypeError) as error:
        raise ArtifactCompatibilityError(
            "missing v2 runtime source provenance inventory"
        ) from error
    by_role = {
        entry.get("role"): entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("role") in RUNTIME_SOURCE_POLICY_V2
    }
    if set(by_role) != set(RUNTIME_SOURCE_POLICY_V2):
        raise ArtifactCompatibilityError("missing v2 runtime source identity")
    package_directory = Path(__file__).parent
    try:
        directory_mode = package_directory.lstat().st_mode
        resolved_directory = package_directory.resolve(strict=True)
    except OSError as error:
        raise ArtifactCompatibilityError(
            "installed v2 runtime package directory is unavailable"
        ) from error
    if stat.S_ISLNK(directory_mode) or not stat.S_ISDIR(directory_mode):
        raise ArtifactCompatibilityError(
            "installed v2 runtime package directory must be real"
        )
    for role, filename in RUNTIME_SOURCE_POLICY_V2.items():
        entry = by_role[role]
        if PurePosixPath(str(entry.get("path", ""))).name != filename:
            raise ArtifactCompatibilityError(
                f"v2 runtime source provenance path mismatch: {role}"
            )
        candidate = package_directory / filename
        try:
            mode = candidate.lstat().st_mode
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ArtifactCompatibilityError(
                f"v2 runtime source is missing or unreadable: {role}"
            ) from error
        if (
            stat.S_ISLNK(mode)
            or not stat.S_ISREG(mode)
            or resolved.parent != resolved_directory
        ):
            raise ArtifactCompatibilityError(
                f"v2 runtime source is not a contained regular file: {role}"
            )
        if sha256_file(candidate) != entry.get("sha256"):
            raise ArtifactCompatibilityError(
                f"v2 runtime source SHA-256 mismatch: {role}"
            )


def _validate_arrays_v2(arrays: Mapping[str, np.ndarray]) -> None:
    user_ids = arrays["user_ids"]
    item_ids = arrays["item_ids"]
    user_factors = arrays["user_factors"]
    item_factors = arrays["item_factors"]
    if user_ids.shape != (5975,) or item_ids.shape != (3489,):
        raise ArtifactIntegrityError("v2 mapping shape mismatch")
    if user_factors.shape != (5975, 32) or item_factors.shape != (3489, 32):
        raise ArtifactIntegrityError("v2 factor shape mismatch")
    for ids in (user_ids, item_ids):
        if (
            ids.dtype != np.dtype("int64")
            or not np.array_equal(ids, np.unique(ids))
            or np.any(ids <= 0)
        ):
            raise ArtifactIntegrityError("v2 IDs must be canonical int64")
    for factors in (user_factors, item_factors):
        if factors.dtype != np.dtype("float32") or not np.isfinite(factors).all():
            raise ArtifactIntegrityError("v2 factors must be finite float32")

    indptr = arrays["seen_indptr"]
    indices = arrays["seen_indices"]
    if indptr.dtype != np.dtype("int64") or indices.dtype != np.dtype("int64"):
        raise ArtifactIntegrityError("v2 seen arrays must be int64")
    if (
        indptr.shape != (5976,)
        or indptr[0] != 0
        or indptr[-1] != len(indices)
        or np.any(np.diff(indptr) < 0)
        or np.any(indices < 0)
        or np.any(indices >= 3489)
    ):
        raise ArtifactIntegrityError("v2 seen-history structure mismatch")
    for row in range(5975):
        values = indices[int(indptr[row]) : int(indptr[row + 1])]
        if len(values) and not np.array_equal(values, np.unique(values)):
            raise ArtifactIntegrityError("v2 seen-history rows must be sorted unique")

    fit_counts = arrays["fit_positive_counts"]
    if fit_counts.shape != user_ids.shape:
        raise ArtifactIntegrityError("fit-positive count alignment mismatch")
    if fit_counts.dtype != np.dtype("int64"):
        raise ArtifactIntegrityError("fit-positive counts must be int64")
    if np.any(fit_counts < 0):
        raise ArtifactIntegrityError("fit-positive counts must be non-negative")
    if int(fit_counts.sum()) != FIT_POSITIVE_TOTAL:
        raise ArtifactIntegrityError("fit-positive count total mismatch")
    if (
        fit_positive_counts_semantic_sha256(user_ids, fit_counts)
        != FIT_POSITIVE_SEMANTIC_SHA256
    ):
        raise ArtifactIntegrityError("fit-positive count semantic hash mismatch")

    counts = arrays["popularity_counts"]
    probabilities = arrays["popularity_probabilities"]
    tails = arrays["tail_groups"]
    if (
        counts.shape != (3489,)
        or counts.dtype != np.dtype("int64")
        or np.any(counts <= 0)
        or int(counts.sum()) != FIT_POSITIVE_TOTAL
    ):
        raise ArtifactIntegrityError("invalid v2 popularity counts")
    expected_probability = (counts + 1).astype(np.float64) / float((counts + 1).sum())
    if (
        probabilities.shape != (3489,)
        or probabilities.dtype != np.dtype("float64")
        or not np.allclose(probabilities, expected_probability, rtol=0, atol=1e-16)
    ):
        raise ArtifactIntegrityError("invalid v2 popularity probabilities")
    popularity_order = np.lexsort((item_ids, -counts))
    expected_tails = np.full(3489, 2, dtype=np.uint8)
    expected_tails[popularity_order[:697]] = 0
    expected_tails[popularity_order[697:1744]] = 1
    if (
        tails.shape != (3489,)
        or tails.dtype != np.dtype("uint8")
        or not np.array_equal(tails, expected_tails)
    ):
        raise ArtifactIntegrityError("invalid v2 tail state")

    genre_indptr = arrays["genre_indptr"]
    genre_indices = arrays["genre_indices"]
    if (
        genre_indptr.dtype != np.dtype("int64")
        or genre_indices.dtype != np.dtype("int32")
        or genre_indptr.shape != (3490,)
        or genre_indptr[0] != 0
        or genre_indptr[-1] != len(genre_indices)
        or np.any(np.diff(genre_indptr) <= 0)
    ):
        raise ArtifactIntegrityError("invalid v2 genre representation")


class ServingBundleV2:
    """Verified v2 runtime with immutable inventory, metadata, and configs."""

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        genres: Sequence[frozenset[str]],
    ) -> None:
        readonly_arrays: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            array = np.asarray(value)
            array.flags.writeable = False
            readonly_arrays[name] = array
        self._arrays = MappingProxyType(readonly_arrays)
        self._genres = tuple(frozenset(labels) for labels in genres)
        self._existing_user_ids = tuple(
            int(value) for value in readonly_arrays["user_ids"]
        )
        self._catalog_item_ids = tuple(
            int(value) for value in readonly_arrays["item_ids"]
        )
        self._user_index = {
            user_id: index for index, user_id in enumerate(self._existing_user_ids)
        }
        self._item_index = {
            movie_id: index for index, movie_id in enumerate(self._catalog_item_ids)
        }
        self._metadata = tuple(
            ItemMetadata(
                movie_id=movie_id,
                genres=self._genres[index],
                popularity_count=int(readonly_arrays["popularity_counts"][index]),
                popularity_probability=float(
                    readonly_arrays["popularity_probabilities"][index]
                ),
                tail_group=TAIL_LABELS[int(readonly_arrays["tail_groups"][index])],
            )
            for index, movie_id in enumerate(self._catalog_item_ids)
        )
        self._fold_in_config = ALSFoldInConfig(
            minimum_positive_interactions=FIT_ELIGIBILITY_MINIMUM,
            recommendation_k=OUTPUT_K,
            confidence_alpha=10.0,
            observed_confidence=11.0,
            implicit_library_alpha=1.0,
        )
        self._fold_in_model = _FoldInModel(readonly_arrays["item_factors"], 0.01)

    @property
    def configuration_registry(
        self,
    ) -> Mapping[str, FrozenServingConfiguration]:
        return _CONFIGURATION_REGISTRY

    @property
    def serving_default(self) -> str:
        return DEFAULT_CONFIGURATION_KEY

    @property
    def candidate_pool_size(self) -> int:
        return CANDIDATE_POOL

    @property
    def recommendation_k(self) -> int:
        return OUTPUT_K

    def existing_user_ids(self) -> tuple[int, ...]:
        return self._existing_user_ids

    def catalog_item_ids(self) -> tuple[int, ...]:
        return self._catalog_item_ids

    def catalog_metadata(self) -> tuple[ItemMetadata, ...]:
        return self._metadata

    def _user_row(self, user_id: Integral) -> int:
        if (
            isinstance(user_id, bool)
            or not isinstance(user_id, Integral)
            or int(user_id) not in self._user_index
        ):
            raise KeyError("unknown canonical user ID")
        return self._user_index[int(user_id)]

    def _configuration(self, key: str) -> FrozenServingConfiguration:
        if not isinstance(key, str) or key not in _CONFIGURATION_REGISTRY:
            raise KeyError("unknown frozen serving configuration key")
        return _CONFIGURATION_REGISTRY[key]

    def fit_positive_count(self, user_id: Integral) -> int:
        row = self._user_row(user_id)
        return int(self._arrays["fit_positive_counts"][row])

    def fit_eligibility(self, user_id: Integral) -> FitEligibility:
        count = self.fit_positive_count(user_id)
        return FitEligibility(
            user_id=int(user_id),
            fit_positive_count=count,
            minimum_required=FIT_ELIGIBILITY_MINIMUM,
            eligible=count >= FIT_ELIGIBILITY_MINIMUM,
        )

    def is_fit_eligible(self, user_id: Integral) -> bool:
        return self.fit_eligibility(user_id).eligible

    def item_metadata(self, movie_id: Integral) -> ItemMetadata:
        if (
            isinstance(movie_id, bool)
            or not isinstance(movie_id, Integral)
            or int(movie_id) not in self._item_index
        ):
            raise KeyError("unknown canonical movie ID")
        return self._metadata[self._item_index[int(movie_id)]]

    def _base_pool(
        self, factor: np.ndarray, excluded: set[int]
    ) -> tuple[FactorRecommendation, ...]:
        return rank_factor_items(
            user_factor=factor,
            item_factors=self._arrays["item_factors"],
            item_ids=self._catalog_item_ids,
            excluded_item_indices=excluded,
            k=CANDIDATE_POOL,
        )

    def _recommend_from_base(
        self,
        base: tuple[FactorRecommendation, ...],
        configuration: FrozenServingConfiguration,
    ) -> tuple[ServingRecommendation, ...]:
        candidate_ids = [row.movie_id for row in base]
        candidate_indices = [self._item_index[value] for value in candidate_ids]
        probabilities = self._arrays["popularity_probabilities"][candidate_indices]
        novelty = [-math.log2(float(value)) for value in probabilities]
        genres = {
            movie_id: self._genres[index]
            for movie_id, index in zip(candidate_ids, candidate_indices, strict=True)
        }
        beta = 0.5 if configuration.beta is None else configuration.beta
        reranked: tuple[RerankedItem, ...] = greedy_rerank(
            candidate_ids,
            [row.score for row in base],
            novelty,
            genres,
            k=OUTPUT_K,
            lambda_=configuration.lambda_,
            beta=beta,
        )
        return tuple(
            ServingRecommendation(
                row.rank, row.movie_id, row.base_rank, row.final_score
            )
            for row in reranked
        )

    def _existing_base_pool(
        self, user_id: Integral
    ) -> tuple[FactorRecommendation, ...]:
        row = self._user_row(user_id)
        start = int(self._arrays["seen_indptr"][row])
        end = int(self._arrays["seen_indptr"][row + 1])
        excluded = {int(value) for value in self._arrays["seen_indices"][start:end]}
        return self._base_pool(self._arrays["user_factors"][row], excluded)

    def recommend_existing_user(
        self,
        user_id: Integral,
        configuration_key: str = DEFAULT_CONFIGURATION_KEY,
    ) -> tuple[ServingRecommendation, ...]:
        configuration = self._configuration(configuration_key)
        base = self._existing_base_pool(user_id)
        return self._recommend_from_base(base, configuration)

    def recommend_existing_user_configurations(
        self, user_id: Integral
    ) -> Mapping[str, tuple[ServingRecommendation, ...]]:
        """Recommend all frozen variants from one generated base pool."""

        base = self._existing_base_pool(user_id)
        result = {
            key: self._recommend_from_base(base, _CONFIGURATION_REGISTRY[key])
            for key in ACCEPTED_CONFIGURATION_KEYS
        }
        return MappingProxyType(result)

    def recommend_new_user(
        self,
        positive_movie_ids: Sequence[int],
        configuration_key: str = DEFAULT_CONFIGURATION_KEY,
    ) -> tuple[ServingRecommendation, ...]:
        configuration = self._configuration(configuration_key)
        row = build_fold_in_confidence_row(
            positive_movie_ids,
            self._item_index,
            len(self._catalog_item_ids),
            self._fold_in_config,
        )
        factor = self._fold_in_model.recalculate_user(0, row)
        excluded = {self._item_index[int(movie_id)] for movie_id in positive_movie_ids}
        return self._recommend_from_base(
            self._base_pool(factor, excluded), configuration
        )


def _genres_from_payload(
    arrays: Mapping[str, np.ndarray], vocabulary: object
) -> tuple[frozenset[str], ...]:
    if (
        not isinstance(vocabulary, list)
        or not vocabulary
        or vocabulary != sorted(set(vocabulary))
        or any(not isinstance(value, str) or not value for value in vocabulary)
    ):
        raise ArtifactIntegrityError("invalid v2 genre vocabulary")
    indices = arrays["genre_indices"]
    if len(indices) and (np.any(indices < 0) or np.any(indices >= len(vocabulary))):
        raise ArtifactIntegrityError("v2 genre index outside vocabulary")
    indptr = arrays["genre_indptr"]
    return tuple(
        frozenset(
            vocabulary[int(index)]
            for index in indices[int(indptr[row]) : int(indptr[row + 1])]
        )
        for row in range(3489)
    )


def load_serving_bundle_v2(
    directory: Path, *, verify_golden: bool = True
) -> ServingBundleV2:
    path = Path(directory)
    if path.is_symlink() or not path.is_dir():
        raise ArtifactIntegrityError("v2 bundle path must be a real directory")
    entries = list(path.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ArtifactIntegrityError("v2 bundle entries must be regular files")
    names = {entry.name for entry in entries}
    if names != EXPECTED_FILES_V2:
        raise ArtifactIntegrityError(
            "v2 exact file set mismatch; "
            f"missing={sorted(EXPECTED_FILES_V2 - names)}, "
            f"extra={sorted(names - EXPECTED_FILES_V2)}"
        )
    checksums = _read_checksums_v2(path)
    manifest = _load_json(path / "manifest.json")
    config = _load_json(path / "serving_config.json")
    provenance = _load_json(path / "provenance.json")
    requests = _load_json(path / "golden_requests.json")
    outputs = _load_json(path / "golden_outputs.json")
    if manifest.get("schema_version") != 2 or manifest.get("bundle_id") != BUNDLE_ID_V2:
        raise ArtifactIntegrityError("unsupported v2 artifact identity")
    if config.get("schema_version") != 2:
        raise ArtifactIntegrityError("unsupported v2 artifact schema")
    if manifest.get("expected_files") != sorted(EXPECTED_FILES_V2):
        raise ArtifactIntegrityError("v2 manifest file inventory mismatch")
    metadata = {
        name: {
            "sha256": checksums[name],
            "size_bytes": (path / name).stat().st_size,
        }
        for name in sorted(PAYLOAD_FILES_V2)
    }
    if manifest.get("files") != metadata:
        raise ArtifactIntegrityError("v2 manifest file metadata mismatch")
    bundle_sha256 = _bundle_identity_v2(path)
    if manifest.get("bundle_sha256") != bundle_sha256:
        raise ArtifactIntegrityError("v2 bundle identity mismatch")
    distribution_sha256 = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    if manifest.get("distribution_sha256") != distribution_sha256:
        raise ArtifactIntegrityError("v2 distribution identity mismatch")
    if manifest.get("source_final_fit_state_digest") != FINAL_FIT_STATE_DIGEST:
        raise ArtifactIntegrityError("v2 FINAL-001 state mismatch")
    if manifest.get("superseded_bundle") != {
        "bundle_id": SUPERSEDED_BUNDLE_ID,
        "bundle_sha256": SUPERSEDED_BUNDLE_SHA256,
    }:
        raise ArtifactIntegrityError("v2 superseded bundle identity mismatch")
    _expected_config_v2(config)
    _validate_provenance_v2(provenance, config, bundle_sha256)
    _validate_runtime_sources_v2(provenance)
    arrays = {
        key: _load_npy(path / filename) for key, filename in ARRAY_FILES_V2.items()
    }
    _validate_arrays_v2(arrays)
    vocabulary = _load_json(path / "genre_vocabulary.json").get("genres")
    genres = _genres_from_payload(arrays, vocabulary)
    bundle = ServingBundleV2(arrays, genres)
    if verify_golden:
        if (
            requests
            != {
                "configuration_keys": list(ACCEPTED_CONFIGURATION_KEYS),
                "existing_user": {"user_id": 56},
                "schema_version": 2,
            }
            or outputs.get("schema_version") != 2
        ):
            raise ArtifactIntegrityError("v2 golden fixture shape mismatch")
        actual = {
            key: [row.movie_id for row in rows]
            for key, rows in bundle.recommend_existing_user_configurations(
                requests["existing_user"]["user_id"]
            ).items()
        }
        if actual != outputs.get("existing_user"):
            raise ArtifactIntegrityError("v2 golden verification mismatch")
        parity = outputs.get("v1_parity")
        if parity != {
            "configuration_key": "A0_L1",
            "movie_ids": actual["A0_L1"],
        }:
            raise ArtifactIntegrityError("v2 A0/v1 golden parity mismatch")
    return bundle


def _write_json_v2(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value))


def write_bundle_files_v2(
    stage: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    genres: Sequence[Sequence[str]],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    golden_requests: Mapping[str, Any],
    v1_golden_existing_ids: Sequence[int],
) -> None:
    """Write deterministic v2 payloads, goldens, identities, and checksums."""

    stage.mkdir(parents=True, exist_ok=False)
    expected_array_names = set(ARRAY_FILES_V2) - {"genre_indptr", "genre_indices"}
    if set(arrays) != expected_array_names:
        raise ArtifactIntegrityError("v2 writer array inventory mismatch")
    for key in sorted(arrays):
        with (stage / f"{key}.npy").open("wb") as stream:
            np.save(stream, np.asarray(arrays[key]), allow_pickle=False)
    vocabulary = sorted({label for labels in genres for label in labels})
    lookup = {label: index for index, label in enumerate(vocabulary)}
    indptr = [0]
    indices: list[int] = []
    normalized_genres: list[frozenset[str]] = []
    for labels in genres:
        normalized = frozenset(labels)
        normalized_genres.append(normalized)
        indices.extend(sorted(lookup[label] for label in normalized))
        indptr.append(len(indices))
    genre_indptr = np.asarray(indptr, dtype=np.int64)
    genre_indices = np.asarray(indices, dtype=np.int32)
    with (stage / "genre_indptr.npy").open("wb") as stream:
        np.save(stream, genre_indptr, allow_pickle=False)
    with (stage / "genre_indices.npy").open("wb") as stream:
        np.save(stream, genre_indices, allow_pickle=False)
    _write_json_v2(
        stage / "genre_vocabulary.json",
        {"genres": vocabulary, "schema_version": 2},
    )
    _write_json_v2(stage / "serving_config.json", config)
    _write_json_v2(stage / "golden_requests.json", golden_requests)
    runtime_arrays = dict(arrays)
    runtime_arrays["genre_indptr"] = genre_indptr
    runtime_arrays["genre_indices"] = genre_indices
    _validate_arrays_v2(runtime_arrays)
    bundle = ServingBundleV2(runtime_arrays, normalized_genres)
    generated = {
        key: [row.movie_id for row in rows]
        for key, rows in bundle.recommend_existing_user_configurations(
            golden_requests["existing_user"]["user_id"]
        ).items()
    }
    if generated["A0_L1"] != [int(value) for value in v1_golden_existing_ids]:
        raise ArtifactIntegrityError("v2 writer failed A0/v1 golden parity")
    _write_json_v2(
        stage / "golden_outputs.json",
        {
            "existing_user": generated,
            "schema_version": 2,
            "v1_parity": {
                "configuration_key": "A0_L1",
                "movie_ids": generated["A0_L1"],
            },
        },
    )
    bundle_sha256 = _bundle_identity_v2(stage)
    provenance_with_identity = dict(provenance)
    provenance_with_identity["bundle"] = {
        "bundle_id": BUNDLE_ID_V2,
        "bundle_sha256": bundle_sha256,
    }
    _write_json_v2(stage / "provenance.json", provenance_with_identity)
    _seal_v2(stage)
    load_serving_bundle_v2(stage)


def _seal_v2(stage: Path) -> None:
    for name in ("manifest.json", "CHECKSUMS.sha256"):
        candidate = stage / name
        if candidate.exists():
            candidate.unlink()
    metadata = _file_metadata(stage, PAYLOAD_FILES_V2)
    manifest = {
        "bundle_id": BUNDLE_ID_V2,
        "bundle_sha256": _bundle_identity_v2(stage),
        "distribution_sha256": hashlib.sha256(
            canonical_json_bytes(metadata)
        ).hexdigest(),
        "expected_files": sorted(EXPECTED_FILES_V2),
        "files": metadata,
        "schema_version": SCHEMA_VERSION_V2,
        "source_final_fit_state_digest": FINAL_FIT_STATE_DIGEST,
        "superseded_bundle": {
            "bundle_id": SUPERSEDED_BUNDLE_ID,
            "bundle_sha256": SUPERSEDED_BUNDLE_SHA256,
        },
    }
    _write_json_v2(stage / "manifest.json", manifest)
    covered = sorted({"manifest.json", *PAYLOAD_FILES_V2})
    lines = [f"{sha256_file(stage / name)}  {name}\n" for name in covered]
    (stage / "CHECKSUMS.sha256").write_text(
        "".join(lines), encoding="ascii", newline="\n"
    )
