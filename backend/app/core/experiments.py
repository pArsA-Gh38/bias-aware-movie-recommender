"""Lazy, fail-closed reader for the committed EXP-003 presentation bundle."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import stat
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, TypeVar

from pydantic import ValidationError

from app.schemas.experiments import (
    ActivityGroupRow,
    ExperimentSummaryResponse,
    ExposureRow,
    FrozenTestRow,
    FrozenTestSummary,
    HypothesisOutcome,
    ModelComparisonRow,
    RerankingRow,
    ValidationSummary,
)

PROTOCOL_ID = "exp003-stored-evidence-publication-v1"
SCHEMA_VERSION = 1
EVIDENCE_BASE_GIT_SHA = "8d3d2e27b81676648b982bb3e0f656c6f8d36888"
MANIFEST_SHA256 = "2015148fba12d453f92fa275e01ee735e5460337fc32fa7cb8fd6e67aa0bb63c"
CHECKSUMS_SHA256 = "93353edc5179008d0bd63aba7bb9c93782521484e989c8add51da3e578107233"

MANIFEST_PATH = "reports/experiments/exp003/manifest.json"
CHECKSUMS_PATH = "reports/experiments/exp003/CHECKSUMS.sha256"
TABLE_PATHS = {
    "models": "reports/tables/exp003_model_comparison_validation.csv",
    "reranking": "reports/tables/exp003_reranking_tradeoff_validation.csv",
    "exposure": "reports/tables/exp003_exposure_distribution_validation.csv",
    "activity": "reports/tables/exp003_activity_group_validation.csv",
    "frozen_test": "reports/tables/exp003_frozen_test_confirmation.csv",
    "hypotheses": "reports/tables/exp003_hypothesis_outcomes.csv",
}
FIGURE_PATHS = {
    f"reports/figures/exp003_{name}.{suffix}"
    for name in (
        "ndcg_vs_long_tail",
        "ndcg_vs_ild",
        "exposure_head_mid_tail",
        "gini_coverage_by_lambda",
    )
    for suffix in ("svg", "png")
}
EXPECTED_OUTPUTS = set(TABLE_PATHS.values()) | FIGURE_PATHS

MODELS = (
    "Popularity",
    "ItemKNN-50",
    "ItemKNN-100",
    "ItemKNN-200",
    "ALS-selected-seed42",
)
CONFIGURATIONS = (
    "A0_L1",
    "A1_L1",
    "A1_L09",
    "A1_L08",
    "A1_L07",
    "A1_L06",
    "A2_L1",
    "A2_L09",
    "A2_L08",
    "A2_L07",
    "A2_L06",
    "A3_L1",
    "A3_L09",
    "A3_L08",
    "A3_L07",
    "A3_L06",
)
METRICS = (
    "precision_at_10",
    "recall_at_10",
    "hit_rate_at_10",
    "ndcg_at_10",
    "catalog_coverage_at_10",
    "novelty_at_10",
    "ild_at_10",
    "exposure_gini_at_10",
    "long_tail_exposure_at_10",
)
ACTIVITY_GROUPS = ("low", "medium", "high")
POPULARITY_GROUPS = ("head", "mid_tail", "long_tail")
FROZEN_CONFIGURATIONS = ("A0_L1", "A1_L07", "A1_L06")
HYPOTHESES = ("H1", "H2", "H3", "H4")

HEADERS = {
    "models": (
        "model",
        "metric",
        "mean",
        "ci_low",
        "ci_high",
        "confidence_level",
        "ci_method",
        "bootstrap_resamples",
        "bootstrap_seed",
        "bootstrap_unit",
        "fit_scope",
        "target_scope",
        "eligible_user_count",
        "recommendation_k",
        "catalog_size",
        "evidence_scope",
        "source_role",
    ),
    "reranking": (
        "configuration_key",
        "mode",
        "lambda_",
        "beta",
        "metric",
        "observed_value",
        "ci_low",
        "ci_high",
        "confidence_level",
        "ci_method",
        "bootstrap_resamples",
        "bootstrap_seed",
        "bootstrap_unit",
        "fit_scope",
        "target_scope",
        "eligible_user_count",
        "recommendation_k",
        "catalog_size",
        "evidence_scope",
        "source_role",
    ),
    "exposure": (
        "configuration_key",
        "popularity_group",
        "displayed_slot_count",
        "total_displayed_slot_count",
        "displayed_slot_share",
        "share_sum_tolerance",
        "fit_scope",
        "target_scope",
        "evidence_scope",
        "source_role",
    ),
    "activity": (
        "configuration_key",
        "activity_group",
        "activity_group_user_count",
        "metric",
        "observed_value",
        "ci_low",
        "ci_high",
        "confidence_level",
        "ci_method",
        "bootstrap_resamples",
        "bootstrap_seed",
        "bootstrap_unit",
        "fit_scope",
        "target_scope",
        "evidence_scope",
        "source_role",
    ),
    "frozen_test": (
        "configuration_key",
        "metric",
        "point_estimate",
        "confidence_interval_status",
        "evidence_kind",
        "fit_scope",
        "target_scope",
        "eligible_user_count",
        "recommendation_k",
        "evidence_scope",
        "source_role",
    ),
    "hypotheses": (
        "hypothesis",
        "scientific_outcome",
        "engineering_acceptance",
        "verdict",
        "evidence_scope",
        "reason",
        "source_path",
    ),
}


class ExperimentSummaryUnavailableError(RuntimeError):
    """The committed summary cannot be safely served."""


class ExperimentSummaryProvider(Protocol):
    def get_summary(self) -> ExperimentSummaryResponse: ...


Row = TypeVar("Row")


class CommittedExperimentSummaryProvider:
    """Validate and parse exactly the accepted EXP-003 publication files."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._cached: ExperimentSummaryResponse | None = None
        self._lock = Lock()

    def get_summary(self) -> ExperimentSummaryResponse:
        if self._cached is not None:
            return self._cached
        with self._lock:
            if self._cached is not None:
                return self._cached
            try:
                summary = self._load()
            except ExperimentSummaryUnavailableError:
                raise
            except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
                raise ExperimentSummaryUnavailableError(
                    "invalid EXP-003 bundle"
                ) from exc
            self._cached = summary
            return summary

    def _open_root(self) -> int:
        required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
        if (
            any(not hasattr(os, name) for name in required)
            or os.open not in os.supports_dir_fd
        ):
            raise ExperimentSummaryUnavailableError(
                "safe EXP-003 file opening is unavailable"
            )
        descriptor = os.open(
            self._root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise ExperimentSummaryUnavailableError(
                    "EXP-003 root is not a directory"
                )
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    @staticmethod
    def _read_file(root_fd: int, relative: str) -> bytes:
        path = Path(relative)
        parts = relative.split("/")
        if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
            raise ExperimentSummaryUnavailableError("unsafe EXP-003 path")

        parent_fd = os.dup(root_fd)
        final_fd = -1
        try:
            for component in parts[:-1]:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=parent_fd,
                )
                os.close(parent_fd)
                parent_fd = next_fd
            final_fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=parent_fd,
            )
            if not stat.S_ISREG(os.fstat(final_fd).st_mode):
                raise ExperimentSummaryUnavailableError(
                    "EXP-003 input is not a regular file"
                )
            chunks: list[bytes] = []
            while chunk := os.read(final_fd, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            if final_fd >= 0:
                os.close(final_fd)
            os.close(parent_fd)

    @staticmethod
    def _verify_hash(content: bytes, expected: str) -> None:
        if hashlib.sha256(content).hexdigest() != expected:
            raise ExperimentSummaryUnavailableError("EXP-003 hash mismatch")

    def _load(self) -> ExperimentSummaryResponse:
        root_fd = self._open_root()
        try:
            manifest_bytes = self._read_file(root_fd, MANIFEST_PATH)
            checksums_bytes = self._read_file(root_fd, CHECKSUMS_PATH)
            self._verify_hash(manifest_bytes, MANIFEST_SHA256)
            self._verify_hash(checksums_bytes, CHECKSUMS_SHA256)
            manifest = json.loads(manifest_bytes)
            output_contract = self._validate_manifest(manifest)
            self._validate_checksums(checksums_bytes, output_contract)

            contents: dict[str, bytes] = {}
            for name, path in TABLE_PATHS.items():
                content = self._read_file(root_fd, path)
                expected_hash, expected_bytes = output_contract[path]
                self._verify_hash(content, expected_hash)
                if len(content) != expected_bytes:
                    raise ExperimentSummaryUnavailableError(
                        "EXP-003 byte count mismatch"
                    )
                contents[name] = content
        finally:
            os.close(root_fd)

        models = self._parse_rows("models", contents["models"], ModelComparisonRow)
        reranking = self._parse_rows("reranking", contents["reranking"], RerankingRow)
        exposure = self._parse_rows("exposure", contents["exposure"], ExposureRow)
        activity = self._parse_rows("activity", contents["activity"], ActivityGroupRow)
        frozen = self._parse_rows("frozen_test", contents["frozen_test"], FrozenTestRow)
        hypotheses = self._parse_rows(
            "hypotheses", contents["hypotheses"], HypothesisOutcome
        )
        self._validate_rows(models, reranking, exposure, activity, frozen, hypotheses)

        return ExperimentSummaryResponse(
            protocol_id=PROTOCOL_ID,
            schema_version=SCHEMA_VERSION,
            evidence_base_git_sha=EVIDENCE_BASE_GIT_SHA,
            manifest_sha256=MANIFEST_SHA256,
            validation=ValidationSummary(
                fit_scope="train",
                target_scope="validation",
                eligible_user_count=480,
                models=models,
                reranking=reranking,
                exposure=exposure,
                activity_groups=activity,
            ),
            frozen_test=FrozenTestSummary(
                fit_scope="train+validation",
                target_scope="test",
                eligible_user_count=1114,
                results=frozen,
            ),
            hypotheses=hypotheses,
        )

    @staticmethod
    def _validate_manifest(manifest: dict[str, Any]) -> dict[str, tuple[str, int]]:
        if (
            manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("schema_version") != 1
        ):
            raise ExperimentSummaryUnavailableError("unexpected EXP-003 identity")
        if manifest.get("base_git_sha") != EVIDENCE_BASE_GIT_SHA:
            raise ExperimentSummaryUnavailableError("unexpected EXP-003 base")
        identities = manifest.get("identities")
        if (
            not isinstance(identities, dict)
            or tuple(identities.get("models", ())) != MODELS
            or tuple(identities.get("configurations", ())) != CONFIGURATIONS
            or tuple(identities.get("metrics", ())) != METRICS
        ):
            raise ExperimentSummaryUnavailableError("unexpected EXP-003 identifiers")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list):
            raise ExperimentSummaryUnavailableError("invalid EXP-003 outputs")
        contract: dict[str, tuple[str, int]] = {}
        for output in outputs:
            if not isinstance(output, dict):
                raise ExperimentSummaryUnavailableError("invalid EXP-003 output")
            path = output.get("path")
            if path in contract or path not in EXPECTED_OUTPUTS:
                raise ExperimentSummaryUnavailableError("unexpected EXP-003 output")
            digest, size = output.get("sha256"), output.get("bytes")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or not isinstance(size, int)
            ):
                raise ExperimentSummaryUnavailableError(
                    "invalid EXP-003 output identity"
                )
            contract[path] = (digest, size)
        if set(contract) != EXPECTED_OUTPUTS:
            raise ExperimentSummaryUnavailableError("incomplete EXP-003 outputs")
        expected_counts = {
            "model_rows": 45,
            "reranking_rows": 144,
            "exposure_rows": 48,
            "activity_rows": 432,
            "test_rows": 27,
            "hypothesis_rows": 4,
        }
        if any(
            manifest.get("row_counts", {}).get(key) != value
            for key, value in expected_counts.items()
        ):
            raise ExperimentSummaryUnavailableError("unexpected EXP-003 row counts")
        return contract

    @staticmethod
    def _validate_checksums(
        content: bytes, outputs: dict[str, tuple[str, int]]
    ) -> None:
        rows: dict[str, str] = {}
        try:
            lines = content.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ExperimentSummaryUnavailableError(
                "invalid checksums encoding"
            ) from exc
        for line in lines:
            digest, separator, relative = line.partition("  ")
            if not separator or relative in rows:
                raise ExperimentSummaryUnavailableError("invalid checksums entry")
            rows[relative] = digest
        expected = {f"../../../{path}": digest for path, (digest, _) in outputs.items()}
        expected["manifest.json"] = MANIFEST_SHA256
        if rows != expected:
            raise ExperimentSummaryUnavailableError("unexpected checksums contract")

    @staticmethod
    def _parse_rows(name: str, content: bytes, model: type[Row]) -> tuple[Row, ...]:
        try:
            stream = io.StringIO(content.decode("utf-8"), newline="")
        except UnicodeDecodeError as exc:
            raise ExperimentSummaryUnavailableError("invalid CSV encoding") from exc
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != HEADERS[name]:
            raise ExperimentSummaryUnavailableError("unexpected CSV header")
        parsed: list[Row] = []
        for raw in reader:
            if None in raw or any(value is None for value in raw.values()):
                raise ExperimentSummaryUnavailableError("malformed CSV row")
            values = dict(raw)
            if name == "reranking":
                values["lambda"] = values.pop("lambda_")
            for field in _FLOAT_FIELDS[name]:
                value = values[field]
                if value == "" and field == "beta":
                    values[field] = None
                else:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ExperimentSummaryUnavailableError("non-finite CSV number")
                    values[field] = number
            for field in _INT_FIELDS[name]:
                text = values[field]
                if not text or any(character not in "0123456789" for character in text):
                    raise ExperimentSummaryUnavailableError("invalid CSV integer")
                values[field] = int(text)
            parsed.append(model.model_validate(values))  # type: ignore[attr-defined]
        return tuple(parsed)

    @staticmethod
    def _validate_rows(
        models: tuple[ModelComparisonRow, ...],
        reranking: tuple[RerankingRow, ...],
        exposure: tuple[ExposureRow, ...],
        activity: tuple[ActivityGroupRow, ...],
        frozen: tuple[FrozenTestRow, ...],
        hypotheses: tuple[HypothesisOutcome, ...],
    ) -> None:
        keys = (
            tuple((row.model, row.metric) for row in models),
            tuple((row.configuration_key, row.metric) for row in reranking),
            tuple((row.configuration_key, row.popularity_group) for row in exposure),
            tuple(
                (row.configuration_key, row.activity_group, row.metric)
                for row in activity
            ),
            tuple((row.configuration_key, row.metric) for row in frozen),
            tuple(row.hypothesis for row in hypotheses),
        )
        expected = (
            tuple((model, metric) for model in MODELS for metric in METRICS),
            tuple((config, metric) for config in CONFIGURATIONS for metric in METRICS),
            tuple(
                (config, group)
                for config in CONFIGURATIONS
                for group in POPULARITY_GROUPS
            ),
            tuple(
                (config, group, metric)
                for config in CONFIGURATIONS
                for group in ACTIVITY_GROUPS
                for metric in METRICS
            ),
            tuple(
                (config, metric)
                for config in FROZEN_CONFIGURATIONS
                for metric in METRICS
            ),
            HYPOTHESES,
        )
        if keys != expected:
            raise ExperimentSummaryUnavailableError("unexpected EXP-003 row ordering")
        validation_rows = (*models, *reranking, *exposure, *activity)
        if any(
            row.fit_scope != "train" or row.target_scope != "validation"
            for row in validation_rows
        ):
            raise ExperimentSummaryUnavailableError("unexpected validation scope")
        if any(
            row.fit_scope != "train+validation"
            or row.target_scope != "test"
            or row.eligible_user_count != 1114
            for row in frozen
        ):
            raise ExperimentSummaryUnavailableError("unexpected frozen-test scope")
        if any(row.eligible_user_count != 480 for row in (*models, *reranking)):
            raise ExperimentSummaryUnavailableError("unexpected validation population")
        outcomes = tuple(
            (
                row.scientific_outcome,
                row.engineering_acceptance,
                row.verdict,
                row.evidence_scope,
            )
            for row in hypotheses
        )
        if outcomes != (
            ("rejected", "accepted_evidence", "rejected", "validation"),
            ("accepted", "accepted_evidence", "accepted", "validation"),
            ("not_evaluable", "accepted_evidence", "not evaluable", "validation"),
            ("accepted", "PASS", "accepted", "engineering_cpu_latency"),
        ):
            raise ExperimentSummaryUnavailableError("unexpected hypothesis outcomes")


_FLOAT_FIELDS = {
    "models": {"mean", "ci_low", "ci_high", "confidence_level"},
    "reranking": {
        "lambda",
        "beta",
        "observed_value",
        "ci_low",
        "ci_high",
        "confidence_level",
    },
    "exposure": {"displayed_slot_share", "share_sum_tolerance"},
    "activity": {"observed_value", "ci_low", "ci_high", "confidence_level"},
    "frozen_test": {"point_estimate"},
    "hypotheses": set(),
}
_INT_FIELDS = {
    "models": {
        "bootstrap_resamples",
        "bootstrap_seed",
        "eligible_user_count",
        "recommendation_k",
        "catalog_size",
    },
    "reranking": {
        "bootstrap_resamples",
        "bootstrap_seed",
        "eligible_user_count",
        "recommendation_k",
        "catalog_size",
    },
    "exposure": {"displayed_slot_count", "total_displayed_slot_count"},
    "activity": {"activity_group_user_count", "bootstrap_resamples", "bootstrap_seed"},
    "frozen_test": {"eligible_user_count", "recommendation_k"},
    "hypotheses": set(),
}
