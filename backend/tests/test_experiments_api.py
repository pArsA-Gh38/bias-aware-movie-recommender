"""Behavioral coverage for API-006 Slice B."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.auth import AUTHENTICATION_FAILED
from app.api.experiments import EXPERIMENT_SUMMARY_UNAVAILABLE
from app.core.auth import AuthService, AuthUser, DuplicateEmailError
from app.core.config import Settings
from app.core.experiments import (
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    TABLE_PATHS,
    CommittedExperimentSummaryProvider,
    ExperimentSummaryUnavailableError,
)
from app.core.security import PasswordManager, TokenManager
from app.main import create_app

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SECRET = "experiment-summary-test-secret-with-sufficient-entropy"
ALLOWED_READS = {MANIFEST_PATH, CHECKSUMS_PATH, *TABLE_PATHS.values()}


class MemoryUserStore:
    def __init__(self, user: AuthUser) -> None:
        self.user = user

    def create_user(self, *, email: str, password_hash: str) -> AuthUser:
        raise DuplicateEmailError

    def get_by_email(self, email: str) -> AuthUser | None:
        return self.user if email == self.user.email else None

    def get_by_id(self, user_id: UUID) -> AuthUser | None:
        return self.user if user_id == self.user.id else None


class ExplodingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def get_summary(self) -> Any:
        self.calls += 1
        raise RuntimeError("private provider detail")


def authenticated_client(**kwargs: Any) -> tuple[TestClient, dict[str, str]]:
    user = AuthUser(
        id=uuid4(),
        email="experiments@example.com",
        password_hash="unused",
        created_at=datetime.now(UTC),
        is_active=True,
    )
    tokens = TokenManager(secret_key=SECRET)
    auth = AuthService(
        users=MemoryUserStore(user),
        passwords=PasswordManager(),
        tokens=tokens,
    )
    client = TestClient(create_app(settings=Settings(), auth_service=auth, **kwargs))
    return client, {"Authorization": f"Bearer {tokens.create(user.id)}"}


def test_exact_success_contract_cardinality_order_scopes_and_null_beta() -> None:
    provider = CommittedExperimentSummaryProvider(REPOSITORY_ROOT)
    client, headers = authenticated_client(experiment_summary_provider=provider)

    response = client.get("/experiments/summary", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "protocol_id",
        "schema_version",
        "evidence_base_git_sha",
        "manifest_sha256",
        "validation",
        "frozen_test",
        "hypotheses",
    }
    assert body["protocol_id"] == "exp003-stored-evidence-publication-v1"
    assert body["schema_version"] == 1
    assert body["evidence_base_git_sha"] == "8d3d2e27b81676648b982bb3e0f656c6f8d36888"
    validation = body["validation"]
    assert (
        validation["fit_scope"],
        validation["target_scope"],
        validation["eligible_user_count"],
    ) == ("train", "validation", 480)
    assert [
        len(validation[key])
        for key in ("models", "reranking", "exposure", "activity_groups")
    ] == [45, 144, 48, 432]
    assert [row["model"] for row in validation["models"][::9]] == [
        "Popularity",
        "ItemKNN-50",
        "ItemKNN-100",
        "ItemKNN-200",
        "ALS-selected-seed42",
    ]
    assert validation["reranking"][0]["lambda"] == 1.0
    assert "lambda_" not in validation["reranking"][0]
    assert validation["reranking"][0]["beta"] is None
    frozen = body["frozen_test"]
    assert (
        frozen["fit_scope"],
        frozen["target_scope"],
        frozen["eligible_user_count"],
    ) == ("train+validation", "test", 1114)
    assert len(frozen["results"]) == 27
    assert [row["hypothesis"] for row in body["hypotheses"]] == ["H1", "H2", "H3", "H4"]
    assert [
        (row["scientific_outcome"], row["engineering_acceptance"])
        for row in body["hypotheses"]
    ] == [
        ("rejected", "accepted_evidence"),
        ("accepted", "accepted_evidence"),
        ("not_evaluable", "accepted_evidence"),
        ("accepted", "PASS"),
    ]
    assert "A1_L07 and A1_L06" in body["hypotheses"][1]["reason"]
    assert "7.43526535 ms <= 300 ms" in body["hypotheses"][3]["reason"]


def test_success_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = CommittedExperimentSummaryProvider(REPOSITORY_ROOT)
    calls = 0
    original = provider._read_file

    def observed(root_fd: int, path: str) -> bytes:
        nonlocal calls
        calls += 1
        return original(root_fd, path)

    monkeypatch.setattr(provider, "_read_file", observed)
    provider.get_summary()
    first_calls = calls
    provider.get_summary()
    assert calls == first_calls == 8


def test_authentication_precedes_provider_access() -> None:
    provider = ExplodingProvider()
    response = TestClient(
        create_app(settings=Settings(), experiment_summary_provider=provider)
    ).get("/experiments/summary")
    assert response.status_code == 401
    assert response.json() == {"detail": AUTHENTICATION_FAILED}
    assert provider.calls == 0


def test_unset_and_exploding_providers_are_sanitized() -> None:
    client, headers = authenticated_client()
    missing = client.get("/experiments/summary", headers=headers)
    assert missing.status_code == 503
    assert missing.json() == {"detail": EXPERIMENT_SUMMARY_UNAVAILABLE}

    provider = ExplodingProvider()
    client, headers = authenticated_client(experiment_summary_provider=provider)
    failed = client.get("/experiments/summary", headers=headers)
    assert failed.status_code == 503
    assert failed.json() == {"detail": EXPERIMENT_SUMMARY_UNAVAILABLE}
    assert "private" not in failed.text


def test_provider_not_loaded_by_construction_openapi_or_health() -> None:
    provider = ExplodingProvider()
    client, _ = authenticated_client(experiment_summary_provider=provider)
    client.app.openapi()
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    assert provider.calls == 0


def test_openapi_contract() -> None:
    client, _ = authenticated_client()
    operation = client.app.openapi()["paths"]["/experiments/summary"]["get"]
    assert operation["security"] == [{"HTTPBearer": []}]
    assert {"200", "401", "503"} == set(operation["responses"])
    assert "422" not in operation["responses"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("ExperimentSummaryResponse")


def copy_bundle(destination: Path) -> Path:
    for relative in ALLOWED_READS:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative, target)
    return destination


@pytest.mark.parametrize("relative", sorted(ALLOWED_READS))
def test_missing_component_fails_closed(tmp_path: Path, relative: str) -> None:
    root = copy_bundle(tmp_path)
    (root / relative).unlink()
    with pytest.raises(ExperimentSummaryUnavailableError):
        CommittedExperimentSummaryProvider(root).get_summary()


@pytest.mark.parametrize(
    "relative", [MANIFEST_PATH, CHECKSUMS_PATH, *TABLE_PATHS.values()]
)
def test_every_hash_drift_fails_closed(tmp_path: Path, relative: str) -> None:
    root = copy_bundle(tmp_path)
    with (root / relative).open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ExperimentSummaryUnavailableError):
        CommittedExperimentSummaryProvider(root).get_summary()


def test_symlink_and_out_of_root_are_rejected(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path / "root")
    manifest = root / MANIFEST_PATH
    outside = tmp_path / "outside.json"
    outside.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(outside)
    with pytest.raises(ExperimentSummaryUnavailableError):
        CommittedExperimentSummaryProvider(root).get_summary()


def test_configured_root_symlink_is_rejected(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path / "root")
    alias = tmp_path / "root-alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ExperimentSummaryUnavailableError):
        CommittedExperimentSummaryProvider(alias).get_summary()


@pytest.mark.parametrize("alias_kind", ["external", "relative", "chained"])
def test_reports_directory_symlink_aliases_are_rejected(
    tmp_path: Path, alias_kind: str
) -> None:
    root = copy_bundle(tmp_path / "root")
    reports = root / "reports"
    original = root / "real-reports"
    reports.rename(original)
    if alias_kind == "external":
        target = tmp_path / "external-reports"
        shutil.copytree(original, target)
        reports.symlink_to(target, target_is_directory=True)
    elif alias_kind == "relative":
        reports.symlink_to(original.name, target_is_directory=True)
    else:
        first = root / "reports-alias-one"
        second = root / "reports-alias-two"
        first.symlink_to(second.name, target_is_directory=True)
        second.symlink_to(original.name, target_is_directory=True)
        reports.symlink_to(first.name, target_is_directory=True)
    with pytest.raises(ExperimentSummaryUnavailableError):
        CommittedExperimentSummaryProvider(root).get_summary()


def test_nested_parent_directory_symlink_is_rejected(tmp_path: Path) -> None:
    root = copy_bundle(tmp_path / "root")
    tables = root / "reports/tables"
    original = root / "reports/real-tables"
    tables.rename(original)
    tables.symlink_to(original.name, target_is_directory=True)
    with pytest.raises(ExperimentSummaryUnavailableError):
        CommittedExperimentSummaryProvider(root).get_summary()


@pytest.mark.parametrize(
    "relative", ["/absolute", "", ".", "../escape", "a//b", "a/./b"]
)
def test_noncanonical_relative_paths_are_rejected(
    tmp_path: Path, relative: str
) -> None:
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ExperimentSummaryUnavailableError):
            CommittedExperimentSummaryProvider._read_file(root_fd, relative)
    finally:
        os.close(root_fd)


@pytest.mark.parametrize("replacement", ["directory", "fifo"])
def test_non_regular_terminal_files_are_rejected(
    tmp_path: Path, replacement: str
) -> None:
    root = copy_bundle(tmp_path / "root")
    manifest = root / MANIFEST_PATH
    manifest.unlink()
    if replacement == "directory":
        manifest.mkdir()
    else:
        os.mkfifo(manifest)
    with pytest.raises(ExperimentSummaryUnavailableError):
        CommittedExperimentSummaryProvider(root).get_summary()


def test_missing_safe_open_capability_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.core.experiments.os.supports_dir_fd", set())
    with pytest.raises(ExperimentSummaryUnavailableError):
        CommittedExperimentSummaryProvider(REPOSITORY_ROOT).get_summary()


def test_filesystem_allowlist_excludes_upstream_inputs_and_protected_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[str] = []
    provider = CommittedExperimentSummaryProvider(REPOSITORY_ROOT)
    original = provider._read_file

    def guarded(root_fd: int, relative: str) -> bytes:
        reads.append(relative)
        assert relative in ALLOWED_READS
        assert not relative.startswith("data/processed/")
        assert "exp002" not in relative.lower()
        assert not relative.startswith("ml/src/")
        return original(root_fd, relative)

    monkeypatch.setattr(provider, "_read_file", guarded)
    provider.get_summary()
    assert set(reads) == ALLOWED_READS


def test_provider_never_reopens_an_approved_path_with_pathlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_reopen(_path: Path) -> bytes:
        raise AssertionError("unsafe pathname reopen")

    monkeypatch.setattr(Path, "read_bytes", forbidden_reopen)
    assert (
        len(
            CommittedExperimentSummaryProvider(REPOSITORY_ROOT)
            .get_summary()
            .validation.models
        )
        == 45
    )


def permit_wrapped_open(monkeypatch: pytest.MonkeyPatch, wrapped: Any) -> None:
    monkeypatch.setattr("app.core.experiments.os.open", wrapped)
    monkeypatch.setattr("app.core.experiments.os.supports_dir_fd", {wrapped})


def test_directory_swap_before_component_open_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = copy_bundle(tmp_path / "root")
    external = copy_bundle(tmp_path / "external") / "reports"
    original_open = os.open
    swapped = False

    def swapping_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags, *args, **kwargs)
        if Path(path) == root and kwargs.get("dir_fd") is None and not swapped:
            swapped = True
            (root / "reports").rename(root / "original-reports")
            (root / "reports").symlink_to(external, target_is_directory=True)
        return descriptor

    permit_wrapped_open(monkeypatch, swapping_open)
    with pytest.raises(ExperimentSummaryUnavailableError):
        CommittedExperimentSummaryProvider(root).get_summary()
    assert swapped


def test_directory_swap_after_component_open_remains_anchored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = copy_bundle(tmp_path / "root")
    external_root = copy_bundle(tmp_path / "external")
    (external_root / MANIFEST_PATH).write_bytes(b"substituted")
    original_open = os.open
    swapped = False

    def swapping_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == "reports" and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            reports = root / "reports"
            reports.rename(root / "original-reports")
            reports.symlink_to(external_root / "reports", target_is_directory=True)
        elif path == "experiments" and swapped and (root / "reports").is_symlink():
            (root / "reports").unlink()
            (root / "original-reports").rename(root / "reports")
        return descriptor

    permit_wrapped_open(monkeypatch, swapping_open)
    summary = CommittedExperimentSummaryProvider(root).get_summary()
    assert swapped
    assert len(summary.validation.models) == 45


def test_terminal_replacement_after_fstat_reads_opened_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = copy_bundle(tmp_path / "root")
    target = root / TABLE_PATHS["models"]
    original_open = os.open
    original_fstat = os.fstat
    target_fd = -1
    replaced = False

    def observed_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal target_fd
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == target.name:
            target_fd = descriptor
        return descriptor

    def replacing_fstat(descriptor: int) -> os.stat_result:
        nonlocal replaced
        metadata = original_fstat(descriptor)
        if descriptor == target_fd and not replaced:
            replaced = True
            target.rename(target.with_suffix(".original"))
            target.write_bytes(b"substituted")
        return metadata

    permit_wrapped_open(monkeypatch, observed_open)
    monkeypatch.setattr("app.core.experiments.os.fstat", replacing_fstat)
    summary = CommittedExperimentSummaryProvider(root).get_summary()
    assert replaced
    assert len(summary.validation.models) == 45


def test_descriptor_cleanup_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = copy_bundle(tmp_path / "good")
    bad = copy_bundle(tmp_path / "bad")
    (bad / MANIFEST_PATH).write_bytes(b"invalid")
    original_open, original_dup, original_close = os.open, os.dup, os.close
    active: set[int] = set()

    def tracked_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        descriptor = original_open(path, flags, *args, **kwargs)
        active.add(descriptor)
        return descriptor

    def tracked_dup(descriptor: int) -> int:
        duplicate = original_dup(descriptor)
        active.add(duplicate)
        return duplicate

    def tracked_close(descriptor: int) -> None:
        original_close(descriptor)
        active.discard(descriptor)

    permit_wrapped_open(monkeypatch, tracked_open)
    monkeypatch.setattr("app.core.experiments.os.dup", tracked_dup)
    monkeypatch.setattr("app.core.experiments.os.close", tracked_close)
    for _ in range(3):
        assert (
            len(
                CommittedExperimentSummaryProvider(good).get_summary().validation.models
            )
            == 45
        )
        assert not active
        with pytest.raises(ExperimentSummaryUnavailableError):
            CommittedExperimentSummaryProvider(bad).get_summary()
        assert not active


@pytest.mark.parametrize("failure", ["missing", "malformed", "permission"])
def test_native_bundle_failures_are_api_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root = copy_bundle(tmp_path / "root")
    if failure == "missing":
        (root / MANIFEST_PATH).unlink()
    elif failure == "malformed":
        (root / MANIFEST_PATH).write_bytes(b"malformed")
    else:
        original_open = os.open

        def denied_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            if path == "manifest.json":
                raise PermissionError("private permission detail")
            return original_open(path, flags, *args, **kwargs)

        permit_wrapped_open(monkeypatch, denied_open)

    client, headers = authenticated_client(
        experiment_summary_provider=CommittedExperimentSummaryProvider(root)
    )
    response = client.get("/experiments/summary", headers=headers)
    assert response.status_code == 503
    assert response.json() == {"detail": EXPERIMENT_SUMMARY_UNAVAILABLE}
    assert "private" not in response.text


def resign_bundle(root: Path, mutate: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = root / MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest, root)
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    checksum_lines = []
    for output in manifest["outputs"]:
        checksum_lines.append(f"{output['sha256']}  ../../../{output['path']}\n")
    checksum_lines.append(f"{manifest_hash}  manifest.json\n")
    checksum_bytes = "".join(checksum_lines).encode()
    (root / CHECKSUMS_PATH).write_bytes(checksum_bytes)
    monkeypatch.setattr("app.core.experiments.MANIFEST_SHA256", manifest_hash)
    monkeypatch.setattr(
        "app.core.experiments.CHECKSUMS_SHA256",
        hashlib.sha256(checksum_bytes).hexdigest(),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "protocol",
        "schema",
        "missing_output",
        "duplicate_output",
        "unknown_output",
        "relocated_output",
        "row_count",
    ],
)
def test_manifest_contract_mutations_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = copy_bundle(tmp_path)

    def mutate(manifest: dict[str, Any], _root: Path) -> None:
        if mutation == "protocol":
            manifest["protocol_id"] = "wrong"
        elif mutation == "schema":
            manifest["schema_version"] = 2
        elif mutation == "missing_output":
            manifest["outputs"].pop()
        elif mutation == "duplicate_output":
            manifest["outputs"].append(dict(manifest["outputs"][0]))
        elif mutation == "unknown_output":
            manifest["outputs"][0]["path"] = "reports/tables/unknown.csv"
        elif mutation == "relocated_output":
            manifest["outputs"][0]["path"] = "reports/experiments/exp003/table.csv"
        else:
            manifest["row_counts"]["model_rows"] = 44

    resign_bundle(root, mutate, monkeypatch)
    with pytest.raises(ExperimentSummaryUnavailableError):
        CommittedExperimentSummaryProvider(root).get_summary()


@pytest.mark.parametrize(
    "mutation", ["header", "count", "key", "type", "scope", "order", "nan", "infinity"]
)
def test_csv_schema_mutations_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = copy_bundle(tmp_path)
    table_path = root / TABLE_PATHS["models"]
    lines = table_path.read_text().splitlines()
    if mutation == "header":
        lines[0] = lines[0].replace("model,", "wrong,")
    elif mutation == "count":
        lines.pop()
    elif mutation == "key":
        lines[2] = lines[1]
    elif mutation == "type":
        lines[1] = lines[1].replace("0.11770833333333333", "text")
    elif mutation == "scope":
        lines[1] = lines[1].replace(",train,validation,", ",test,validation,")
    elif mutation == "order":
        lines[1], lines[2] = lines[2], lines[1]
    elif mutation == "nan":
        lines[1] = lines[1].replace("0.11770833333333333", "NaN")
    else:
        lines[1] = lines[1].replace("0.11770833333333333", "Infinity")
    content = ("\n".join(lines) + "\n").encode()
    table_path.write_bytes(content)

    def mutate(manifest: dict[str, Any], _root: Path) -> None:
        output = next(
            row for row in manifest["outputs"] if row["path"] == TABLE_PATHS["models"]
        )
        output["sha256"] = hashlib.sha256(content).hexdigest()
        output["bytes"] = len(content)

    resign_bundle(root, mutate, monkeypatch)
    with pytest.raises(ExperimentSummaryUnavailableError):
        CommittedExperimentSummaryProvider(root).get_summary()
