import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.evaluation_cohorts import (
    EvaluationArtifacts,
    build_cohorts_manifest_section,
    build_evaluation_artifacts,
    update_manifest_document,
    validate_evaluation_artifacts,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"


@dataclass(frozen=True)
class EvaluationCohortPaths:
    """Resolved DATA-006 sources, outputs, and manifest."""

    repository_root: Path
    train: Path
    validation: Path
    test: Path
    evaluation_cohorts: Path
    user_mapping: Path
    item_mapping: Path
    item_popularity_train: Path
    manifest: Path


def default_paths(
    repository_root: Path = REPOSITORY_ROOT,
) -> EvaluationCohortPaths:
    """Return project-default paths below a resolved repository root."""

    root = repository_root.resolve()
    processed = root / "data" / "processed"
    return EvaluationCohortPaths(
        repository_root=root,
        train=processed / "train.parquet",
        validation=processed / "validation.parquet",
        test=processed / "test.parquet",
        evaluation_cohorts=processed / "evaluation_cohorts.parquet",
        user_mapping=processed / "user_mapping.parquet",
        item_mapping=processed / "item_mapping.parquet",
        item_popularity_train=(processed / "item_popularity_train.parquet"),
        manifest=processed / "data_manifest.json",
    )


def _calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def _repository_relative(path: Path, repository_root: Path) -> str:
    root = repository_root.resolve()
    resolved = path.resolve()

    try:
        return str(resolved.relative_to(root))
    except ValueError as error:
        raise ValueError(
            f"DATA-006 path is outside the repository: {resolved}"
        ) from error


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("Data manifest is missing. Run make split first.")

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Data manifest contains invalid JSON: {error}") from error

    if not isinstance(manifest, dict):
        raise TypeError("Data manifest must contain a JSON object")

    return manifest


def _validate_source_manifest(
    manifest: dict[str, Any],
    sources: dict[str, pd.DataFrame],
    source_paths: dict[str, Path],
    repository_root: Path,
    source_sha256: dict[str, str],
) -> None:
    temporal_metadata = manifest.get("temporal_split")

    if not isinstance(temporal_metadata, dict):
        raise TypeError("Manifest has no temporal_split metadata; run make split first")

    expected_output_paths = {
        split_name: _repository_relative(path, repository_root)
        for split_name, path in source_paths.items()
    }
    expected_row_counts = {
        split_name: len(dataframe) for split_name, dataframe in sources.items()
    }
    expected_columns = sources["train"].columns.tolist()
    expected_dtypes = sources["train"].dtypes.astype(str).to_dict()
    checks = {
        "output_paths": (
            temporal_metadata.get("output_paths"),
            expected_output_paths,
        ),
        "output_sha256": (
            temporal_metadata.get("output_sha256"),
            source_sha256,
        ),
        "split_row_counts": (
            temporal_metadata.get("split_row_counts"),
            expected_row_counts,
        ),
        "columns": (
            temporal_metadata.get("columns"),
            expected_columns,
        ),
        "dtypes": (
            temporal_metadata.get("dtypes"),
            expected_dtypes,
        ),
    }
    mismatches = {
        field: {"manifest": manifest_value, "actual": actual_value}
        for field, (manifest_value, actual_value) in checks.items()
        if manifest_value != actual_value
    }

    if mismatches:
        raise ValueError(
            f"Split sources do not match temporal_split manifest metadata: {mismatches}"
        )


def _new_temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    ) as temporary_file:
        return Path(temporary_file.name)


def _write_staged_parquet(
    dataframe: pd.DataFrame,
    target: Path,
) -> Path:
    temporary_path = _new_temporary_path(target)

    try:
        dataframe.to_parquet(
            temporary_path,
            engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
            index=False,
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return temporary_path


def _serialize_manifest(manifest: dict[str, Any]) -> str:
    return (
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def _publish_manifest_last(
    manifest: dict[str, Any],
    paths: EvaluationCohortPaths,
    expected_output_sha256: dict[str, str],
) -> None:
    temporary_path = _new_temporary_path(paths.manifest)

    try:
        temporary_path.write_text(
            _serialize_manifest(manifest),
            encoding="utf-8",
        )
        parsed = json.loads(temporary_path.read_text(encoding="utf-8"))

        if parsed != manifest:
            raise RuntimeError(
                "Staged data manifest did not parse back to the same document"
            )

        output_paths = {
            "evaluation_cohorts": paths.evaluation_cohorts,
            "user_mapping": paths.user_mapping,
            "item_mapping": paths.item_mapping,
            "item_popularity_train": paths.item_popularity_train,
        }
        actual_output_sha256 = {
            output_name: _calculate_sha256(path)
            for output_name, path in output_paths.items()
        }

        if actual_output_sha256 != expected_output_sha256:
            raise RuntimeError("Published DATA-006 hashes do not match manifest hashes")

        os.replace(temporary_path, paths.manifest)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_artifacts(paths: dict[str, Path]) -> EvaluationArtifacts:
    return EvaluationArtifacts(
        evaluation_cohorts=pd.read_parquet(paths["evaluation_cohorts"]),
        user_mapping=pd.read_parquet(paths["user_mapping"]),
        item_mapping=pd.read_parquet(paths["item_mapping"]),
        item_popularity_train=pd.read_parquet(paths["item_popularity_train"]),
    )


def build_and_publish_evaluation_cohorts(
    paths: EvaluationCohortPaths,
) -> dict[str, Any]:
    """Build, verify, hash, and atomically replace each DATA-006 output."""

    root = paths.repository_root.resolve()
    path_values = {
        "train": paths.train,
        "validation": paths.validation,
        "test": paths.test,
        "evaluation_cohorts": paths.evaluation_cohorts,
        "user_mapping": paths.user_mapping,
        "item_mapping": paths.item_mapping,
        "item_popularity_train": paths.item_popularity_train,
        "manifest": paths.manifest,
    }

    for path in path_values.values():
        _repository_relative(path, root)

    source_paths = {
        "train": paths.train,
        "validation": paths.validation,
        "test": paths.test,
    }
    missing_sources = [
        split_name
        for split_name, source_path in source_paths.items()
        if not source_path.exists()
    ]

    if missing_sources:
        raise FileNotFoundError(
            f"Missing split sources {missing_sources}. Run make split first."
        )

    manifest = _load_manifest(paths.manifest)
    sources = {
        split_name: pd.read_parquet(source_path)
        for split_name, source_path in source_paths.items()
    }
    source_sha256 = {
        split_name: _calculate_sha256(source_path)
        for split_name, source_path in source_paths.items()
    }
    _validate_source_manifest(
        manifest,
        sources,
        source_paths,
        root,
        source_sha256,
    )
    artifacts = build_evaluation_artifacts(
        sources["train"],
        sources["validation"],
        sources["test"],
    )
    target_paths = {
        "evaluation_cohorts": paths.evaluation_cohorts,
        "user_mapping": paths.user_mapping,
        "item_mapping": paths.item_mapping,
        "item_popularity_train": paths.item_popularity_train,
    }
    dataframes = {
        output_name: getattr(artifacts, output_name) for output_name in target_paths
    }
    staged_paths: dict[str, Path] = {}

    try:
        for output_name, dataframe in dataframes.items():
            staged_paths[output_name] = _write_staged_parquet(
                dataframe,
                target_paths[output_name],
            )

        stored_artifacts = _read_artifacts(staged_paths)
        validate_evaluation_artifacts(
            stored_artifacts,
            sources["train"],
            sources["validation"],
            sources["test"],
        )
        output_sha256 = {
            output_name: _calculate_sha256(path)
            for output_name, path in staged_paths.items()
        }
        relative_source_paths = {
            split_name: _repository_relative(source_path, root)
            for split_name, source_path in source_paths.items()
        }
        relative_output_paths = {
            output_name: _repository_relative(output_path, root)
            for output_name, output_path in target_paths.items()
        }
        section = build_cohorts_manifest_section(
            stored_artifacts,
            sources["train"],
            sources["validation"],
            sources["test"],
            source_paths=relative_source_paths,
            source_sha256=source_sha256,
            output_paths=relative_output_paths,
            output_sha256=output_sha256,
            parquet_engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
        )
        updated_manifest = update_manifest_document(manifest, section)

        for output_name, target_path in target_paths.items():
            os.replace(staged_paths[output_name], target_path)
            staged_paths.pop(output_name)

        _publish_manifest_last(
            updated_manifest,
            paths,
            output_sha256,
        )
        return section
    finally:
        for temporary_path in staged_paths.values():
            temporary_path.unlink(missing_ok=True)


def _path_argument(raw_path: str, repository_root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repository_root / path


def _parse_args() -> EvaluationCohortPaths:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-free MovieLens evaluation cohorts, mappings, "
            "and train item popularity"
        )
    )
    parser.add_argument(
        "--repository-root",
        default=str(REPOSITORY_ROOT),
        help="repository root used to resolve and validate all paths",
    )
    parser.add_argument("--train", default="data/processed/train.parquet")
    parser.add_argument(
        "--validation",
        default="data/processed/validation.parquet",
    )
    parser.add_argument("--test", default="data/processed/test.parquet")
    parser.add_argument(
        "--evaluation-cohorts-output",
        default="data/processed/evaluation_cohorts.parquet",
    )
    parser.add_argument(
        "--user-mapping-output",
        default="data/processed/user_mapping.parquet",
    )
    parser.add_argument(
        "--item-mapping-output",
        default="data/processed/item_mapping.parquet",
    )
    parser.add_argument(
        "--item-popularity-train-output",
        default="data/processed/item_popularity_train.parquet",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/data_manifest.json",
    )
    arguments = parser.parse_args()
    root = Path(arguments.repository_root).resolve()
    return EvaluationCohortPaths(
        repository_root=root,
        train=_path_argument(arguments.train, root),
        validation=_path_argument(arguments.validation, root),
        test=_path_argument(arguments.test, root),
        evaluation_cohorts=_path_argument(
            arguments.evaluation_cohorts_output,
            root,
        ),
        user_mapping=_path_argument(arguments.user_mapping_output, root),
        item_mapping=_path_argument(arguments.item_mapping_output, root),
        item_popularity_train=_path_argument(
            arguments.item_popularity_train_output,
            root,
        ),
        manifest=_path_argument(arguments.manifest, root),
    )


def _print_context_summary(
    evaluation_split: str,
    context: dict[str, Any],
) -> None:
    print(f"- {evaluation_split} context ({context['fit_scope']} fit):")
    print(f"  fit interactions: {context['fit_interaction_count']}")
    print(f"  fit users: {context['fit_user_count']}")
    print(f"  fit items: {context['fit_item_count']}")
    print(f"  target interactions: {context['target_interaction_count']}")
    print(f"  target users: {context['target_user_count']}")
    print(f"  eligible users: {context['eligible_user_count']}")
    print(f"  excluded users: {context['excluded_user_count']}")
    print(f"  cold-start users: {context['cold_start_user_count']}")
    print(f"  warm target interactions: {context['warm_target_interaction_count']}")
    print(
        "  cold-start target interactions: "
        f"{context['cold_start_target_interaction_count']}"
    )
    print(
        "  cold-start target unique items: "
        f"{context['cold_start_target_unique_item_count']}"
    )
    print(f"  exclusion reasons: {context['exclusion_reason_counts']}")
    print(f"  eligible activity groups: {context['eligible_activity_group_counts']}")


def _print_summary(section: dict[str, Any]) -> None:
    print("MovieLens 1M DATA-006 artifacts generated:")

    for evaluation_split in ("validation", "test"):
        _print_context_summary(
            evaluation_split,
            section["contexts"][evaluation_split],
        )

    print(f"- mapping counts: {section['mapping_counts']}")
    print(f"- train item popularity: {section['item_popularity_train']}")
    print(f"- output rows: {section['output_row_counts']}")
    print(f"- output SHA-256: {section['output_sha256']}")
    print("- each Parquet atomically replaced; manifest published last")


def main() -> None:
    """CLI entrypoint with a concise nonzero failure."""

    paths = _parse_args()

    try:
        section = build_and_publish_evaluation_cohorts(paths)
    except Exception as error:
        raise SystemExit(f"Evaluation cohort build failed: {error}") from error

    _print_summary(section)


if __name__ == "__main__":
    main()
