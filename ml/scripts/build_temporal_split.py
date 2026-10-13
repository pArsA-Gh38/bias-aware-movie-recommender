import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.temporal_split import (
    TemporalSplits,
    build_global_temporal_split,
    build_temporal_split_manifest_section,
    update_manifest_document,
    validate_temporal_split,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"


@dataclass(frozen=True)
class TemporalSplitPaths:
    """Resolved input, output, and manifest paths for one split execution."""

    repository_root: Path
    source: Path
    train: Path
    validation: Path
    test: Path
    manifest: Path


def default_paths(
    repository_root: Path = REPOSITORY_ROOT,
) -> TemporalSplitPaths:
    """Return DATA-005 project defaults under the resolved repository root."""

    root = repository_root.resolve()
    processed = root / "data" / "processed"
    return TemporalSplitPaths(
        repository_root=root,
        source=processed / "positives.parquet",
        train=processed / "train.parquet",
        validation=processed / "validation.parquet",
        test=processed / "test.parquet",
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
            f"DATA-005 path is outside the repository: {resolved}"
        ) from error


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            "Data manifest is missing. Run make data and make positives first."
        )

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Data manifest contains invalid JSON: {error}") from error

    if not isinstance(manifest, dict):
        raise TypeError("Data manifest must contain a JSON object")

    return manifest


def _validate_source_manifest(
    manifest: dict[str, Any],
    source: pd.DataFrame,
    source_path: Path,
    repository_root: Path,
    source_sha256: str,
) -> None:
    positive_metadata = manifest.get("positive_interactions")

    if not isinstance(positive_metadata, dict):
        raise TypeError(
            "Manifest has no positive_interactions metadata; run make positives first"
        )

    expected_values = {
        "output_path": _repository_relative(
            source_path,
            repository_root,
        ),
        "positive_row_count": len(source),
        "output_sha256": source_sha256,
        "columns": source.columns.tolist(),
        "dtypes": source.dtypes.astype(str).to_dict(),
    }
    mismatches = {
        key: {
            "manifest": positive_metadata.get(key),
            "actual": actual_value,
        }
        for key, actual_value in expected_values.items()
        if positive_metadata.get(key) != actual_value
    }

    if mismatches:
        raise ValueError(
            f"Positive source does not match manifest metadata: {mismatches}"
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
    paths: TemporalSplitPaths,
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
            "train": paths.train,
            "validation": paths.validation,
            "test": paths.test,
        }
        actual_output_sha256 = {
            split_name: _calculate_sha256(path)
            for split_name, path in output_paths.items()
        }

        if actual_output_sha256 != expected_output_sha256:
            raise RuntimeError(
                "Published output hashes do not match staged manifest hashes"
            )

        os.replace(temporary_path, paths.manifest)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def build_and_publish_temporal_split(
    paths: TemporalSplitPaths,
) -> dict[str, Any]:
    """Build, read back, validate, hash, and atomically publish DATA-005."""

    root = paths.repository_root.resolve()
    path_values = {
        "source": paths.source,
        "train": paths.train,
        "validation": paths.validation,
        "test": paths.test,
        "manifest": paths.manifest,
    }

    for path in path_values.values():
        _repository_relative(path, root)

    if not paths.source.exists():
        raise FileNotFoundError(
            "Positive interactions are missing. Run make positives first."
        )

    manifest = _load_manifest(paths.manifest)
    source = pd.read_parquet(paths.source)
    source_sha256 = _calculate_sha256(paths.source)
    _validate_source_manifest(
        manifest,
        source,
        paths.source,
        root,
        source_sha256,
    )
    splits = build_global_temporal_split(source)
    validate_temporal_split(source, splits)

    target_paths = {
        "train": paths.train,
        "validation": paths.validation,
        "test": paths.test,
    }
    dataframes = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    staged_paths: dict[str, Path] = {}

    try:
        for split_name, dataframe in dataframes.items():
            staged_paths[split_name] = _write_staged_parquet(
                dataframe,
                target_paths[split_name],
            )

        stored_splits = TemporalSplits(
            train=pd.read_parquet(staged_paths["train"]),
            validation=pd.read_parquet(staged_paths["validation"]),
            test=pd.read_parquet(staged_paths["test"]),
            train_end=splits.train_end,
            validation_end=splits.validation_end,
        )
        validate_temporal_split(source, stored_splits)
        output_sha256 = {
            split_name: _calculate_sha256(path)
            for split_name, path in staged_paths.items()
        }

        for split_name, target_path in target_paths.items():
            os.replace(staged_paths[split_name], target_path)
            staged_paths.pop(split_name)

        output_paths = {
            split_name: _repository_relative(path, root)
            for split_name, path in target_paths.items()
        }
        temporal_section = build_temporal_split_manifest_section(
            source,
            stored_splits,
            source_path=_repository_relative(paths.source, root),
            source_sha256=source_sha256,
            output_paths=output_paths,
            output_sha256=output_sha256,
            parquet_engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
        )
        updated_manifest = update_manifest_document(
            manifest,
            temporal_section,
        )
        _publish_manifest_last(
            updated_manifest,
            paths,
            output_sha256,
        )
        return temporal_section
    finally:
        for temporary_path in staged_paths.values():
            temporary_path.unlink(missing_ok=True)


def _path_argument(raw_path: str, repository_root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repository_root / path


def _parse_args() -> TemporalSplitPaths:
    parser = argparse.ArgumentParser(
        description=("Build the deterministic MovieLens global temporal 80/10/10 split")
    )
    parser.add_argument(
        "--repository-root",
        default=str(REPOSITORY_ROOT),
        help="repository root used to resolve and validate all paths",
    )
    parser.add_argument(
        "--source",
        default="data/processed/positives.parquet",
    )
    parser.add_argument(
        "--train-output",
        default="data/processed/train.parquet",
    )
    parser.add_argument(
        "--validation-output",
        default="data/processed/validation.parquet",
    )
    parser.add_argument(
        "--test-output",
        default="data/processed/test.parquet",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/data_manifest.json",
    )
    arguments = parser.parse_args()
    root = Path(arguments.repository_root).resolve()
    return TemporalSplitPaths(
        repository_root=root,
        source=_path_argument(arguments.source, root),
        train=_path_argument(arguments.train_output, root),
        validation=_path_argument(arguments.validation_output, root),
        test=_path_argument(arguments.test_output, root),
        manifest=_path_argument(arguments.manifest, root),
    )


def _print_summary(section: dict[str, Any]) -> None:
    counts = section["split_row_counts"]
    cutoffs = section["cutoff_indices"]
    print("MovieLens 1M global temporal split generated:")
    print(f"- source rows: {section['source_row_count']}")
    print(f"- train rows: {counts['train']}")
    print(f"- validation rows: {counts['validation']}")
    print(f"- test rows: {counts['test']}")
    print(
        "- cutoff indices: "
        f"train={cutoffs['train_end_exclusive']}, "
        f"validation={cutoffs['validation_end_exclusive']}"
    )
    print(f"- output SHA-256: {section['output_sha256']}")
    print("- data manifest published after all split outputs")


def main() -> None:
    """CLI entrypoint with a concise error and nonzero failure status."""

    paths = _parse_args()

    try:
        section = build_and_publish_temporal_split(paths)
    except Exception as error:
        raise SystemExit(f"Temporal split failed: {error}") from error

    _print_summary(section)


if __name__ == "__main__":
    main()
