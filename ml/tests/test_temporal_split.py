import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import scripts.build_temporal_split as split_script
from scripts.build_temporal_split import (
    TemporalSplitPaths,
    build_and_publish_temporal_split,
)
from src.data.temporal_split import (
    INTERACTION_KEY,
    TemporalSplits,
    allocation_boundaries,
    build_global_temporal_split,
    build_temporal_split_manifest_section,
    calculate_overlap_counts,
    canonical_sort,
    update_manifest_document,
    validate_temporal_split,
)


def _source_frame(
    row_count: int,
    *,
    shuffled: bool = True,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "user_id": pd.Series(
                [(index % 4) + 1 for index in range(row_count)],
                dtype="int32",
            ),
            "movie_id": pd.Series(
                [100 + index for index in range(row_count)],
                dtype="int32",
            ),
            "rating": pd.Series(
                [4 + (index % 2) for index in range(row_count)],
                dtype="int8",
            ),
            "timestamp": pd.Series(
                [1_000 + (index // 3) for index in range(row_count)],
                dtype="int64",
            ),
        }
    )

    if shuffled and row_count:
        frame = frame.sample(frac=1, random_state=17).reset_index(drop=True)

    return frame


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test_repository(
    tmp_path: Path,
    source: pd.DataFrame,
) -> tuple[TemporalSplitPaths, dict[str, Any]]:
    root = tmp_path / "repository"
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    source_path = processed / "positives.parquet"
    source.to_parquet(
        source_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    manifest: dict[str, Any] = {
        "manifest_schema_version": 1,
        "unknown_section": {
            "nested": ["preserve", 7],
        },
        "positive_interactions": {
            "output_path": "data/processed/positives.parquet",
            "positive_row_count": len(source),
            "output_sha256": _sha256(source_path),
            "columns": source.columns.tolist(),
            "dtypes": source.dtypes.astype(str).to_dict(),
        },
    }
    manifest_path = processed / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    paths = TemporalSplitPaths(
        repository_root=root,
        source=source_path,
        train=processed / "train.parquet",
        validation=processed / "validation.parquet",
        test=processed / "test.parquet",
        manifest=manifest_path,
    )
    return paths, manifest


def _manifest_section(
    source: pd.DataFrame,
    splits: TemporalSplits,
) -> dict[str, Any]:
    return build_temporal_split_manifest_section(
        source,
        splits,
        source_path="data/processed/positives.parquet",
        source_sha256="a" * 64,
        output_paths={
            "train": "data/processed/train.parquet",
            "validation": "data/processed/validation.parquet",
            "test": "data/processed/test.parquet",
        },
        output_sha256={
            "train": "b" * 64,
            "validation": "c" * 64,
            "test": "d" * 64,
        },
        parquet_engine="pyarrow",
        compression="snappy",
    )


def test_allocation_for_ten_rows_is_exact() -> None:
    assert allocation_boundaries(10) == (8, 9)
    splits = build_global_temporal_split(_source_frame(10))
    assert [len(splits.train), len(splits.validation), len(splits.test)] == [
        8,
        1,
        1,
    ]


def test_allocation_for_eleven_rows_uses_fixed_integer_boundaries() -> None:
    assert allocation_boundaries(11) == (8, 9)
    splits = build_global_temporal_split(_source_frame(11))
    assert [len(splits.train), len(splits.validation), len(splits.test)] == [
        8,
        1,
        2,
    ]


def test_allocation_for_non_divisible_count_is_exact() -> None:
    assert allocation_boundaries(17) == (13, 15)
    splits = build_global_temporal_split(_source_frame(17))
    assert [len(splits.train), len(splits.validation), len(splits.test)] == [
        13,
        2,
        2,
    ]


def test_empty_source_fails_fast() -> None:
    with pytest.raises(ValueError, match="at least 6"):
        build_global_temporal_split(_source_frame(0))


@pytest.mark.parametrize("row_count", [1, 2, 3, 4, 5])
def test_source_smaller_than_six_rows_fails_fast(row_count: int) -> None:
    with pytest.raises(ValueError, match="at least 6"):
        build_global_temporal_split(_source_frame(row_count))


@pytest.mark.parametrize(
    ("row_count", "expected_counts"),
    [
        (6, [4, 1, 1]),
        (7, [5, 1, 1]),
        (8, [6, 1, 1]),
        (9, [7, 1, 1]),
    ],
)
def test_small_valid_source_uses_fixed_integer_boundaries(
    row_count: int,
    expected_counts: list[int],
) -> None:
    splits = build_global_temporal_split(_source_frame(row_count))

    assert allocation_boundaries(row_count) == (
        (row_count * 8) // 10,
        (row_count * 9) // 10,
    )
    assert [
        len(splits.train),
        len(splits.validation),
        len(splits.test),
    ] == expected_counts


def test_missing_required_sort_column_is_rejected() -> None:
    source = _source_frame(10).drop(columns=["movie_id"])

    with pytest.raises(ValueError, match="missing sort columns.*movie_id"):
        build_global_temporal_split(source)


def test_null_sort_key_is_rejected() -> None:
    source = _source_frame(10).astype({"timestamp": "Int64"})
    source.loc[3, "timestamp"] = pd.NA

    with pytest.raises(ValueError, match="null"):
        build_global_temporal_split(source)


def test_non_integer_sort_key_dtype_is_rejected() -> None:
    source = _source_frame(10)
    source["timestamp"] = source["timestamp"].astype("string")

    with pytest.raises(TypeError, match="comparable integer dtypes"):
        build_global_temporal_split(source)


def test_unsorted_input_becomes_canonical_global_order() -> None:
    source = _source_frame(17)
    splits = build_global_temporal_split(source)
    combined = pd.concat(
        [splits.train, splits.validation, splits.test],
        ignore_index=True,
    )

    pd.testing.assert_frame_equal(combined, canonical_sort(source))


def test_timestamp_ties_break_by_user_then_movie() -> None:
    source = pd.DataFrame(
        {
            "user_id": pd.Series(
                [2, 1, 1, 3, 2, 4, 3, 4, 5, 5],
                dtype="int32",
            ),
            "movie_id": pd.Series(
                [8, 9, 4, 3, 2, 6, 1, 5, 7, 0],
                dtype="int32",
            ),
            "rating": pd.Series([4] * 10, dtype="int8"),
            "timestamp": pd.Series([100] * 10, dtype="int64"),
        }
    )
    splits = build_global_temporal_split(source)
    combined = pd.concat(
        [splits.train, splits.validation, splits.test],
        ignore_index=True,
    )

    assert combined[["user_id", "movie_id"]].to_records(index=False).tolist() == [
        (1, 4),
        (1, 9),
        (2, 2),
        (2, 8),
        (3, 1),
        (3, 3),
        (4, 5),
        (4, 6),
        (5, 0),
        (5, 7),
    ]


def test_equal_timestamp_may_cross_fixed_cutoff_deterministically() -> None:
    source = _source_frame(10)
    source["timestamp"] = pd.Series([100] * 10, dtype="int64")
    splits = build_global_temporal_split(source)

    assert splits.train["timestamp"].max() == 100
    assert splits.validation["timestamp"].min() == 100
    assert splits.validation["timestamp"].max() == 100
    assert splits.test["timestamp"].min() == 100
    assert splits.train_end == 8
    assert splits.validation_end == 9


def test_repeated_logic_execution_is_identical() -> None:
    source = _source_frame(23)
    first = build_global_temporal_split(source)
    second = build_global_temporal_split(source.copy())

    for split_name in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(
            getattr(first, split_name),
            getattr(second, split_name),
        )


def test_schema_is_preserved_for_every_split() -> None:
    source = _source_frame(13)
    source["weight"] = pd.Series(range(13), dtype="float32")
    splits = build_global_temporal_split(source)

    assert all(
        dataframe.columns.tolist() == source.columns.tolist()
        for dataframe in (splits.train, splits.validation, splits.test)
    )


def test_dtypes_are_preserved_for_every_split() -> None:
    source = _source_frame(13)
    source["weight"] = pd.Series(range(13), dtype="float32")
    splits = build_global_temporal_split(source)
    expected = source.dtypes.astype(str).to_dict()

    assert all(
        dataframe.dtypes.astype(str).to_dict() == expected
        for dataframe in (splits.train, splits.validation, splits.test)
    )


def test_column_order_is_preserved_for_every_split() -> None:
    source = _source_frame(13)[["rating", "timestamp", "movie_id", "user_id"]]
    splits = build_global_temporal_split(source)

    assert all(
        dataframe.columns.tolist() == source.columns.tolist()
        for dataframe in (splits.train, splits.validation, splits.test)
    )


def test_all_source_rows_are_conserved_exactly() -> None:
    source = _source_frame(29)
    splits = build_global_temporal_split(source)
    combined = pd.concat(
        [splits.train, splits.validation, splits.test],
        ignore_index=True,
    )

    assert len(combined) == len(source)
    pd.testing.assert_frame_equal(combined, canonical_sort(source))


def test_pairwise_interaction_and_full_row_overlap_are_zero() -> None:
    splits = build_global_temporal_split(_source_frame(29))
    counts = calculate_overlap_counts(splits)

    assert counts == {
        "train_validation": {
            "interaction_key": 0,
            "full_row_fingerprint": 0,
        },
        "train_test": {
            "interaction_key": 0,
            "full_row_fingerprint": 0,
        },
        "validation_test": {
            "interaction_key": 0,
            "full_row_fingerprint": 0,
        },
    }


def test_validator_rejects_pairwise_overlap() -> None:
    source = _source_frame(20)
    splits = build_global_temporal_split(source)
    overlapping_validation = splits.validation.copy()
    overlapping_validation.iloc[0] = splits.train.iloc[0]
    invalid = replace(splits, validation=overlapping_validation)

    with pytest.raises(ValueError, match="overlap"):
        validate_temporal_split(source, invalid)


def test_temporal_ordering_is_non_strict_across_boundaries() -> None:
    splits = build_global_temporal_split(_source_frame(31))

    assert splits.train["timestamp"].max() <= splits.validation["timestamp"].min()
    assert splits.validation["timestamp"].max() <= splits.test["timestamp"].min()


def test_validator_rejects_noncanonical_partition_order() -> None:
    source = _source_frame(20)
    splits = build_global_temporal_split(source)
    invalid = replace(
        splits,
        train=splits.train.iloc[::-1].reset_index(drop=True),
    )

    with pytest.raises(ValueError, match="canonical temporal order"):
        validate_temporal_split(source, invalid)


def test_validator_rejects_temporal_boundary_inversion() -> None:
    source = _source_frame(20)
    splits = build_global_temporal_split(source)
    shifted_validation = splits.validation.copy()
    shifted_validation["timestamp"] = pd.Series(
        [0] * len(shifted_validation),
        dtype="int64",
    )
    invalid = replace(splits, validation=shifted_validation)

    with pytest.raises(ValueError, match="Train timestamps"):
        validate_temporal_split(source, invalid)


def test_validator_detects_a_missing_row() -> None:
    source = _source_frame(20)
    splits = build_global_temporal_split(source)
    invalid = replace(
        splits,
        train=splits.train.iloc[:-1].reset_index(drop=True),
    )

    with pytest.raises(ValueError, match="missing source rows"):
        validate_temporal_split(source, invalid)


def test_validator_detects_an_extra_row() -> None:
    source = _source_frame(20)
    splits = build_global_temporal_split(source)
    invalid = replace(
        splits,
        train=pd.concat(
            [splits.train, splits.train.iloc[[0]]],
            ignore_index=True,
        ),
    )

    with pytest.raises(ValueError, match="extra rows"):
        validate_temporal_split(source, invalid)


def test_validator_detects_a_modified_non_key_value() -> None:
    source = _source_frame(20)
    splits = build_global_temporal_split(source)
    modified_train = splits.train.copy()
    modified_train.loc[0, "rating"] = 5 - (int(modified_train.loc[0, "rating"]) - 4)
    invalid = replace(splits, train=modified_train)

    with pytest.raises(ValueError, match="modified"):
        validate_temporal_split(source, invalid)


def test_duplicate_interaction_key_is_rejected() -> None:
    source = _source_frame(10)
    source.loc[9, INTERACTION_KEY] = source.loc[0, INTERACTION_KEY].to_numpy()

    with pytest.raises(ValueError, match="duplicate user-movie"):
        build_global_temporal_split(source)


def test_manifest_update_preserves_all_existing_sections() -> None:
    source = _source_frame(10)
    section = _manifest_section(
        source,
        build_global_temporal_split(source),
    )
    original = {
        "manifest_schema_version": 1,
        "unknown": {"nested": [1, 2, 3]},
        "positive_interactions": {"threshold": 4},
    }
    updated = update_manifest_document(original, section)

    assert original == {
        "manifest_schema_version": 1,
        "unknown": {"nested": [1, 2, 3]},
        "positive_interactions": {"threshold": 4},
    }
    assert {
        key: value for key, value in updated.items() if key != "temporal_split"
    } == original


def test_manifest_update_is_idempotent() -> None:
    source = _source_frame(10)
    section = _manifest_section(
        source,
        build_global_temporal_split(source),
    )
    once = update_manifest_document({"existing": True}, section)
    twice = update_manifest_document(once, section)

    assert once == twice
    assert (
        json.dumps(once, indent=2, ensure_ascii=False) + "\n"
        == json.dumps(twice, indent=2, ensure_ascii=False) + "\n"
    )


def test_manifest_output_paths_are_repository_relative() -> None:
    source = _source_frame(10)
    section = _manifest_section(
        source,
        build_global_temporal_split(source),
    )

    assert not Path(section["source_path"]).is_absolute()
    assert all(
        not Path(path).is_absolute() for path in section["output_paths"].values()
    )


def test_manifest_has_no_variable_execution_metadata() -> None:
    source = _source_frame(10)
    section = _manifest_section(
        source,
        build_global_temporal_split(source),
    )
    serialized = json.dumps(section)

    for forbidden_field in (
        "created_at_utc",
        "updated_at_utc",
        "executed_at",
        "branch_name",
        "hostname",
        "temporary_path",
    ):
        assert forbidden_field not in serialized


def test_write_read_round_trip_preserves_content_and_manifest(
    tmp_path: Path,
) -> None:
    source = _source_frame(23)
    paths, original_manifest = _test_repository(tmp_path, source)
    section = build_and_publish_temporal_split(paths)
    stored = TemporalSplits(
        train=pd.read_parquet(paths.train),
        validation=pd.read_parquet(paths.validation),
        test=pd.read_parquet(paths.test),
        train_end=section["cutoff_indices"]["train_end_exclusive"],
        validation_end=section["cutoff_indices"]["validation_end_exclusive"],
    )
    validate_temporal_split(source, stored)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

    assert manifest["unknown_section"] == original_manifest["unknown_section"]
    assert manifest["temporal_split"] == section
    assert section["output_sha256"] == {
        "train": _sha256(paths.train),
        "validation": _sha256(paths.validation),
        "test": _sha256(paths.test),
    }
    assert paths.manifest.read_bytes().endswith(b"\n")


def test_repeated_file_build_is_byte_for_byte_identical(
    tmp_path: Path,
) -> None:
    paths, _ = _test_repository(tmp_path, _source_frame(27))
    build_and_publish_temporal_split(paths)
    first_hashes = {
        path.name: _sha256(path)
        for path in (
            paths.train,
            paths.validation,
            paths.test,
            paths.manifest,
        )
    }
    build_and_publish_temporal_split(paths)
    second_hashes = {
        path.name: _sha256(path)
        for path in (
            paths.train,
            paths.validation,
            paths.test,
            paths.manifest,
        )
    }

    assert first_hashes == second_hashes


def test_failure_before_publish_preserves_existing_outputs_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _ = _test_repository(tmp_path, _source_frame(20))
    sentinel = {
        paths.train: b"existing-train",
        paths.validation: b"existing-validation",
        paths.test: b"existing-test",
    }

    for path, content in sentinel.items():
        path.write_bytes(content)

    manifest_before = paths.manifest.read_bytes()
    original_validator = split_script.validate_temporal_split
    call_count = 0

    def fail_on_staged_readback(
        source: pd.DataFrame,
        splits: TemporalSplits,
    ) -> None:
        nonlocal call_count
        call_count += 1

        if call_count == 2:
            raise ValueError("injected read-back validation failure")

        original_validator(source, splits)

    monkeypatch.setattr(
        split_script,
        "validate_temporal_split",
        fail_on_staged_readback,
    )

    with pytest.raises(ValueError, match="injected read-back"):
        build_and_publish_temporal_split(paths)

    assert {path: path.read_bytes() for path in sentinel} == sentinel
    assert paths.manifest.read_bytes() == manifest_before
    assert not list(paths.train.parent.glob(".*.tmp"))


def test_source_hash_mismatch_with_manifest_fails_before_publish(
    tmp_path: Path,
) -> None:
    paths, _ = _test_repository(tmp_path, _source_frame(20))
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["positive_interactions"]["output_sha256"] = "0" * 64
    paths.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match manifest"):
        build_and_publish_temporal_split(paths)

    assert not paths.train.exists()
    assert not paths.validation.exists()
    assert not paths.test.exists()
