import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import scripts.build_evaluation_cohorts as cohort_script
from scripts.build_evaluation_cohorts import (
    EvaluationCohortPaths,
    build_and_publish_evaluation_cohorts,
)
from src.data.evaluation_cohorts import (
    COHORT_COLUMNS,
    EXCLUSION_REASONS,
    ITEM_MAPPING_COLUMNS,
    ITEM_POPULARITY_COLUMNS,
    MINIMUM_FIT_POSITIVES,
    SOURCE_COLUMNS,
    USER_MAPPING_COLUMNS,
    EvaluationArtifacts,
    build_cohorts_manifest_section,
    build_contiguous_mapping,
    build_evaluation_artifacts,
    build_evaluation_context,
    build_item_popularity_train,
    update_manifest_document,
    validate_evaluation_artifacts,
    validate_mapping_table,
    validate_source_frame,
)


def _frame(rows: list[tuple[int, int, int, int]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        rows,
        columns=SOURCE_COLUMNS,
    ).astype(
        {
            "user_id": "int32",
            "movie_id": "int32",
            "rating": "int8",
            "timestamp": "int64",
        }
    )
    return dataframe.sort_values(
        ["timestamp", "user_id", "movie_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _policy_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[tuple[int, int, int, int]] = []
    timestamp = 1

    for user_id, movie_start, count in (
        (1, 100, 5),
        (2, 200, 4),
        (3, 300, 5),
        (6, 600, 1),
    ):
        for offset in range(count):
            rows.append((user_id, movie_start + offset, 4, timestamp))
            timestamp += 1

    target = _frame(
        [
            (1, 200, 4, 100),
            (1, 900, 5, 101),
            (2, 100, 4, 102),
            (3, 901, 4, 103),
            (4, 101, 5, 104),
            (5, 902, 4, 105),
        ]
    )
    return _frame(rows), target


def _project_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = _frame(
        [
            (1, 10, 4, 1),
            (1, 11, 5, 2),
            (1, 12, 4, 3),
            (1, 13, 5, 4),
            (1, 14, 4, 5),
            (2, 10, 4, 6),
            (2, 20, 4, 7),
            (2, 21, 5, 8),
            (2, 22, 4, 9),
            (3, 30, 4, 10),
            (3, 31, 5, 11),
        ]
    )
    validation = _frame(
        [
            (1, 20, 4, 20),
            (1, 900, 5, 21),
            (2, 11, 4, 22),
            (4, 10, 4, 23),
            (5, 901, 5, 24),
        ]
    )
    test = _frame(
        [
            (1, 21, 4, 30),
            (2, 12, 5, 31),
            (4, 13, 4, 32),
            (6, 902, 4, 33),
            (7, 14, 5, 34),
        ]
    )
    return train, validation, test


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test_repository(
    tmp_path: Path,
) -> tuple[EvaluationCohortPaths, dict[str, Any]]:
    train, validation, test = _project_splits()
    root = tmp_path / "repository"
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    source_frames = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    source_paths = {
        split_name: processed / f"{split_name}.parquet" for split_name in source_frames
    }

    for split_name, dataframe in source_frames.items():
        dataframe.to_parquet(
            source_paths[split_name],
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

    source_hashes = {
        split_name: _sha256(path) for split_name, path in source_paths.items()
    }
    manifest: dict[str, Any] = {
        "manifest_schema_version": 1,
        "unknown_section": {"nested": ["preserve", 17]},
        "positive_interactions": {"threshold": 4},
        "temporal_split": {
            "output_paths": {
                split_name: f"data/processed/{split_name}.parquet"
                for split_name in source_frames
            },
            "output_sha256": source_hashes,
            "split_row_counts": {
                split_name: len(dataframe)
                for split_name, dataframe in source_frames.items()
            },
            "columns": train.columns.tolist(),
            "dtypes": train.dtypes.astype(str).to_dict(),
        },
    }
    manifest_path = processed / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    paths = EvaluationCohortPaths(
        repository_root=root,
        train=source_paths["train"],
        validation=source_paths["validation"],
        test=source_paths["test"],
        evaluation_cohorts=processed / "evaluation_cohorts.parquet",
        user_mapping=processed / "user_mapping.parquet",
        item_mapping=processed / "item_mapping.parquet",
        item_popularity_train=(processed / "item_popularity_train.parquet"),
        manifest=manifest_path,
    )
    return paths, manifest


def _manifest_section(
    artifacts: EvaluationArtifacts,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    return build_cohorts_manifest_section(
        artifacts,
        train,
        validation,
        test,
        source_paths={
            "train": "data/processed/train.parquet",
            "validation": "data/processed/validation.parquet",
            "test": "data/processed/test.parquet",
        },
        source_sha256={
            "train": "a" * 64,
            "validation": "b" * 64,
            "test": "c" * 64,
        },
        output_paths={
            "evaluation_cohorts": ("data/processed/evaluation_cohorts.parquet"),
            "user_mapping": "data/processed/user_mapping.parquet",
            "item_mapping": "data/processed/item_mapping.parquet",
            "item_popularity_train": ("data/processed/item_popularity_train.parquet"),
        },
        output_sha256={
            "evaluation_cohorts": "d" * 64,
            "user_mapping": "e" * 64,
            "item_mapping": "f" * 64,
            "item_popularity_train": "0" * 64,
        },
        parquet_engine="pyarrow",
        compression="snappy",
    )


def _row_for(cohort: pd.DataFrame, user_id: int) -> pd.Series:
    return cohort.loc[cohort["user_id"].eq(user_id)].iloc[0]


def test_project_threshold_is_fixed_at_five() -> None:
    assert MINIMUM_FIT_POSITIVES == 5


@pytest.mark.parametrize(
    ("fit_count", "eligible", "activity_group"),
    [
        (4, False, None),
        (5, True, "low"),
        (9, True, "low"),
        (10, True, "medium"),
        (29, True, "medium"),
        (30, True, "high"),
    ],
)
def test_fit_threshold_and_activity_boundaries(
    fit_count: int,
    eligible: bool,
    activity_group: str | None,
) -> None:
    fit = _frame(
        [(1, 100 + index, 4, index + 1) for index in range(fit_count)]
        + [(2, 999, 4, fit_count + 1)]
    )
    target = _frame([(1, 999, 4, fit_count + 10)])
    cohort = build_evaluation_context(
        fit,
        target,
        evaluation_split="validation",
        fit_scope="train",
    )
    row = cohort.iloc[0]

    assert bool(row["is_eligible"]) is eligible
    assert (
        None if pd.isna(row["activity_group"]) else row["activity_group"]
    ) == activity_group


def test_cohort_policy_reasons_are_exact_and_mutually_exclusive() -> None:
    fit, target = _policy_frames()
    cohort = build_evaluation_context(
        fit,
        target,
        evaluation_split="validation",
        fit_scope="train",
    )

    assert _row_for(cohort, 1)["exclusion_reason"] is pd.NA
    assert _row_for(cohort, 2)["exclusion_reason"] == "insufficient_fit_positives"
    assert _row_for(cohort, 3)["exclusion_reason"] == "no_warm_target_items"
    assert _row_for(cohort, 4)["exclusion_reason"] == "insufficient_fit_positives"
    assert (
        _row_for(cohort, 5)["exclusion_reason"]
        == "insufficient_fit_positives_and_no_warm_target_items"
    )
    assert set(cohort["exclusion_reason"].dropna()) == set(EXCLUSION_REASONS)
    assert cohort.loc[cohort["is_eligible"], "exclusion_reason"].isna().all()


def test_warm_and_cold_target_counts_and_eligibility_are_exact() -> None:
    fit, target = _policy_frames()
    cohort = build_evaluation_context(
        fit,
        target,
        evaluation_split="validation",
        fit_scope="train",
    )
    user_one = _row_for(cohort, 1)
    user_three = _row_for(cohort, 3)

    assert user_one["target_positive_count"] == 2
    assert user_one["warm_target_positive_count"] == 1
    assert user_one["cold_start_target_positive_count"] == 1
    assert bool(user_one["is_eligible"])
    assert user_three["warm_target_positive_count"] == 0
    assert user_three["cold_start_target_positive_count"] == 1
    assert not bool(user_three["is_eligible"])
    assert cohort["target_positive_count"].equals(
        cohort["warm_target_positive_count"]
        + cohort["cold_start_target_positive_count"]
    )


def test_cold_start_user_flag_is_independent_from_exclusion_reason() -> None:
    fit, target = _policy_frames()
    cohort = build_evaluation_context(
        fit,
        target,
        evaluation_split="validation",
        fit_scope="train",
    )

    assert bool(_row_for(cohort, 4)["is_cold_start_user"])
    assert _row_for(cohort, 4)["exclusion_reason"] == "insufficient_fit_positives"
    assert bool(_row_for(cohort, 5)["is_cold_start_user"])
    assert not bool(_row_for(cohort, 1)["is_cold_start_user"])


def test_cohort_has_one_sorted_row_per_target_user_only() -> None:
    fit, target = _policy_frames()
    cohort = build_evaluation_context(
        fit,
        target,
        evaluation_split="validation",
        fit_scope="train",
    )

    assert cohort["user_id"].tolist() == [1, 2, 3, 4, 5]
    assert not cohort["user_id"].eq(6).any()
    assert len(cohort) == target["user_id"].nunique()
    assert not cohort["user_id"].duplicated().any()


def test_context_build_is_deterministic() -> None:
    fit, target = _policy_frames()
    first = build_evaluation_context(
        fit,
        target,
        evaluation_split="validation",
        fit_scope="train",
    )
    second = build_evaluation_context(
        fit.copy(),
        target.copy(),
        evaluation_split="validation",
        fit_scope="train",
    )

    pd.testing.assert_frame_equal(first, second)


def test_artifact_schemas_and_dtypes_are_exact() -> None:
    train, validation, test = _project_splits()
    artifacts = build_evaluation_artifacts(train, validation, test)

    assert artifacts.evaluation_cohorts.columns.tolist() == COHORT_COLUMNS
    assert artifacts.user_mapping.columns.tolist() == USER_MAPPING_COLUMNS
    assert artifacts.item_mapping.columns.tolist() == ITEM_MAPPING_COLUMNS
    assert artifacts.item_popularity_train.columns.tolist() == ITEM_POPULARITY_COLUMNS
    assert artifacts.evaluation_cohorts.dtypes.astype(str).to_dict() == {
        "evaluation_split": "string",
        "fit_scope": "string",
        "user_id": "int32",
        "fit_positive_count": "int64",
        "target_positive_count": "int64",
        "warm_target_positive_count": "int64",
        "cold_start_target_positive_count": "int64",
        "is_cold_start_user": "bool",
        "is_eligible": "bool",
        "activity_group": "string",
        "exclusion_reason": "string",
    }


def test_validation_context_uses_train_only() -> None:
    train, validation, test = _project_splits()
    artifacts = build_evaluation_artifacts(train, validation, test)
    cohort = artifacts.evaluation_cohorts.loc[
        artifacts.evaluation_cohorts["evaluation_split"].eq("validation")
    ]

    assert _row_for(cohort, 1)["fit_positive_count"] == 5
    assert _row_for(cohort, 4)["fit_positive_count"] == 0
    assert set(
        artifacts.user_mapping.loc[
            artifacts.user_mapping["fit_scope"].eq("train"),
            "user_id",
        ]
    ) == set(train["user_id"])
    assert 4 not in set(
        artifacts.user_mapping.loc[
            artifacts.user_mapping["fit_scope"].eq("train"),
            "user_id",
        ]
    )
    assert 900 not in set(
        artifacts.item_mapping.loc[
            artifacts.item_mapping["fit_scope"].eq("train"),
            "movie_id",
        ]
    )


def test_test_context_uses_train_and_validation_only() -> None:
    train, validation, test = _project_splits()
    artifacts = build_evaluation_artifacts(train, validation, test)
    cohort = artifacts.evaluation_cohorts.loc[
        artifacts.evaluation_cohorts["evaluation_split"].eq("test")
    ]

    assert _row_for(cohort, 1)["fit_positive_count"] == 7
    assert _row_for(cohort, 4)["fit_positive_count"] == 1
    train_validation_users = set(train["user_id"]).union(validation["user_id"])
    mapped_users = set(
        artifacts.user_mapping.loc[
            artifacts.user_mapping["fit_scope"].eq("train_validation"),
            "user_id",
        ]
    )
    mapped_items = set(
        artifacts.item_mapping.loc[
            artifacts.item_mapping["fit_scope"].eq("train_validation"),
            "movie_id",
        ]
    )

    assert mapped_users == train_validation_users
    assert 6 not in mapped_users
    assert 7 not in mapped_users
    assert 902 not in mapped_items


def test_test_only_rows_do_not_change_fit_mappings_or_fit_counts() -> None:
    train, validation, test = _project_splits()
    changed_test = test.copy()
    changed_test["user_id"] = pd.Series(
        [101, 102, 103, 104, 105],
        dtype="int32",
    )
    changed_test["movie_id"] = pd.Series(
        [801, 802, 803, 804, 805],
        dtype="int32",
    )
    first = build_evaluation_artifacts(train, validation, test)
    second = build_evaluation_artifacts(train, validation, changed_test)

    pd.testing.assert_frame_equal(first.user_mapping, second.user_mapping)
    pd.testing.assert_frame_equal(first.item_mapping, second.item_mapping)
    pd.testing.assert_frame_equal(
        first.item_popularity_train,
        second.item_popularity_train,
    )
    assert set(
        first.evaluation_cohorts.loc[
            first.evaluation_cohorts["evaluation_split"].eq("test"),
            "fit_positive_count",
        ]
    ).issubset({0, 1, 5, 7})


def test_replacing_test_with_only_unseen_ids_yields_zero_fit_counts() -> None:
    train, validation, test = _project_splits()
    unseen_test = test.copy()
    unseen_test["user_id"] = pd.Series(
        [101, 102, 103, 104, 105],
        dtype="int32",
    )
    unseen_test["movie_id"] = pd.Series(
        [801, 802, 803, 804, 805],
        dtype="int32",
    )
    artifacts = build_evaluation_artifacts(train, validation, unseen_test)
    test_cohort = artifacts.evaluation_cohorts.loc[
        artifacts.evaluation_cohorts["evaluation_split"].eq("test")
    ]

    assert test_cohort["fit_positive_count"].eq(0).all()
    assert test_cohort["warm_target_positive_count"].eq(0).all()
    assert test_cohort["is_cold_start_user"].all()


@pytest.mark.parametrize(
    ("id_column", "index_column"),
    [
        ("user_id", "user_index"),
        ("movie_id", "item_index"),
    ],
)
def test_mapping_is_zero_based_contiguous_sorted_and_deterministic(
    id_column: str,
    index_column: str,
) -> None:
    train, _, _ = _project_splits()
    first = build_contiguous_mapping(
        train,
        fit_scope="train",
        id_column=id_column,
        index_column=index_column,
    )
    second = build_contiguous_mapping(
        train.copy(),
        fit_scope="train",
        id_column=id_column,
        index_column=index_column,
    )

    assert first[index_column].tolist() == list(range(len(first)))
    assert first[id_column].tolist() == sorted(train[id_column].unique())
    pd.testing.assert_frame_equal(first, second)


def test_mapping_contains_all_fit_ids_not_only_eligible_users() -> None:
    train, validation, test = _project_splits()
    artifacts = build_evaluation_artifacts(train, validation, test)
    train_mapping = artifacts.user_mapping.loc[
        artifacts.user_mapping["fit_scope"].eq("train")
    ]

    assert set(train_mapping["user_id"]) == set(train["user_id"])
    assert 2 in set(train_mapping["user_id"])
    assert not _row_for(
        artifacts.evaluation_cohorts.loc[
            artifacts.evaluation_cohorts["evaluation_split"].eq("validation")
        ],
        2,
    )["is_eligible"]


def test_duplicate_mapped_id_is_rejected() -> None:
    train, _, _ = _project_splits()
    mapping = build_contiguous_mapping(
        train,
        fit_scope="train",
        id_column="user_id",
        index_column="user_index",
    )
    mapping.loc[1, "user_id"] = mapping.loc[0, "user_id"]

    with pytest.raises(ValueError, match="duplicate user_id"):
        validate_mapping_table(
            mapping,
            fit_scope="train",
            id_column="user_id",
            index_column="user_index",
        )


def test_duplicate_mapped_index_is_rejected() -> None:
    train, _, _ = _project_splits()
    mapping = build_contiguous_mapping(
        train,
        fit_scope="train",
        id_column="movie_id",
        index_column="item_index",
    )
    mapping.loc[1, "item_index"] = mapping.loc[0, "item_index"]

    with pytest.raises(ValueError, match="duplicate item_index"):
        validate_mapping_table(
            mapping,
            fit_scope="train",
            id_column="movie_id",
            index_column="item_index",
        )


def test_mapping_gap_is_rejected() -> None:
    train, _, _ = _project_splits()
    mapping = build_contiguous_mapping(
        train,
        fit_scope="train",
        id_column="movie_id",
        index_column="item_index",
    )
    mapping.loc[1, "item_index"] = 999

    with pytest.raises(ValueError, match="not zero-based and contiguous"):
        validate_mapping_table(
            mapping,
            fit_scope="train",
            id_column="movie_id",
            index_column="item_index",
        )


def test_popularity_uses_train_only_and_has_deterministic_ties() -> None:
    train, validation, test = _project_splits()
    popularity = build_item_popularity_train(train)
    counts = dict(
        popularity[["movie_id", "positive_count"]].itertuples(
            index=False,
            name=None,
        )
    )

    assert counts[10] == 2
    assert counts[11] == 1
    assert 900 not in counts
    assert 901 not in counts
    assert 902 not in counts
    tied = popularity.loc[popularity["positive_count"].eq(1), "movie_id"]
    assert tied.tolist() == sorted(tied.tolist())
    assert int(popularity["positive_count"].sum()) == len(train)
    assert len(popularity) == train["movie_id"].nunique()
    assert len(validation) + len(test) > 0


def test_popularity_item_count_matches_train_item_mapping() -> None:
    train, validation, test = _project_splits()
    artifacts = build_evaluation_artifacts(train, validation, test)
    train_item_mapping = artifacts.item_mapping.loc[
        artifacts.item_mapping["fit_scope"].eq("train")
    ]

    assert len(artifacts.item_popularity_train) == len(train_item_mapping)


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        ("missing_column", ValueError, "unexpected columns"),
        ("null_id", ValueError, "null values"),
        ("float_id", TypeError, "integer dtypes"),
        ("duplicate", ValueError, "duplicate interactions"),
        ("invalid_rating", ValueError, "positive rating"),
        ("nonpositive_user", ValueError, "non-positive user_id"),
        ("noncanonical", ValueError, "canonical temporal order"),
    ],
)
def test_invalid_source_is_rejected(
    mutation: str,
    error_type: type[Exception],
    message: str,
) -> None:
    train, _, _ = _project_splits()
    invalid = train.copy()

    if mutation == "missing_column":
        invalid = invalid.drop(columns=["rating"])
    elif mutation == "null_id":
        invalid = invalid.astype({"user_id": "Int32"})
        invalid.loc[0, "user_id"] = pd.NA
    elif mutation == "float_id":
        invalid["movie_id"] = invalid["movie_id"].astype("float64")
    elif mutation == "duplicate":
        invalid.loc[1, ["user_id", "movie_id"]] = invalid.loc[
            0, ["user_id", "movie_id"]
        ].to_numpy()
    elif mutation == "invalid_rating":
        invalid.loc[0, "rating"] = 3
    elif mutation == "nonpositive_user":
        invalid.loc[0, "user_id"] = 0
    elif mutation == "noncanonical":
        invalid = invalid.iloc[::-1].reset_index(drop=True)

    with pytest.raises(error_type, match=message):
        validate_source_frame(invalid, split_name="invalid")


@pytest.mark.parametrize("empty_side", ["fit", "target"])
def test_empty_fit_or_target_fails_fast(empty_side: str) -> None:
    fit, target = _policy_frames()
    empty = fit.iloc[0:0].copy()

    with pytest.raises(ValueError, match="must not be empty"):
        build_evaluation_context(
            empty if empty_side == "fit" else fit,
            empty if empty_side == "target" else target,
            evaluation_split="validation",
            fit_scope="train",
        )


def test_incompatible_split_schema_is_rejected() -> None:
    train, validation, test = _project_splits()
    validation = validation.astype({"user_id": "int64"})

    with pytest.raises(TypeError, match="incompatible with train"):
        build_evaluation_artifacts(train, validation, test)


def test_cross_split_duplicate_interaction_is_rejected() -> None:
    train, validation, test = _project_splits()
    validation.loc[0, ["user_id", "movie_id"]] = train.loc[
        0, ["user_id", "movie_id"]
    ].to_numpy()

    with pytest.raises(ValueError, match="overlap"):
        build_evaluation_artifacts(train, validation, test)


def test_temporal_boundary_inversion_is_rejected() -> None:
    train, validation, test = _project_splits()
    validation["timestamp"] = pd.Series(
        range(100, 100 + len(validation)),
        dtype="int64",
    )
    test["timestamp"] = pd.Series(
        range(50, 50 + len(test)),
        dtype="int64",
    )

    with pytest.raises(ValueError, match="Validation timestamps"):
        build_evaluation_artifacts(train, validation, test)


@pytest.mark.parametrize("mutation", ["missing", "extra", "modified"])
def test_cohort_output_row_mutation_is_detected(mutation: str) -> None:
    train, validation, test = _project_splits()
    artifacts = build_evaluation_artifacts(train, validation, test)
    cohort = artifacts.evaluation_cohorts.copy()

    if mutation == "missing":
        cohort = cohort.iloc[:-1].reset_index(drop=True)
    elif mutation == "extra":
        cohort = pd.concat([cohort, cohort.iloc[[0]]], ignore_index=True)
    else:
        cohort.loc[0, "fit_positive_count"] += 1

    with pytest.raises(ValueError):
        validate_evaluation_artifacts(
            replace(artifacts, evaluation_cohorts=cohort),
            train,
            validation,
            test,
        )


@pytest.mark.parametrize(
    "field_name",
    ["user_mapping", "item_mapping", "item_popularity_train"],
)
def test_non_cohort_output_mutation_is_detected(field_name: str) -> None:
    train, validation, test = _project_splits()
    artifacts = build_evaluation_artifacts(train, validation, test)
    modified = getattr(artifacts, field_name).copy()

    if field_name == "user_mapping":
        modified.loc[0, "user_index"] = 999
    elif field_name == "item_mapping":
        modified.loc[0, "item_index"] = 999
    else:
        modified.loc[0, "positive_count"] += 1

    with pytest.raises(ValueError):
        validate_evaluation_artifacts(
            replace(artifacts, **{field_name: modified}),
            train,
            validation,
            test,
        )


def test_combined_cohort_order_is_validation_then_test_then_user() -> None:
    train, validation, test = _project_splits()
    cohort = build_evaluation_artifacts(
        train,
        validation,
        test,
    ).evaluation_cohorts

    assert cohort["evaluation_split"].tolist() == (
        ["validation"] * validation["user_id"].nunique()
        + ["test"] * test["user_id"].nunique()
    )
    for split_name in ("validation", "test"):
        users = cohort.loc[
            cohort["evaluation_split"].eq(split_name),
            "user_id",
        ].tolist()
        assert users == sorted(users)


def test_manifest_update_preserves_unknown_sections_and_input() -> None:
    train, validation, test = _project_splits()
    artifacts = build_evaluation_artifacts(train, validation, test)
    section = _manifest_section(artifacts, train, validation, test)
    original = {
        "manifest_schema_version": 1,
        "unknown": {"nested": [1, 2, 3]},
        "temporal_split": {"keep": True},
    }
    updated = update_manifest_document(original, section)

    assert original == {
        "manifest_schema_version": 1,
        "unknown": {"nested": [1, 2, 3]},
        "temporal_split": {"keep": True},
    }
    assert {
        key: value for key, value in updated.items() if key != "cohorts_and_mappings"
    } == original
    assert updated["cohorts_and_mappings"] == section


def test_manifest_update_is_idempotent() -> None:
    train, validation, test = _project_splits()
    artifacts = build_evaluation_artifacts(train, validation, test)
    section = _manifest_section(artifacts, train, validation, test)
    once = update_manifest_document({"existing": True}, section)
    twice = update_manifest_document(once, section)

    assert once == twice
    assert (
        json.dumps(once, indent=2, ensure_ascii=False) + "\n"
        == json.dumps(twice, indent=2, ensure_ascii=False) + "\n"
    )


def test_manifest_paths_are_relative_and_has_no_runtime_metadata() -> None:
    train, validation, test = _project_splits()
    artifacts = build_evaluation_artifacts(train, validation, test)
    section = _manifest_section(artifacts, train, validation, test)
    serialized = json.dumps(section)

    assert all(
        not Path(path).is_absolute() for path in section["source_paths"].values()
    )
    assert all(
        not Path(path).is_absolute() for path in section["output_paths"].values()
    )

    for forbidden in (
        "timestamp",
        "created_at",
        "updated_at",
        "hostname",
        "temporary_path",
        "uuid",
    ):
        assert forbidden not in serialized


def test_manifest_context_counts_satisfy_all_invariants() -> None:
    train, validation, test = _project_splits()
    artifacts = build_evaluation_artifacts(train, validation, test)
    section = _manifest_section(artifacts, train, validation, test)

    for context in section["contexts"].values():
        assert (
            context["eligible_user_count"] + context["excluded_user_count"]
            == context["target_user_count"]
        )
        assert (
            sum(context["exclusion_reason_counts"].values())
            == context["excluded_user_count"]
        )
        assert (
            sum(context["eligible_activity_group_counts"].values())
            == context["eligible_user_count"]
        )
        assert (
            context["warm_target_interaction_count"]
            + context["cold_start_target_interaction_count"]
            == context["target_interaction_count"]
        )


def test_manifest_records_fixed_policy_and_train_popularity() -> None:
    train, validation, test = _project_splits()
    artifacts = build_evaluation_artifacts(train, validation, test)
    section = _manifest_section(artifacts, train, validation, test)

    assert section["policy"]["minimum_fit_positives"] == 5
    assert section["policy"]["activity_groups"] == {
        "low": {"minimum": 5, "maximum": 9},
        "medium": {"minimum": 10, "maximum": 29},
        "high": {"minimum": 30, "maximum": None},
    }
    assert section["item_popularity_train"]["positive_count_sum"] == len(train)
    assert section["mapping_counts"]["train"]["items"] == train["movie_id"].nunique()


def test_write_read_round_trip_validates_outputs_hashes_and_manifest(
    tmp_path: Path,
) -> None:
    paths, original_manifest = _test_repository(tmp_path)
    section = build_and_publish_evaluation_cohorts(paths)
    train = pd.read_parquet(paths.train)
    validation = pd.read_parquet(paths.validation)
    test = pd.read_parquet(paths.test)
    stored = EvaluationArtifacts(
        evaluation_cohorts=pd.read_parquet(paths.evaluation_cohorts),
        user_mapping=pd.read_parquet(paths.user_mapping),
        item_mapping=pd.read_parquet(paths.item_mapping),
        item_popularity_train=pd.read_parquet(paths.item_popularity_train),
    )
    validate_evaluation_artifacts(stored, train, validation, test)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

    assert manifest["unknown_section"] == original_manifest["unknown_section"]
    assert manifest["cohorts_and_mappings"] == section
    assert section["output_sha256"] == {
        "evaluation_cohorts": _sha256(paths.evaluation_cohorts),
        "user_mapping": _sha256(paths.user_mapping),
        "item_mapping": _sha256(paths.item_mapping),
        "item_popularity_train": _sha256(paths.item_popularity_train),
    }
    assert paths.manifest.read_bytes().endswith(b"\n")


def test_repeated_file_build_is_byte_for_byte_identical(
    tmp_path: Path,
) -> None:
    paths, _ = _test_repository(tmp_path)
    build_and_publish_evaluation_cohorts(paths)
    output_paths = (
        paths.evaluation_cohorts,
        paths.user_mapping,
        paths.item_mapping,
        paths.item_popularity_train,
        paths.manifest,
    )
    first_hashes = {path.name: _sha256(path) for path in output_paths}
    build_and_publish_evaluation_cohorts(paths)
    second_hashes = {path.name: _sha256(path) for path in output_paths}

    assert first_hashes == second_hashes


def test_failure_before_publish_preserves_outputs_and_cleans_temps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _ = _test_repository(tmp_path)
    sentinel = {
        paths.evaluation_cohorts: b"existing-cohort",
        paths.user_mapping: b"existing-user-mapping",
        paths.item_mapping: b"existing-item-mapping",
        paths.item_popularity_train: b"existing-popularity",
    }

    for path, content in sentinel.items():
        path.write_bytes(content)

    manifest_before = paths.manifest.read_bytes()

    def fail_staged_validation(
        artifacts: EvaluationArtifacts,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        del artifacts, train, validation, test
        raise ValueError("injected staged validation failure")

    monkeypatch.setattr(
        cohort_script,
        "validate_evaluation_artifacts",
        fail_staged_validation,
    )

    with pytest.raises(ValueError, match="injected staged"):
        build_and_publish_evaluation_cohorts(paths)

    assert {path: path.read_bytes() for path in sentinel} == sentinel
    assert paths.manifest.read_bytes() == manifest_before
    assert not list(paths.manifest.parent.glob(".*.tmp"))


def test_source_hash_mismatch_fails_before_publish(tmp_path: Path) -> None:
    paths, _ = _test_repository(tmp_path)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["temporal_split"]["output_sha256"]["train"] = "0" * 64
    paths.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match"):
        build_and_publish_evaluation_cohorts(paths)

    assert not paths.evaluation_cohorts.exists()
    assert not paths.user_mapping.exists()
    assert not paths.item_mapping.exists()
    assert not paths.item_popularity_train.exists()


def test_missing_source_fails_before_publish(tmp_path: Path) -> None:
    paths, _ = _test_repository(tmp_path)
    paths.test.unlink()

    with pytest.raises(FileNotFoundError, match="Missing split sources"):
        build_and_publish_evaluation_cohorts(paths)

    assert not paths.evaluation_cohorts.exists()


def test_cli_integration_in_temporary_repository(tmp_path: Path) -> None:
    paths, _ = _test_repository(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_evaluation_cohorts",
            "--repository-root",
            str(paths.repository_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "DATA-006 artifacts generated" in completed.stdout
    assert paths.evaluation_cohorts.exists()
    assert paths.user_mapping.exists()
    assert paths.item_mapping.exists()
    assert paths.item_popularity_train.exists()
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert "cohorts_and_mappings" in manifest
