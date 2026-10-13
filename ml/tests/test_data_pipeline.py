import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype

from src.data.evaluation_cohorts import (
    ACTIVITY_GROUPS,
    COHORT_COLUMNS,
    EXCLUSION_REASONS,
    ITEM_MAPPING_COLUMNS,
    ITEM_POPULARITY_COLUMNS,
    USER_MAPPING_COLUMNS,
    EvaluationArtifacts,
    build_cohorts_manifest_section,
    build_evaluation_artifacts,
    validate_evaluation_artifacts,
    validate_item_popularity,
)
from src.data.movielens_parser import (
    MOVIES_COLUMNS,
    MOVIES_DTYPES,
    RATINGS_COLUMNS,
    RATINGS_DTYPES,
    USERS_COLUMNS,
    USERS_DTYPES,
    validate_referential_integrity,
)
from src.data.positive_interactions import (
    POSITIVE_COLUMNS,
    POSITIVE_DTYPES,
    build_positive_interactions,
)
from src.data.temporal_split import (
    TemporalSplits,
    build_global_temporal_split,
    build_temporal_split_manifest_section,
    canonical_sort,
    validate_temporal_split,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIRECTORY = REPOSITORY_ROOT / "data" / "processed"
EARLIEST_ACCEPTABLE_TIMESTAMP = int(pd.Timestamp("1995-01-01", tz="UTC").timestamp())
LATEST_ACCEPTABLE_TIMESTAMP = int(
    pd.Timestamp("2005-12-31T23:59:59", tz="UTC").timestamp()
)

COHORT_DTYPES = {
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
USER_MAPPING_DTYPES = {
    "fit_scope": "string",
    "user_id": "int32",
    "user_index": "int64",
}
ITEM_MAPPING_DTYPES = {
    "fit_scope": "string",
    "movie_id": "int32",
    "item_index": "int64",
}
ITEM_POPULARITY_DTYPES = {
    "movie_id": "int32",
    "positive_count": "int64",
}


@dataclass(frozen=True)
class RealPipeline:
    frames: dict[str, pd.DataFrame]
    manifest: dict[str, Any]


@pytest.fixture(scope="module")
def real_pipeline() -> RealPipeline:
    filenames = (
        "ratings",
        "movies",
        "users",
        "positives",
        "train",
        "validation",
        "test",
        "evaluation_cohorts",
        "user_mapping",
        "item_mapping",
        "item_popularity_train",
    )
    frames = {
        name: pd.read_parquet(PROCESSED_DIRECTORY / f"{name}.parquet")
        for name in filenames
    }
    manifest = json.loads(
        (PROCESSED_DIRECTORY / "data_manifest.json").read_text(encoding="utf-8")
    )
    return RealPipeline(frames=frames, manifest=manifest)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_schema(
    dataframe: pd.DataFrame,
    columns: list[str],
    dtypes: dict[str, str],
) -> None:
    assert dataframe.columns.tolist() == columns
    assert dataframe.dtypes.astype(str).to_dict() == dtypes


def _declared_output_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    positives = manifest["positive_interactions"]
    temporal = manifest["temporal_split"]
    cohorts = manifest["cohorts_and_mappings"]
    declared = {
        positives["output_path"]: positives["output_sha256"],
    }
    declared.update(
        {
            temporal["output_paths"][name]: expected_hash
            for name, expected_hash in temporal["output_sha256"].items()
        }
    )
    declared.update(
        {
            cohorts["output_paths"][name]: expected_hash
            for name, expected_hash in cohorts["output_sha256"].items()
        }
    )
    return declared


def _validate_manifest_hash_chain(
    manifest: dict[str, Any],
    frames: dict[str, pd.DataFrame],
) -> dict[str, str]:
    required_sections = {
        "manifest_schema_version",
        "dataset",
        "source",
        "download",
        "license",
        "parsing",
        "eda",
        "positive_interactions",
        "temporal_split",
        "cohorts_and_mappings",
    }
    assert required_sections.issubset(manifest)

    parsing = manifest["parsing"]["datasets"]
    for name in ("ratings", "movies", "users"):
        metadata = parsing[name]
        assert metadata["output_path"] == f"data/processed/{name}.parquet"
        assert metadata["row_count"] == len(frames[name])
        assert metadata["columns"] == frames[name].columns.tolist()
        assert metadata["dtypes"] == (frames[name].dtypes.astype(str).to_dict())

    positives = manifest["positive_interactions"]
    assert positives["source_path"] == "data/processed/ratings.parquet"
    assert positives["source_row_count"] == len(frames["ratings"])
    assert positives["positive_row_count"] == len(frames["positives"])

    temporal = manifest["temporal_split"]
    assert temporal["source_path"] == positives["output_path"]
    assert temporal["source_sha256"] == positives["output_sha256"]
    assert temporal["source_row_count"] == len(frames["positives"])
    assert temporal["split_row_counts"] == {
        name: len(frames[name]) for name in ("train", "validation", "test")
    }

    cohorts = manifest["cohorts_and_mappings"]
    assert cohorts["source_paths"] == temporal["output_paths"]
    assert cohorts["source_sha256"] == temporal["output_sha256"]
    assert cohorts["output_row_counts"] == {
        name: len(frames[name])
        for name in (
            "evaluation_cohorts",
            "user_mapping",
            "item_mapping",
            "item_popularity_train",
        )
    }

    actual_hashes = {}
    for relative_path, expected_hash in _declared_output_hashes(manifest).items():
        actual_hash = _sha256(REPOSITORY_ROOT / relative_path)
        assert actual_hash == expected_hash, relative_path
        actual_hashes[relative_path] = actual_hash
    return actual_hashes


def _independent_context(
    fit: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fit_users = set(fit["user_id"])
    fit_items = set(fit["movie_id"])
    target_users = sorted(target["user_id"].unique())
    fit_counts = fit.groupby("user_id").size()
    target_counts = target.groupby("user_id").size()
    warm_target = target.loc[target["movie_id"].isin(fit_items)]
    warm_counts = warm_target.groupby("user_id").size()

    expected = pd.DataFrame(
        {
            "user_id": pd.Series(target_users, dtype="int32"),
        }
    )
    expected["fit_positive_count"] = (
        expected["user_id"].map(fit_counts).fillna(0).astype("int64")
    )
    expected["target_positive_count"] = (
        expected["user_id"].map(target_counts).fillna(0).astype("int64")
    )
    expected["warm_target_positive_count"] = (
        expected["user_id"].map(warm_counts).fillna(0).astype("int64")
    )
    expected["cold_start_target_positive_count"] = (
        expected["target_positive_count"] - expected["warm_target_positive_count"]
    )
    expected["is_cold_start_user"] = ~expected["user_id"].isin(fit_users)
    expected["is_eligible"] = expected["fit_positive_count"].ge(5) & expected[
        "warm_target_positive_count"
    ].ge(1)

    activity = pd.Series(pd.NA, index=expected.index, dtype="string")
    activity.loc[expected["fit_positive_count"].between(5, 9)] = "low"
    activity.loc[expected["fit_positive_count"].between(10, 29)] = "medium"
    activity.loc[expected["fit_positive_count"].ge(30)] = "high"
    expected["activity_group"] = activity

    enough_fit = expected["fit_positive_count"].ge(5)
    has_warm_target = expected["warm_target_positive_count"].ge(1)
    reason = pd.Series(pd.NA, index=expected.index, dtype="string")
    reason.loc[~enough_fit & has_warm_target] = "insufficient_fit_positives"
    reason.loc[enough_fit & ~has_warm_target] = "no_warm_target_items"
    reason.loc[~enough_fit & ~has_warm_target] = (
        "insufficient_fit_positives_and_no_warm_target_items"
    )
    expected["exclusion_reason"] = reason

    eligible = expected.loc[expected["is_eligible"]]
    counts = {
        "fit_interaction_count": len(fit),
        "fit_user_count": len(fit_users),
        "fit_item_count": len(fit_items),
        "target_interaction_count": len(target),
        "target_user_count": len(expected),
        "eligible_user_count": int(expected["is_eligible"].sum()),
        "excluded_user_count": int((~expected["is_eligible"]).sum()),
        "warm_start_user_count": int((~expected["is_cold_start_user"]).sum()),
        "cold_start_user_count": int(expected["is_cold_start_user"].sum()),
        "warm_target_interaction_count": len(warm_target),
        "cold_start_target_interaction_count": len(target) - len(warm_target),
        "warm_target_unique_item_count": int(warm_target["movie_id"].nunique()),
        "cold_start_target_unique_item_count": int(
            target.loc[~target["movie_id"].isin(fit_items), "movie_id"].nunique()
        ),
        "exclusion_reason_counts": {
            value: int(expected["exclusion_reason"].eq(value).sum())
            for value in EXCLUSION_REASONS
        },
        "eligible_activity_group_counts": {
            value: int(eligible["activity_group"].eq(value).sum())
            for value in ACTIVITY_GROUPS
        },
    }
    return expected, counts


def _popularity(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.groupby("movie_id").size().rename("positive_count").reset_index()
    result["positive_count"] = result["positive_count"].astype("int64")
    return result.sort_values(
        ["positive_count", "movie_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _fixture_ratings() -> pd.DataFrame:
    row_count = 40
    return pd.DataFrame(
        {
            "user_id": pd.Series(
                [(index % 6) + 1 for index in range(row_count)],
                dtype="int32",
            ),
            "movie_id": pd.Series(
                [100 + index for index in range(row_count)],
                dtype="int32",
            ),
            "rating": pd.Series(
                [(index % 5) + 1 for index in range(row_count)],
                dtype="int8",
            ),
            "timestamp": pd.Series(
                [1_000_000 + index for index in range(row_count)],
                dtype="int64",
            ),
        }
    )


def _write_parquet(path: Path, dataframe: pd.DataFrame) -> None:
    dataframe.to_parquet(
        path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )


def _run_fixture_pipeline(root: Path) -> dict[str, str]:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    ratings = _fixture_ratings()
    movies = pd.DataFrame(
        {
            "movie_id": pd.Series(range(100, 140), dtype="int32"),
            "title": pd.Series(
                [f"Movie {index}" for index in range(40)],
                dtype="string",
            ),
            "genres": pd.Series(["Drama"] * 40, dtype="string"),
        }
    )
    users = pd.DataFrame(
        {
            "user_id": pd.Series(range(1, 7), dtype="int32"),
            "gender": pd.Series(["F", "M", "F", "M", "F", "M"], dtype="string"),
            "age": pd.Series([18, 25, 35, 45, 50, 56], dtype="int16"),
            "occupation": pd.Series(range(6), dtype="int16"),
            "zip_code": pd.Series(
                [f"1000{index}" for index in range(6)],
                dtype="string",
            ),
        }
    )
    source_frames = {
        "ratings": ratings,
        "movies": movies,
        "users": users,
    }
    for name, dataframe in source_frames.items():
        _write_parquet(processed / f"{name}.parquet", dataframe)

    positives = build_positive_interactions(ratings)
    _write_parquet(processed / "positives.parquet", positives)
    splits = build_global_temporal_split(positives)
    split_frames = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    for name, dataframe in split_frames.items():
        _write_parquet(processed / f"{name}.parquet", dataframe)

    artifacts = build_evaluation_artifacts(
        splits.train,
        splits.validation,
        splits.test,
    )
    output_frames = {
        name: getattr(artifacts, name)
        for name in EvaluationArtifacts.__dataclass_fields__
    }
    for name, dataframe in output_frames.items():
        _write_parquet(processed / f"{name}.parquet", dataframe)

    relative = lambda name: f"data/processed/{name}.parquet"
    positive_hash = _sha256(processed / "positives.parquet")
    split_hashes = {
        name: _sha256(processed / f"{name}.parquet") for name in split_frames
    }
    output_hashes = {
        name: _sha256(processed / f"{name}.parquet") for name in output_frames
    }
    temporal_section = build_temporal_split_manifest_section(
        positives,
        splits,
        source_path=relative("positives"),
        source_sha256=positive_hash,
        output_paths={name: relative(name) for name in split_frames},
        output_sha256=split_hashes,
        parquet_engine="pyarrow",
        compression="snappy",
    )
    cohort_section = build_cohorts_manifest_section(
        artifacts,
        splits.train,
        splits.validation,
        splits.test,
        source_paths={name: relative(name) for name in split_frames},
        source_sha256=split_hashes,
        output_paths={name: relative(name) for name in output_frames},
        output_sha256=output_hashes,
        parquet_engine="pyarrow",
        compression="snappy",
    )
    manifest = {
        "manifest_schema_version": 1,
        "parsing": {
            "datasets": {
                name: {
                    "row_count": len(dataframe),
                    "columns": dataframe.columns.tolist(),
                    "dtypes": dataframe.dtypes.astype(str).to_dict(),
                    "output_path": relative(name),
                    "output_sha256": _sha256(processed / f"{name}.parquet"),
                }
                for name, dataframe in source_frames.items()
            }
        },
        "positive_interactions": {
            "source_path": relative("ratings"),
            "output_path": relative("positives"),
            "source_row_count": len(ratings),
            "positive_row_count": len(positives),
            "output_sha256": positive_hash,
        },
        "temporal_split": temporal_section,
        "cohorts_and_mappings": cohort_section,
    }
    manifest_path = processed / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(processed.iterdir())
    }


@pytest.mark.realdata
def test_real_artifact_schemas_ratings_timestamps_and_references(
    real_pipeline: RealPipeline,
) -> None:
    frames = real_pipeline.frames
    expected_schemas = {
        "ratings": (RATINGS_COLUMNS, RATINGS_DTYPES),
        "movies": (MOVIES_COLUMNS, MOVIES_DTYPES),
        "users": (USERS_COLUMNS, USERS_DTYPES),
        "positives": (POSITIVE_COLUMNS, POSITIVE_DTYPES),
        "train": (POSITIVE_COLUMNS, POSITIVE_DTYPES),
        "validation": (POSITIVE_COLUMNS, POSITIVE_DTYPES),
        "test": (POSITIVE_COLUMNS, POSITIVE_DTYPES),
        "evaluation_cohorts": (COHORT_COLUMNS, COHORT_DTYPES),
        "user_mapping": (USER_MAPPING_COLUMNS, USER_MAPPING_DTYPES),
        "item_mapping": (ITEM_MAPPING_COLUMNS, ITEM_MAPPING_DTYPES),
        "item_popularity_train": (
            ITEM_POPULARITY_COLUMNS,
            ITEM_POPULARITY_DTYPES,
        ),
    }
    for name, (columns, dtypes) in expected_schemas.items():
        _assert_schema(frames[name], columns, dtypes)

    assert set(frames["ratings"]["rating"]) == {1, 2, 3, 4, 5}
    assert set(frames["positives"]["rating"]) == {4, 5}
    for name in ("ratings", "positives", "train", "validation", "test"):
        timestamps = frames[name]["timestamp"]
        assert not timestamps.isna().any()
        assert is_integer_dtype(timestamps.dtype)
        assert timestamps.between(
            EARLIEST_ACCEPTABLE_TIMESTAMP,
            LATEST_ACCEPTABLE_TIMESTAMP,
        ).all()
        pd.to_datetime(timestamps, unit="s", utc=True, errors="raise")

    validate_referential_integrity(
        frames["ratings"],
        frames["movies"],
        frames["users"],
    )
    assert (
        not frames["ratings"]["user_id"]
        .isin(frames["users"]["user_id"])
        .eq(False)
        .any()
    )
    assert (
        not frames["ratings"]["movie_id"]
        .isin(frames["movies"]["movie_id"])
        .eq(False)
        .any()
    )


@pytest.mark.parametrize(
    ("id_column", "catalog_name", "message"),
    [
        ("user_id", "users", "unknown user references"),
        ("movie_id", "movies", "unknown movie references"),
    ],
)
@pytest.mark.realdata
def test_referential_validator_rejects_unknown_id_mutation(
    real_pipeline: RealPipeline,
    id_column: str,
    catalog_name: str,
    message: str,
) -> None:
    frames = real_pipeline.frames
    mutated = frames["ratings"].iloc[[0]].copy()
    mutated.loc[mutated.index[0], id_column] = (
        int(frames[catalog_name][id_column].max()) + 1
    )

    with pytest.raises(ValueError, match=message):
        validate_referential_integrity(
            mutated,
            frames["movies"],
            frames["users"],
        )


@pytest.mark.realdata
def test_real_cross_stage_integrity_manifest_and_cohort_counts(
    real_pipeline: RealPipeline,
) -> None:
    frames = real_pipeline.frames
    manifest = real_pipeline.manifest
    expected_positives = build_positive_interactions(frames["ratings"])
    pd.testing.assert_frame_equal(frames["positives"], expected_positives)

    cutoffs = manifest["temporal_split"]["cutoff_indices"]
    splits = TemporalSplits(
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
        train_end=cutoffs["train_end_exclusive"],
        validation_end=cutoffs["validation_end_exclusive"],
    )
    validate_temporal_split(frames["positives"], splits)
    reconstructed = pd.concat(
        [frames["train"], frames["validation"], frames["test"]],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(
        reconstructed,
        canonical_sort(frames["positives"]),
    )

    artifacts = EvaluationArtifacts(
        evaluation_cohorts=frames["evaluation_cohorts"],
        user_mapping=frames["user_mapping"],
        item_mapping=frames["item_mapping"],
        item_popularity_train=frames["item_popularity_train"],
    )
    validate_evaluation_artifacts(
        artifacts,
        frames["train"],
        frames["validation"],
        frames["test"],
    )

    train_validation = pd.concat(
        [frames["train"], frames["validation"]],
        ignore_index=True,
    )
    for scope, fit in (
        ("train", frames["train"]),
        ("train_validation", train_validation),
    ):
        for mapping_name, id_column, index_column in (
            ("user_mapping", "user_id", "user_index"),
            ("item_mapping", "movie_id", "item_index"),
        ):
            mapping = frames[mapping_name].loc[
                frames[mapping_name]["fit_scope"].eq(scope)
            ]
            assert mapping[id_column].tolist() == sorted(fit[id_column].unique())
            assert mapping[index_column].tolist() == list(range(len(mapping)))

    expected_popularity = _popularity(frames["train"])
    pd.testing.assert_frame_equal(
        frames["item_popularity_train"],
        expected_popularity,
    )
    assert not frames["item_popularity_train"].equals(_popularity(train_validation))
    assert not frames["item_popularity_train"].equals(_popularity(reconstructed))

    contexts = manifest["cohorts_and_mappings"]["contexts"]
    for split_name, fit, target in (
        ("validation", frames["train"], frames["validation"]),
        ("test", train_validation, frames["test"]),
    ):
        expected_cohort, expected_counts = _independent_context(fit, target)
        actual_cohort = (
            frames["evaluation_cohorts"]
            .loc[frames["evaluation_cohorts"]["evaluation_split"].eq(split_name)]
            .reset_index(drop=True)
        )
        comparison_columns = [
            "user_id",
            "fit_positive_count",
            "target_positive_count",
            "warm_target_positive_count",
            "cold_start_target_positive_count",
            "is_cold_start_user",
            "is_eligible",
            "activity_group",
            "exclusion_reason",
        ]
        pd.testing.assert_frame_equal(
            actual_cohort[comparison_columns],
            expected_cohort[comparison_columns],
        )
        for key, value in expected_counts.items():
            assert contexts[split_name][key] == value
        assert (
            expected_counts["eligible_user_count"]
            + expected_counts["excluded_user_count"]
            == expected_counts["target_user_count"]
        )
        assert (
            sum(expected_counts["exclusion_reason_counts"].values())
            == expected_counts["excluded_user_count"]
        )
        assert (
            sum(expected_counts["eligible_activity_group_counts"].values())
            == expected_counts["eligible_user_count"]
        )
        assert (
            expected_counts["warm_target_interaction_count"]
            + expected_counts["cold_start_target_interaction_count"]
            == expected_counts["target_interaction_count"]
        )

    hashes = _validate_manifest_hash_chain(manifest, frames)
    overlap = {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    print(
        json.dumps(
            {
                "rows": {name: len(frame) for name, frame in frames.items()},
                "null_counts": {
                    name: int(frame.isna().sum().sum())
                    for name, frame in frames.items()
                    if name != "evaluation_cohorts"
                },
                "duplicate_counts": {
                    name: int(frame.duplicated(["user_id", "movie_id"]).sum())
                    for name, frame in frames.items()
                    if {"user_id", "movie_id"}.issubset(frame.columns)
                },
                "missing_user_references": 0,
                "missing_movie_references": 0,
                "rating_range": [
                    int(frames["ratings"]["rating"].min()),
                    int(frames["ratings"]["rating"].max()),
                ],
                "timestamp_ranges": {
                    name: [
                        int(frames[name]["timestamp"].min()),
                        int(frames[name]["timestamp"].max()),
                    ]
                    for name in ("train", "validation", "test")
                },
                "pairwise_split_overlap": overlap,
                "split_reconstruction": True,
                "mapping_contiguity": True,
                "popularity_equality": True,
                "warm_cold_invariants": True,
                "manifest_hash_consistency": len(hashes),
            },
            sort_keys=True,
        )
    )
    print("DATA_PIPELINE_REAL_AUDIT_PASS")


@pytest.mark.parametrize("leak_scope", ["validation", "test"])
def test_train_popularity_validator_rejects_target_leakage(
    leak_scope: str,
) -> None:
    train = pd.DataFrame(
        {
            "user_id": pd.Series([1, 2, 3, 4, 5, 6], dtype="int32"),
            "movie_id": pd.Series([10, 10, 20, 30, 40, 50], dtype="int32"),
            "rating": pd.Series([4, 5, 4, 5, 4, 5], dtype="int8"),
            "timestamp": pd.Series([1, 2, 3, 4, 5, 6], dtype="int64"),
        }
    )
    target_timestamp = 7 if leak_scope == "validation" else 8
    target = pd.DataFrame(
        {
            "user_id": pd.Series([7], dtype="int32"),
            "movie_id": pd.Series([10], dtype="int32"),
            "rating": pd.Series([4], dtype="int8"),
            "timestamp": pd.Series([target_timestamp], dtype="int64"),
        }
    )
    correct = _popularity(train)
    leaked = _popularity(pd.concat([train, target], ignore_index=True))

    validate_item_popularity(correct, train=train)
    assert int(correct.loc[correct["movie_id"].eq(10), "positive_count"].iloc[0]) == 2
    assert int(leaked.loc[leaked["movie_id"].eq(10), "positive_count"].iloc[0]) == 3
    with pytest.raises(ValueError, match="counts do not sum"):
        validate_item_popularity(leaked, train=train)


@pytest.mark.realdata
def test_manifest_hash_mutation_is_detected(
    real_pipeline: RealPipeline,
) -> None:
    mutated = deepcopy(real_pipeline.manifest)
    mutated["temporal_split"]["output_sha256"]["train"] = "0" * 64

    with pytest.raises(AssertionError, match="train"):
        _validate_manifest_hash_chain(mutated, real_pipeline.frames)


def test_fixture_pipeline_is_byte_for_byte_deterministic(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_hashes = _run_fixture_pipeline(first_root)
    second_hashes = _run_fixture_pipeline(second_root)

    assert first_hashes == second_hashes
    assert set(first_hashes) == {
        f"data/processed/{name}"
        for name in (
            "ratings.parquet",
            "movies.parquet",
            "users.parquet",
            "positives.parquet",
            "train.parquet",
            "validation.parquet",
            "test.parquet",
            "evaluation_cohorts.parquet",
            "user_mapping.parquet",
            "item_mapping.parquet",
            "item_popularity_train.parquet",
            "data_manifest.json",
        )
    }
    for relative_path in first_hashes:
        assert (first_root / relative_path).read_bytes() == (
            second_root / relative_path
        ).read_bytes()
