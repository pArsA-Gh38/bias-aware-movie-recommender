from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

SORT_COLUMNS = ["timestamp", "user_id", "movie_id"]
INTERACTION_KEY = ["user_id", "movie_id"]
MINIMUM_SOURCE_ROWS = 6
TRAIN_NUMERATOR = 8
VALIDATION_END_NUMERATOR = 9
RATIO_DENOMINATOR = 10


@dataclass(frozen=True)
class TemporalSplits:
    """The three fixed-row temporal partitions and their boundaries."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: int
    validation_end: int


def allocation_boundaries(source_row_count: int) -> tuple[int, int]:
    """Return exact 80% and 90% exclusive boundaries using integers."""

    if isinstance(source_row_count, bool) or not isinstance(source_row_count, int):
        raise TypeError("source_row_count must be an integer")

    if source_row_count < MINIMUM_SOURCE_ROWS:
        raise ValueError(
            "Temporal split requires at least 6 source rows so train, "
            "validation, and test are all non-empty"
        )

    train_end = (source_row_count * TRAIN_NUMERATOR) // RATIO_DENOMINATOR
    validation_end = (source_row_count * VALIDATION_END_NUMERATOR) // RATIO_DENOMINATOR

    split_row_counts = (
        train_end,
        validation_end - train_end,
        source_row_count - validation_end,
    )

    if any(row_count <= 0 for row_count in split_row_counts):
        raise ValueError(
            "Temporal split allocation must produce non-empty train, "
            "validation, and test partitions"
        )

    return train_end, validation_end


def validate_temporal_source(source: pd.DataFrame) -> None:
    """Validate the DATA-005 source contract before sorting or splitting."""

    if not isinstance(source, pd.DataFrame):
        raise TypeError("Temporal split source must be a pandas DataFrame")

    allocation_boundaries(len(source))

    missing_columns = [
        column for column in SORT_COLUMNS if column not in source.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Temporal split source is missing sort columns: {missing_columns}"
        )

    null_counts = source[SORT_COLUMNS].isna().sum()

    if int(null_counts.sum()) != 0:
        raise ValueError(
            f"Temporal split sort keys contain null values: {null_counts.to_dict()}"
        )

    invalid_dtypes = {
        column: str(source[column].dtype)
        for column in SORT_COLUMNS
        if (
            is_bool_dtype(source[column].dtype)
            or not is_integer_dtype(source[column].dtype)
        )
    }

    if invalid_dtypes:
        raise TypeError(
            "Temporal split sort keys must have comparable integer dtypes: "
            f"{invalid_dtypes}"
        )

    duplicate_count = int(source.duplicated(INTERACTION_KEY).sum())

    if duplicate_count:
        raise ValueError(
            "Temporal split source contains "
            f"{duplicate_count} duplicate user-movie interaction keys"
        )


def canonical_sort(source: pd.DataFrame) -> pd.DataFrame:
    """Return a stable global `(timestamp, user_id, movie_id)` ordering."""

    missing_columns = [
        column for column in SORT_COLUMNS if column not in source.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Temporal split source is missing sort columns: {missing_columns}"
        )

    try:
        return source.sort_values(
            SORT_COLUMNS,
            kind="mergesort",
        ).reset_index(drop=True)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Temporal split sort keys are not mutually comparable"
        ) from error


def build_global_temporal_split(source: pd.DataFrame) -> TemporalSplits:
    """Build the deterministic global 80/10/10 temporal split."""

    validate_temporal_source(source)
    sorted_source = canonical_sort(source)
    train_end, validation_end = allocation_boundaries(len(sorted_source))

    splits = TemporalSplits(
        train=sorted_source.iloc[:train_end].copy().reset_index(drop=True),
        validation=(
            sorted_source.iloc[train_end:validation_end].copy().reset_index(drop=True)
        ),
        test=(sorted_source.iloc[validation_end:].copy().reset_index(drop=True)),
        train_end=train_end,
        validation_end=validation_end,
    )
    validate_temporal_split(source, splits)
    return splits


def _schema_signature(dataframe: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    return (
        dataframe.columns.tolist(),
        dataframe.dtypes.astype(str).to_dict(),
    )


def _assert_frame_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    message: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as error:
        raise ValueError(message) from error


def calculate_overlap_counts(
    splits: TemporalSplits,
) -> dict[str, dict[str, int]]:
    """Count pairwise interaction-key and full-row fingerprint overlap."""

    named_splits = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    overlap: dict[str, dict[str, int]] = {}

    for (left_name, left), (right_name, right) in combinations(
        named_splits.items(),
        2,
    ):
        label = f"{left_name}_{right_name}"
        key_overlap = left[INTERACTION_KEY].merge(
            right[INTERACTION_KEY],
            on=INTERACTION_KEY,
            how="inner",
            validate="one_to_one",
        )
        left_fingerprints = set(
            pd.util.hash_pandas_object(left, index=False).astype("uint64")
        )
        right_fingerprints = set(
            pd.util.hash_pandas_object(right, index=False).astype("uint64")
        )
        overlap[label] = {
            "interaction_key": len(key_overlap),
            "full_row_fingerprint": len(
                left_fingerprints.intersection(right_fingerprints)
            ),
        }

    return overlap


def validate_temporal_split(
    source: pd.DataFrame,
    splits: TemporalSplits,
) -> None:
    """Validate allocation, schema, content, overlap, and temporal ordering."""

    validate_temporal_source(source)
    expected_train_end, expected_validation_end = allocation_boundaries(len(source))
    expected_schema = _schema_signature(source)
    named_splits = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }

    for split_name, dataframe in named_splits.items():
        if _schema_signature(dataframe) != expected_schema:
            raise ValueError(
                f"{split_name} schema, column order, or dtypes differ from the source"
            )

    combined_count = sum(len(dataframe) for dataframe in named_splits.values())

    if combined_count < len(source):
        raise ValueError(
            f"Temporal split is missing source rows: {combined_count} < {len(source)}"
        )

    if combined_count > len(source):
        raise ValueError(
            f"Temporal split contains extra rows: {combined_count} > {len(source)}"
        )

    expected_counts = {
        "train": expected_train_end,
        "validation": expected_validation_end - expected_train_end,
        "test": len(source) - expected_validation_end,
    }
    actual_counts = {
        split_name: len(dataframe) for split_name, dataframe in named_splits.items()
    }

    if (
        splits.train_end != expected_train_end
        or splits.validation_end != expected_validation_end
        or actual_counts != expected_counts
    ):
        raise ValueError(
            "Temporal split allocation does not match fixed row boundaries: "
            f"{actual_counts} != {expected_counts}"
        )

    for split_name, dataframe in named_splits.items():
        null_counts = dataframe[SORT_COLUMNS].isna().sum()

        if int(null_counts.sum()):
            raise ValueError(
                f"{split_name} contains null sort keys: {null_counts.to_dict()}"
            )

        duplicate_count = int(dataframe.duplicated(INTERACTION_KEY).sum())

        if duplicate_count:
            raise ValueError(
                f"{split_name} contains {duplicate_count} duplicate interaction keys"
            )

    overlap_counts = calculate_overlap_counts(splits)

    if any(
        count
        for pair_counts in overlap_counts.values()
        for count in pair_counts.values()
    ):
        raise ValueError(f"Temporal split partitions overlap: {overlap_counts}")

    for split_name, dataframe in named_splits.items():
        _assert_frame_equal(
            dataframe,
            canonical_sort(dataframe),
            f"{split_name} is not in canonical temporal order",
        )

    if splits.train["timestamp"].max() > splits.validation["timestamp"].min():
        raise ValueError("Train timestamps extend beyond validation")

    if splits.validation["timestamp"].max() > splits.test["timestamp"].min():
        raise ValueError("Validation timestamps extend beyond test")

    combined = pd.concat(
        named_splits.values(),
        ignore_index=True,
    )
    _assert_frame_equal(
        combined,
        canonical_sort(source),
        "Temporal split contains a modified, missing, extra, or misplaced row",
    )


def _sort_key(dataframe: pd.DataFrame, index: int) -> dict[str, int]:
    row = dataframe.iloc[index]
    return {column: int(row[column]) for column in SORT_COLUMNS}


def build_temporal_split_manifest_section(
    source: pd.DataFrame,
    splits: TemporalSplits,
    *,
    source_path: str,
    source_sha256: str,
    output_paths: dict[str, str],
    output_sha256: dict[str, str],
    parquet_engine: str,
    compression: str,
) -> dict[str, Any]:
    """Build deterministic DATA-005 metadata with no execution-time fields."""

    validate_temporal_split(source, splits)
    source_row_count = len(source)
    named_splits = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }

    return {
        "strategy": "global_temporal",
        "source_path": source_path,
        "source_sha256": source_sha256,
        "source_row_count": source_row_count,
        "interaction_key": INTERACTION_KEY,
        "columns": source.columns.tolist(),
        "dtypes": source.dtypes.astype(str).to_dict(),
        "sort_order": SORT_COLUMNS,
        "ratios": {
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1,
        },
        "allocation": {
            "method": ("fixed_row_boundaries_after_global_lexicographic_sort"),
            "train_end_formula": "(n * 8) // 10",
            "validation_end_formula": "(n * 9) // 10",
            "minimum_source_rows": MINIMUM_SOURCE_ROWS,
        },
        "tie_policy": {
            "timestamp_ties_may_cross_boundaries": True,
            "tie_breakers": ["user_id", "movie_id"],
            "boundary_rule": ("fixed_row_indices_after_lexicographic_sort"),
        },
        "split_row_counts": {
            split_name: len(dataframe) for split_name, dataframe in named_splits.items()
        },
        "actual_ratios": {
            split_name: len(dataframe) / source_row_count
            for split_name, dataframe in named_splits.items()
        },
        "cutoff_indices": {
            "train_end_exclusive": splits.train_end,
            "validation_end_exclusive": splits.validation_end,
        },
        "timestamp_ranges": {
            split_name: {
                "minimum": int(dataframe["timestamp"].min()),
                "maximum": int(dataframe["timestamp"].max()),
            }
            for split_name, dataframe in named_splits.items()
        },
        "boundary_sort_keys": {
            "last_train": _sort_key(splits.train, -1),
            "first_validation": _sort_key(splits.validation, 0),
            "last_validation": _sort_key(splits.validation, -1),
            "first_test": _sort_key(splits.test, 0),
        },
        "output_paths": output_paths,
        "output_sha256": output_sha256,
        "parquet_engine": parquet_engine,
        "compression": compression,
    }


def update_manifest_document(
    manifest: dict[str, Any],
    temporal_split_section: dict[str, Any],
) -> dict[str, Any]:
    """Insert or replace only `temporal_split`, preserving every other field."""

    if not isinstance(manifest, dict):
        raise TypeError("Data manifest must contain a JSON object")

    updated = deepcopy(manifest)
    updated["temporal_split"] = deepcopy(temporal_split_section)
    return updated
