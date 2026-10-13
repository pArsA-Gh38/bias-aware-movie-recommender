from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

SOURCE_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
SOURCE_SORT_COLUMNS = ["timestamp", "user_id", "movie_id"]
INTERACTION_KEY = ["user_id", "movie_id"]
POSITIVE_RATING_VALUES = (4, 5)

MINIMUM_FIT_POSITIVES = 5
LOW_ACTIVITY_MINIMUM = 5
LOW_ACTIVITY_MAXIMUM = 9
MEDIUM_ACTIVITY_MINIMUM = 10
MEDIUM_ACTIVITY_MAXIMUM = 29
HIGH_ACTIVITY_MINIMUM = 30

EVALUATION_SPLITS = ("validation", "test")
FIT_SCOPES = ("train", "train_validation")
EXCLUSION_REASONS = (
    "insufficient_fit_positives",
    "no_warm_target_items",
    "insufficient_fit_positives_and_no_warm_target_items",
)
ACTIVITY_GROUPS = ("low", "medium", "high")

COHORT_COLUMNS = [
    "evaluation_split",
    "fit_scope",
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
USER_MAPPING_COLUMNS = ["fit_scope", "user_id", "user_index"]
ITEM_MAPPING_COLUMNS = ["fit_scope", "movie_id", "item_index"]
ITEM_POPULARITY_COLUMNS = ["movie_id", "positive_count"]


@dataclass(frozen=True)
class EvaluationArtifacts:
    """All deterministic DATA-006 tabular outputs."""

    evaluation_cohorts: pd.DataFrame
    user_mapping: pd.DataFrame
    item_mapping: pd.DataFrame
    item_popularity_train: pd.DataFrame


def _schema_signature(
    dataframe: pd.DataFrame,
) -> tuple[list[str], dict[str, str]]:
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


def validate_source_frame(
    dataframe: pd.DataFrame,
    *,
    split_name: str,
) -> None:
    """Fail fast on a malformed positive-interaction split."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{split_name} must be a pandas DataFrame")

    if dataframe.empty:
        raise ValueError(f"{split_name} must not be empty")

    if dataframe.columns.tolist() != SOURCE_COLUMNS:
        raise ValueError(
            f"{split_name} has unexpected columns: {dataframe.columns.tolist()}"
        )

    null_counts = dataframe.isna().sum()

    if int(null_counts.sum()):
        raise ValueError(f"{split_name} contains null values: {null_counts.to_dict()}")

    invalid_dtypes = {
        column: str(dataframe[column].dtype)
        for column in SOURCE_COLUMNS
        if (
            is_bool_dtype(dataframe[column].dtype)
            or not is_integer_dtype(dataframe[column].dtype)
        )
    }

    if invalid_dtypes:
        raise TypeError(
            f"{split_name} columns must use integer dtypes: {invalid_dtypes}"
        )

    if not dataframe["rating"].isin(POSITIVE_RATING_VALUES).all():
        raise ValueError(
            f"{split_name} must contain only positive rating values 4 and 5"
        )

    if not dataframe["user_id"].gt(0).all():
        raise ValueError(f"{split_name} contains a non-positive user_id")

    if not dataframe["movie_id"].gt(0).all():
        raise ValueError(f"{split_name} contains a non-positive movie_id")

    duplicate_count = int(dataframe.duplicated(INTERACTION_KEY).sum())

    if duplicate_count:
        raise ValueError(
            f"{split_name} contains {duplicate_count} duplicate interactions"
        )

    canonical = dataframe.sort_values(
        SOURCE_SORT_COLUMNS,
        kind="mergesort",
    ).reset_index(drop=True)
    _assert_frame_equal(
        dataframe,
        canonical,
        f"{split_name} is not in canonical temporal order",
    )


def validate_source_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Validate compatible schema, ordering, time, and split isolation."""

    named_splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for split_name, dataframe in named_splits.items():
        validate_source_frame(dataframe, split_name=split_name)

    expected_schema = _schema_signature(train)

    for split_name, dataframe in named_splits.items():
        if _schema_signature(dataframe) != expected_schema:
            raise TypeError(
                f"{split_name} schema or dtypes are incompatible with train"
            )

    for (left_name, left), (right_name, right) in combinations(
        named_splits.items(),
        2,
    ):
        overlap = left[INTERACTION_KEY].merge(
            right[INTERACTION_KEY],
            on=INTERACTION_KEY,
            how="inner",
            validate="one_to_one",
        )

        if not overlap.empty:
            raise ValueError(
                f"{left_name} and {right_name} overlap on "
                f"{len(overlap)} interaction keys"
            )

    if train["timestamp"].max() > validation["timestamp"].min():
        raise ValueError("Train timestamps extend beyond validation")

    if validation["timestamp"].max() > test["timestamp"].min():
        raise ValueError("Validation timestamps extend beyond test")


def _string_series(
    values: list[str] | pd.Series,
    *,
    index: pd.Index | None = None,
) -> pd.Series:
    return pd.Series(values, index=index, dtype="string")


def _build_evaluation_context_unvalidated(
    fit: pd.DataFrame,
    target: pd.DataFrame,
    *,
    evaluation_split: str,
    fit_scope: str,
) -> pd.DataFrame:
    fit_users = pd.Index(fit["user_id"].unique())
    fit_items = pd.Index(fit["movie_id"].unique())
    target_users = (
        target["user_id"]
        .drop_duplicates()
        .sort_values(kind="mergesort")
        .reset_index(drop=True)
    )
    fit_counts = fit.groupby("user_id", sort=False).size()
    target_counts = target.groupby("user_id", sort=False).size()
    warm_target = target.loc[target["movie_id"].isin(fit_items)]
    warm_counts = warm_target.groupby("user_id", sort=False).size()

    cohort = pd.DataFrame({"user_id": target_users})
    cohort["fit_positive_count"] = (
        cohort["user_id"].map(fit_counts).fillna(0).astype("int64")
    )
    cohort["target_positive_count"] = (
        cohort["user_id"].map(target_counts).fillna(0).astype("int64")
    )
    cohort["warm_target_positive_count"] = (
        cohort["user_id"].map(warm_counts).fillna(0).astype("int64")
    )
    cohort["cold_start_target_positive_count"] = (
        cohort["target_positive_count"] - cohort["warm_target_positive_count"]
    ).astype("int64")
    cohort["is_cold_start_user"] = (~cohort["user_id"].isin(fit_users)).astype("bool")
    enough_fit = cohort["fit_positive_count"].ge(MINIMUM_FIT_POSITIVES)
    has_warm_target = cohort["warm_target_positive_count"].ge(1)
    cohort["is_eligible"] = (enough_fit & has_warm_target).astype("bool")

    activity_group = pd.Series(
        pd.NA,
        index=cohort.index,
        dtype="string",
    )
    activity_group.loc[
        cohort["fit_positive_count"].between(
            LOW_ACTIVITY_MINIMUM,
            LOW_ACTIVITY_MAXIMUM,
        )
    ] = "low"
    activity_group.loc[
        cohort["fit_positive_count"].between(
            MEDIUM_ACTIVITY_MINIMUM,
            MEDIUM_ACTIVITY_MAXIMUM,
        )
    ] = "medium"
    activity_group.loc[cohort["fit_positive_count"].ge(HIGH_ACTIVITY_MINIMUM)] = "high"
    cohort["activity_group"] = activity_group

    exclusion_reason = pd.Series(
        pd.NA,
        index=cohort.index,
        dtype="string",
    )
    exclusion_reason.loc[~enough_fit & has_warm_target] = "insufficient_fit_positives"
    exclusion_reason.loc[enough_fit & ~has_warm_target] = "no_warm_target_items"
    exclusion_reason.loc[~enough_fit & ~has_warm_target] = (
        "insufficient_fit_positives_and_no_warm_target_items"
    )
    cohort["exclusion_reason"] = exclusion_reason
    cohort.insert(
        0,
        "fit_scope",
        _string_series([fit_scope] * len(cohort)),
    )
    cohort.insert(
        0,
        "evaluation_split",
        _string_series([evaluation_split] * len(cohort)),
    )
    return cohort[COHORT_COLUMNS].reset_index(drop=True)


def build_evaluation_context(
    fit: pd.DataFrame,
    target: pd.DataFrame,
    *,
    evaluation_split: str,
    fit_scope: str,
) -> pd.DataFrame:
    """Build one warm-start cohort using only its declared fit data."""

    if evaluation_split not in EVALUATION_SPLITS:
        raise ValueError(f"Unknown evaluation_split: {evaluation_split}")

    if fit_scope not in FIT_SCOPES:
        raise ValueError(f"Unknown fit_scope: {fit_scope}")

    validate_source_frame(fit, split_name=f"{fit_scope} fit")
    validate_source_frame(target, split_name=f"{evaluation_split} target")

    if _schema_signature(fit) != _schema_signature(target):
        raise TypeError("Fit and target schemas or dtypes are incompatible")

    cohort = _build_evaluation_context_unvalidated(
        fit,
        target,
        evaluation_split=evaluation_split,
        fit_scope=fit_scope,
    )
    validate_cohort_table(
        cohort,
        user_dtype=str(fit["user_id"].dtype),
    )
    return cohort


def build_contiguous_mapping(
    fit: pd.DataFrame,
    *,
    fit_scope: str,
    id_column: str,
    index_column: str,
) -> pd.DataFrame:
    """Map every sorted fit ID onto a zero-based contiguous index."""

    if fit_scope not in FIT_SCOPES:
        raise ValueError(f"Unknown fit_scope: {fit_scope}")

    if id_column not in ("user_id", "movie_id"):
        raise ValueError(f"Unsupported mapping ID column: {id_column}")

    validate_source_frame(fit, split_name=f"{fit_scope} fit")
    identifiers = (
        fit[id_column]
        .drop_duplicates()
        .sort_values(kind="mergesort")
        .reset_index(drop=True)
    )
    mapping = pd.DataFrame(
        {
            "fit_scope": _string_series([fit_scope] * len(identifiers)),
            id_column: identifiers,
            index_column: pd.Series(
                range(len(identifiers)),
                dtype="int64",
            ),
        }
    )
    validate_mapping_table(
        mapping,
        fit_scope=fit_scope,
        id_column=id_column,
        index_column=index_column,
        expected_ids=identifiers,
    )
    return mapping


def validate_mapping_table(
    mapping: pd.DataFrame,
    *,
    fit_scope: str,
    id_column: str,
    index_column: str,
    expected_ids: pd.Series | None = None,
) -> None:
    """Validate one mapping scope, including duplicate and gap detection."""

    expected_columns = ["fit_scope", id_column, index_column]

    if mapping.columns.tolist() != expected_columns:
        raise ValueError(f"Unexpected mapping columns: {mapping.columns.tolist()}")

    if mapping.empty:
        raise ValueError(f"{fit_scope} mapping must not be empty")

    if mapping.isna().any().any():
        raise ValueError(f"{fit_scope} mapping contains null values")

    if mapping["fit_scope"].dtype != pd.StringDtype():
        raise TypeError("Mapping fit_scope must use nullable string dtype")

    if not mapping["fit_scope"].eq(fit_scope).all():
        raise ValueError(f"Mapping contains rows outside {fit_scope}")

    if is_bool_dtype(mapping[id_column].dtype) or not is_integer_dtype(
        mapping[id_column].dtype
    ):
        raise TypeError(f"{id_column} must use an integer dtype")

    if str(mapping[index_column].dtype) != "int64":
        raise TypeError(f"{index_column} must use int64 dtype")

    if mapping[id_column].duplicated().any():
        raise ValueError(f"Mapping contains duplicate {id_column}")

    if mapping[index_column].duplicated().any():
        raise ValueError(f"Mapping contains duplicate {index_column}")

    expected_indices = pd.Series(
        range(len(mapping)),
        dtype="int64",
        name=index_column,
    )

    if not mapping[index_column].reset_index(drop=True).equals(expected_indices):
        raise ValueError(f"{fit_scope} {index_column} is not zero-based and contiguous")

    if not mapping[id_column].is_monotonic_increasing:
        raise ValueError(f"{fit_scope} {id_column} is not ascending")

    if expected_ids is not None:
        actual_ids = mapping[id_column].reset_index(drop=True)
        expected = expected_ids.reset_index(drop=True)

        if not actual_ids.equals(expected):
            raise ValueError(
                f"{fit_scope} mapping has a modified, missing, or extra ID"
            )


def build_item_popularity_train(train: pd.DataFrame) -> pd.DataFrame:
    """Count train positives only and apply the fixed deterministic order."""

    validate_source_frame(train, split_name="train")
    popularity = (
        train.groupby("movie_id", sort=False)
        .size()
        .rename("positive_count")
        .reset_index()
    )
    popularity["positive_count"] = popularity["positive_count"].astype("int64")
    popularity = popularity.sort_values(
        ["positive_count", "movie_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    validate_item_popularity(
        popularity,
        train=train,
    )
    return popularity


def validate_cohort_table(
    cohort: pd.DataFrame,
    *,
    user_dtype: str,
) -> None:
    """Validate output schema and all cohort policy invariants."""

    if cohort.columns.tolist() != COHORT_COLUMNS:
        raise ValueError(f"Unexpected cohort columns: {cohort.columns.tolist()}")

    expected_dtypes = {
        "evaluation_split": "string",
        "fit_scope": "string",
        "user_id": user_dtype,
        "fit_positive_count": "int64",
        "target_positive_count": "int64",
        "warm_target_positive_count": "int64",
        "cold_start_target_positive_count": "int64",
        "is_cold_start_user": "bool",
        "is_eligible": "bool",
        "activity_group": "string",
        "exclusion_reason": "string",
    }
    actual_dtypes = cohort.dtypes.astype(str).to_dict()

    if actual_dtypes != expected_dtypes:
        raise TypeError(f"Unexpected cohort dtypes: {actual_dtypes}")

    required_non_null = [
        column
        for column in COHORT_COLUMNS
        if column not in ("activity_group", "exclusion_reason")
    ]

    if cohort[required_non_null].isna().any().any():
        raise ValueError("Cohort contains null values in required columns")

    if cohort.duplicated(["evaluation_split", "user_id"]).any():
        raise ValueError("Cohort contains duplicate target users")

    if not cohort["evaluation_split"].isin(EVALUATION_SPLITS).all():
        raise ValueError("Cohort contains an unknown evaluation split")

    expected_scope = cohort["evaluation_split"].map(
        {"validation": "train", "test": "train_validation"}
    )

    if not cohort["fit_scope"].equals(expected_scope.astype("string")):
        raise ValueError("Cohort evaluation_split and fit_scope disagree")

    count_columns = [
        "fit_positive_count",
        "target_positive_count",
        "warm_target_positive_count",
        "cold_start_target_positive_count",
    ]

    if cohort[count_columns].lt(0).any().any():
        raise ValueError("Cohort contains a negative count")

    if not cohort["target_positive_count"].equals(
        cohort["warm_target_positive_count"]
        + cohort["cold_start_target_positive_count"]
    ):
        raise ValueError("Warm and cold target counts do not sum to target")

    expected_eligible = cohort["fit_positive_count"].ge(MINIMUM_FIT_POSITIVES) & cohort[
        "warm_target_positive_count"
    ].ge(1)

    if not cohort["is_eligible"].equals(expected_eligible):
        raise ValueError("Cohort eligibility does not match fixed policy")

    expected_activity = pd.Series(
        pd.NA,
        index=cohort.index,
        dtype="string",
    )
    expected_activity.loc[cohort["fit_positive_count"].between(5, 9)] = "low"
    expected_activity.loc[cohort["fit_positive_count"].between(10, 29)] = "medium"
    expected_activity.loc[cohort["fit_positive_count"].ge(30)] = "high"

    if not cohort["activity_group"].equals(expected_activity):
        raise ValueError("Cohort activity_group does not match fixed policy")

    enough_fit = cohort["fit_positive_count"].ge(MINIMUM_FIT_POSITIVES)
    has_warm_target = cohort["warm_target_positive_count"].ge(1)
    expected_reason = pd.Series(
        pd.NA,
        index=cohort.index,
        dtype="string",
    )
    expected_reason.loc[~enough_fit & has_warm_target] = "insufficient_fit_positives"
    expected_reason.loc[enough_fit & ~has_warm_target] = "no_warm_target_items"
    expected_reason.loc[~enough_fit & ~has_warm_target] = (
        "insufficient_fit_positives_and_no_warm_target_items"
    )

    if not cohort["exclusion_reason"].equals(expected_reason):
        raise ValueError("Cohort exclusion_reason does not match fixed policy")

    invalid_activity = (
        cohort["activity_group"]
        .dropna()
        .loc[lambda values: ~values.isin(ACTIVITY_GROUPS)]
    )
    invalid_reasons = (
        cohort["exclusion_reason"]
        .dropna()
        .loc[lambda values: ~values.isin(EXCLUSION_REASONS)]
    )

    if not invalid_activity.empty or not invalid_reasons.empty:
        raise ValueError("Cohort contains an unsupported categorical value")

    split_order = cohort["evaluation_split"].map({"validation": 0, "test": 1})
    order_frame = pd.DataFrame(
        {
            "split_order": split_order,
            "user_id": cohort["user_id"],
        }
    )
    expected_order = order_frame.sort_values(
        ["split_order", "user_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if not order_frame.reset_index(drop=True).equals(expected_order):
        raise ValueError("Cohort rows are not in deterministic order")


def validate_item_popularity(
    popularity: pd.DataFrame,
    *,
    train: pd.DataFrame,
) -> None:
    """Validate exact train-only item counts and deterministic tie order."""

    if popularity.columns.tolist() != ITEM_POPULARITY_COLUMNS:
        raise ValueError(
            f"Unexpected popularity columns: {popularity.columns.tolist()}"
        )

    expected_dtypes = {
        "movie_id": str(train["movie_id"].dtype),
        "positive_count": "int64",
    }

    if popularity.dtypes.astype(str).to_dict() != expected_dtypes:
        raise TypeError("Unexpected train item-popularity dtypes")

    if popularity.empty or popularity.isna().any().any():
        raise ValueError("Train item popularity must be non-empty and non-null")

    if popularity["movie_id"].duplicated().any():
        raise ValueError("Train item popularity contains duplicate movie_id")

    if int(popularity["positive_count"].sum()) != len(train):
        raise ValueError("Train item-popularity counts do not sum to train rows")

    expected = (
        train.groupby("movie_id", sort=False)
        .size()
        .rename("positive_count")
        .reset_index()
    )
    expected["positive_count"] = expected["positive_count"].astype("int64")
    expected = expected.sort_values(
        ["positive_count", "movie_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    _assert_frame_equal(
        popularity,
        expected,
        "Train item popularity has modified, missing, extra, or misordered rows",
    )


def _build_artifacts_unvalidated(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> EvaluationArtifacts:
    train_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )
    validation_cohort = _build_evaluation_context_unvalidated(
        train,
        validation,
        evaluation_split="validation",
        fit_scope="train",
    )
    test_cohort = _build_evaluation_context_unvalidated(
        train_validation,
        test,
        evaluation_split="test",
        fit_scope="train_validation",
    )
    cohorts = pd.concat(
        [validation_cohort, test_cohort],
        ignore_index=True,
    )

    train_user_mapping = build_contiguous_mapping(
        train,
        fit_scope="train",
        id_column="user_id",
        index_column="user_index",
    )
    train_validation_user_mapping = build_contiguous_mapping(
        train_validation,
        fit_scope="train_validation",
        id_column="user_id",
        index_column="user_index",
    )
    user_mapping = pd.concat(
        [train_user_mapping, train_validation_user_mapping],
        ignore_index=True,
    )
    train_item_mapping = build_contiguous_mapping(
        train,
        fit_scope="train",
        id_column="movie_id",
        index_column="item_index",
    )
    train_validation_item_mapping = build_contiguous_mapping(
        train_validation,
        fit_scope="train_validation",
        id_column="movie_id",
        index_column="item_index",
    )
    item_mapping = pd.concat(
        [train_item_mapping, train_validation_item_mapping],
        ignore_index=True,
    )
    return EvaluationArtifacts(
        evaluation_cohorts=cohorts,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        item_popularity_train=build_item_popularity_train(train),
    )


def validate_evaluation_artifacts(
    artifacts: EvaluationArtifacts,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Recompute and compare every output to detect any content mutation."""

    validate_source_splits(train, validation, test)
    validate_cohort_table(
        artifacts.evaluation_cohorts,
        user_dtype=str(train["user_id"].dtype),
    )
    train_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    for scope, fit in (
        ("train", train),
        ("train_validation", train_validation),
    ):
        user_scope = artifacts.user_mapping.loc[
            artifacts.user_mapping["fit_scope"].eq(scope)
        ].reset_index(drop=True)
        expected_user_ids = (
            fit["user_id"]
            .drop_duplicates()
            .sort_values(kind="mergesort")
            .reset_index(drop=True)
        )
        validate_mapping_table(
            user_scope,
            fit_scope=scope,
            id_column="user_id",
            index_column="user_index",
            expected_ids=expected_user_ids,
        )
        item_scope = artifacts.item_mapping.loc[
            artifacts.item_mapping["fit_scope"].eq(scope)
        ].reset_index(drop=True)
        expected_item_ids = (
            fit["movie_id"]
            .drop_duplicates()
            .sort_values(kind="mergesort")
            .reset_index(drop=True)
        )
        validate_mapping_table(
            item_scope,
            fit_scope=scope,
            id_column="movie_id",
            index_column="item_index",
            expected_ids=expected_item_ids,
        )

    validate_item_popularity(
        artifacts.item_popularity_train,
        train=train,
    )
    expected = _build_artifacts_unvalidated(train, validation, test)

    for field_name in EvaluationArtifacts.__dataclass_fields__:
        _assert_frame_equal(
            getattr(artifacts, field_name),
            getattr(expected, field_name),
            f"{field_name} has a modified, missing, or extra output row",
        )


def build_evaluation_artifacts(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> EvaluationArtifacts:
    """Build and validate both contexts, mappings, and train popularity."""

    validate_source_splits(train, validation, test)
    artifacts = _build_artifacts_unvalidated(train, validation, test)
    validate_evaluation_artifacts(artifacts, train, validation, test)
    return artifacts


def _count_values_with_zeros(
    series: pd.Series,
    values: tuple[str, ...],
) -> dict[str, int]:
    counts = series.value_counts()
    return {value: int(counts.get(value, 0)) for value in values}


def build_cohorts_manifest_section(
    artifacts: EvaluationArtifacts,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    source_paths: dict[str, str],
    source_sha256: dict[str, str],
    output_paths: dict[str, str],
    output_sha256: dict[str, str],
    parquet_engine: str,
    compression: str,
) -> dict[str, Any]:
    """Build deterministic DATA-006 metadata from validated real artifacts."""

    validate_evaluation_artifacts(artifacts, train, validation, test)
    train_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )
    contexts: dict[str, dict[str, Any]] = {}

    for evaluation_split, fit_scope, fit, target in (
        ("validation", "train", train, validation),
        ("test", "train_validation", train_validation, test),
    ):
        cohort = artifacts.evaluation_cohorts.loc[
            artifacts.evaluation_cohorts["evaluation_split"].eq(evaluation_split)
        ]
        fit_items = pd.Index(fit["movie_id"].unique())
        warm_mask = target["movie_id"].isin(fit_items)
        eligible = cohort.loc[cohort["is_eligible"]]
        contexts[evaluation_split] = {
            "fit_scope": fit_scope,
            "fit_paths": (
                [source_paths["train"]]
                if fit_scope == "train"
                else [source_paths["train"], source_paths["validation"]]
            ),
            "target_path": source_paths[evaluation_split],
            "fit_interaction_count": len(fit),
            "fit_user_count": int(fit["user_id"].nunique()),
            "fit_item_count": int(fit["movie_id"].nunique()),
            "target_interaction_count": len(target),
            "target_user_count": len(cohort),
            "eligible_user_count": int(cohort["is_eligible"].sum()),
            "excluded_user_count": int((~cohort["is_eligible"]).sum()),
            "warm_start_user_count": int((~cohort["is_cold_start_user"]).sum()),
            "cold_start_user_count": int(cohort["is_cold_start_user"].sum()),
            "warm_target_interaction_count": int(warm_mask.sum()),
            "cold_start_target_interaction_count": int((~warm_mask).sum()),
            "warm_target_unique_item_count": int(
                target.loc[warm_mask, "movie_id"].nunique()
            ),
            "cold_start_target_unique_item_count": int(
                target.loc[~warm_mask, "movie_id"].nunique()
            ),
            "exclusion_reason_counts": _count_values_with_zeros(
                cohort["exclusion_reason"],
                EXCLUSION_REASONS,
            ),
            "eligible_activity_group_counts": _count_values_with_zeros(
                eligible["activity_group"],
                ACTIVITY_GROUPS,
            ),
        }

    return {
        "policy": {
            "minimum_fit_positives": MINIMUM_FIT_POSITIVES,
            "minimum_warm_target_positives": 1,
            "target_item_must_exist_in_fit_catalog": True,
            "activity_groups": {
                "low": {"minimum": 5, "maximum": 9},
                "medium": {"minimum": 10, "maximum": 29},
                "high": {"minimum": 30, "maximum": None},
            },
            "mapping_index_base": 0,
            "mapping_id_order": "ascending",
        },
        "contexts": contexts,
        "mapping_counts": {
            scope: {
                "users": int(artifacts.user_mapping["fit_scope"].eq(scope).sum()),
                "items": int(artifacts.item_mapping["fit_scope"].eq(scope).sum()),
            }
            for scope in FIT_SCOPES
        },
        "item_popularity_train": {
            "row_count": len(artifacts.item_popularity_train),
            "positive_count_sum": int(
                artifacts.item_popularity_train["positive_count"].sum()
            ),
        },
        "source_paths": source_paths,
        "source_sha256": source_sha256,
        "output_paths": output_paths,
        "output_row_counts": {
            "evaluation_cohorts": len(artifacts.evaluation_cohorts),
            "user_mapping": len(artifacts.user_mapping),
            "item_mapping": len(artifacts.item_mapping),
            "item_popularity_train": len(artifacts.item_popularity_train),
        },
        "output_sha256": output_sha256,
        "output_schemas": {
            "evaluation_cohorts": {
                "columns": artifacts.evaluation_cohorts.columns.tolist(),
                "dtypes": (artifacts.evaluation_cohorts.dtypes.astype(str).to_dict()),
            },
            "user_mapping": {
                "columns": artifacts.user_mapping.columns.tolist(),
                "dtypes": (artifacts.user_mapping.dtypes.astype(str).to_dict()),
            },
            "item_mapping": {
                "columns": artifacts.item_mapping.columns.tolist(),
                "dtypes": (artifacts.item_mapping.dtypes.astype(str).to_dict()),
            },
            "item_popularity_train": {
                "columns": (artifacts.item_popularity_train.columns.tolist()),
                "dtypes": (
                    artifacts.item_popularity_train.dtypes.astype(str).to_dict()
                ),
            },
        },
        "parquet_engine": parquet_engine,
        "compression": compression,
    }


def update_manifest_document(
    manifest: dict[str, Any],
    cohorts_and_mappings_section: dict[str, Any],
) -> dict[str, Any]:
    """Replace only DATA-006 metadata and preserve every unknown section."""

    if not isinstance(manifest, dict):
        raise TypeError("Data manifest must contain a JSON object")

    updated = deepcopy(manifest)
    updated["cohorts_and_mappings"] = deepcopy(cohorts_and_mappings_section)
    return updated
