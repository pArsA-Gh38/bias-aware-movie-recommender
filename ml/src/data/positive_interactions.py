import pandas as pd

MINIMUM_POSITIVE_RATING = 4
POSITIVE_RATING_VALUES = (4, 5)
POSITIVE_COLUMNS = [
    "user_id",
    "movie_id",
    "rating",
    "timestamp",
]
POSITIVE_DTYPES = {
    "user_id": "int32",
    "movie_id": "int32",
    "rating": "int8",
    "timestamp": "int64",
}
POSITIVE_SORT_COLUMNS = [
    "timestamp",
    "user_id",
    "movie_id",
]


def _validate_minimum_positive_rating(
    minimum_positive_rating: int,
) -> None:
    if (
        isinstance(minimum_positive_rating, bool)
        or not isinstance(minimum_positive_rating, int)
        or minimum_positive_rating != MINIMUM_POSITIVE_RATING
    ):
        raise ValueError("minimum_positive_rating must be the project-approved value 4")


def _validate_ratings_source(ratings: pd.DataFrame) -> None:
    if ratings.columns.tolist() != POSITIVE_COLUMNS:
        raise ValueError(f"Unexpected ratings columns: {ratings.columns.tolist()}")

    null_counts = ratings.isna().sum()

    if int(null_counts.sum()) != 0:
        raise ValueError(f"Ratings contain null values: {null_counts.to_dict()}")

    actual_dtypes = ratings.dtypes.astype(str).to_dict()

    if actual_dtypes != POSITIVE_DTYPES:
        raise ValueError(f"Unexpected ratings dtypes: {actual_dtypes}")

    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Ratings contain values outside the range 1 to 5")

    duplicate_count = int(ratings.duplicated(["user_id", "movie_id"]).sum())

    if duplicate_count != 0:
        raise ValueError(
            f"Ratings contain {duplicate_count} duplicate user-movie pairs"
        )


def _sorted_positive_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.sort_values(
        POSITIVE_SORT_COLUMNS,
        kind="mergesort",
    ).reset_index(drop=True)


def build_positive_interactions(
    ratings: pd.DataFrame,
    *,
    minimum_positive_rating: int = MINIMUM_POSITIVE_RATING,
) -> pd.DataFrame:
    """Select rating values 4 and 5 and return them in a stable order."""

    _validate_minimum_positive_rating(minimum_positive_rating)
    _validate_ratings_source(ratings)

    positives = ratings.loc[
        ratings["rating"].ge(minimum_positive_rating),
        POSITIVE_COLUMNS,
    ].copy()
    positives = _sorted_positive_rows(positives)

    validate_positive_interactions(
        positives,
        ratings,
        minimum_positive_rating=minimum_positive_rating,
    )
    return positives


def validate_positive_interactions(
    positives: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    minimum_positive_rating: int = MINIMUM_POSITIVE_RATING,
) -> None:
    """Validate schema, rule, source equality, and deterministic ordering."""

    _validate_minimum_positive_rating(minimum_positive_rating)
    _validate_ratings_source(ratings)

    if positives.columns.tolist() != POSITIVE_COLUMNS:
        raise ValueError(f"Unexpected positive columns: {positives.columns.tolist()}")

    null_counts = positives.isna().sum()

    if int(null_counts.sum()) != 0:
        raise ValueError(
            f"Positive interactions contain null values: {null_counts.to_dict()}"
        )

    actual_dtypes = positives.dtypes.astype(str).to_dict()

    if actual_dtypes != POSITIVE_DTYPES:
        raise ValueError(f"Unexpected positive interaction dtypes: {actual_dtypes}")

    if not positives["rating"].isin(POSITIVE_RATING_VALUES).all():
        raise ValueError(
            "Positive interactions must contain only rating values 4 and 5"
        )

    duplicate_count = int(positives.duplicated(["user_id", "movie_id"]).sum())

    if duplicate_count != 0:
        raise ValueError(
            "Positive interactions contain "
            f"{duplicate_count} duplicate user-movie pairs"
        )

    expected = _sorted_positive_rows(
        ratings.loc[
            ratings["rating"].ge(minimum_positive_rating),
            POSITIVE_COLUMNS,
        ].copy()
    )

    if len(positives) != len(expected):
        raise ValueError(
            "Positive interaction row count does not match the source filter: "
            f"{len(positives)} != {len(expected)}"
        )

    try:
        pd.testing.assert_frame_equal(
            positives,
            expected,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as error:
        raise ValueError(
            "Positive interactions are not the exact, deterministically "
            "ordered source subset"
        ) from error
