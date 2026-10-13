from pathlib import Path

import pandas as pd

EXPECTED_RATINGS_ROW_COUNT = 1_000_209

EXPECTED_MOVIES_ROW_COUNT = 3_883

EXPECTED_USERS_ROW_COUNT = 6_040

RATINGS_COLUMNS = [
    "user_id",
    "movie_id",
    "rating",
    "timestamp",
]

RATINGS_DTYPES = {
    "user_id": "int32",
    "movie_id": "int32",
    "rating": "int8",
    "timestamp": "int64",
}

MOVIES_COLUMNS = [
    "movie_id",
    "title",
    "genres",
]

MOVIES_DTYPES = {
    "movie_id": "int32",
    "title": "string",
    "genres": "string",
}

USERS_COLUMNS = [
    "user_id",
    "gender",
    "age",
    "occupation",
    "zip_code",
]

USERS_DTYPES = {
    "user_id": "int32",
    "gender": "string",
    "age": "int16",
    "occupation": "int16",
    "zip_code": "string",
}


def read_ratings(
    path: Path,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read MovieLens 1M ratings with an explicit schema."""

    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=RATINGS_COLUMNS,
        dtype=RATINGS_DTYPES,
        encoding="ascii",
        nrows=nrows,
    )


def read_users(
    path: Path,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read MovieLens 1M users with an explicit schema."""

    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=USERS_COLUMNS,
        dtype=USERS_DTYPES,
        encoding="ascii",
        nrows=nrows,
    )


def read_movies(
    path: Path,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read MovieLens 1M movies with an explicit schema."""

    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=MOVIES_COLUMNS,
        dtype=MOVIES_DTYPES,
        encoding="latin-1",
        nrows=nrows,
    )


def validate_users(users: pd.DataFrame) -> None:
    """Validate the parsed MovieLens users dataset."""

    if users.columns.tolist() != USERS_COLUMNS:
        raise ValueError(f"Unexpected users columns: {users.columns.tolist()}")

    actual_dtypes = users.dtypes.astype(str).to_dict()

    if actual_dtypes != USERS_DTYPES:
        raise ValueError(f"Unexpected users dtypes: {actual_dtypes}")

    if len(users) != EXPECTED_USERS_ROW_COUNT:
        raise ValueError(
            f"Unexpected users row count: {len(users)} != {EXPECTED_USERS_ROW_COUNT}"
        )

    null_counts = users.isna().sum()

    if int(null_counts.sum()) != 0:
        raise ValueError(f"Users contain null values: {null_counts.to_dict()}")

    duplicate_user_ids = int(users["user_id"].duplicated().sum())

    if duplicate_user_ids != 0:
        raise ValueError(f"Users contain {duplicate_user_ids} duplicate user IDs")

    if not users["gender"].isin(["F", "M"]).all():
        raise ValueError("Users contain unexpected gender values")

    expected_age_values = {1, 18, 25, 35, 45, 50, 56}
    actual_age_values = set(users["age"].unique().tolist())

    if not actual_age_values.issubset(expected_age_values):
        raise ValueError(f"Users contain unexpected age codes: {actual_age_values}")

    if not users["occupation"].between(0, 20).all():
        raise ValueError("Users contain occupation codes outside 0 to 20")

    if users["zip_code"].str.strip().eq("").any():
        raise ValueError("Users contain empty zip codes")


def validate_movies(movies: pd.DataFrame) -> None:
    """Validate the parsed MovieLens movies dataset."""

    if movies.columns.tolist() != MOVIES_COLUMNS:
        raise ValueError(f"Unexpected movies columns: {movies.columns.tolist()}")

    actual_dtypes = movies.dtypes.astype(str).to_dict()

    if actual_dtypes != MOVIES_DTYPES:
        raise ValueError(f"Unexpected movies dtypes: {actual_dtypes}")

    if len(movies) != EXPECTED_MOVIES_ROW_COUNT:
        raise ValueError(
            f"Unexpected movies row count: {len(movies)} != {EXPECTED_MOVIES_ROW_COUNT}"
        )

    null_counts = movies.isna().sum()

    if int(null_counts.sum()) != 0:
        raise ValueError(f"Movies contain null values: {null_counts.to_dict()}")

    duplicate_movie_ids = int(movies["movie_id"].duplicated().sum())

    if duplicate_movie_ids != 0:
        raise ValueError(f"Movies contain {duplicate_movie_ids} duplicate movie IDs")

    if movies["title"].str.strip().eq("").any():
        raise ValueError("Movies contain empty titles")

    if movies["genres"].str.strip().eq("").any():
        raise ValueError("Movies contain empty genres")


def validate_ratings(ratings: pd.DataFrame) -> None:
    """Validate the parsed MovieLens ratings dataset."""

    if ratings.columns.tolist() != RATINGS_COLUMNS:
        raise ValueError(f"Unexpected ratings columns: {ratings.columns.tolist()}")

    actual_dtypes = ratings.dtypes.astype(str).to_dict()

    if actual_dtypes != RATINGS_DTYPES:
        raise ValueError(f"Unexpected ratings dtypes: {actual_dtypes}")

    if len(ratings) != EXPECTED_RATINGS_ROW_COUNT:
        raise ValueError(
            "Unexpected ratings row count: "
            f"{len(ratings)} != {EXPECTED_RATINGS_ROW_COUNT}"
        )

    null_counts = ratings.isna().sum()

    if int(null_counts.sum()) != 0:
        raise ValueError(f"Ratings contain null values: {null_counts.to_dict()}")

    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Ratings contain values outside the range 1 to 5")

    duplicate_count = int(
        ratings.duplicated(
            subset=["user_id", "movie_id"],
        ).sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Ratings contain {duplicate_count} duplicate user-movie pairs"
        )


def validate_referential_integrity(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
) -> None:
    """Validate references from ratings to users and movies."""

    missing_user_count = int((~ratings["user_id"].isin(users["user_id"])).sum())

    if missing_user_count != 0:
        raise ValueError(
            f"Ratings contain {missing_user_count} unknown user references"
        )

    missing_movie_count = int((~ratings["movie_id"].isin(movies["movie_id"])).sum())

    if missing_movie_count != 0:
        raise ValueError(
            f"Ratings contain {missing_movie_count} unknown movie references"
        )
