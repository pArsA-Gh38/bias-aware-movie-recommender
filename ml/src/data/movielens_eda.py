from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

DATA_QUALITY_COLUMNS = ["dataset", "category", "metric", "value"]
RATING_DISTRIBUTION_COLUMNS = [
    "rating",
    "count",
    "percentage",
    "cumulative_percentage",
]
ACTIVITY_SUMMARY_COLUMNS = ["metric", "value"]
MOVIE_POPULARITY_COLUMNS = [
    "popularity_rank",
    "movie_id",
    "rating_count",
]
TIMELINE_COLUMNS = ["month_utc", "rating_count", "percentage"]


@dataclass(frozen=True)
class MovieLensEDA:
    """Deterministic tables and scalar metrics for MovieLens EDA."""

    metrics: dict[str, int | float | str | bool]
    data_quality_summary: pd.DataFrame
    rating_distribution: pd.DataFrame
    user_activity_summary: pd.DataFrame
    movie_popularity_summary: pd.DataFrame
    movie_popularity: pd.DataFrame
    ratings_timeline: pd.DataFrame


def calculate_density_sparsity(
    rating_count: int,
    user_count: int,
    movie_count: int,
) -> tuple[float, float]:
    """Calculate explicit-rating matrix density and sparsity."""

    possible_interaction_count = user_count * movie_count

    if possible_interaction_count <= 0:
        raise ValueError("User and movie counts must both be positive")

    if rating_count < 0 or rating_count > possible_interaction_count:
        raise ValueError("Rating count must fit the user-movie matrix")

    density = rating_count / possible_interaction_count
    return density, 1.0 - density


def calculate_rating_distribution(
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Count ratings from one through five and calculate percentages."""

    counts = (
        ratings["rating"].value_counts().reindex(range(1, 6), fill_value=0).sort_index()
    )

    if int(counts.sum()) != len(ratings):
        raise ValueError("Rating distribution contains values outside 1 to 5")

    total = len(ratings)

    if total:
        percentages = counts.astype("float64").div(total).mul(100.0)
    else:
        percentages = counts.astype("float64")

    distribution = pd.DataFrame(
        {
            "rating": counts.index,
            "count": counts.to_numpy(),
            "percentage": percentages.to_numpy(),
            "cumulative_percentage": percentages.cumsum().to_numpy(),
        }
    )

    return distribution.astype(
        {
            "rating": "int8",
            "count": "int64",
        }
    )[RATING_DISTRIBUTION_COLUMNS]


def calculate_user_activity(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    """Count ratings for every catalog user, including inactive users."""

    rating_counts = ratings.groupby("user_id", sort=False).size()
    activity = users[["user_id"]].merge(
        rating_counts.rename("rating_count"),
        left_on="user_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    activity["rating_count"] = activity["rating_count"].fillna(0).astype("int64")
    return activity.sort_values("user_id", kind="mergesort").reset_index(drop=True)


def calculate_movie_popularity(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Count all ratings per movie with deterministic popularity ordering."""

    rating_counts = ratings.groupby("movie_id", sort=False).size()
    popularity = movies[["movie_id"]].merge(
        rating_counts.rename("rating_count"),
        left_on="movie_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    popularity["rating_count"] = popularity["rating_count"].fillna(0).astype("int64")
    popularity = popularity.sort_values(
        ["rating_count", "movie_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    popularity.insert(
        0,
        "popularity_rank",
        pd.Series(range(1, len(popularity) + 1), dtype="int64"),
    )
    return popularity[MOVIE_POPULARITY_COLUMNS]


def _distribution_statistics(
    counts: pd.Series,
) -> list[tuple[str, float]]:
    values = counts.astype("float64")
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    return [
        ("minimum", float(values.min())),
        ("maximum", float(values.max())),
        ("mean", float(values.mean())),
        ("median", float(values.median())),
        ("standard_deviation", standard_deviation),
        ("percentile_25", float(values.quantile(0.25))),
        ("percentile_75", float(values.quantile(0.75))),
    ]


def summarize_user_activity(
    user_activity: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize rating counts across all catalog users."""

    counts = user_activity["rating_count"]
    rows = [
        ("user_count", float(len(user_activity))),
        ("rated_user_count", float(counts.gt(0).sum())),
        ("unrated_user_count", float(counts.eq(0).sum())),
        *_distribution_statistics(counts),
    ]
    return pd.DataFrame(rows, columns=ACTIVITY_SUMMARY_COLUMNS)


def summarize_movie_popularity(
    movie_popularity: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize all-rating popularity across the complete movie catalog."""

    counts = movie_popularity["rating_count"]
    rows = [
        ("movie_count", float(len(movie_popularity))),
        ("rated_movie_count", float(counts.gt(0).sum())),
        ("unrated_movie_count", float(counts.eq(0).sum())),
        *_distribution_statistics(counts),
    ]
    return pd.DataFrame(rows, columns=ACTIVITY_SUMMARY_COLUMNS)


def calculate_ratings_timeline(
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate epoch timestamps into deterministic UTC calendar months."""

    datetimes = pd.to_datetime(
        ratings["timestamp"],
        unit="s",
        utc=True,
        errors="raise",
    )
    month_labels = datetimes.dt.strftime("%Y-%m")
    counts = month_labels.value_counts().sort_index()
    total = len(ratings)

    if total:
        percentages = counts.astype("float64").div(total).mul(100.0)
    else:
        percentages = counts.astype("float64")

    timeline = pd.DataFrame(
        {
            "month_utc": counts.index,
            "rating_count": counts.to_numpy(),
            "percentage": percentages.to_numpy(),
        }
    )
    timeline["rating_count"] = timeline["rating_count"].astype("int64")
    return timeline[TIMELINE_COLUMNS].reset_index(drop=True)


def _quality_row(
    dataset: str,
    category: str,
    metric: str,
    value: object,
) -> dict[str, str]:
    return {
        "dataset": dataset,
        "category": category,
        "metric": metric,
        "value": str(value),
    }


def build_data_quality_summary(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
    encodings: Mapping[str, str],
    metrics: Mapping[str, int | float | str | bool],
) -> pd.DataFrame:
    """Build a stable long-form table of size, schema, and quality checks."""

    rows = [
        _quality_row("all", "dataset_summary", key, metrics[key])
        for key in (
            "rating_count",
            "user_count",
            "movie_count",
            "rated_user_count",
            "rated_movie_count",
            "unrated_movie_count",
            "possible_interaction_count",
            "density",
            "sparsity",
            "earliest_timestamp",
            "latest_timestamp",
            "earliest_datetime_utc",
            "latest_datetime_utc",
        )
    ]

    datasets = {
        "ratings": ratings,
        "movies": movies,
        "users": users,
    }

    for dataset_name, dataframe in datasets.items():
        rows.append(
            _quality_row(
                dataset_name,
                "source",
                "encoding",
                encodings[dataset_name],
            )
        )
        rows.append(
            _quality_row(
                dataset_name,
                "schema",
                "columns",
                ",".join(dataframe.columns),
            )
        )

        for column_name, dtype in dataframe.dtypes.items():
            rows.append(
                _quality_row(
                    dataset_name,
                    "schema",
                    f"dtype.{column_name}",
                    dtype,
                )
            )

        null_counts = dataframe.isna().sum()
        rows.append(
            _quality_row(
                dataset_name,
                "quality",
                "null_count",
                int(null_counts.sum()),
            )
        )

        for column_name, null_count in null_counts.items():
            rows.append(
                _quality_row(
                    dataset_name,
                    "quality",
                    f"null_count.{column_name}",
                    int(null_count),
                )
            )

    cross_dataset_checks = [
        (
            "users",
            "duplicate_user_id_count",
            int(users["user_id"].duplicated().sum()),
        ),
        (
            "movies",
            "duplicate_movie_id_count",
            int(movies["movie_id"].duplicated().sum()),
        ),
        (
            "ratings",
            "duplicate_user_movie_count",
            int(ratings.duplicated(subset=["user_id", "movie_id"]).sum()),
        ),
        (
            "ratings",
            "invalid_rating_count",
            int((ratings["rating"].isna() | ~ratings["rating"].between(1, 5)).sum()),
        ),
        (
            "ratings",
            "missing_user_reference_count",
            int((~ratings["user_id"].isin(users["user_id"])).sum()),
        ),
        (
            "ratings",
            "missing_movie_reference_count",
            int((~ratings["movie_id"].isin(movies["movie_id"])).sum()),
        ),
    ]

    rows.extend(
        _quality_row(dataset, "quality", metric, value)
        for dataset, metric, value in cross_dataset_checks
    )
    return pd.DataFrame(rows, columns=DATA_QUALITY_COLUMNS)


def analyze_movielens(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
    encodings: Mapping[str, str],
) -> MovieLensEDA:
    """Compute the complete DATA-003 analysis without writing artifacts."""

    rating_count = len(ratings)
    user_count = len(users)
    movie_count = len(movies)
    possible_interaction_count = user_count * movie_count
    density, sparsity = calculate_density_sparsity(
        rating_count,
        user_count,
        movie_count,
    )
    user_activity = calculate_user_activity(ratings, users)
    movie_popularity = calculate_movie_popularity(ratings, movies)
    datetimes = pd.to_datetime(
        ratings["timestamp"],
        unit="s",
        utc=True,
        errors="raise",
    )

    metrics: dict[str, int | float | str | bool] = {
        "rating_count": rating_count,
        "user_count": user_count,
        "movie_count": movie_count,
        "rated_user_count": int(user_activity["rating_count"].gt(0).sum()),
        "rated_movie_count": int(movie_popularity["rating_count"].gt(0).sum()),
        "unrated_movie_count": int(movie_popularity["rating_count"].eq(0).sum()),
        "possible_interaction_count": possible_interaction_count,
        "density": density,
        "sparsity": sparsity,
        "earliest_timestamp": int(ratings["timestamp"].min()),
        "latest_timestamp": int(ratings["timestamp"].max()),
        "earliest_datetime_utc": datetimes.min().isoformat(),
        "latest_datetime_utc": datetimes.max().isoformat(),
        "all_users_have_at_least_20_ratings": bool(
            user_activity["rating_count"].min() >= 20
        ),
    }

    return MovieLensEDA(
        metrics=metrics,
        data_quality_summary=build_data_quality_summary(
            ratings,
            movies,
            users,
            encodings,
            metrics,
        ),
        rating_distribution=calculate_rating_distribution(ratings),
        user_activity_summary=summarize_user_activity(user_activity),
        movie_popularity_summary=summarize_movie_popularity(movie_popularity),
        movie_popularity=movie_popularity,
        ratings_timeline=calculate_ratings_timeline(ratings),
    )
