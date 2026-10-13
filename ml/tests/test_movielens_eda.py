from pathlib import Path

import pandas as pd
import pytest

from src.data.movielens_eda import (
    ACTIVITY_SUMMARY_COLUMNS,
    DATA_QUALITY_COLUMNS,
    MOVIE_POPULARITY_COLUMNS,
    RATING_DISTRIBUTION_COLUMNS,
    TIMELINE_COLUMNS,
    analyze_movielens,
    calculate_density_sparsity,
    calculate_movie_popularity,
    calculate_rating_distribution,
    calculate_ratings_timeline,
    calculate_user_activity,
    summarize_user_activity,
)
from src.data.movielens_parser import (
    EXPECTED_MOVIES_ROW_COUNT,
    EXPECTED_RATINGS_ROW_COUNT,
    EXPECTED_USERS_ROW_COUNT,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIRECTORY = REPOSITORY_ROOT / "data" / "processed"
TABLES_DIRECTORY = REPOSITORY_ROOT / "reports" / "tables"
ENCODINGS = {
    "ratings": "ascii",
    "movies": "latin-1",
    "users": "ascii",
}


@pytest.fixture
def small_movielens() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "movie_id": [10, 20, 10, 30],
            "rating": [5, 4, 3, 1],
            "timestamp": [0, 86_400, 2_678_400, 2_764_800],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [10, 20, 30, 40],
            "title": ["A", "B", "C", "D"],
            "genres": ["Drama", "Comedy", "Drama", "Action"],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "gender": ["F", "M", "F"],
            "age": [25, 35, 18],
            "occupation": [1, 2, 3],
            "zip_code": ["10000", "20000", "30000"],
        }
    )
    return ratings, movies, users


def test_density_and_sparsity_are_correct() -> None:
    density, sparsity = calculate_density_sparsity(4, 3, 4)

    assert density == pytest.approx(1 / 3)
    assert sparsity == pytest.approx(2 / 3)
    assert density + sparsity == pytest.approx(1.0)


def test_rating_distribution_counts_and_percentages(
    small_movielens: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    ratings, _, _ = small_movielens
    distribution = calculate_rating_distribution(ratings)

    assert distribution.columns.tolist() == RATING_DISTRIBUTION_COLUMNS
    assert distribution["count"].tolist() == [1, 0, 1, 1, 1]
    assert distribution["percentage"].tolist() == pytest.approx(
        [25.0, 0.0, 25.0, 25.0, 25.0]
    )
    assert distribution["cumulative_percentage"].tolist() == pytest.approx(
        [25.0, 25.0, 50.0, 75.0, 100.0]
    )
    assert int(distribution["count"].sum()) == len(ratings)


def test_movie_popularity_includes_unrated_and_breaks_ties_by_movie_id(
    small_movielens: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    ratings, movies, _ = small_movielens
    popularity = calculate_movie_popularity(ratings, movies)

    assert popularity.columns.tolist() == MOVIE_POPULARITY_COLUMNS
    assert popularity["movie_id"].tolist() == [10, 20, 30, 40]
    assert popularity["rating_count"].tolist() == [2, 1, 1, 0]
    assert popularity["popularity_rank"].tolist() == [1, 2, 3, 4]
    assert int(popularity["rating_count"].eq(0).sum()) == 1


def test_user_activity_summary_is_correct(
    small_movielens: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    ratings, _, users = small_movielens
    activity = calculate_user_activity(ratings, users)
    summary = summarize_user_activity(activity).set_index("metric")["value"]

    assert activity["rating_count"].tolist() == [2, 1, 1]
    assert summary["minimum"] == 1
    assert summary["maximum"] == 2
    assert summary["mean"] == pytest.approx(4 / 3)
    assert summary["median"] == 1
    assert summary["standard_deviation"] == pytest.approx(1 / 3**0.5)
    assert summary["percentile_25"] == 1
    assert summary["percentile_75"] == pytest.approx(1.5)


def test_timestamp_conversion_and_monthly_buckets_are_utc(
    small_movielens: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    ratings, movies, users = small_movielens
    analysis = analyze_movielens(ratings, movies, users, ENCODINGS)
    timeline = calculate_ratings_timeline(ratings)

    assert analysis.metrics["earliest_timestamp"] == 0
    assert analysis.metrics["earliest_datetime_utc"] == ("1970-01-01T00:00:00+00:00")
    assert analysis.metrics["latest_datetime_utc"] == ("1970-02-02T00:00:00+00:00")
    assert timeline.columns.tolist() == TIMELINE_COLUMNS
    assert timeline["month_utc"].tolist() == ["1970-01", "1970-02"]
    assert timeline["rating_count"].tolist() == [2, 2]
    assert int(timeline["rating_count"].sum()) == len(ratings)


def test_analysis_outputs_have_stable_schemas_and_no_nulls(
    small_movielens: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    ratings, movies, users = small_movielens
    analysis = analyze_movielens(ratings, movies, users, ENCODINGS)
    outputs = [
        analysis.data_quality_summary,
        analysis.rating_distribution,
        analysis.user_activity_summary,
        analysis.movie_popularity_summary,
        analysis.movie_popularity,
        analysis.ratings_timeline,
    ]

    assert analysis.data_quality_summary.columns.tolist() == (DATA_QUALITY_COLUMNS)
    assert analysis.user_activity_summary.columns.tolist() == (ACTIVITY_SUMMARY_COLUMNS)
    assert analysis.movie_popularity_summary.columns.tolist() == (
        ACTIVITY_SUMMARY_COLUMNS
    )
    assert all(not output.isna().any().any() for output in outputs)


@pytest.mark.realdata
def test_real_movielens_eda_artifacts_match_parsed_data() -> None:
    ratings = pd.read_parquet(PROCESSED_DIRECTORY / "ratings.parquet")
    movies = pd.read_parquet(PROCESSED_DIRECTORY / "movies.parquet")
    users = pd.read_parquet(PROCESSED_DIRECTORY / "users.parquet")
    quality = pd.read_csv(
        TABLES_DIRECTORY / "data_quality_summary.csv",
        dtype={"value": "string"},
    )
    distribution = pd.read_csv(TABLES_DIRECTORY / "rating_distribution.csv")
    timeline = pd.read_csv(TABLES_DIRECTORY / "ratings_timeline.csv")

    def summary_value(metric: str) -> str:
        values = quality.loc[
            quality["metric"].eq(metric),
            "value",
        ]
        assert len(values) == 1
        return str(values.iloc[0])

    assert len(ratings) == EXPECTED_RATINGS_ROW_COUNT == 1_000_209
    assert len(movies) == EXPECTED_MOVIES_ROW_COUNT == 3_883
    assert len(users) == EXPECTED_USERS_ROW_COUNT == 6_040
    assert int(summary_value("rating_count")) == len(ratings)
    assert int(summary_value("movie_count")) == len(movies)
    assert int(summary_value("user_count")) == len(users)
    assert int(distribution["count"].sum()) == len(ratings)
    assert int(timeline["rating_count"].sum()) == len(ratings)
    assert float(summary_value("density")) + float(
        summary_value("sparsity")
    ) == pytest.approx(1.0)
