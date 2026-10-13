from pathlib import Path

import pandas as pd
import pytest

from src.data.positive_interactions import (
    MINIMUM_POSITIVE_RATING,
    POSITIVE_COLUMNS,
    POSITIVE_DTYPES,
    build_positive_interactions,
    validate_positive_interactions,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIRECTORY = REPOSITORY_ROOT / "data" / "processed"


@pytest.fixture
def rating_scale_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": pd.Series([5, 4, 3, 2, 1], dtype="int32"),
            "movie_id": pd.Series(
                [50, 40, 30, 20, 10],
                dtype="int32",
            ),
            "rating": pd.Series([1, 2, 3, 4, 5], dtype="int8"),
            "timestamp": pd.Series(
                [500, 400, 300, 200, 100],
                dtype="int64",
            ),
        }
    )


def test_rating_values_four_and_five_are_the_exact_positive_set(
    rating_scale_fixture: pd.DataFrame,
) -> None:
    positives = build_positive_interactions(rating_scale_fixture)

    assert positives["rating"].tolist() == [5, 4]
    assert set(positives["rating"]) == {4, 5}
    assert 3 not in positives["rating"].array
    assert 2 not in positives["rating"].array
    assert 1 not in positives["rating"].array


def test_source_values_columns_and_dtypes_are_preserved(
    rating_scale_fixture: pd.DataFrame,
) -> None:
    positives = build_positive_interactions(rating_scale_fixture)

    assert positives.columns.tolist() == POSITIVE_COLUMNS
    assert positives.dtypes.astype(str).to_dict() == POSITIVE_DTYPES
    assert positives.to_dict("records") == [
        {
            "user_id": 1,
            "movie_id": 10,
            "rating": 5,
            "timestamp": 100,
        },
        {
            "user_id": 2,
            "movie_id": 20,
            "rating": 4,
            "timestamp": 200,
        },
    ]


def test_sort_is_deterministic_and_breaks_timestamp_ties() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": pd.Series([2, 1, 1, 3], dtype="int32"),
            "movie_id": pd.Series([5, 9, 4, 1], dtype="int32"),
            "rating": pd.Series([4, 5, 4, 5], dtype="int8"),
            "timestamp": pd.Series([10, 10, 10, 5], dtype="int64"),
        }
    )

    first = build_positive_interactions(ratings)
    second = build_positive_interactions(ratings.sample(frac=1, random_state=7))

    assert first[["timestamp", "user_id", "movie_id"]].to_records(
        index=False
    ).tolist() == [
        (5, 3, 1),
        (10, 1, 4),
        (10, 1, 9),
        (10, 2, 5),
    ]
    pd.testing.assert_frame_equal(first, second)


def test_null_source_value_is_rejected(
    rating_scale_fixture: pd.DataFrame,
) -> None:
    invalid = rating_scale_fixture.astype({"user_id": "Int32"})
    invalid.loc[0, "user_id"] = pd.NA

    with pytest.raises(ValueError, match="null"):
        build_positive_interactions(invalid)


def test_duplicate_source_user_movie_is_rejected(
    rating_scale_fixture: pd.DataFrame,
) -> None:
    duplicate = pd.concat(
        [rating_scale_fixture, rating_scale_fixture.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_positive_interactions(duplicate)


def test_duplicate_positive_user_movie_is_rejected(
    rating_scale_fixture: pd.DataFrame,
) -> None:
    positives = build_positive_interactions(rating_scale_fixture)
    duplicate = pd.concat(
        [positives, positives.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_positive_interactions(duplicate, rating_scale_fixture)


@pytest.mark.parametrize("invalid_rating", [0, 6])
def test_source_rating_outside_valid_range_is_rejected(
    rating_scale_fixture: pd.DataFrame,
    invalid_rating: int,
) -> None:
    invalid = rating_scale_fixture.copy()
    invalid.loc[0, "rating"] = invalid_rating

    with pytest.raises(ValueError, match="outside the range"):
        build_positive_interactions(invalid)


def test_empty_source_returns_empty_frame_with_exact_schema() -> None:
    ratings = pd.DataFrame(
        {column: pd.Series(dtype=dtype) for column, dtype in POSITIVE_DTYPES.items()}
    )

    positives = build_positive_interactions(ratings)

    assert positives.empty
    assert positives.columns.tolist() == POSITIVE_COLUMNS
    assert positives.dtypes.astype(str).to_dict() == POSITIVE_DTYPES
    validate_positive_interactions(positives, ratings)


@pytest.mark.parametrize("invalid_threshold", [3, 5, True, 4.0])
def test_invalid_threshold_fails_fast(
    rating_scale_fixture: pd.DataFrame,
    invalid_threshold: object,
) -> None:
    with pytest.raises(ValueError, match="project-approved value 4"):
        build_positive_interactions(
            rating_scale_fixture,
            minimum_positive_rating=invalid_threshold,
        )


def test_validation_rejects_a_missing_positive_row(
    rating_scale_fixture: pd.DataFrame,
) -> None:
    positives = build_positive_interactions(rating_scale_fixture).iloc[[0]]

    with pytest.raises(ValueError, match="row count"):
        validate_positive_interactions(positives, rating_scale_fixture)


def test_validation_rejects_an_added_non_positive_row(
    rating_scale_fixture: pd.DataFrame,
) -> None:
    positives = build_positive_interactions(rating_scale_fixture)
    non_positive = rating_scale_fixture.loc[rating_scale_fixture["rating"].eq(3)]
    invalid = pd.concat([positives, non_positive], ignore_index=True)

    with pytest.raises(ValueError, match="only rating values 4 and 5"):
        validate_positive_interactions(invalid, rating_scale_fixture)


def test_validation_rejects_modified_source_values(
    rating_scale_fixture: pd.DataFrame,
) -> None:
    positives = build_positive_interactions(rating_scale_fixture)
    invalid = positives.copy()
    invalid.loc[0, "timestamp"] += 1

    with pytest.raises(ValueError, match="exact"):
        validate_positive_interactions(invalid, rating_scale_fixture)


def test_validation_rejects_non_deterministic_order(
    rating_scale_fixture: pd.DataFrame,
) -> None:
    positives = build_positive_interactions(rating_scale_fixture)
    invalid = positives.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="deterministically"):
        validate_positive_interactions(invalid, rating_scale_fixture)


def test_validation_rejects_wrong_columns_and_dtypes(
    rating_scale_fixture: pd.DataFrame,
) -> None:
    positives = build_positive_interactions(rating_scale_fixture)

    with pytest.raises(ValueError, match="columns"):
        validate_positive_interactions(
            positives[POSITIVE_COLUMNS[::-1]],
            rating_scale_fixture,
        )

    with pytest.raises(ValueError, match="dtypes"):
        validate_positive_interactions(
            positives.astype({"user_id": "int64"}),
            rating_scale_fixture,
        )


@pytest.mark.realdata
def test_real_positive_parquet_is_exact_source_subset() -> None:
    ratings = pd.read_parquet(PROCESSED_DIRECTORY / "ratings.parquet")
    positives = pd.read_parquet(PROCESSED_DIRECTORY / "positives.parquet")
    expected = build_positive_interactions(ratings)

    assert len(ratings) == 1_000_209
    assert len(positives) == int(ratings["rating"].ge(4).sum())
    assert len(positives) == int(
        ratings["rating"].eq(4).sum() + ratings["rating"].eq(5).sum()
    )
    assert set(positives["rating"]) == {4, 5}
    assert not positives.isna().any().any()
    assert not positives.duplicated(["user_id", "movie_id"]).any()
    pd.testing.assert_frame_equal(positives, expected)
    validate_positive_interactions(positives, ratings)


def test_default_threshold_is_the_approved_value() -> None:
    assert MINIMUM_POSITIVE_RATING == 4
