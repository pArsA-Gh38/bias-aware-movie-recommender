from pathlib import Path

import pandas as pd
import pytest

from src.data.movielens_parser import (
    EXPECTED_MOVIES_ROW_COUNT,
    EXPECTED_RATINGS_ROW_COUNT,
    EXPECTED_USERS_ROW_COUNT,
    MOVIES_COLUMNS,
    MOVIES_DTYPES,
    RATINGS_COLUMNS,
    RATINGS_DTYPES,
    USERS_COLUMNS,
    USERS_DTYPES,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIRECTORY = REPOSITORY_ROOT / "data" / "processed"


@pytest.mark.parametrize(
    (
        "filename",
        "expected_row_count",
        "expected_columns",
        "expected_dtypes",
    ),
    [
        (
            "ratings.parquet",
            EXPECTED_RATINGS_ROW_COUNT,
            RATINGS_COLUMNS,
            RATINGS_DTYPES,
        ),
        (
            "movies.parquet",
            EXPECTED_MOVIES_ROW_COUNT,
            MOVIES_COLUMNS,
            MOVIES_DTYPES,
        ),
        (
            "users.parquet",
            EXPECTED_USERS_ROW_COUNT,
            USERS_COLUMNS,
            USERS_DTYPES,
        ),
    ],
)
@pytest.mark.realdata
def test_parquet_count_and_types(
    filename: str,
    expected_row_count: int,
    expected_columns: list[str],
    expected_dtypes: dict[str, str],
) -> None:
    dataframe = pd.read_parquet(PROCESSED_DIRECTORY / filename)

    assert len(dataframe) == expected_row_count
    assert dataframe.columns.tolist() == expected_columns
    assert dataframe.dtypes.astype(str).to_dict() == expected_dtypes
