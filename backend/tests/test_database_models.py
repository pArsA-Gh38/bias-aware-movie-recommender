"""Focused metadata/configuration tests for DB-001."""

from __future__ import annotations

from sqlalchemy import Integer
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

import app.models  # noqa: F401
from app.core.config import Settings
from app.db.base import Base

EXPECTED_TABLES = {
    "users",
    "movies",
    "ratings",
    "model_versions",
    "recommendation_events",
    "recommendation_items",
    "feedback",
}


def test_settings_load_database_url_only_from_explicit_environment() -> None:
    settings = Settings.from_environment(
        {"MOVIE_RECOMMENDER_DATABASE_URL": "postgresql+psycopg://db/test"}
    )

    assert settings.database_url == "postgresql+psycopg://db/test"
    assert Settings.from_environment({}).database_url is None


def test_model_metadata_is_importable_without_database_connection() -> None:
    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_postgresql_specific_types_are_declared() -> None:
    tables = Base.metadata.tables

    assert isinstance(tables["users"].c.id.type, UUID)
    assert isinstance(tables["movies"].c.genres.type, ARRAY)
    assert isinstance(tables["model_versions"].c.config.type, JSONB)
    assert isinstance(tables["model_versions"].c.metrics.type, JSONB)


def test_movie_id_is_the_only_integer_identity_and_has_no_surrogate_mapping() -> None:
    movie = Base.metadata.tables["movies"]

    assert list(movie.primary_key.columns.keys()) == ["id"]
    assert isinstance(movie.c.id.type, Integer)
    assert movie.c.id.autoincrement is False
    assert "movielens_id" not in movie.c


def test_required_table_constraints_and_indexes_are_present_in_metadata() -> None:
    tables = Base.metadata.tables
    constraint_names = {
        constraint.name
        for table in tables.values()
        for constraint in table.constraints
        if constraint.name
    }
    index_names = {index.name for table in tables.values() for index in table.indexes}

    assert {
        "uq_ratings_user_movie",
        "uq_model_versions_name_version",
        "uq_recommendation_items_event_rank",
        "uq_recommendation_items_event_movie",
        "fk_feedback_event_movie_recommendation_items",
    } <= constraint_names
    assert {
        "ix_ratings_user_id",
        "ix_ratings_movie_id",
        "uq_model_versions_single_active",
        "ix_recommendation_events_user_created_at",
        "ix_recommendation_items_event_rank",
        "ix_feedback_user_created_at",
    } <= index_names
