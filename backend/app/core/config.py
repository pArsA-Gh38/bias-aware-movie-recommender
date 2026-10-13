"""Environment-backed configuration for the API application shell."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration needed to construct the API without external services."""

    app_name: str = "Movie Recommender API"
    app_version: str = "0.1.0"
    database_url: str | None = None
    auth_secret_key: str | None = None
    auth_access_token_expire_minutes: int | None = 30
    artifact_path: Path | None = None
    exp003_path: Path | None = None

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Load supported overrides without reading files or contacting services."""
        source = os.environ if environ is None else environ
        defaults = cls()
        raw_expiration = source.get(
            "MOVIE_RECOMMENDER_AUTH_ACCESS_TOKEN_EXPIRE_MINUTES"
        )
        expiration: int | None = defaults.auth_access_token_expire_minutes
        if raw_expiration is not None:
            try:
                parsed_expiration = int(raw_expiration)
            except ValueError:
                expiration = None
            else:
                expiration = (
                    parsed_expiration if 1 <= parsed_expiration <= 1_440 else None
                )

        return cls(
            app_name=source.get("MOVIE_RECOMMENDER_APP_NAME", defaults.app_name),
            app_version=source.get(
                "MOVIE_RECOMMENDER_APP_VERSION", defaults.app_version
            ),
            database_url=source.get("MOVIE_RECOMMENDER_DATABASE_URL"),
            auth_secret_key=source.get("MOVIE_RECOMMENDER_AUTH_SECRET_KEY"),
            auth_access_token_expire_minutes=expiration,
            artifact_path=(
                Path(raw_path)
                if (raw_path := source.get("MOVIE_RECOMMENDER_ARTIFACT_PATH"))
                else None
            ),
            exp003_path=(
                Path(raw_path)
                if (raw_path := source.get("MOVIE_RECOMMENDER_EXP003_PATH"))
                else None
            ),
        )
