"""Register the configured ART-002 bundle for local recommendation persistence."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings
from app.db.session import create_database_engine
from app.models import ModelVersion

MODEL_NAME = "implicit_als"
MODEL_VERSION = "art-002-v2"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelVersionSeedError(RuntimeError):
    """The serving artifact cannot be safely registered."""


@dataclass(frozen=True, slots=True)
class ArtifactRegistration:
    path: Path
    artifact_sha256: str
    data_hash: str
    config: dict[str, Any]


def read_registration(artifact_path: Path) -> ArtifactRegistration:
    path = artifact_path.expanduser().resolve()
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
        serving_config = json.loads(
            (path / "serving_config.json").read_text(encoding="utf-8")
        )
        artifact_sha256 = manifest["bundle_sha256"]
        data_hash = provenance["fit_positive_counts"]["semantic_sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ModelVersionSeedError(
            f"invalid ART-002 registration metadata under {path}: {exc}"
        ) from exc
    if manifest.get("schema_version") != 2:
        raise ModelVersionSeedError(
            "only an ART-002 schema-v2 bundle can be registered"
        )
    if not SHA256_PATTERN.fullmatch(
        str(artifact_sha256)
    ) or not SHA256_PATTERN.fullmatch(str(data_hash)):
        raise ModelVersionSeedError("ART-002 registration hashes must be SHA-256 hex")
    if not isinstance(serving_config, dict):
        raise ModelVersionSeedError("ART-002 serving configuration must be an object")
    return ArtifactRegistration(
        path=path,
        artifact_sha256=str(artifact_sha256),
        data_hash=str(data_hash),
        config=serving_config,
    )


def seed_model_version(engine: Engine, registration: ArtifactRegistration) -> str:
    """Insert the active model once and fail closed on conflicting metadata."""
    with engine.begin() as connection:
        model_versions = ModelVersion.__table__
        existing = (
            connection.execute(
                select(model_versions).where(
                    ModelVersion.name == MODEL_NAME,
                    ModelVersion.version == MODEL_VERSION,
                )
            )
            .mappings()
            .one_or_none()
        )
        if existing is not None:
            if (
                existing["artifact_sha256"] != registration.artifact_sha256
                or existing["data_hash"] != registration.data_hash
            ):
                raise ModelVersionSeedError(
                    "registered ART-002 v2 hashes conflict with the configured artifact"
                )
            if existing["status"] != "active":
                raise ModelVersionSeedError(
                    "ART-002 v2 is registered but is not the active model"
                )
            return "already_registered"

        active = connection.execute(
            select(ModelVersion.id).where(ModelVersion.status == "active")
        ).scalar_one_or_none()
        if active is not None:
            raise ModelVersionSeedError(
                "another model version is active; refusing to change model state"
            )
        connection.execute(
            model_versions.insert().values(
                name=MODEL_NAME,
                version=MODEL_VERSION,
                artifact_path=str(registration.path),
                artifact_sha256=registration.artifact_sha256,
                data_hash=registration.data_hash,
                config=registration.config,
                metrics={},
                status="active",
            )
        )
    return "registered"


def main() -> int:
    settings = Settings.from_environment()
    if settings.artifact_path is None:
        print(
            "Model registration failed: MOVIE_RECOMMENDER_ARTIFACT_PATH is required",
            file=sys.stderr,
        )
        return 1
    engine: Engine | None = None
    try:
        registration = read_registration(settings.artifact_path)
        engine = create_database_engine(settings)
        result = seed_model_version(engine, registration)
    except (ModelVersionSeedError, RuntimeError, SQLAlchemyError) as exc:
        print(f"Model registration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    print(f"artifact_path={registration.path}")
    print(f"model_version={MODEL_VERSION}")
    print(f"status={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
