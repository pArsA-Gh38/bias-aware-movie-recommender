"""Initialize local demo state without overwriting existing accounts or models."""

from __future__ import annotations

import json
from pathlib import Path

from recommender_core.local_artifact import load_local_bundle
from sqlalchemy import DefaultClause, MetaData, select, text

from app.core.config import Settings
from app.db.base import Base
from app.db.seed_catalog import validate_catalog_rows
from app.db.session import create_database_engine
from app.models import ModelVersion, Movie


def initialize(settings: Settings) -> None:
    if settings.artifact_path is None:
        raise ValueError("local artifact path is required")
    path = Path(settings.artifact_path)
    load_local_bundle(path)
    manifest = json.loads((path / "manifest.json").read_text())
    rows = json.loads((path / "catalog.json").read_text())
    catalog = validate_catalog_rows(
        [
            {
                "movie_id": row["movie_id"],
                "title": row["title"],
                "genres": "|".join(row["genres"]),
            }
            for row in rows
        ],
        expected_count=len(rows),
    )
    engine = create_database_engine(settings)
    try:
        if engine.dialect.name == "sqlite":
            schema = MetaData()
            for table in Base.metadata.sorted_tables:
                table.to_metadata(schema)
            schema.tables["movies"].c.genres.server_default = DefaultClause(
                text("'[]'")
            )
            schema.create_all(engine)
        else:
            from alembic.config import Config

            from alembic import command

            config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
            config.attributes["settings"] = settings
            config.set_main_option(
                "script_location", str(Path(__file__).resolve().parents[2] / "alembic")
            )
            command.upgrade(config, "head")
        with engine.begin() as connection:
            current = set(connection.execute(select(Movie.id)).scalars())
            expected = {row.id for row in catalog}
            if current and current != expected:
                raise ValueError("database catalog differs; use separate local state")
            if not current:
                connection.execute(
                    Movie.__table__.insert(), [row.database_values() for row in catalog]
                )
            active = (
                connection.execute(
                    select(ModelVersion.__table__).where(
                        ModelVersion.status == "active"
                    )
                )
                .mappings()
                .one_or_none()
            )
            if active is not None:
                if active["artifact_sha256"] != manifest["bundle_sha256"]:
                    raise ValueError(
                        "active model differs; refusing to overwrite database state"
                    )
            else:
                connection.execute(
                    ModelVersion.__table__.insert().values(
                        name="implicit_als",
                        version="local-" + manifest["bundle_sha256"][:16],
                        artifact_path=str(path.resolve()),
                        artifact_sha256=manifest["bundle_sha256"],
                        data_hash=manifest["provenance"]["training_matrix_sha256"],
                        config=manifest["provenance"],
                        metrics={},
                        status="active",
                    )
                )
    finally:
        engine.dispose()


if __name__ == "__main__":
    initialize(Settings.from_environment())
    print("Local catalog and model registration ready.")
