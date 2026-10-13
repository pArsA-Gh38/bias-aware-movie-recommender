"""Database engine and session construction."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings


def create_database_engine(
    settings: Settings | None = None,
    **engine_options: object,
) -> Engine:
    """Create an engine from the environment-backed database URL."""
    configured = settings or Settings.from_environment()
    if not configured.database_url:
        raise RuntimeError("MOVIE_RECOMMENDER_DATABASE_URL is required")
    if configured.database_url.startswith("postgresql"):
        connect_args = dict(engine_options.pop("connect_args", {}))  # type: ignore[arg-type]
        connect_args.setdefault("connect_timeout", 5)
        engine_options["connect_args"] = connect_args
    engine = create_engine(configured.database_url, **engine_options)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create the application's explicit SQLAlchemy 2.x session factory."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
