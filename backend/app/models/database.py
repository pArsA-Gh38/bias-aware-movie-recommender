"""SQLAlchemy 2.x models for the MVP PostgreSQL persistence schema."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

UUID_TYPE = PostgreSQLUUID(as_uuid=True)


class User(Base):
    """Application user; removal is represented by ``is_active=False``."""

    __tablename__ = "users"
    __table_args__ = (CheckConstraint("email = lower(email)", name="email_lowercase"),)

    id: Mapped[UUID] = mapped_column(UUID_TYPE, primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )


class Movie(Base):
    """Canonical MovieLens catalog movie keyed directly by MovieID."""

    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    release_year: Mapped[int | None] = mapped_column(SmallInteger)
    genres: Mapped[list[str]] = mapped_column(
        ARRAY(Text).with_variant(JSON(), "sqlite"),
        nullable=False,
        server_default=text("'{}'::text[]"),
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )


class Rating(Base):
    """One current explicit rating per application user and movie."""

    __tablename__ = "ratings"
    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 5", name="rating_range"),
        UniqueConstraint("user_id", "movie_id", name="uq_ratings_user_movie"),
        Index("ix_ratings_user_id", "user_id"),
        Index("ix_ratings_movie_id", "movie_id"),
    )

    id: Mapped[UUID] = mapped_column(UUID_TYPE, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT", onupdate="CASCADE"), nullable=False
    )
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="RESTRICT", onupdate="CASCADE"), nullable=False
    )
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ModelVersion(Base):
    """Registered, immutable model artifact version metadata."""

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
        Index(
            "uq_model_versions_single_active",
            "status",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(UUID_TYPE, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    data_hash: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict[str, object]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"), nullable=False
    )
    metrics: Mapped[dict[str, object]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"), nullable=False
    )
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class RecommendationEvent(Base):
    """One persisted recommendation request and its serving parameters."""

    __tablename__ = "recommendation_events"
    __table_args__ = (
        CheckConstraint("lambda BETWEEN 0 AND 1", name="lambda_range"),
        CheckConstraint("beta BETWEEN 0 AND 1", name="beta_range"),
        CheckConstraint("k > 0", name="positive_k"),
        Index(
            "ix_recommendation_events_user_created_at",
            "user_id",
            text("created_at DESC"),
        ),
    )

    id: Mapped[UUID] = mapped_column(UUID_TYPE, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT", onupdate="CASCADE"), nullable=False
    )
    model_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT", onupdate="CASCADE"),
        nullable=False,
    )
    strategy: Mapped[str] = mapped_column(Text, nullable=False)
    lambda_: Mapped[Decimal] = mapped_column("lambda", Numeric(5, 4), nullable=False)
    beta: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    k: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_count: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class RecommendationItem(Base):
    """Ranked catalog movie emitted for one recommendation event."""

    __tablename__ = "recommendation_items"
    __table_args__ = (
        UniqueConstraint("event_id", "rank", name="uq_recommendation_items_event_rank"),
        UniqueConstraint(
            "event_id", "movie_id", name="uq_recommendation_items_event_movie"
        ),
        Index("ix_recommendation_items_event_rank", "event_id", "rank"),
    )

    id: Mapped[UUID] = mapped_column(UUID_TYPE, primary_key=True, default=uuid4)
    event_id: Mapped[UUID] = mapped_column(
        ForeignKey("recommendation_events.id", ondelete="RESTRICT", onupdate="CASCADE"),
        nullable=False,
    )
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="RESTRICT", onupdate="CASCADE"), nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    base_score: Mapped[float] = mapped_column(Float, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    popularity_group: Mapped[str] = mapped_column(Text, nullable=False)


class Feedback(Base):
    """User feedback whose movie must belong to the referenced event."""

    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("action IN ('click', 'like', 'dislike')", name="valid_action"),
        ForeignKeyConstraint(
            ["event_id", "movie_id"],
            ["recommendation_items.event_id", "recommendation_items.movie_id"],
            name="fk_feedback_event_movie_recommendation_items",
            ondelete="RESTRICT",
            onupdate="CASCADE",
        ),
        Index("ix_feedback_user_created_at", "user_id", text("created_at DESC")),
    )

    id: Mapped[UUID] = mapped_column(UUID_TYPE, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT", onupdate="CASCADE"), nullable=False
    )
    event_id: Mapped[UUID] = mapped_column(
        ForeignKey("recommendation_events.id", ondelete="RESTRICT", onupdate="CASCADE"),
        nullable=False,
    )
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="RESTRICT", onupdate="CASCADE"), nullable=False
    )
    action: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
