"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from recommender_core import ArtifactError
from sqlalchemy.exc import SQLAlchemyError

from app.api.auth import create_auth_router, create_current_user_dependency
from app.api.experiments import create_experiments_router
from app.api.feedback import create_feedback_router
from app.api.health import create_health_router
from app.api.movies import create_movies_router
from app.api.ratings import create_ratings_router
from app.api.recommendations import create_recommendations_router
from app.core.auth import AuthService
from app.core.config import Settings
from app.core.experiments import (
    CommittedExperimentSummaryProvider,
    ExperimentSummaryProvider,
)
from app.core.feedback import FeedbackService
from app.core.movies import MovieCatalogService
from app.core.ratings import RatingService
from app.core.readiness import (
    ReadinessProvider,
    RuntimeReadinessProvider,
)
from app.core.recommendations import RecommendationService
from app.core.security import PasswordManager, TokenManager
from app.db.auth import SQLAlchemyUserStore
from app.db.feedback import SQLAlchemyFeedbackRepository
from app.db.movies import SQLAlchemyMovieCatalogRepository
from app.db.ratings import SQLAlchemyRatingRepository
from app.db.recommendations import SQLAlchemyRecommendationRepository
from app.db.session import create_database_engine, create_session_factory
from app.recommendation.artifact_loader import ArtifactLoader


def _create_environment_auth_service(
    settings: Settings,
) -> AuthService | None:
    if (
        not settings.database_url
        or not settings.auth_secret_key
        or settings.auth_access_token_expire_minutes is None
    ):
        return None

    try:
        engine = create_database_engine(settings)

        users = SQLAlchemyUserStore(create_session_factory(engine))

        tokens = TokenManager(
            secret_key=settings.auth_secret_key,
            lifetime_minutes=settings.auth_access_token_expire_minutes,
        )

        return AuthService(
            users=users,
            passwords=PasswordManager(),
            tokens=tokens,
        )

    except (RuntimeError, SQLAlchemyError, ValueError):
        return None


def _create_environment_movie_catalog_service(
    settings: Settings,
) -> MovieCatalogService | None:
    if not settings.database_url:
        return None

    try:
        engine = create_database_engine(settings)

        repository = SQLAlchemyMovieCatalogRepository(create_session_factory(engine))

        return MovieCatalogService(repository)

    except (RuntimeError, SQLAlchemyError, ValueError):
        return None


def _create_environment_rating_service(
    settings: Settings,
) -> RatingService | None:
    if not settings.database_url:
        return None

    try:
        engine = create_database_engine(settings)

        repository = SQLAlchemyRatingRepository(create_session_factory(engine))

        return RatingService(repository)

    except (RuntimeError, SQLAlchemyError, ValueError):
        return None


def _create_environment_recommendation_service(
    settings: Settings,
    artifact_loader: ArtifactLoader | None,
) -> RecommendationService | None:
    if not settings.database_url or artifact_loader is None:
        return None
    try:
        engine = create_database_engine(settings)
        repository = SQLAlchemyRecommendationRepository(create_session_factory(engine))
        return RecommendationService(repository, artifact_loader)
    except (RuntimeError, SQLAlchemyError, ValueError):
        return None


def _create_environment_feedback_service(
    settings: Settings,
) -> FeedbackService | None:
    if not settings.database_url:
        return None
    try:
        engine = create_database_engine(settings)
        repository = SQLAlchemyFeedbackRepository(create_session_factory(engine))
        return FeedbackService(repository)
    except (RuntimeError, SQLAlchemyError, ValueError):
        return None


def _create_environment_readiness_provider(
    settings: Settings,
    artifact_loader: ArtifactLoader | None,
) -> ReadinessProvider:
    database_engine = None
    if settings.database_url:
        try:
            database_engine = create_database_engine(settings, pool_pre_ping=True)
        except (RuntimeError, SQLAlchemyError, ValueError):
            pass
    return RuntimeReadinessProvider(database_engine, artifact_loader)


def create_app(
    *,
    settings: Settings | None = None,
    readiness_provider: ReadinessProvider | None = None,
    auth_service: AuthService | None = None,
    movie_catalog_service: MovieCatalogService | None = None,
    rating_service: RatingService | None = None,
    artifact_loader: ArtifactLoader | None = None,
    recommendation_service: RecommendationService | None = None,
    feedback_service: FeedbackService | None = None,
    experiment_summary_provider: ExperimentSummaryProvider | None = None,
) -> FastAPI:
    """
    Construct the API without forcing DB connections
    or loading recommendation artifacts.
    """

    resolved_settings = settings or Settings.from_environment()
    resolved_loader = artifact_loader
    if resolved_loader is None and resolved_settings.artifact_path is not None:
        resolved_loader = ArtifactLoader(resolved_settings.artifact_path)
    resolved_readiness = readiness_provider or _create_environment_readiness_provider(
        resolved_settings,
        resolved_loader,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if resolved_loader is not None:
            try:
                resolved_loader.load()
            except ArtifactError:
                pass
        yield

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        lifespan=lifespan,
    )
    application.state.artifact_loader = resolved_loader

    resolved_auth_service = auth_service or _create_environment_auth_service(
        resolved_settings
    )

    resolved_movie_catalog_service = (
        movie_catalog_service
        or _create_environment_movie_catalog_service(resolved_settings)
    )

    resolved_rating_service = rating_service or _create_environment_rating_service(
        resolved_settings
    )

    resolved_recommendation_service = (
        recommendation_service
        or _create_environment_recommendation_service(
            resolved_settings,
            resolved_loader,
        )
    )

    resolved_feedback_service = (
        feedback_service or _create_environment_feedback_service(resolved_settings)
    )
    resolved_experiment_summary_provider = experiment_summary_provider
    if (
        resolved_experiment_summary_provider is None
        and resolved_settings.exp003_path is not None
    ):
        resolved_experiment_summary_provider = CommittedExperimentSummaryProvider(
            resolved_settings.exp003_path
        )

    current_user_dependency = create_current_user_dependency(resolved_auth_service)

    application.include_router(create_health_router(resolved_readiness))

    application.include_router(
        create_auth_router(
            resolved_auth_service,
            current_user_dependency,
        )
    )

    application.include_router(
        create_movies_router(
            resolved_movie_catalog_service,
            current_user_dependency,
        )
    )

    application.include_router(
        create_ratings_router(
            resolved_rating_service,
            current_user_dependency,
        )
    )

    application.include_router(
        create_recommendations_router(
            resolved_recommendation_service,
            current_user_dependency,
        )
    )

    application.include_router(
        create_feedback_router(
            resolved_feedback_service,
            current_user_dependency,
        )
    )

    application.include_router(
        create_experiments_router(
            resolved_experiment_summary_provider,
            current_user_dependency,
        )
    )

    return application


app = create_app()
