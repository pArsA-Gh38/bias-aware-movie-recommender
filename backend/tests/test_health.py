from __future__ import annotations

import builtins
import socket
from collections.abc import Mapping
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.core.config import Settings
from app.core.readiness import ReadinessState, StaticReadinessProvider
from app.main import create_app
from app.recommendation.artifact_loader import ArtifactLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_ARTIFACT = REPOSITORY_ROOT / (
    "artifacts/serving/art-002-final-als-f32-r0p01-i30-a10-s42-train-validation-v2"
)


class ExplodingReadinessProvider:
    """Fails if app construction or liveness performs a dependency check."""

    def get_state(self) -> ReadinessState:
        raise AssertionError("readiness provider must not be called")


def client_for(state: ReadinessState) -> TestClient:
    app = create_app(readiness_provider=StaticReadinessProvider(state))
    return TestClient(app)


def test_application_and_openapi_are_created_without_external_dependencies() -> None:
    app = create_app(readiness_provider=ExplodingReadinessProvider())

    assert app.title == "Movie Recommender API"
    assert app.openapi()["paths"].keys() == {
        "/health/live",
        "/health/ready",
        "/auth/register",
        "/auth/login",
        "/auth/me",
        "/movies",
        "/ratings/{movie_id}",
        "/ratings/me",
        "/recommendations",
        "/feedback",
        "/experiments/summary",
    }


def test_application_construction_does_not_read_artifact_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("application construction must not open files")

    monkeypatch.setattr(builtins, "open", fail_open)

    app = create_app(
        settings=Settings(
            database_url="postgresql+psycopg://db/test",
            auth_secret_key="construction-test-secret-with-sufficient-entropy",
        )
    )

    assert app.title == "Movie Recommender API"


def test_application_construction_does_not_connect_to_database(
    monkeypatch: pytest.MonkeyPatch,
    local_artifact: Path,
) -> None:
    def fail_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("application construction must not connect to services")

    monkeypatch.setattr(socket, "create_connection", fail_connect)
    monkeypatch.setattr(socket.socket, "connect", fail_connect)

    app = create_app(
        settings=Settings(
            database_url="postgresql+psycopg://db/test",
            auth_secret_key="construction-test-secret-with-sufficient-entropy",
        )
    )

    assert app.title == "Movie Recommender API"


def test_settings_support_explicit_environment_overrides() -> None:
    environ: Mapping[str, str] = {
        "MOVIE_RECOMMENDER_APP_NAME": "Test Recommender API",
        "MOVIE_RECOMMENDER_APP_VERSION": "test-version",
        "MOVIE_RECOMMENDER_AUTH_SECRET_KEY": "environment-test-secret",
        "MOVIE_RECOMMENDER_AUTH_ACCESS_TOKEN_EXPIRE_MINUTES": "45",
        "MOVIE_RECOMMENDER_EXP003_PATH": "/srv/movie-recommender",
    }

    settings = Settings.from_environment(environ)
    app = create_app(settings=settings)

    assert app.title == "Test Recommender API"
    assert app.version == "test-version"
    assert settings.auth_secret_key == "environment-test-secret"
    assert settings.auth_access_token_expire_minutes == 45
    assert settings.exp003_path == Path("/srv/movie-recommender")


def test_liveness_is_unauthenticated_and_skips_readiness_checks() -> None:
    client = TestClient(create_app(readiness_provider=ExplodingReadinessProvider()))

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_default_readiness_fails_closed_with_explicit_unknown_state() -> None:
    response = TestClient(create_app(settings=Settings())).get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "database_reachable": False,
            "artifact_loaded": False,
            "artifact_schema_compatible": None,
        },
    }


@pytest.mark.parametrize(
    ("state", "failed_check"),
    [
        (ReadinessState(False, True, True), "database_reachable"),
        (ReadinessState(True, False, True), "artifact_loaded"),
        (ReadinessState(True, True, False), "artifact_schema_compatible"),
    ],
)
def test_one_failed_dependency_keeps_readiness_false(
    state: ReadinessState, failed_check: str
) -> None:
    response = client_for(state).get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"][failed_check] is False


def test_all_ready_simulated_state_returns_success_without_authentication() -> None:
    response = client_for(ReadinessState(True, True, True)).get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "database_reachable": True,
            "artifact_loaded": True,
            "artifact_schema_compatible": True,
        },
    }


def test_runtime_readiness_reports_reachable_database(
    monkeypatch: pytest.MonkeyPatch,
    local_artifact: Path,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    def database_engine(*args: object, **kwargs: object):
        return engine

    monkeypatch.setattr(
        "app.main.create_database_engine",
        database_engine,
    )

    with TestClient(
        create_app(
            settings=Settings(database_url="postgresql+psycopg://db/runtime"),
            artifact_loader=ArtifactLoader(local_artifact),
        )
    ) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "database_reachable": True,
            "artifact_loaded": True,
            "artifact_schema_compatible": True,
        },
    }
