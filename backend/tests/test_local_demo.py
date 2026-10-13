"""Verify real database persistence, event ownership and application restart."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.core.config import Settings
from app.db.seed_local import initialize
from app.db.session import create_database_engine
from app.main import create_app


def token(client, email):
    credentials = {"email": email, "password": "Synthetic-demo-only-2026"}
    assert client.post("/auth/register", json=credentials).status_code == 201
    response = client.post("/auth/login", json=credentials)
    assert response.status_code == 200
    return {"Authorization": "Bearer " + response.json()["access_token"]}


@pytest.mark.parametrize("database", ["sqlite", "postgresql"])
def test_persistent_register_rate_recommend_feedback_and_isolation(
    tmp_path: Path, local_artifact: Path, database: str
):
    url = "sqlite+pysqlite:///" + str(tmp_path / "app.sqlite")
    if database == "postgresql":
        url = os.environ.get("MOVIE_RECOMMENDER_TEST_DATABASE_URL")
        if not url:
            pytest.skip("PostgreSQL acceptance requires an isolated test database")
    settings = Settings(
        database_url=url,
        auth_secret_key="synthetic-local-integration-secret-2026",
        artifact_path=local_artifact,
        exp003_path=Path(__file__).resolve().parents[2],
    )
    initialize(settings)
    initialize(settings)
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/health/ready").status_code == 200
        alice = token(client, "alice@example.com")
        bob = token(client, "bob@example.com")
        filtered = client.get("/movies?genre=comedy", headers=alice)
        assert filtered.status_code == 200
        assert all("Comedy" in row["genres"] for row in filtered.json()["items"])
        assert client.post("/recommendations", headers=alice).status_code == 409
        for i in range(1, 7):
            response = client.put(
                f"/ratings/{i}", headers=alice, json={"rating": 5 if i <= 5 else 1}
            )
            assert response.status_code == 200
        first = client.post(
            "/recommendations", headers=alice, json={"configuration_key": "A1_L07"}
        )
        assert first.status_code == 200, first.text
        rows = first.json()["items"]
        assert len(rows) == 10
        assert not {1, 2, 3, 4, 5, 6} & {row["movie_id"] for row in rows}
        intent = {
            "event_id": first.json()["event_id"],
            "movie_id": rows[0]["movie_id"],
            "action": "like",
        }
        assert client.post("/feedback", headers=alice, json=intent).status_code == 201
        assert client.post("/feedback", headers=bob, json=intent).status_code == 404
        assert client.get("/experiments/summary", headers=alice).status_code == 200
        assert (
            client.post(
                "/recommendations", headers=alice, json={"configuration_key": "unknown"}
            ).status_code
            == 422
        )
    with TestClient(create_app(settings=settings)) as restarted:
        login = restarted.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "Synthetic-demo-only-2026"},
        )
        assert login.status_code == 200
        response = restarted.post(
            "/recommendations",
            headers={"Authorization": "Bearer " + login.json()["access_token"]},
            json={"configuration_key": "A1_L07"},
        )
        assert response.status_code == 200
        assert [r["movie_id"] for r in response.json()["items"]] == [
            r["movie_id"] for r in rows
        ]
    engine = create_database_engine(settings)
    with engine.connect() as connection:
        if database == "sqlite":
            assert connection.scalar(text("PRAGMA foreign_keys")) == 1
        assert connection.scalar(text("SELECT count(*) FROM feedback")) == 1
        assert (
            connection.scalar(text("SELECT count(*) FROM recommendation_events")) == 2
        )
    engine.dispose()
