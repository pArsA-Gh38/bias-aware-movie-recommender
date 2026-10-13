# Setup

## Default portable demo

Install Python 3.11+, uv, Node.js 22 and npm, then run `make demo`.
The runner installs locked dependencies, generates synthetic data, trains an
item-only ALS model, initializes persistent SQLite and starts the real app on loopback.

Open http://127.0.0.1:3000. Backend OpenAPI is at http://127.0.0.1:8000/docs.
Register your own account and rate at least five films with 4–5 stars.
The catalog is fictional and clearly labeled.

Ctrl+C stops runner-owned services. The database, artifact, logs and generated
secret remain in ignored `.local/synthetic/`. Restarting preserves accounts and
ratings; the runner refuses catalog/model conflicts.

For occupied ports:

```bash
BACKEND_PORT=18000 FRONTEND_PORT=13000 make demo
```

The BFF uses the selected backend port. The runner does not source `.env` files
by itself; `.env.example` documents overrides.

## Optional MovieLens model

Review the [MovieLens 1M terms](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt), then run:

```bash
make demo DATASET=movielens
```

Download is directly from GroupLens, with TLS and SHA-256 verification.
Ratings are globally sorted by timestamp, user ID and film ID. Positives
(rating ≥4) in the earliest 80% fit the model and popularity counts.
Titles/genres align with the fit catalog. No source-user factors or histories
are saved in the bundle. No test evaluation is run.

State is separate in `.local/movielens/`. The MIT code license does not grant
permission to redistribute locally downloaded or generated MovieLens payloads.

## Optional PostgreSQL

Create an empty development database and supply your local values:

```bash
export MOVIE_RECOMMENDER_DATABASE_URL='postgresql+psycopg://USER:PASSWORD@127.0.0.1:5432/DATABASE'
make demo
```

The initializer runs Alembic migrations and seeds that development database.
It refuses mismatched existing catalog/model state. Never target production
or commit credentials.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Port unavailable | Override ports or stop the prior runner |
| Recommendations return 409 | Five positive eligible ratings are required |
| Dashboard unavailable | Inspect local logs and exported report checksums |
| Artifact incompatible | Preserve old state; use a new directory with matching numerical locks |
| Dataset download fails | Check network/CA certificates; TLS is not bypassed |
| Login unavailable | Inspect ignored backend logs and database/secret configuration |
| Browser cannot launch | Run `npx playwright install chromium` in frontend |

Internet hosting requires a separate deployment/security design.
