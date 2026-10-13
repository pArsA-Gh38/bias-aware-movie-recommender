# Architecture

Next.js renders the UI and forwards same-origin `/api` requests through an
authenticated BFF. Its HttpOnly cookie holds the session token. FastAPI validates
the bearer token and calls application services. SQLAlchemy repositories handle
transactional persistence.

The shared `recommender_core` package owns deterministic fold-in, scoring,
normalization, genre-distance reranking and artifact validation. Offline `ml`
modules own preparation/training and evaluation utilities. Backend requests
never invoke training or import the offline package.

## Recommendation flow

1. Load the account's ratings and active model registration.
2. Require at least five positive catalog ratings (4–5 stars).
3. Recalculate a new-user factor from fixed item factors.
4. Score eligible films, excluding every rated film.
5. Select a deterministic pool of 100 candidates and rerank to ten results.
6. Validate metadata and persist the event/items atomically.

Serving styles are `A0_L1`, `A1_L07` and `A1_L06`. Their configuration semantics
are retained, while the local model has an independent identity.
The synthetic catalog can exhaust a full candidate pool after extensive rating;
it is intended for a short portfolio walkthrough.

## Persistence and integrity

Seven tables cover users, films, ratings, model versions, recommendation events,
ranked event items and feedback. Ratings are unique per account/film. Event
items are unique by rank and film. Feedback requires an owned event and an
actual member film; cross-user event access is denied.

PostgreSQL uses ARRAY/JSONB and Alembic. SQLite is a local-demo adapter using JSON
genres and enforced foreign keys, without a production portability claim.

The independent `local-serving-v1` bundle contains a catalog, item IDs, item
factors, popularity counts and a manifest. Exact inventory, SHA-256,
numerical-version checks, geometry/type/finite-value validation and symlink
rejection protect loading. NumPy pickle loading is disabled. No source-user
factors or histories are written to the bundle.

Checksums do not authenticate an untrusted producer. Build locally or use a
trusted source. Historical loaders remain strict; old research artifacts
require their original matching source/provenance.

## Dashboard

The backend validates stored EXP-003 manifests/checksums and serves supported
aggregate tables. Validation and test remain labeled; the new local model is
never silently substituted for historical evidence.
