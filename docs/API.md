# API guide

With the backend running, [OpenAPI](http://127.0.0.1:8000/docs) is the detailed
source of schemas and status codes. The web frontend uses authenticated,
same-origin BFF routes under `/api`; backend routes have no `/api` prefix.

| Backend endpoint | Authentication | Purpose |
| --- | --- | --- |
| `GET /health/live` | No | Process liveness |
| `GET /health/ready` | No | DB/artifact readiness |
| `POST /auth/register` | No | Create an account |
| `POST /auth/login` | No | Issue a bearer token |
| `GET /auth/me` | Bearer | Current account |
| `GET /movies` | Bearer | Filtered/paginated catalog |
| `PUT /ratings/{movie_id}` | Bearer | Set/update own rating |
| `GET /ratings/me` | Bearer | Own rating state |
| `POST /recommendations` | Bearer | Generate and persist recommendations |
| `POST /feedback` | Bearer | Record owned event-item feedback |
| `GET /experiments/summary` | Bearer | Read validated EXP-003 evidence |

Recommendation input:

```json
{"configuration_key": "A1_L07"}
```

Only `A0_L1`, `A1_L07` and `A1_L06` are accepted. An omitted request uses
the backend's frozen default. Arbitrary lambda, beta and user IDs are not
part of this API.

Results include event/model identity, applied configuration and ranked movie
IDs, titles, genres and scores. Treat backend-confirmed metadata as authoritative.

Feedback accepts `event_id`, `movie_id` and one of `click`, `like`,
`dislike`. The backend checks event ownership and membership.

401 means authentication failed; 409 can indicate insufficient positive
eligible ratings; 422 means invalid input; 503 means a required service or
validated evidence is unavailable. Consult OpenAPI for each route's contract.
