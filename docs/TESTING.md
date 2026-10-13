# Testing

`make setup` installs Python and npm lockfiles.

| Command | Scope |
| --- | --- |
| `make lint` | Python/core/scripts and frontend lint |
| `make test` | Core, ML, backend, frontend and publication tests; architecture audit |
| `make frontend-check` | TypeScript and production build |
| `npm run test:e2e` in frontend | Real register → rate → recommend → feedback → dashboard flow |
| `make public-release-check` | Restricted payload paths across fetched history |

Core tests verify geometry, checksums, symlink/extra-payload rejection,
determinism and rated-item exclusion. Real SQLAlchemy/SQLite tests verify
account isolation, feedback, restart behavior and foreign-key enforcement.

Default ML tests are pure/synthetic/fixture-based. Historical MovieLens tests
requiring locally prepared research data are marked `realdata` and excluded
by default. They are not covered by the synthetic demo.

This is the public suite, not the entire private accepted-artifact suite.
Synthetic fixtures replace private serving-artifact acceptance cases;
accepted historical results remain unchanged.

CI installs locked dependencies, lints/tests/builds, runs actual PostgreSQL
migrations/persistence, checks paths and scans fetched history with redacted
Gitleaks. Its browser job runs the real app and captures screenshots. Full-
history publication checks apply to public distribution; a private preparation
branch can retain the old research history.

To capture actual screenshots:

```bash
cd frontend
npx playwright install chromium
CAPTURE_PORTFOLIO_SCREENSHOTS=1 npm run test:e2e
```

A build alone is not browser or PostgreSQL evidence. Production npm audits
are separate from development advisories; see [SECURITY.md](../SECURITY.md).

## Verified preparation run

On 2026-10-13, all four hosted CI jobs passed: quality, PostgreSQL, browser
and secrets. This included 40 core, 716 ML, 180 backend and 180 frontend tests,
five publication tests, a real PostgreSQL persistence test and the full browser
flow. Ten historical real-data ML cases were intentionally deselected; the
ordinary backend run skips PostgreSQL, which is tested by its separate CI job.

Production npm audit: zero known advisories. Installed Python audit: 54 dependency
entries, zero known advisories. Development npm audit: five High lint-tool entries,
documented in SECURITY.md. The new local Git history passed the publication
inventory and Gitleaks. All README images are captures of the real application
with fictional films and a synthetic test account.
