# Security

This is a local academic prototype. The demo binds to loopback. Public source
availability does not establish a hardened public service.

JWTs stay in an HttpOnly cookie; Argon2 protects passwords. Secrets are generated
into ignored private local state. Data, models, accounts, DB files and logs
must remain untracked.

Use private vulnerability reporting when enabled on GitHub. Do not post live
credentials or exploitable details in public issues. Only current main is
maintained; no response-time guarantee is offered.

The loader uses `allow_pickle=False`, exact inventory, checksums, numerical-
version checks, geometry/type validation and symlink rejection. Checksums
protect byte consistency, not third-party authenticity. Build locally or
use a trusted source.

Production npm and installed Python audits were clean during preparation.
The development npm audit retains five High entries in the eslint/fast-glob/
micromatch/braces chain ([advisory](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm)).
A lint-tool issue was not demonstrated as an application exploit. An incompatible
forced lint downgrade was not applied. Review upstream fixes and distinguish
production audit results from a clean development audit.

External deployment requires TLS, abuse controls, CSRF review, protected DB
access, backup/restore, monitoring and a threat model.
