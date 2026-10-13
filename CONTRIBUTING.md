# Contributing

Read [setup](docs/SETUP.md), [architecture](docs/ARCHITECTURE.md) and
[testing](docs/TESTING.md). Branch from current main and use `make setup`.

Before opening a PR:

```bash
make lint
make test
make frontend-check
git diff --check
```

Describe the concrete problem, resulting behavior and relevant validation.
Report unavailable checks honestly. Run the browser workflow for user-flow
changes. Discuss broad changes in an issue first.

Do not tune on frozen test results or rewrite historical manifests/results.
New experiments require a separate protocol, identity, fit scope and provenance.
Numerical changes require a newly built compatible local artifact.

Never commit source datasets, user histories, accounts, credentials, databases,
logs or generated models. A clean tip does not establish clean Git history.

Preserve compatible rights and attribution. Original code is MIT; MovieLens
has separate terms. Use [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
