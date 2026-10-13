# Working on this repository

Use source-backed claims. Preserve deterministic ranking, rated-item exclusion,
transactional event membership and user isolation. Keep backend independent of
offline ML modules. Use locked dependencies.

Run make lint, make test and make frontend-check. Use the browser workflow for
user-flow changes. Synthetic fixtures belong in public CI; source datasets,
user histories, credentials and generated models do not.

Stored research evidence is historical. Do not rewrite it to match a new model,
rerun the original frozen experiment or describe the synthetic demo as research
validation. Build a separately identified local artifact for new runs.
