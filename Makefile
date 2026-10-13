.PHONY: setup lint test frontend-check prepare demo architecture-audit public-release-check
DATASET ?= synthetic

setup:
	cd packages/recommender_core && uv sync --locked
	cd backend && uv sync --locked
	cd ml && uv sync --locked
	cd frontend && npm ci --no-fund --no-audit

lint:
	cd packages/recommender_core && uv run ruff check . ../../scripts
	cd backend && uv run ruff check .
	cd ml && uv run ruff check .
	cd frontend && npm run lint

test:
	cd packages/recommender_core && uv run pytest -q
	cd ml && uv run pytest -q
	cd backend && uv run pytest -q
	cd frontend && npm test
	python3 -m unittest discover -s scripts/release/tests -p 'test_*.py'
	$(MAKE) architecture-audit

frontend-check:
	cd frontend && npm run type-check && npm run build

prepare:
	cd ml && uv run python -m scripts.prepare_demo --source $(DATASET)

demo:
	python3 scripts/dev.py --dataset $(DATASET)

architecture-audit:
	python3 scripts/architecture/check_runtime_boundaries.py

public-release-check:
	python3 scripts/release/check_publication.py
