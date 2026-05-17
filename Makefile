SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PYTHON ?= python3.12
UV     := $(shell command -v uv 2>/dev/null)

# When uv is present, prefix runtime commands with `uv run`; otherwise fall back to the env's binary.
ifeq ($(UV),)
    RUN :=
else
    RUN := uv run
endif

.PHONY: help install install-pip hooks lint format typecheck test test-fast test-all check \
        dev-up dev-down dev-reset dev-logs mlflow-ui minio-ui \
        data-pull data-push train serve drift retrain build clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- install ------------------------------------------------------------------
install: ## Install all deps via uv (preferred); falls back to pip
ifeq ($(UV),)
	@echo "[install] uv not found — falling back to pip. Install uv: https://docs.astral.sh/uv/"
	@$(MAKE) install-pip
else
	uv sync --extra dev
	$(RUN) pre-commit install
endif

install-pip: ## Install via pip (fallback path)
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

hooks: ## (Re)install pre-commit hooks
	$(RUN) pre-commit install --install-hooks

# --- quality gates ------------------------------------------------------------
lint: ## Ruff lint
	$(RUN) ruff check .

format: ## Ruff format (write) + fix lints
	$(RUN) ruff format .
	$(RUN) ruff check . --fix

typecheck: ## Mypy strict typecheck
	$(RUN) mypy src/

test: ## Run default test subset (no integration, no data-dependent)
	$(RUN) pytest -ra -m "not integration and not needs_data"

test-fast: ## Fastest subset — no coverage
	$(RUN) pytest -ra -m "not integration and not needs_data and not slow" --no-cov

test-all: ## Run everything including integration + data-dependent
	$(RUN) pytest -ra

check: lint typecheck test ## All quality gates (lint + typecheck + test)

# --- dev services -------------------------------------------------------------
dev-up: ## Start local MLflow + MinIO + Postgres
	docker compose -f docker-compose.dev.yml --env-file .env up -d
	@echo ""
	@echo "  MLflow:  http://localhost:5000"
	@echo "  MinIO:   http://localhost:9001  (login: minioadmin / minioadmin)"
	@echo "  Postgres: localhost:5432"

dev-down: ## Stop local services (keep volumes)
	docker compose -f docker-compose.dev.yml down

dev-reset: ## Stop and DELETE volumes (destructive — confirms via Docker prompt)
	docker compose -f docker-compose.dev.yml down -v

dev-logs: ## Tail dev service logs
	docker compose -f docker-compose.dev.yml logs -f --tail=200

mlflow-ui: ## Open MLflow UI
	@open http://localhost:5000 2>/dev/null || xdg-open http://localhost:5000 2>/dev/null || echo "open manually: http://localhost:5000"

minio-ui: ## Open MinIO console
	@open http://localhost:9001 2>/dev/null || xdg-open http://localhost:9001 2>/dev/null || echo "open manually: http://localhost:9001"

# --- DVC ----------------------------------------------------------------------
data-pull: ## Pull DVC-tracked data
	$(RUN) dvc pull

data-push: ## Push DVC-tracked data
	$(RUN) dvc push

# --- pipeline (filled in across phases) ---------------------------------------
train: ## Train model (Phase 3+)
	$(RUN) python -m cluster_canary.training.train

serve: ## Launch BentoML inference service (Phase 4+)
	$(RUN) bentoml serve src.cluster_canary.serving.service:ClusterCanaryService --port $${BENTOML_PORT:-3000}

drift: ## Run drift detection (Phase 6+)
	$(RUN) python -m cluster_canary.monitoring.drift

retrain: ## Trigger Prefect retrain flow (Phase 7+)
	$(RUN) python -m cluster_canary.pipelines.retrain

# --- build / clean ------------------------------------------------------------
build: ## Build wheel + sdist
	$(RUN) python -c "import cluster_canary; print('cluster_canary v' + cluster_canary.__version__)"
ifeq ($(UV),)
	$(PYTHON) -m build
else
	uv build
endif

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
