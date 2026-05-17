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
        lab-up lab-up-kind lab-up-chaos lab-up-observability lab-up-workloads lab-up-chaos-plans \
        lab-build-leaky lab-down lab-reset lab-status lab-scrape \
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

# --- Phase 1: data lab + scraper ----------------------------------------------
lab-up: lab-up-kind lab-up-chaos lab-up-observability lab-up-workloads lab-up-chaos-plans ## Bring up kind + chaos-mesh + Prometheus + workloads + chaos
	@echo ""
	@echo "  Prometheus UI: http://localhost:9090"
	@echo "  Run 'make lab-scrape' once you've let the cluster run for a while."

lab-up-kind: ## Create the kind canary-lab cluster
	@if kind get clusters | grep -q '^canary-lab$$'; then \
	  echo "[lab] kind cluster canary-lab already exists — skipping create"; \
	else \
	  kind create cluster --config infra/lab/kind-config.yaml; \
	fi
	kubectl cluster-info --context kind-canary-lab

lab-up-chaos: ## Install chaos-mesh into the cluster
	bash infra/lab/chaos-mesh/install.sh

lab-up-observability: ## Apply Prometheus + kube-state-metrics
	kubectl apply -f infra/lab/observability/

lab-up-workloads: lab-build-leaky ## Apply workloads (nginx, postgres, redis, leaky-flask, go-batch, fluent-bit)
	kubectl apply -f infra/lab/workloads/namespace.yaml
	kubectl apply -f infra/lab/workloads/nginx.yaml
	kubectl apply -f infra/lab/workloads/postgres.yaml
	kubectl apply -f infra/lab/workloads/redis.yaml
	kubectl apply -f infra/lab/workloads/leaky-flask/deployment.yaml
	kubectl apply -f infra/lab/workloads/go-batch.yaml
	kubectl apply -f infra/lab/workloads/fluent-bit.yaml

lab-up-chaos-plans: ## Apply chaos plans (instant-oom + random-crash schedule)
	kubectl apply -f infra/lab/chaos-plans/instant-oom.yaml
	kubectl apply -f infra/lab/chaos-plans/random-crash.yaml

lab-build-leaky: ## Build leaky-flask image and load into kind
	docker build -t cluster-canary/leaky-flask:dev infra/lab/workloads/leaky-flask
	kind load docker-image cluster-canary/leaky-flask:dev --name canary-lab

lab-down: ## Delete the kind canary-lab cluster
	-kind delete cluster --name canary-lab

lab-reset: lab-down lab-up ## Tear down and rebuild (destructive)

lab-status: ## Show cluster + pod status
	kubectl get nodes
	@echo ""
	@echo "--- observability ---"
	kubectl -n observability get pods
	@echo ""
	@echo "--- workloads ---"
	kubectl -n workloads get pods
	@echo ""
	@echo "--- chaos-mesh ---"
	kubectl -n chaos-mesh get pods

lab-scrape: ## Run a 24h scrape of Prometheus → parquet (uses env: SCRAPE_START, SCRAPE_END, PROMETHEUS_URL)
	$(RUN) python -m cluster_canary.pipelines.scrape

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
