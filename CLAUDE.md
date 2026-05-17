# CLAUDE.md

This file is loaded automatically by Claude Code in this repo. Read it before making changes.

## What this project is

**cluster-canary** — predict Kubernetes pod failures (OOMKill, CrashLoopBackOff) 30 minutes ahead, serve predictions from an in-cluster sidecar at p95 < 50 ms, retrain when workload mix drifts. Pairs with [`kubeai-ops`](https://github.com/sharankumarreddyk/kubeai-ops) as its proactive layer.

Bootstrapped from [`mlops-reference-template`](https://github.com/sharankumarreddyk/mlops-reference-template). The 10-phase roadmap lives in [`docs/PLAN.md`](docs/PLAN.md). Phase 1 research context (kind+chaos-mesh lab setup, Alibaba / Google trace schemas, OOM-signal sources) is in [`docs/PHASE_1_CONTEXT.md`](docs/PHASE_1_CONTEXT.md).

## Project-level Claude skills

Five custom skills under `.claude/skills/` are auto-discovered. **Apply them whenever their triggers fire** — invoke via the `Skill` tool, don't just mention them.

| Skill | Trigger |
|---|---|
| `mlflow-tracking-conventions` | Any MLflow run, training script, registry call |
| `feature-leakage-detector` | Feature engineering, CV splits, group aggregates, `fit_transform` |
| `evidently-drift-runner` | Drift metrics, monitoring setup, retraining triggers |
| `bentoml-service-scaffolder` | Serving endpoints, BentoML services, model Dockerfiles |
| `model-card-writer` | Model promotion, release, registry push |

Each skill's `SKILL.md` documents the contract. **Skill violations fail the task** — they are not style suggestions.

## Core conventions

### Code
- **Python 3.12**, strict mypy, ruff format + lint. No new dependency without a one-line rationale in the PR description.
- **`src/` layout** — package is `cluster_canary`, imports go through `cluster_canary.*`.
- **Pydantic v2** for any I/O boundary (API, config, file schemas).
- **`structlog`** for logs — never `print()` in `src/`. JSON output for any service.
- **`pathlib` over `os.path`**.
- **No `Optional[T]`** — use `T | None`.
- **No `# TODO`** in source. Finish or open an issue.

### Tests
- Tests live in `tests/`, mirroring `src/` where it helps.
- Markers: `slow`, `integration`, `needs_data`. CI runs `not integration and not needs_data` by default.
- Use `tests/test_no_leakage.py` as the template for any new data-quality test.
- New code with real branching → at least one test per branch that matters.

### MLflow
- Read `.claude/skills/mlflow-tracking-conventions/SKILL.md` first.
- Experiment name: `cluster_canary__oom_30min` (or `__crashloop_30min` for the secondary task). Run name: `<model>__<dataset>__<sha[:7]>__<ts>`.
- Required tags on every run: `git_sha`, `git_dirty`, `dataset_version`, `chaos_plan_version`, `model_family`, `stage`, `author`.
- Required metrics — **binary classification at <1 % positive rate**: `brier_score`, `pr_auc`, `precision_at_recall_0_8`, `precision_at_recall_0_5`, `lead_time_median_min`, plus latency percentiles. Report ROC-AUC as secondary only — it is misleading at this class balance.
- Always report metrics at multiple lead-time horizons (5 / 10 / 15 / 30 min) so the precision-vs-lead-time tradeoff is visible.
- No "untitled run #N". Ever.

### DVC
- `data/raw/`, `data/interim/`, `data/processed/`, `data/features/` are DVC-tracked.
- Never commit raw/processed parquet to git. Use `dvc add` + `dvc push`.
- Remote `minio` points at the local MinIO bucket `s3://cluster-canary-dvc`.

### Serving
- Read `.claude/skills/bentoml-service-scaffolder/SKILL.md` first.
- gRPC, not HTTP — this runs in-cluster as a sidecar, latency budget 50 ms p95.
- `/healthz` + `/readyz` mandatory. `X-Model-Version` header in every response.
- Adaptive batching enabled. Prometheus `/metrics` exposed.

### K8s data lab (Phase 1)
- `kind` 3-worker cluster, K8s 1.30, cgroup v2 enabled.
- `chaos-mesh` 2.7+ for failure injection (StressChaos = memory leak; PodChaos = crash; NetworkChaos = jitter).
- **`chaosDaemon.runtime=containerd`** is non-negotiable — kind uses containerd, not Docker; missing this is the #1 setup failure.
- Prometheus scrape interval **15 s** (NOT the default 30 s) — 30-min lead time requires 120+ samples per window to detect derivative trends.
- OOM ground truth: `container_oom_events_total` (cAdvisor) + `kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}` (kube-state-metrics v2.13+). Pin ksm version — older versions emit `Error` instead of `OOMKilled`.

### Commits / PRs
- Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `ci:`, `chore:`.
- Reference the phase in the commit body: `Phase 3: train LightGBM baseline with Optuna`.
- Pre-commit must pass locally (`make check`). CI runs the same gates.

## Local dev

```bash
make install   # uv sync --extra dev + pre-commit install
make dev-up    # MLflow (5000) + MinIO (9001) + Postgres (5432)
make check     # lint + typecheck + test
```

## Hard NOs

- ❌ No `# TODO` / `# FIXME` / `# XXX` in committed source.
- ❌ No `try/except` that just logs and re-raises.
- ❌ No `random_state` train/test splits on time-series data — temporal only.
- ❌ No fitting transformers on the union of train+test. Pipeline-wrapped fit-on-train.
- ❌ No naive target encoding (`groupby.transform('mean')` on full train). Use sklearn's `TargetEncoder` with CV.
- ❌ No promoting a model without a model card.
- ❌ No `print()` in `src/`. Use `structlog`.
- ❌ No introducing a new abstraction to solve a one-instance problem.
- ❌ No commented-out code. Delete it; git remembers.

## Cross-session memory

User-level memory lives at `~/.claude/projects/-Users-sharan-Documents-mlp1/memory/` (project root tied to the template work — same Sharan, all memories carry over). Always read `MEMORY.md` at session start. Key entries:
- `user-profile` — Sharan, early-career, targeting ML Engineer (MLOps); GitHub `sharankumarreddyk`, git noreply `167206944+sharankumarreddyk@users.noreply.github.com`.
- `feedback-use-skills` — invoke skills proactively via the `Skill` tool; don't just mention them.
- `feedback-no-claude-mentions` — keep portfolio-visible docs neutral; CLAUDE.md (this file) is config, not user-facing.
- `feedback-parallel-agents` — spawn 2-4 background research subagents proactively for non-trivial phases.
- `project-pivot-to-cluster-canary` — this project's positioning (proactive ML for K8s platform).

When this session ends, append an `impl-log-YYYYMMDD.md` entry summarizing the task, files changed, and any conventions confirmed.
