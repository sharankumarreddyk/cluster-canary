# Convention Skills

Five project-level skill specs that encode senior-MLE conventions for end-to-end MLOps work. They are auto-discovered by Claude Code from this directory and apply whenever their triggers fire.

| Skill | Triggers on | Stops you from |
|---|---|---|
| [`mlflow-tracking-conventions`](mlflow-tracking-conventions/SKILL.md) | Any MLflow run, model registry call, training script | Untitled runs, missing tags, ambiguous metric names, untracked artifacts |
| [`feature-leakage-detector`](feature-leakage-detector/SKILL.md) | Feature engineering, CV splits, group aggregates, fit_transform | Target leakage, temporal leakage, group leakage, train-test contamination, target-encoding leakage |
| [`evidently-drift-runner`](evidently-drift-runner/SKILL.md) | Any drift metric, monitoring setup, retraining trigger | Wrong reference window, low-power tests, missing prediction drift, noisy alerts |
| [`bentoml-service-scaffolder`](bentoml-service-scaffolder/SKILL.md) | Serving endpoints, BentoML services, model Dockerfiles | Missing batching, no `/readyz`, no model version in response, no load test |
| [`model-card-writer`](model-card-writer/SKILL.md) | Model promotion, release, registry push | Cards without sliced metrics, missing out-of-scope, no changelog |

## Why these five, in this order

They map to the five points in the ML lifecycle where senior interviewers and review committees most often find junior work lacking:

1. **Reproducibility** (mlflow-tracking-conventions) — Can someone else run your experiment? If not, it didn't happen.
2. **Statistical correctness** (feature-leakage-detector) — Are your metrics real? Leaked features inflate numbers and crash in prod.
3. **Production awareness** (evidently-drift-runner) — Does the model still work next month? Most portfolio projects skip this entirely.
4. **Serving discipline** (bentoml-service-scaffolder) — Will it survive 100 RPS? Most portfolios show a script, not a service.
5. **Ownership and transparency** (model-card-writer) — Do you know your model's failure modes? This separates "I trained a model" from "I shipped one."

## How to use them

Just work normally. When you write code matching a skill's triggers (described in each `SKILL.md`'s top section), Claude will apply the skill's conventions automatically. You can also invoke explicitly with `/<skill-name>` or by referencing it in a prompt: "use the feature-leakage-detector skill to audit `src/cluster_canary/features/`".

## When to promote a skill to `~/.claude/skills/` (user-level)

These currently live at the project level so they version with `cluster_canary`. Once a skill is battle-tested across two or more projects, copy it to `~/.claude/skills/` to make it available everywhere. The natural promotion candidates after `cluster_canary`:
- `mlflow-tracking-conventions` — useful for every future ML project
- `feature-leakage-detector` — useful for every tabular/time-series project
- `model-card-writer` — useful for every shipped model

Project-specific tunings (the deny-list in `feature-leakage-detector`, the slice columns in `model-card-writer`) should stay in the project-level copy.

## Adding to the `AI-Skills` repo

Once stable, copy these into the `sharankumarreddyk/AI-Skills` repo under `claude-global-skills/skills/`. They'll join the existing collection (architectural-review, debugging, root-cause-analysis, etc.) and become part of your public skill portfolio.

---

## What's next (Phase 0)

With the skills in place, the next session can kick off Phase 0 of the project:

1. `pyproject.toml` with deps (pandas, lightgbm, pytorch, mlflow, dvc, evidently, bentoml, prefect, optuna, ruff, mypy, pytest)
2. `src/cluster_canary/` package skeleton (data/, features/, models/, training/, serving/, monitoring/, pipelines/)
3. `dvc init` + S3/MinIO remote stub
4. MLflow local server compose file
5. Pre-commit (ruff, mypy, dvc status)
6. GitHub Actions skeleton (lint, test, train-on-PR placeholder)
7. README with architecture diagram + results table placeholder
8. `tests/test_no_leakage.py` from `feature-leakage-detector`

Estimated effort: 2 days part-time, or one focused 4-hour session.
