---
name: mlflow-tracking-conventions
description: Enforces a strict, grep-able MLflow logging convention so every run is reproducible and comparable. Use whenever code creates an MLflow run, trains a model, or compares experiments. Prevents "untitled run #847" anti-patterns and ensures every artifact a hiring manager or future-you would want is logged.
---

# MLflow Tracking Conventions

## When this skill applies

Apply this skill whenever the code touches any of these:
- `mlflow.start_run(...)`, `mlflow.log_*`, `mlflow.set_experiment(...)`
- A training script, hyperparameter sweep, or eval script
- The MLflow model registry (`register_model`, stage transitions)
- Any new file in `src/**/training/`, `src/**/models/`, or `notebooks/` that fits a model

If a script does any of the above without following the rules below, fix it before declaring the task done.

## The contract — every run must log all of this

### 1. Experiment & run naming
- Experiment name: `<project>__<task>` — e.g. `myproject__main_task`. Set via `MLFLOW_EXPERIMENT_NAME` or `mlflow.set_experiment(...)` at the top of every entrypoint.
- Run name: `<model_family>__<dataset_version>__<git_sha[:7]>__<timestamp>` — e.g. `lgbm__v1_2026_q1__a4f9c12__20260518T1430`.
- No untitled runs. Ever.

### 2. Required tags (use `mlflow.set_tags`)
```python
mlflow.set_tags({
    "git_sha": git_sha,                 # full SHA, dirty=true|false
    "git_branch": branch,
    "git_dirty": str(is_dirty),
    "dataset_version": dvc_data_version, # `dvc status` short hash of input
    "author": os.environ["USER"],
    "model_family": "lgbm",              # lgbm | xgboost | pytorch_mlp | sklearn_baseline
    "stage": "experiment",               # experiment | candidate | challenger | champion
    "data_window_start": "2024-07-01",
    "data_window_end":   "2024-09-30",
})
```

### 3. Required params (use `mlflow.log_params`)
- All model hyperparameters (LightGBM/XGBoost params; PyTorch lr/batch/epochs/optimizer).
- All data params: `n_train`, `n_val`, `n_test`, `train_window`, `val_window`, `cv_strategy`.
- All feature params: `n_features`, `feature_set_name`, `target_transform` (e.g. `"log1p"` or `"none"`).

### 4. Required metrics (use `mlflow.log_metrics`)
For regression:
- `rmse`, `mae`, `mape`, `r2` — on train, val, test, AND a **temporal holdout** (later window if time-series).
- `p50_abs_err`, `p95_abs_err`, `p99_abs_err`.
- `inference_latency_ms_p50`, `inference_latency_ms_p95` — measured on a 1k-row sample.
- `model_size_mb`.

For classification:
- `roc_auc`, `pr_auc`, `f1`, `precision`, `recall` per class.
- Per-slice metrics (see slicing below).

### 5. Required artifacts (use `mlflow.log_artifact` / `mlflow.<flavor>.log_model`)
- The serialized model (use the right flavor — `mlflow.lightgbm.log_model`, `mlflow.pytorch.log_model`).
- `requirements.txt` (frozen at training time) and `conda.yaml`.
- `feature_list.json` — exact ordered feature names.
- `eval_report.html` — full eval breakdown.
- `residual_plot.png` (regression) or `confusion_matrix.png` (classification).
- `feature_importance.png` (or SHAP summary) for tree models.
- `slice_metrics.json` — metrics broken down by every slice column (see below).
- `input_schema.json` — column names + dtypes + nullable.

### 6. Slicing (don't skip — interviewers ask)
Always log metrics sliced by the natural subgroups in your data — typical slice axes include:
- Geographic / location bucket (region, country, zone)
- Temporal bucket (hour-of-day, day-of-week, season)
- Magnitude bucket (small / medium / large value bins)
- Categorical (top-N category, then "other")
- Demographic / cohort (when relevant — surface fairness gaps)

Save to `slice_metrics.json` and assert no slice has worst-metric > 1.5× the overall (if it does, log a warning tag `imbalanced_slice=true`).

## The starter template

Every training entrypoint starts with this block. Copy it verbatim.

```python
import os, subprocess, datetime as dt
from pathlib import Path
import mlflow

def _git_meta() -> dict:
    sha    = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    dirty  = bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())
    return {"git_sha": sha, "git_branch": branch, "git_dirty": str(dirty)}

def start_tracked_run(*, project: str, task: str, model_family: str,
                      dataset_version: str, data_window: tuple[str, str]):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(f"{project}__{task}")
    git = _git_meta()
    ts  = dt.datetime.utcnow().strftime("%Y%m%dT%H%M")
    run_name = f"{model_family}__{dataset_version}__{git['git_sha'][:7]}__{ts}"
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tags({
        **git,
        "dataset_version": dataset_version,
        "author": os.environ.get("USER", "unknown"),
        "model_family": model_family,
        "stage": "experiment",
        "data_window_start": data_window[0],
        "data_window_end":   data_window[1],
    })
    return run
```

## Registry promotion rules

Promotions go: `experiment` → `candidate` → `challenger` → `champion`.

- A run becomes a `candidate` only if `temporal_holdout_rmse <= 0.98 * current_champion_rmse` AND `p95_abs_err <= 1.05 * champion_p95`.
- A `candidate` becomes a `challenger` after a successful shadow-deploy for ≥ 24h with `p95_latency_ms <= 100`.
- A `challenger` becomes `champion` only via an explicit promotion PR with the comparison plot attached.

Never overwrite the `Production` stage in the registry. Always use stage transitions with `archive_existing_versions=True`.

## Anti-patterns to refuse

- Logging metrics with names like `loss`, `score`, `acc` — these are ambiguous. Use `val_rmse`, `test_mape`, etc.
- Calling `mlflow.start_run()` without `mlflow.set_experiment(...)` upstream.
- Logging `model.pkl` as a generic artifact instead of using the correct flavor.
- Hyperparameter sweeps that log only the best run. Log all of them, mark best with tag `is_best=true`.
- Logging plots as base64 strings in a metric. Use `log_figure`/`log_artifact`.
