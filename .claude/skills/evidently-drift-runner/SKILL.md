---
name: evidently-drift-runner
description: Scaffolds Evidently AI drift reports correctly — reference vs current window selection, the right statistical test per feature type, threshold tuning, HTML+JSON+Prometheus outputs, and CI/CD integration. Use whenever code monitors model inputs, predictions, or performance over time. Most engineers misconfigure the reference window — this skill prevents that.
---

# Evidently Drift Runner

## When this skill applies

Trigger when code:
- Imports `evidently` or computes any drift metric (PSI, KS, JS divergence, Wasserstein).
- Sets up monitoring for an ML model in production.
- Builds a scheduled job that compares "current" production data against a baseline.
- Adds drift-based alerts or auto-retraining triggers.

If drift detection is being added without these conventions, fix before declaring done.

## The fundamental decision: what is "reference" vs "current"?

This is where 80% of drift setups go wrong. Get it right at the start.

### Reference window — the "what good looks like" baseline

Reference = **the data the currently-deployed model was trained on**. NOT the most recent N days. NOT a rolling window. It's frozen at training time and updates only when a new model is promoted to production.

```python
# Right
reference = pd.read_parquet("data/processed/train_2024Q3.parquet")  # the prod model's training data

# Wrong — this is a rolling window comparison, useful for change detection but not for "is the model stale"
reference = current.shift(-30)
```

Store reference alongside the model in the registry. When the model is promoted, snapshot the reference. When drift is computed, pull reference from registry.

### Current window — what's happening now

Current = **the most recent production window** sized for statistical power.
- For high-volume models (>10k preds/day): rolling 7-day window, computed daily.
- For low-volume (<1k preds/day): rolling 30-day window, computed weekly.
- Minimum 1000 rows for any drift test to be statistically meaningful. Below that, report `n_too_small=true` and skip.

```python
current = predictions[predictions["pred_ts"] >= now - timedelta(days=7)]
if len(current) < 1000:
    log.warning("Skipping drift: n=%d below minimum", len(current))
    return {"status": "skipped", "reason": "insufficient_data"}
```

## Right statistical test per feature type

Evidently picks defaults; verify they match the table below. Override with `column_mapping` when needed.

| Feature type | Test | Threshold | Why |
|---|---|---|---|
| Numeric, low cardinality | Wasserstein | 0.1 | Robust, scale-aware |
| Numeric, continuous | KS or PSI | 0.2 (PSI) / 0.05 (KS p-value) | Industry standard for tabular |
| High-cardinality categorical | Chi-square or PSI on top-k | 0.2 | Avoids combinatorial blow-up |
| Low-cardinality categorical | Chi-square | p < 0.05 | Standard |
| Text features (embeddings) | MMD or cosine drift | task-dependent | Distribution in embedding space |
| Target | Wasserstein + label distribution chi-sq | 0.1 + p<0.05 | Catches concept drift early |
| Predictions | Wasserstein on output distribution | 0.1 | Cheap proxy when ground-truth lags |

## The four reports to run (in this order)

### 1. Data drift (features)
Have feature distributions shifted from reference?
```python
from evidently.report import Report
from evidently.metrics import DataDriftPreset
report = Report(metrics=[DataDriftPreset(num_stattest='wasserstein', num_stattest_threshold=0.1)])
report.run(reference_data=reference, current_data=current, column_mapping=mapping)
```

### 2. Prediction drift
Has the model's output distribution shifted? Often the earliest signal because predictions are observed instantly.
```python
from evidently.metrics import ColumnDriftMetric
report = Report(metrics=[ColumnDriftMetric(column_name="prediction", stattest="wasserstein", stattest_threshold=0.1)])
```

### 3. Target drift (when ground truth available)
Has the label distribution shifted? When ground truth is observed quickly (e.g. completion times, immediate purchase outcomes), this is the most actionable signal. When truth lags (credit default, churn over 90 days), expect delayed detection here.

### 4. Performance regression
When truth is in, recompute RMSE/MAPE on the current window and compare to the model's MLflow-logged test RMSE.
```python
from evidently.metrics import RegressionPerformanceMetrics
```

## Outputs every drift run produces

A run is incomplete without all of these:

1. **`drift_report_<YYYYMMDD>.html`** — human-readable report saved to S3/MinIO under `reports/drift/`.
2. **`drift_metrics.json`** — machine-readable metrics for downstream automation:
   ```json
   {
     "run_id": "drift_20260518T1200",
     "model_version": "5",
     "reference_window": ["2024-07-01", "2024-09-30"],
     "current_window": ["2026-05-11", "2026-05-17"],
     "n_reference": 4500000, "n_current": 87000,
     "data_drift_share": 0.34, "data_drift_detected": true,
     "drifted_columns": ["feature_a", "feature_b", "feature_c"],
     "prediction_drift": {"stattest": "wasserstein", "value": 0.18, "threshold": 0.1},
     "target_drift":     {"stattest": "wasserstein", "value": 0.04, "threshold": 0.1, "detected": false},
     "current_rmse": 3.41, "training_rmse": 2.18, "rmse_regression_pct": 56.4
   }
   ```
3. **Prometheus metrics** — emit via `prometheus_client` Pushgateway or sidecar:
   - `model_drift_share{model="<model_name>", version="5"} 0.34`
   - `model_prediction_drift_wasserstein{...} 0.18`
   - `model_performance_rmse_current{...} 3.41`
4. **MLflow tag** on the production model run: `latest_drift_check=<timestamp>`, `latest_drift_detected=true|false`.

## Alert thresholds (the retraining trigger)

A retraining run is triggered when **any** of the following fires for 2 consecutive drift checks:
- `data_drift_share > 0.5` — majority of features drifting
- `prediction_drift_wasserstein > 0.15` — strong output shift
- `target_drift_detected == true` — concept drift
- `current_rmse > 1.3 * training_rmse` — performance regression

Single-check triggers cause noisy retrains. The 2-of-N rule is the standard pattern.

## Starter template

```python
# src/cluster_canary/monitoring/drift.py
from pathlib import Path
import json, datetime as dt
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metrics import (
    DataDriftPreset, ColumnDriftMetric, RegressionPerformanceMetrics,
)
from evidently.pipeline.column_mapping import ColumnMapping
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

MIN_ROWS = 1_000

def run_drift_check(model_uri: str, reference: pd.DataFrame, current: pd.DataFrame,
                    feature_cols: list[str], pred_col: str = "prediction",
                    target_col: str | None = None) -> dict:
    if len(current) < MIN_ROWS:
        return {"status": "skipped", "n_current": len(current)}

    mapping = ColumnMapping(
        numerical_features=[c for c in feature_cols if reference[c].dtype.kind in "fi"],
        categorical_features=[c for c in feature_cols if reference[c].dtype.kind == "O"],
        prediction=pred_col,
        target=target_col,
    )
    metrics = [
        DataDriftPreset(num_stattest="wasserstein", num_stattest_threshold=0.1),
        ColumnDriftMetric(column_name=pred_col, stattest="wasserstein", stattest_threshold=0.1),
    ]
    if target_col:
        metrics.append(RegressionPerformanceMetrics())

    report = Report(metrics=metrics)
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M")
    html_path = Path(f"reports/drift/drift_report_{ts}.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(html_path))

    metrics_dict = report.as_dict()
    summary = _summarize(metrics_dict, model_uri)
    Path(f"reports/drift/drift_metrics_{ts}.json").write_text(json.dumps(summary, indent=2, default=str))

    _push_to_prometheus(summary)
    return summary
```

## Anti-patterns to refuse

- Reference window = "last 30 days, rolling". That measures change, not staleness.
- Running drift on <1000 rows and reporting "no drift" — false negative from low power.
- Single-feature drift alerts firing all the time. Aggregate to `data_drift_share` for the alert; keep per-feature for diagnosis.
- No prediction drift check (people often skip this; it's the cheapest and earliest signal).
- Storing reports as PNGs only. HTML + JSON, always.
