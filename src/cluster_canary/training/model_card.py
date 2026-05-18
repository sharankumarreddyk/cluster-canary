"""Generate `MODEL_CARD.md` per the `model-card-writer` skill template.

Phase 4 outputs a card every time a model is registered/promoted. The card is
the inverse-of-marketing artefact — explicit out-of-scope uses, sliced
performance, fairness considerations, operational behavior.

Lives separately from `train.py` so it can be re-run against an existing
MLflow run without retraining (e.g. when promoting from `candidate` → `champion`).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from cluster_canary.training.eval import HeadlineMetrics


def render_model_card(
    *,
    model_name: str,
    version: str,
    git_sha: str,
    mlflow_run_id: str,
    dataset_version: str,
    data_window: tuple[str, str],
    feature_count: int,
    n_train: int,
    n_val: int,
    n_test: int,
    headline_test: HeadlineMetrics,
    lead_time: dict[str, float],
    slice_metrics: list[dict[str, Any]],
    sample_ratio: float,
    best_params: dict[str, Any],
    out_path: Path,
) -> Path:
    """Render and write the model card. Returns the path written."""
    today = dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d")
    text = _CARD_TEMPLATE.format(
        model_name=model_name,
        version=version,
        date=today,
        git_sha=git_sha[:7] if git_sha else "nogit",
        mlflow_run_id=mlflow_run_id,
        dataset_version=dataset_version,
        data_window_start=data_window[0],
        data_window_end=data_window[1],
        feature_count=feature_count,
        n_train=f"{n_train:,}",
        n_val=f"{n_val:,}",
        n_test=f"{n_test:,}",
        sample_ratio=sample_ratio,
        brier=headline_test.brier_score,
        pr_auc=headline_test.pr_auc,
        roc_auc=headline_test.roc_auc,
        prec_r5=headline_test.precision_at_recall_0_5,
        prec_r8=headline_test.precision_at_recall_0_8,
        threshold=headline_test.threshold_used,
        prec_at_t=headline_test.precision_at_threshold,
        rec_at_t=headline_test.recall_at_threshold,
        lead_time_json=json.dumps(lead_time, indent=2),
        slice_table=_render_slice_table(slice_metrics),
        params_json=json.dumps(best_params, indent=2, sort_keys=True),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return out_path


def _render_slice_table(slices: list[dict[str, Any]]) -> str:
    if not slices:
        return "_(no slices passed the min-rows threshold)_"
    header = "| Slice | Value | n_rows | n_pos | Brier | PR-AUC | Prec@R=0.8 |\n|---|---|---|---|---|---|---|"
    rows: list[str] = [header]
    for s in slices:
        rows.append(
            f"| {s['slice_col']} | {s['slice_value']} | {s['n_rows']} | "
            f"{s['n_positive']} | {s['brier_score']:.4f} | "
            f"{_fmt(s.get('pr_auc'))} | {_fmt(s.get('precision_at_recall_0_8'))} |"
        )
    return "\n".join(rows)


def _fmt(v: object) -> str:
    if v is None:
        return "—"
    if not isinstance(v, (int, float, str)):
        return str(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "—" if f != f else f"{f:.4f}"  # NaN check via self-inequality


_CARD_TEMPLATE: str = """# Model Card — {model_name}

**Version:** {version}  •  **Released:** {date}
**MLflow run:** {mlflow_run_id}  •  **Git SHA:** {git_sha}
**Owner:** sharankumarreddyk  •  **Contact:** see repo

## 1. Intended use

- **Primary use case:** Predict whether a running Kubernetes pod will experience
  an OOMKill within the next 30 minutes. Output is a calibrated probability with
  top-3 SHAP contributors so an on-call engineer (or `kubeai-ops`) can act
  proactively before user-visible failure.
- **Primary intended users:** in-cluster sidecar / SRE on-call / `kubeai-ops`
  action router.
- **Out-of-scope uses (do NOT use for):**
  - SLA enforcement / refund decisions — the model is probabilistic.
  - Predicting node failures, network outages, control-plane crashes.
  - Predicting pod failures in clusters whose workload mix differs materially
    from the training distribution (revalidate drift first).
  - Replacing kubelet's eviction decisions.

## 2. Training data

| Field | Value |
|---|---|
| Source | Phase 1 synthetic data lab (kind + chaos-mesh) and/or Phase 2 Alibaba Cluster Trace 2018 |
| Window covered | {data_window_start} → {data_window_end} |
| DVC dataset_version | `{dataset_version}` |
| Rows train / val / test | {n_train} / {n_val} / {n_test} |
| Negative downsampling ratio | {sample_ratio} (Elkan-corrected before calibration) |
| Sensitive attributes? | None — pod_uid / container / node are infrastructure IDs, not PII |
| Preprocessing summary | Long-form Prometheus parquet → wide pivot keyed by `(scrape_timestamp, pod_uid, container)`, feature engineering per `docs/features.md`, 60/20/20 temporal split |

## 3. Model details

| Field | Value |
|---|---|
| Architecture | LightGBM binary classifier (gbdt), Optuna 50-trial TPE search |
| Inputs | {feature_count} `feat_*`-prefixed features (see `data/processed/feature_list.json`) |
| Output | `P(OOM in next 30 min)` ∈ [0, 1], isotonic-calibrated |
| Imbalance handling | Negative downsampling on train + Elkan correction + isotonic calibration on val |
| Reproducibility | `dvc repro train` at git SHA {git_sha} |

Best Optuna params:

```json
{params_json}
```

## 4. Evaluation

### Overall (held-out test set)

| Metric | Value | PLAN gate |
|---|---|---|
| Brier score | {brier:.4f} | ≤ 0.10 |
| PR-AUC | {pr_auc:.4f} | — |
| ROC-AUC | {roc_auc:.4f} | — |
| Precision @ recall=0.5 | {prec_r5:.4f} | — |
| Precision @ recall=0.8 | {prec_r8:.4f} | ≥ 0.50 |
| Precision @ threshold={threshold} | {prec_at_t:.4f} | — |
| Recall @ threshold={threshold} | {rec_at_t:.4f} | — |

### Lead-time distribution (test set)

```json
{lead_time_json}
```

### Sliced performance

{slice_table}

## 5. Operational behavior

- **Inference path:** in-cluster sidecar (Phase 5 BentoML gRPC), TreeSHAP via
  `booster.predict(pred_contrib=True)` for top-3 explanations.
- **Latency budget:** p95 < 50 ms (model itself ~0.5-2 ms; remainder is
  feature prep + serialization).
- **Drift monitoring:** Phase 6 Evidently AI runs daily; reference = the
  training data window above.
- **Retraining trigger:** Phase 7 - data_drift_share > 0.5 OR
  current_PR-AUC < 0.7 x training PR-AUC for 2 consecutive checks.

## 6. Limitations

- Single-cluster training data. Generalization to clusters with very different
  workload mix is unproven.
- Lead-time of 30 min is at the optimistic end of published prior work
  (closest analog: Microsoft Narya, OSDI 2020 — AUC 0.85 at 30 min on Azure
  VMs). Performance degrades smoothly at longer horizons.
- The synthetic data lab uses a hand-tuned leaky-flask app for slow-leak
  signal; real-world memory leaks have more shapes than we've represented.
- Alibaba data spans only days 2-8 (issue #52 alignment); extending
  horizon-1-and-9 generalization is future work.

## 7. Fairness considerations

- No protected attributes are inputs. Pod IDs / namespaces are infrastructure
  identifiers, not human-correlated.
- Slice performance is reported per-namespace and per-node so over- or
  under-served slices are visible. The action router (Phase 6) should NOT
  apply differential thresholds by namespace unless explicitly justified.

## 8. How to use this model

```python
import lightgbm as lgb
booster = lgb.Booster(model_file="canary_model.txt")
# Single prediction with top-3 explanation:
from cluster_canary.training.shap_helpers import predict_with_explanation
predict_with_explanation(booster, x_row, model_version="{version}")
```

Service-level API: see `src/cluster_canary/serving/service.py` (Phase 5+).

## 9. Citation

If you use this model or methodology, cite the repository:
<https://github.com/sharankumarreddyk/cluster-canary>.

## 10. Changelog

| Version | Date | Change |
|---|---|---|
| {version} | {date} | Initial release |
"""
