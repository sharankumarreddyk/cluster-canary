# Modeling — Phase 4

`make train` runs the full Phase 4 pipeline against `data/processed/{train,val,test}.parquet`:

1. **Baseline rule** — `feat_mem_pct_of_limit__5min_max > 0.9` → fire. The
   reference everything else has to beat.
2. **LightGBM** — Optuna 50-trial TPE search; isotonic-calibrated;
   MLflow-tracked.
3. **Evaluation** — Brier, PR-AUC, precision @ recall=0.5 / 0.8, lead-time at
   5 / 10 / 15 / 30 min, plus slice metrics by namespace and node.
4. **Model card** — `MODEL_CARD.md` generated and logged as an MLflow artifact.

## The PLAN.md acceptance gates

```
Brier score             ≤ 0.10
Precision @ recall=0.8  ≥ 0.50
```

Both reported as `acceptance_*` MLflow metrics so promotion gates can read them
directly.

## Class imbalance handling

At ~0.1 % positive pod-minutes (typical for cluster-canary):

1. **Negative downsample at training time** to bring positives to
   `TARGET_POSITIVE_RATE` (default 0.10). Implementation:
   `cluster_canary.training.imbalance.downsample_negatives`. Preserves
   temporal order so the model can't see future negatives by accident.
2. **Train LightGBM** with `scale_pos_weight=1.0` (data already balanced via
   downsampling).
3. **Elkan probability correction** on val/test predictions to map back to the
   ORIGINAL production prior:
   ```
   p_true = p_train / (p_train + (1 - p_train) / sample_ratio)
   ```
   Source: Elkan, IJCAI 2001.
4. **Isotonic calibration** on val (held-out, original prior) absorbs residual
   miscalibration. Platt is a fallback for slices with < 1000 positives.

## Optuna search

50 trials. `TPESampler(multivariate=True, group=True, n_startup_trials=10)` —
TPE outperforms random at this budget; CMA-ES would need > 1000 trials. Pruner
is `HyperbandPruner` paired with `LightGBMPruningCallback`. The objective is
**average precision (AUC-PR)** — Brier alone is misleading at extreme
imbalance (a constant `p=0` baseline scores Brier ≈ 0.001 — useless target).

Search space:

| Param | Range |
|---|---|
| `learning_rate` | log [0.01, 0.1] |
| `num_leaves` | int [15, 255] |
| `max_depth` | int [4, 12] |
| `min_data_in_leaf` | int log [5, 200] — CRITICAL at 0.1 % positive (default 20 too high) |
| `feature_fraction` | [0.5, 1.0] |
| `bagging_fraction` | [0.5, 1.0] |
| `lambda_l1`, `lambda_l2` | log [1e-8, 10.0] |
| `min_gain_to_split` | [0.0, 0.5] |
| `n_estimators` | not tuned — early stopping at 75 rounds picks |

## Why these design decisions

| Choice | Why |
|---|---|
| LightGBM over XGBoost | Faster at the search budget; near-identical accuracy on tabular |
| Isotonic over Platt by default | Dominates Platt at ≥ 1000 positives in the calibration set; our val typically has > 10k |
| TreeSHAP via `pred_contrib=True` | Same algorithm as `shap.TreeExplainer`, faster, no extra dep, stable API since 2017 |
| MLflow tags via `start_tracked_run` | Enforces the contract in `.claude/skills/mlflow-tracking-conventions/SKILL.md` — every run has the same grep-able tag schema |
| Lead-time at multiple horizons | The 30-min target is at the optimistic edge of published prior (Narya OSDI 2020); reporting 5/10/15/30 makes the precision-vs-horizon tradeoff visible |

## Inference path (Phase 5 preview)

`predict_with_explanation` in `src/cluster_canary/training/shap_helpers.py`
returns the JSON schema the in-cluster gRPC sidecar serves:

```json
{
  "prediction": 0.87,
  "prediction_logodds": 1.89,
  "base_value": -2.10,
  "model_version": "canary-v0.1.0",
  "top_contributors": [
    {
      "feature": "feat_mem_pct_of_limit__5min_max",
      "contribution_logodds": 0.42,
      "contribution_prob_delta": 0.08,
      "direction": "up",
      "rank": 1
    }
  ]
}
```

`contribution_logodds` is the TreeSHAP value (log-odds; LightGBM's binary
objective is log-odds). `contribution_prob_delta` is the approximate marginal
probability change from this feature — sigmoid is non-linear, so it doesn't
compose additively, but it's the right thing to show on an on-call dashboard
labelled clearly. The `base_value` is stripped from `top_contributors`
(it's E[f(X)], not a feature contribution).

Per-row TreeSHAP latency is ~0.5–2 ms on a single CPU core for a 50-feature,
500-tree model — well under the 50 ms p95 inference budget.

## How to run

```bash
# Once Phase 3 has produced data/processed/:
make train             # 50-trial Optuna search, ~30 min depending on data size
make train-fast        # OPTUNA_N_TRIALS=5 — for smoke testing the path

# Or via DVC:
dvc repro train

# Outputs:
#   models/canary_model.txt
#   models/calibrator.pkl
#   models/feature_list.json
#   models/MODEL_CARD.md
#   reports/eval/test_metrics.json
#   reports/eval/slice_metrics.json
#   reports/eval/baseline_test_metrics.json

# MLflow:
make mlflow-ui   # http://localhost:5000 — find the runs in cluster_canary__oom_30min
```

## Acceptance verification

The two PLAN.md gates are computed automatically and logged to MLflow as
`acceptance_brier_le_0_10` and `acceptance_precision_at_recall_0_8_ge_0_5`.
The orchestrator promotes the run from `stage=experiment` to
`stage=candidate` only when BOTH gates pass. The model card records
which gates passed for any reviewer.
