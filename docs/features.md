# Features — Phase 3

Phase 3 turns the labeled wide parquet from Phase 1 (synthetic) and Phase 2
(Alibaba) into a model-ready train/val/test split under `data/processed/`.

The whole module is built around two non-negotiables:

1. **Strict left-closedness.** Every rolling feature is computed from samples
   in `[t − window, t)` — strictly past, never `t` itself. The leakage audit
   (`tests/test_no_leakage.py::test_rolling_windows_are_left_closed`) greps
   every `.rolling(` call in `src/` and fails the build if any is missing
   `closed='left'`. The canonical primitive is `_windowing.grouped_time_rolling`.
2. **Unified contract across sources.** A model trained on synthetic data is
   the same shape as one trained on Alibaba data. The unification happens at
   the feature layer, not the metric layer — see the dispatch logic in
   `memory.add_mem_pct_of_limit` for the canonical pattern.

## Feature catalogue

All feature columns start with `feat_*`. The list below is what the build
records to `data/processed/feature_list.json` for the leakage audit.

### Temporal (always present)
- `feat_hour_of_day` — `int8`, 0–23
- `feat_day_of_week` — `int8`, 0–6 (Mon = 0)
- `feat_is_weekend` — `int8`, 0/1
- `feat_is_business_hours` — `int8`, 0/1 (Mon–Fri, 09–18 UTC)

### Memory (the load-bearing predictors)
- `feat_mem_pct_of_limit` — unified `[0, 1]` (synthetic: `working_set / limit`; Alibaba: `mem_util_percent / 100`)
- `feat_mem_pct_of_limit__{5min,15min,30min}_{mean,max}` — rolling stats, left-closed
- `feat_mem_growth_rate__{5min,15min,30min}` — `(value(t) − value(t − w)) / value(t − w)`; the leading indicator per Microsoft Narya (OSDI 2020)
- `feat_mem_ws_rss_gap_pct` — synthetic only; `(working_set − rss) / limit`. The OOM-killer evaluates working-set, RSS lags — the gap matters
- `feat_mem_failcnt__5min_rate` — synthetic only; positive deltas of cgroup `memory.failcnt` over 5 min. The strongest *leading* indicator of OOM in published K8s anomaly-detection work (cgroup hit limit but didn't OOM yet)

### CPU
- `feat_cpu_util` — unified `[0, 1]` for Alibaba, raw rate for synthetic
- `feat_cpu_throttle_ratio` — synthetic only; `throttled_periods / total_periods`
- `feat_cpu_util__{5min,15min}_mean`
- `feat_cpu_throttle_ratio__{5min,15min}_mean`

### Lifecycle
- `feat_pod_age_sec` — seconds since first observation of `(pod_uid, container)`
- `feat_restart_rate_1h` — synthetic only; positive delta of `kube_pod_container_status_restarts_total` over 1 h

### Aggregations
- `feat_node_mem_pct_sum` — at each `(scrape_timestamp, node)`, sum of `feat_mem_pct_of_limit` across all containers
- `feat_node_mem_pct_max` — same, max
- `feat_node_n_containers` — count of distinct containers reporting on that node
- `feat_lineage_24h_oom_rate` — for each row, fraction of same-lineage rows in `[t − 24h, t)` labeled positive. Lineage = `app_du` for Alibaba; for synthetic, the leading prefix of `pod` (the deployment-name root) as a stand-in until an image-hash column is wired through `scraper.py`

## Source dispatch

The feature modules detect which source they're operating on by checking which
raw metric columns are present:

| Column | Source | Effect |
|---|---|---|
| `alibaba_container_mem_util_pct` | Alibaba 2018 | `feat_mem_pct_of_limit` = `<col> / 100` |
| `container_memory_working_set_bytes` + `kube_pod_container_resource_limits_memory_bytes` | Synthetic Phase 1 lab | `feat_mem_pct_of_limit` = ratio |
| neither | unknown | `NaN` (model handles via imputation) |

Same pattern for CPU. Synthetic-only features (`feat_mem_failcnt__5min_rate`,
`feat_cpu_throttle_ratio*`, `feat_restart_rate_1h`) produce `NaN` for Alibaba
rows; tree models handle this natively.

## Output layout

```
data/processed/
├── train.parquet                # ~60 % of rows, earliest in time
├── val.parquet                  # ~20 %, middle
├── test.parquet                 # ~20 %, latest
├── feature_list.json            # ordered feature_columns (deny-list-checked)
└── entity_overlap.json          # opt-in list — currently ["pod_uid"]

reports/eval/
├── feature_metrics.json         # rows in/out, n_features
└── split_metrics.json           # n_train/val/test + per-split positive rates
```

`entity_overlap.json` matters because the leakage audit
(`test_entity_overlap_documented`) flags any `_id` column whose values
overlap across splits. For cluster-canary `pod_uid` IS expected to overlap by
design (we're learning per-pod patterns over time, not generalizing to unseen
pods), so we opt it in.

## What's NOT in this phase

- **Image hash as a feature** for the synthetic side. Phase 1 doesn't scrape
  it; the lineage feature falls back to the pod-name prefix as a soft proxy.
  Promotion to a real feature is a Phase 4 polish item.
- **Z-scoring or normalization across sources.** Tree models don't need it.
  If we add a neural challenger in Phase 4 (PyTorch 1D-CNN per
  `docs/PLAN.md`), per-source z-scoring would happen there.
- **Smarter imputation.** `NaN`s pass through to the model; LightGBM handles
  them natively, but a linear / NN model would need explicit imputation.

## How to run

```bash
# Once Phase 1 and Phase 2 outputs exist:
make features          # → data/processed/{train,val,test}.parquet + feature_list.json

# Or via DVC:
dvc repro features

# Inspect:
uv run python -c "import pandas as pd; print(pd.read_parquet('data/processed/train.parquet').head())"
cat data/processed/feature_list.json
cat reports/eval/split_metrics.json | jq .
```

## Verification

The acceptance criterion from `docs/PLAN.md` is:

> All tests in `test_no_leakage.py` pass on real data (`pytest -m needs_data`).
> `data/processed/feature_list.json` exists and contains no entries from the
> deny-list.

The 5 deselected leakage tests become runnable as soon as
`data/processed/train.parquet` is present:

```bash
make features
uv run pytest -m needs_data
```
