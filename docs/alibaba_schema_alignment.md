# Alibaba 2018 → cluster-canary schema alignment

The Phase 2 harmonizer ports the Alibaba Cluster Trace 2018 CSV files into the
same long-form parquet contract used by the Phase 1 Prometheus scraper. This
doc lists every column mapping so feature engineering (Phase 3) treats Alibaba
and the synthetic lab uniformly.

## Long-form contract (recap)

Every row in `data/interim/alibaba/*.parquet` matches the contract from
[`src/cluster_canary/data/schema.py`](../src/cluster_canary/data/schema.py):

| Column | Type | Source for Alibaba rows |
|---|---|---|
| `scrape_timestamp` | `datetime64[ns, UTC]` | `time_stamp` (seconds since trace start) → `2018-01-01T00:00:00Z + Δseconds` |
| `metric_name` | `string` | derived per source column — see mapping tables below |
| `value` | `float64` | source column value, NaN-coerced |
| `namespace` | `string` | constant `"alibaba_2018"` |
| `pod` / `pod_uid` / `container` | `string` | `container_id` for container-level rows, `machine_id` for machine-level rows |
| `node` | `string` | `machine_id` |

## `container_usage.csv` → 8 metric_names

Per-container metrics, ~10 s non-uniform sampling, ~28 GB raw.

| Alibaba column | metric_name | Units | Notes |
|---|---|---|---|
| `cpu_util_percent` | `alibaba_container_cpu_util_pct` | 0–100 | Per-container CPU utilization |
| `mem_util_percent` | `alibaba_container_mem_util_pct` | 0–100 | **Primary heartbeat + OOM-predictor metric.** Memory utilization. |
| `cpi` | `alibaba_container_cpi` | cycles/instr | Hardware perf counter |
| `mem_gps` | `alibaba_container_mem_gps_pct` | 0–100 | Memory bandwidth (normalized) |
| `mpki` | `alibaba_container_mpki` | misses/1k instr | Cache misses per 1000 instructions |
| `net_in` | `alibaba_container_net_in_pct` | 0–100 | Network ingress (normalized) |
| `net_out` | `alibaba_container_net_out_pct` | 0–100 | Network egress (normalized) |
| `disk_io_percent` | `alibaba_container_disk_io_pct` | 0–100 | Disk I/O (rows with `-1` / `101` sentinels dropped) |

## `container_meta.csv` → 3 numeric metrics + dynamic one-hot status

Container lifecycle events, ~2.4 MB.

| Alibaba column | metric_name | Units |
|---|---|---|
| `cpu_request` | `alibaba_container_cpu_request` | 100 = 1 core |
| `cpu_limit` | `alibaba_container_cpu_limit` | 100 = 1 core |
| `mem_size` | `alibaba_container_mem_size_pct` | 0–100 (normalized) |
| `status` | `alibaba_container_status__<observed-value>` | 0 or 1 — one-hot per observed string |

The `status` enum is **undocumented** (issue #223 open since 2026-04 — Alibaba never published it). The harmonizer enumerates unique values empirically and emits one metric per value. Logs the discovered set at runtime so we can audit unknowns.

Common community-reported values: `started`, `stopped`, possibly `Terminated` / `allocating` — treat as unverified until empirically observed.

## Important: unit difference vs the Phase 1 synthetic lab

The synthetic Prometheus scraper produces **absolute-byte** memory metrics
(`container_memory_working_set_bytes` etc.) because cAdvisor exposes absolute
values. Alibaba 2018 ships **normalized 0–100 utilization** with the actual
normalization factor (RAM-per-machine, cores-per-machine) undisclosed.

→ The two datasets are NOT directly stackable on memory-bytes. The Phase 3
feature engineering must compute the same DERIVED features
(`mem_pct_of_limit`, `mem_growth_rate_per_min`, etc.) on both sources;
alignment happens at the feature layer, not the metric layer.

## Day-misalignment correction

Per [issue #52](https://github.com/alibaba/clusterdata/issues/52), the trace's
`machine_*` files cover days 1–8 but `container_*` files cover days 2–9. Only
**days 2–8 (7 days)** are aligned across all sources. The harmonizer drops
rows outside this window with the constants `ALIGNED_DAY_RANGE = (2, 8)` in
[`alibaba_schema.py`](../src/cluster_canary/data/alibaba_schema.py).

## OOM event derivation (Phase 2 specific)

The Alibaba trace has no explicit OOM-kill label. The
[`alibaba_oom_detector`](../src/cluster_canary/data/alibaba_oom_detector.py)
module derives events via:

1. **Disappearance.** Container's last `alibaba_container_mem_util_pct` sample is
   more than `disappearance_threshold_sec` (default 5 min) before the trace end.
2. **Memory saturation.** Average of the last `mem_saturation_lookback_samples`
   memory-util samples ≥ `mem_saturation_threshold_pct` (default 95 %).

Both conditions must hold. Status transitions in `container_meta` are not
used as primary signal because the enum is undocumented; once the empirical
enum is known after the first real run, we can promote it from soft to hard
signal.

## Constants reference

All constants live in [`alibaba_schema.py`](../src/cluster_canary/data/alibaba_schema.py):

- `ALIGNED_DAY_RANGE = (2, 8)`
- `TRACE_ORIGIN_UTC = "2018-01-01T00:00:00Z"` (arbitrary but stable origin)
- `NAMESPACE_TAG = "alibaba_2018"`
- `CONTAINER_USAGE_METRIC_MAP` — the 8-entry mapping above
- `CONTAINER_META_METRIC_MAP` — the 3-entry mapping above
- `DISK_IO_INVALID_SENTINELS = (-1, 101)`
