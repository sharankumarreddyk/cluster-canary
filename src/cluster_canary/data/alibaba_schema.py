"""Alibaba Cluster Trace 2018 schema constants — verified against schema.txt
on 2026-05-18.

Five CSV files (no headers). Column orders are exact and authoritative; pass
`names=` when loading via pandas.

Trade-off note: Alibaba columns are normalized to [0, 100] (memory, network)
or per-100-cores (cpu_request/limit, where 100 == 1 core). The absolute
normalization factor is undisclosed (issue #223 open), so derived features must
stay *relative*. We surface the raw normalized values under `alibaba_` prefixed
metric_names rather than force-mapping them to the synthetic-lab's cAdvisor
metrics (which are absolute bytes). Phase 3 feature engineering computes the
SAME derived features (e.g. `mem_pct_of_limit`) on both datasets — alignment
happens at the feature layer, not at the metric layer.
"""

from __future__ import annotations

# Trace coverage: 8 days, ~4 K machines, one production cluster.
# Day-misalignment bug (issue #52): machine_* covers days 1-8 but
# container_* covers days 2-9. Only days 2-8 (7 days) are aligned.
ALIGNED_DAY_RANGE: tuple[int, int] = (2, 8)
SECONDS_PER_DAY: int = 86_400
# Synthetic origin for the trace's "seconds since start" timestamps. Arbitrary
# but stable so the parquet `scrape_timestamp` column has wall-clock-shaped
# values that downstream pandas plotting + temporal-split code handle naturally.
TRACE_ORIGIN_UTC: str = "2018-01-01T00:00:00Z"

# Sentinel values in `disk_io_percent`. Drop or null at harmonize time.
DISK_IO_INVALID_SENTINELS: tuple[int, ...] = (-1, 101)


# -- Column orderings (NO HEADER in the CSVs — pass these to read_csv `names=`) --

MACHINE_META_COLS: tuple[str, ...] = (
    "machine_id",
    "time_stamp",
    "failure_domain_1",
    "failure_domain_2",
    "cpu_num",
    "mem_size",
    "status",
)

MACHINE_USAGE_COLS: tuple[str, ...] = (
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "mkpi",
    "net_in",
    "net_out",
    "disk_io_percent",
)

CONTAINER_META_COLS: tuple[str, ...] = (
    "container_id",
    "machine_id",
    "time_stamp",
    "app_du",
    "status",
    "cpu_request",   # 100 == 1 core
    "cpu_limit",     # 100 == 1 core
    "mem_size",      # normalized [0, 100]
)

CONTAINER_USAGE_COLS: tuple[str, ...] = (
    "container_id",
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "cpi",
    "mem_gps",
    "mpki",
    "net_in",
    "net_out",
    "disk_io_percent",
)

BATCH_TASK_COLS: tuple[str, ...] = (
    "task_name",
    "instance_num",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
)

# -- Mapping: Alibaba container_usage columns → cluster-canary metric_names --

# These names are kept distinct from the synthetic-lab's `container_*_bytes`
# cAdvisor metrics because the units differ (normalized [0,100] here vs.
# absolute bytes there). Phase 3 features unify them.
CONTAINER_USAGE_METRIC_MAP: dict[str, str] = {
    "cpu_util_percent":  "alibaba_container_cpu_util_pct",
    "mem_util_percent":  "alibaba_container_mem_util_pct",
    "cpi":               "alibaba_container_cpi",
    "mem_gps":           "alibaba_container_mem_gps_pct",
    "mpki":              "alibaba_container_mpki",
    "net_in":            "alibaba_container_net_in_pct",
    "net_out":           "alibaba_container_net_out_pct",
    "disk_io_percent":   "alibaba_container_disk_io_pct",
}

# container_meta non-time, non-id columns → metric_names. Status is a string;
# we one-hot it at harmonize time into per-status boolean metric rows.
CONTAINER_META_METRIC_MAP: dict[str, str] = {
    "cpu_request":  "alibaba_container_cpu_request",
    "cpu_limit":    "alibaba_container_cpu_limit",
    "mem_size":     "alibaba_container_mem_size_pct",
}

# Status one-hot output metric. The actual enum values are not published; we
# encode them as `alibaba_container_status__<value>` (one metric per observed
# string value, == 1 at the timestamp the transition happens, 0 elsewhere).
CONTAINER_META_STATUS_METRIC_PREFIX: str = "alibaba_container_status__"

# -- Constants for the long-form output frame --

# Synthetic namespace for the entire trace. Phase 3 group/entity-overlap audits
# use this to keep alibaba rows out of join surface with synthetic-lab rows.
NAMESPACE_TAG: str = "alibaba_2018"
