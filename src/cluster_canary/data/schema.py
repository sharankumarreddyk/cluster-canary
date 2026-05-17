"""Shared column constants + Prometheus query catalogue for cluster-canary.

Single source of truth for:
- the metric set the scraper pulls
- the column names the labeler reads
- the dtype contract the parquet writer enforces

If a downstream module (features, training) needs to add a column, add the
PromQL here, give it a stable column name, and the scraper will pick it up
automatically on the next run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    """One Prometheus query that becomes one column in the long-form frame.

    `query` is a PromQL expression evaluated with `query_range`. It SHOULD use
    `topk()` or aggregation when the underlying metric is high-cardinality, to
    stay under Prometheus's `max_samples` ceiling.

    `column` is the canonical name the scraper writes; downstream code reads it
    from `schema.METRIC_COLUMNS`.
    """

    column: str
    query: str
    description: str


# -- Memory features (the primary OOM predictors) --------------------------- #
MEMORY_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        column="container_memory_working_set_bytes",
        query='container_memory_working_set_bytes{container!="",container!="POD"}',
        description="What the OOM-killer evaluates. Primary OOM predictor.",
    ),
    MetricSpec(
        column="container_memory_rss",
        query='container_memory_rss{container!="",container!="POD"}',
        description="Resident set size; lags working-set. The gap matters.",
    ),
    MetricSpec(
        column="container_memory_cache",
        query='container_memory_cache{container!="",container!="POD"}',
        description="Page cache — distinguishes leak vs file-backed memory.",
    ),
    MetricSpec(
        column="container_memory_failcnt",
        query='container_memory_failcnt{container!="",container!="POD"}',
        description=(
            "Times the cgroup hit its limit but did NOT OOM. Strongest leading indicator."
        ),
    ),
    MetricSpec(
        column="container_memory_max_usage_bytes",
        query='container_memory_max_usage_bytes{container!="",container!="POD"}',
        description="Peak observed memory; not reset per restart.",
    ),
)

# -- CPU / throttling features --------------------------------------------- #
CPU_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        column="container_cpu_usage_seconds_rate",
        query=(
            'rate(container_cpu_usage_seconds_total{container!="",container!="POD"}[1m])'
        ),
        description="1-min rate of CPU consumption (cores).",
    ),
    MetricSpec(
        column="container_cpu_throttled_periods_rate",
        query=(
            'rate(container_cpu_cfs_throttled_periods_total'
            '{container!="",container!="POD"}[1m])'
        ),
        description="Throttled CFS periods per second. Counterintuitive predictor.",
    ),
    MetricSpec(
        column="container_cpu_periods_rate",
        query=(
            'rate(container_cpu_cfs_periods_total{container!="",container!="POD"}[1m])'
        ),
        description="Total CFS periods per second. Pair with throttled for a ratio.",
    ),
)

# -- OOM ground-truth signals (the LABEL inputs, NOT features) ------------- #
OOM_GROUND_TRUTH_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        column="container_oom_events_total",
        query='container_oom_events_total{container!="",container!="POD"}',
        description=(
            "cAdvisor counter. Primary OOM ground truth — delta > 0 means OOM happened."
        ),
    ),
    MetricSpec(
        column="kube_pod_container_status_last_terminated_reason_oomkilled",
        query=(
            'kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}'
        ),
        description=(
            "ksm gauge; transitions 0→1 when last termination reason was OOMKilled. "
            "Pin ksm >= v2.13 — older versions emit `Error` instead."
        ),
    ),
    MetricSpec(
        column="kube_pod_container_status_restarts_total",
        query='kube_pod_container_status_restarts_total',
        description="Restart counter; delta > 0 means a restart happened.",
    ),
)

# -- Pod / container metadata (no rate; just current state) ---------------- #
METADATA_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        column="kube_pod_container_resource_requests_memory_bytes",
        query='kube_pod_container_resource_requests{resource="memory"}',
        description="Memory request — denominator for utilization.",
    ),
    MetricSpec(
        column="kube_pod_container_resource_limits_memory_bytes",
        query='kube_pod_container_resource_limits{resource="memory"}',
        description="Memory limit — denominator for OOM-proximity.",
    ),
    MetricSpec(
        column="kube_pod_info",
        query="kube_pod_info",
        description="Pod -> node, namespace, uid. Join key for everything.",
    ),
)

ALL_METRICS: tuple[MetricSpec, ...] = (
    MEMORY_METRICS + CPU_METRICS + OOM_GROUND_TRUTH_METRICS + METADATA_METRICS
)

METRIC_COLUMNS: tuple[str, ...] = tuple(m.column for m in ALL_METRICS)

# -- Long-form frame contract --------------------------------------------- #
# After scraping, the canonical layout is:
#   scrape_timestamp  | datetime64[ns, UTC] | scrape time, == evaluation time
#   metric_name       | string              | one of METRIC_COLUMNS
#   value             | float64             | observation
#   namespace         | string              | from label
#   pod               | string              | from label
#   pod_uid           | string              | join key across restarts (preferred over pod name)
#   container         | string              | from label, empty for pod-level metrics
#   node              | string              | from label or kube_pod_info
LONG_FRAME_COLUMNS: tuple[str, ...] = (
    "scrape_timestamp",
    "metric_name",
    "value",
    "namespace",
    "pod",
    "pod_uid",
    "container",
    "node",
)

LONG_FRAME_DTYPES: dict[str, str] = {
    "scrape_timestamp": "datetime64[ns, UTC]",
    "metric_name": "string",
    "value": "float64",
    "namespace": "string",
    "pod": "string",
    "pod_uid": "string",
    "container": "string",
    "node": "string",
}

# Labels to keep from each Prometheus result. Anything not in this set is
# discarded at scrape time to keep parquet rows narrow and joinable.
KEPT_LABELS: tuple[str, ...] = ("namespace", "pod", "uid", "container", "node", "reason")

# Window-labeling parameters (overridable via env in pipelines/scrape.py).
DEFAULT_LEAD_MINUTES: int = 30
DEFAULT_COOLDOWN_MINUTES: int = 5
DEFAULT_SCRAPE_STEP_SEC: int = 15
DEFAULT_CHUNK_MINUTES: int = 15  # query_range chunk size to stay under max_samples
DEFAULT_MAX_WORKERS: int = 8     # ThreadPoolExecutor for per-metric parallelism
