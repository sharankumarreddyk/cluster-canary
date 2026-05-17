"""Prometheus HTTP API → parquet scraper for cluster-canary.

Pulls the metric set defined in `schema.ALL_METRICS` from a Prometheus
instance via `query_range`, chunks large windows to stay under
`--query.max-samples`, runs per-metric requests in a thread pool, and
writes one parquet row group per (date, hour) partition under
`data/raw/scrape/`.

Why HTTP API + parquet, not `remote_write`:
- Re-queryable when feature definitions change without re-running the lab.
- No Thanos / Cortex / sidecar — fits in a 16 GB MacBook dev loop.
- Parquet partitioning by hour falls out for free.

Idempotency: re-running over an already-scraped window overwrites the
target parquet file deterministically (same chunk → same path).

Usage:
    python -m cluster_canary.data.scraper \\
        --start 2026-05-18T00:00 --end 2026-05-19T00:00

    # or via env:
    PROMETHEUS_URL=http://localhost:9090 \\
    SCRAPE_START=2026-05-18T00:00 \\
    SCRAPE_END=2026-05-19T00:00 \\
    python -m cluster_canary.data.scraper
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.exceptions import PrometheusApiClientException

from cluster_canary.data.schema import (
    ALL_METRICS,
    DEFAULT_CHUNK_MINUTES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_SCRAPE_STEP_SEC,
    KEPT_LABELS,
    LONG_FRAME_COLUMNS,
    LONG_FRAME_DTYPES,
    MetricSpec,
)

log = structlog.get_logger(__name__)

DEFAULT_RAW_DIR = Path("data/raw/scrape")


class ScrapeError(RuntimeError):
    """Raised when a scrape cannot complete after the configured retries."""


@dataclass(frozen=True)
class Chunk:
    """A (start, end) sub-window of a larger scrape range."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"chunk end {self.end} must be > start {self.start}")


def _chunks(
    start: datetime,
    end: datetime,
    *,
    chunk: timedelta,
) -> Iterator[Chunk]:
    """Yield fixed-width sub-windows of [start, end). The last window may be short."""
    if end <= start:
        raise ValueError(f"window end {end} must be > start {start}")
    cur = start
    while cur < end:
        nxt = min(cur + chunk, end)
        yield Chunk(start=cur, end=nxt)
        cur = nxt


def _result_to_rows(
    raw: list[dict[str, Any]],
    *,
    metric_name: str,
) -> list[dict[str, Any]]:
    """Flatten a Prometheus `data.result` payload to long-form rows.

    Drops labels not in `KEPT_LABELS` to keep parquet narrow. The Prometheus
    wire format ships values as strings (per the prometheus-api-client agent
    finding); we cast here so downstream code sees float64.
    """
    rows: list[dict[str, Any]] = []
    for series in raw:
        labels = series.get("metric", {})
        kept = {k: labels.get(k, "") for k in KEPT_LABELS}
        # `uid` is the kubernetes UID; rename to canonical pod_uid.
        pod_uid = kept.pop("uid")
        for ts_sec, value_str in series.get("values", []):
            try:
                value = float(value_str)
            except (TypeError, ValueError):
                # Prometheus returns "NaN" / "+Inf" / "-Inf" as strings; skip them.
                continue
            rows.append(
                {
                    "scrape_timestamp": pd.to_datetime(float(ts_sec), unit="s", utc=True),
                    "metric_name": metric_name,
                    "value": value,
                    "namespace": kept.get("namespace", ""),
                    "pod": kept.get("pod", ""),
                    "pod_uid": pod_uid,
                    "container": kept.get("container", ""),
                    "node": kept.get("node", ""),
                }
            )
    return rows


def _scrape_one_metric(
    pc: PrometheusConnect,
    spec: MetricSpec,
    chunk: Chunk,
    *,
    step: str,
) -> list[dict[str, Any]]:
    """Run a single (metric, chunk) query_range and return long-form rows.

    Wraps `prometheus_api_client.PrometheusApiClientException` and the
    underlying `requests` errors in `ScrapeError` so callers see one type.
    """
    try:
        raw = pc.custom_query_range(
            query=spec.query,
            start_time=chunk.start,
            end_time=chunk.end,
            step=step,
        )
    except PrometheusApiClientException as exc:
        raise ScrapeError(
            f"prometheus error for metric={spec.column} window=[{chunk.start},{chunk.end}): {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — network errors bubble unchanged; tag them
        raise ScrapeError(
            f"network error for metric={spec.column} window=[{chunk.start},{chunk.end}): {exc}"
        ) from exc
    rows = _result_to_rows(raw, metric_name=spec.column)
    log.debug(
        "scrape.metric.done",
        metric=spec.column,
        chunk_start=chunk.start.isoformat(),
        chunk_end=chunk.end.isoformat(),
        rows=len(rows),
    )
    return rows


def _coerce_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a typed long-form DataFrame, enforcing the schema contract."""
    if not rows:
        df = pd.DataFrame(columns=list(LONG_FRAME_COLUMNS))
    else:
        df = pd.DataFrame(rows, columns=list(LONG_FRAME_COLUMNS))
    for col, dtype in LONG_FRAME_DTYPES.items():
        df[col] = df[col].astype(dtype)  # type: ignore[call-overload]
    return df


def _partition_path(out_dir: Path, ts: pd.Timestamp) -> Path:
    """Return the parquet path for an hour-partition."""
    dt = ts.tz_convert("UTC")
    return (
        out_dir
        / f"dt={dt.strftime('%Y-%m-%d')}"
        / f"hour={dt.strftime('%H')}"
        / "metrics.parquet"
    )


def scrape_window(
    *,
    prom_url: str,
    start: datetime,
    end: datetime,
    metrics: Iterable[MetricSpec] = ALL_METRICS,
    step: str = f"{DEFAULT_SCRAPE_STEP_SEC}s",
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES,
    max_workers: int = DEFAULT_MAX_WORKERS,
    out_dir: Path = DEFAULT_RAW_DIR,
    timeout_sec: int = 30,
) -> dict[str, int]:
    """Scrape `[start, end)` from `prom_url` and write hour-partitioned parquet.

    Returns `{partition_path: rows_written}` for the partitions touched.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be timezone-aware (use UTC)")
    if end <= start:
        raise ValueError(f"end {end} must be > start {start}")

    pc = PrometheusConnect(url=prom_url, disable_ssl=True, timeout=timeout_sec)
    chunk = timedelta(minutes=chunk_minutes)
    metric_list = list(metrics)

    log.info(
        "scrape.start",
        prom_url=prom_url,
        start=start.isoformat(),
        end=end.isoformat(),
        step=step,
        chunk_minutes=chunk_minutes,
        n_metrics=len(metric_list),
    )

    all_rows: list[dict[str, Any]] = []
    for c in _chunks(start, end, chunk=chunk):
        # Per-chunk thread pool: one query per metric in parallel against Prom.
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_scrape_one_metric, pc, spec, c, step=step): spec
                for spec in metric_list
            }
            for fut in as_completed(futures):
                rows = fut.result()
                all_rows.extend(rows)

    df = _coerce_frame(all_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    if df.empty:
        log.warning("scrape.empty", start=start.isoformat(), end=end.isoformat())
        return written

    # Partition by (date, hour) of the scrape_timestamp.
    df["_partition_key"] = df["scrape_timestamp"].dt.floor("h")
    for partition_ts, part_df in df.groupby("_partition_key", sort=True):
        # groupby key is a Timestamp because of `.dt.floor`; mypy widens to a union.
        ts = pd.Timestamp(partition_ts)  # type: ignore[arg-type]
        path = _partition_path(out_dir, ts)
        path.parent.mkdir(parents=True, exist_ok=True)
        part_df.drop(columns=["_partition_key"]).to_parquet(path, index=False)
        written[str(path)] = len(part_df)
        log.info("scrape.partition.written", path=str(path), rows=len(part_df))

    log.info(
        "scrape.done",
        total_rows=len(df),
        partitions=len(written),
        out_dir=str(out_dir),
    )
    return written


def _parse_dt(value: str) -> datetime:
    """Parse an ISO-ish datetime; ensure UTC."""
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cluster_canary.data.scraper", description=__doc__)
    p.add_argument("--prom-url", default=os.environ.get("PROMETHEUS_URL", "http://localhost:9090"))
    p.add_argument(
        "--start",
        default=os.environ.get("SCRAPE_START"),
        help="ISO datetime; defaults to (now - 24h).",
    )
    p.add_argument(
        "--end",
        default=os.environ.get("SCRAPE_END"),
        help="ISO datetime; defaults to now.",
    )
    p.add_argument("--step", default=os.environ.get("SCRAPE_STEP", f"{DEFAULT_SCRAPE_STEP_SEC}s"))
    p.add_argument("--chunk-minutes", type=int, default=DEFAULT_CHUNK_MINUTES)
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_DIR)
    return p


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)

    end = _parse_dt(args.end) if args.end else datetime.now(tz=UTC)
    start = _parse_dt(args.start) if args.start else end - timedelta(hours=24)

    written = scrape_window(
        prom_url=args.prom_url,
        start=start,
        end=end,
        step=args.step,
        chunk_minutes=args.chunk_minutes,
        max_workers=args.max_workers,
        out_dir=args.out_dir,
    )
    log.info("scrape.complete", partitions=len(written), total_rows=sum(written.values()))
    return 0 if written else 2  # 2 = empty result


if __name__ == "__main__":
    raise SystemExit(main())
