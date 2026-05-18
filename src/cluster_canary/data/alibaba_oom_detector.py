"""OOM event detection for the Alibaba 2018 trace.

The trace has no explicit OOM-kill label. We infer events via three signals,
in order of confidence:

1. **Container disappearance.** The last `alibaba_container_mem_util_pct`
   observation for a given `container_id` is more than `disappearance_threshold`
   before the trace end — strong proxy for the container being killed.

2. **Memory saturation just before disappearance.** Average of the last
   `mem_saturation_lookback` `alibaba_container_mem_util_pct` samples is
   ≥ `mem_saturation_threshold` (default 95). Combined with (1), this isolates
   OOM-likely deaths from clean shutdowns.

3. **Status transition** (optional, secondary). If `container_meta.status`
   transitions away from a "running"-class value (e.g. "started" → "stopped")
   at the disappearance timestamp, that's confirmatory. The enum is
   undocumented; the detector treats this as a soft signal only.

Reflects `docs/PHASE_1_CONTEXT.md` § 7 (the Alibaba ground-truth section).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

from cluster_canary.data.alibaba_schema import (
    CONTAINER_META_STATUS_METRIC_PREFIX,
    NAMESPACE_TAG,
)
from cluster_canary.data.label_generator import (
    LABEL_COL,
    OomEvent,
    _label_intervals_for_events,
    _row_in_intervals,
    long_to_wide,
)
from cluster_canary.data.schema import (
    DEFAULT_COOLDOWN_MINUTES,
    DEFAULT_LEAD_MINUTES,
    LONG_FRAME_DTYPES,
)

log = structlog.get_logger(__name__)

DEFAULT_INTERIM_DIR = Path("data/interim/alibaba")
DEFAULT_LABELED_DIR = Path("data/interim/alibaba_labeled")

# Default "is this container gone?" threshold. Alibaba's sampling is non-uniform
# (10-60 s gaps). 5 minutes without a sample is well beyond normal jitter.
DEFAULT_DISAPPEARANCE_THRESHOLD_SEC: int = 5 * 60

# Default memory saturation thresholds.
DEFAULT_MEM_SATURATION_THRESHOLD_PCT: float = 95.0
DEFAULT_MEM_SATURATION_LOOKBACK_SAMPLES: int = 10

# Metric column we treat as "container heartbeat" — its presence at a timestamp
# means the container is alive then.
HEARTBEAT_METRIC: str = "alibaba_container_mem_util_pct"


class AlibabaOomDetectorError(RuntimeError):
    """Raised when OOM detection cannot complete."""


@dataclass(frozen=True)
class DetectorConfig:
    """Tunable parameters for the Alibaba OOM detector."""

    disappearance_threshold_sec: int = DEFAULT_DISAPPEARANCE_THRESHOLD_SEC
    mem_saturation_threshold_pct: float = DEFAULT_MEM_SATURATION_THRESHOLD_PCT
    mem_saturation_lookback_samples: int = DEFAULT_MEM_SATURATION_LOOKBACK_SAMPLES


def _validate_long_frame(df: pd.DataFrame) -> None:
    required = set(LONG_FRAME_DTYPES) | {"metric_name", "value"}
    missing = required - set(df.columns)
    if missing:
        raise AlibabaOomDetectorError(f"input frame missing columns: {sorted(missing)}")


def detect_oom_events(
    long_df: pd.DataFrame,
    *,
    config: DetectorConfig | None = None,
) -> list[OomEvent]:
    """Detect probable OOM-kill events for Alibaba containers.

    For each `(pod_uid, container)` with at least one `HEARTBEAT_METRIC` sample,
    locate the last observation. If that observation is more than
    `disappearance_threshold_sec` before the global trace end AND the
    last `mem_saturation_lookback_samples` memory-utilization samples averaged
    ≥ `mem_saturation_threshold_pct`, classify as an OOM event at the
    last-observation timestamp.

    Returns events sorted by `(pod_uid, container, timestamp)`.
    """
    _validate_long_frame(long_df)
    cfg = config or DetectorConfig()

    heart = long_df[long_df["metric_name"] == HEARTBEAT_METRIC]
    if heart.empty:
        log.warning("alibaba_oom.no_heartbeat_metric", metric=HEARTBEAT_METRIC)
        return []

    trace_end = heart["scrape_timestamp"].max()
    threshold_td = pd.Timedelta(seconds=cfg.disappearance_threshold_sec)

    events: list[OomEvent] = []
    grouped = heart.sort_values(["pod_uid", "container", "scrape_timestamp"]).groupby(
        ["pod_uid", "container"], sort=False
    )
    for (pod_uid, container), group in grouped:
        last_ts = group["scrape_timestamp"].iloc[-1]
        if (trace_end - last_ts) <= threshold_td:
            continue  # still alive at trace end

        tail = group["value"].tail(cfg.mem_saturation_lookback_samples)
        if tail.empty:
            continue
        avg_tail = float(tail.mean())
        if avg_tail < cfg.mem_saturation_threshold_pct:
            continue

        events.append(
            OomEvent(
                pod_uid=str(pod_uid),
                container=str(container),
                timestamp=last_ts,
                source=f"alibaba_disappearance+mem={avg_tail:.1f}",
            )
        )

    events.sort(key=lambda e: (e.pod_uid, e.container, e.timestamp))
    log.info(
        "alibaba_oom.events.detected",
        n_events=len(events),
        n_pods=len({(e.pod_uid, e.container) for e in events}),
        config=cfg.__dict__,
    )
    return events


def _censored_intervals_for_events(
    events: list[OomEvent],
    *,
    cooldown: pd.Timedelta,
) -> dict[tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Inline copy of the censoring helper from `label_generator` (private there)."""
    from cluster_canary.data.label_generator import _union_intervals

    raw: dict[tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for e in events:
        key = (e.pod_uid, e.container)
        raw.setdefault(key, []).append((e.timestamp, e.timestamp + cooldown))
    return {k: _union_intervals(v) for k, v in raw.items()}


def generate_labels(
    long_df: pd.DataFrame,
    *,
    lead_minutes: int = DEFAULT_LEAD_MINUTES,
    cooldown_minutes: int = DEFAULT_COOLDOWN_MINUTES,
    detector_config: DetectorConfig | None = None,
) -> pd.DataFrame:
    """End-to-end: harmonized long-form → wide + `event_within_30min` + censoring.

    Same output contract as `label_generator.generate_labels`. Internally uses
    the Alibaba-specific `detect_oom_events` instead of the cAdvisor/ksm-based
    one. Censoring is closed-closed; positive windows are half-open from the left.
    """
    _validate_long_frame(long_df)
    if long_df.empty:
        wide = long_to_wide(long_df)
        wide[LABEL_COL] = pd.Series(dtype="int8")
        return wide

    lead = pd.Timedelta(minutes=lead_minutes)
    cooldown = pd.Timedelta(minutes=cooldown_minutes)

    events = detect_oom_events(long_df, config=detector_config)
    positive_intervals = _label_intervals_for_events(events, lead=lead)
    censored_intervals = _censored_intervals_for_events(events, cooldown=cooldown)

    wide = long_to_wide(long_df)

    def _label(row: pd.Series) -> int:
        return int(
            _row_in_intervals(
                row["scrape_timestamp"],
                positive_intervals.get((row["pod_uid"], row["container"]), []),
            )
        )

    def _censored(row: pd.Series) -> bool:
        ts = row["scrape_timestamp"]
        for start, end in censored_intervals.get((row["pod_uid"], row["container"]), []):
            if start <= ts <= end:
                return True
        return False

    wide[LABEL_COL] = wide.apply(_label, axis=1).astype("int8")
    wide["_censor"] = wide.apply(_censored, axis=1)

    n_in = len(wide)
    wide = wide[~wide["_censor"]].drop(columns=["_censor"]).reset_index(drop=True)
    log.info(
        "alibaba_oom.label.done",
        n_rows_in=n_in,
        n_rows_out=len(wide),
        n_positive=int(wide[LABEL_COL].sum()),
        positive_pct=round(float(wide[LABEL_COL].mean()) * 100, 4) if len(wide) else 0.0,
        n_events=len(events),
    )
    return wide


def _read_parquets(paths: Iterable[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cluster_canary.data.alibaba_oom_detector", description=__doc__
    )
    p.add_argument(
        "--interim-dir",
        type=Path,
        default=Path(os.environ.get("ALIBABA_INTERIM_DIR", str(DEFAULT_INTERIM_DIR))),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("ALIBABA_LABELED_DIR", str(DEFAULT_LABELED_DIR))),
    )
    p.add_argument("--lead-minutes", type=int, default=DEFAULT_LEAD_MINUTES)
    p.add_argument("--cooldown-minutes", type=int, default=DEFAULT_COOLDOWN_MINUTES)
    return p


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)

    parquets = sorted(args.interim_dir.glob("*.parquet"))
    if not parquets:
        log.error("alibaba_oom.no_input", interim_dir=str(args.interim_dir))
        return 1

    long_df = _read_parquets(parquets)
    # Restrict to the synthetic Alibaba namespace so cross-source data can't leak in.
    long_df = long_df[long_df["namespace"] == NAMESPACE_TAG].reset_index(drop=True)
    log.info("alibaba_oom.read.done", n_files=len(parquets), n_rows=len(long_df))

    wide = generate_labels(
        long_df, lead_minutes=args.lead_minutes, cooldown_minutes=args.cooldown_minutes
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "labeled.parquet"
    wide.to_parquet(out, index=False)
    log.info("alibaba_oom.complete", out=str(out), n_rows=len(wide))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
