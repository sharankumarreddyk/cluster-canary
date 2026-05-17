"""OOM event detection + window-labeling for cluster-canary.

Given the long-form scrape frame produced by `scraper.py`, this module:

1. **Detects OOM events** per `(pod_uid, container)`:
   - Primary: positive delta in `container_oom_events_total` (cAdvisor counter).
   - Backup: 0→1 transition of
     `kube_pod_container_status_last_terminated_reason_oomkilled` (ksm gauge).
   The two signals are unioned — either one fires an event at its observed
   timestamp.

2. **Generates the supervised label** `event_within_30min`:
   - Every row in `[T_oom - LEAD_MINUTES, T_oom)` for the same
     `(pod_uid, container)` is positive (1).
   - Repeated OOMs within the lead window are unioned (the rare two-OOMs-in-15-min
     case still yields one continuous positive window, not double-counting).
   - Rows in `[T_oom, T_oom + COOLDOWN_MINUTES]` are dropped (censored) —
     post-event signals leak the label via restart counters and stale memory.

3. **Pivots** the long-form frame to a wide per-`(scrape_timestamp, pod_uid, container)`
   row with one column per metric, plus the `event_within_30min` label.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import structlog

from cluster_canary.data.schema import (
    DEFAULT_COOLDOWN_MINUTES,
    DEFAULT_LEAD_MINUTES,
    LONG_FRAME_DTYPES,
    METRIC_COLUMNS,
)

log = structlog.get_logger(__name__)

DEFAULT_RAW_DIR = Path("data/raw/scrape")
DEFAULT_LABELED_DIR = Path("data/interim/labeled")
LABEL_COL = "event_within_30min"


class LabelGenerationError(RuntimeError):
    """Raised when label generation cannot complete."""


@dataclass(frozen=True)
class OomEvent:
    """A detected OOM event for one (pod_uid, container) at a specific time."""

    pod_uid: str
    container: str
    timestamp: pd.Timestamp
    source: str  # "oom_events_total" | "ksm_last_terminated_reason"


def _validate_long_frame(df: pd.DataFrame) -> None:
    """Sanity-check the input frame conforms to the scraper schema."""
    required = set(LONG_FRAME_DTYPES) | {"metric_name", "value"}
    missing = required - set(df.columns)
    if missing:
        raise LabelGenerationError(f"input frame missing columns: {sorted(missing)}")


def detect_oom_events(long_df: pd.DataFrame) -> list[OomEvent]:
    """Detect OOM events from the long-form scrape frame.

    Returns events sorted by (pod_uid, container, timestamp), de-duplicated
    when both signals fire within the same scrape timestamp.
    """
    _validate_long_frame(long_df)

    events: list[OomEvent] = []

    # 1. cAdvisor counter — positive delta means OOM happened in that interval.
    cadv = long_df[long_df["metric_name"] == "container_oom_events_total"].copy()
    if not cadv.empty:
        cadv = cadv.sort_values(["pod_uid", "container", "scrape_timestamp"])
        cadv["prev_value"] = cadv.groupby(["pod_uid", "container"])["value"].shift(1)
        new_oom = cadv[cadv["value"] > cadv["prev_value"].fillna(cadv["value"])]
        for _, row in new_oom.iterrows():
            events.append(
                OomEvent(
                    pod_uid=row["pod_uid"],
                    container=row["container"],
                    timestamp=row["scrape_timestamp"],
                    source="oom_events_total",
                )
            )

    # 2. ksm last_terminated_reason — 0→1 transition.
    ksm = long_df[
        long_df["metric_name"]
        == "kube_pod_container_status_last_terminated_reason_oomkilled"
    ].copy()
    if not ksm.empty:
        ksm = ksm.sort_values(["pod_uid", "container", "scrape_timestamp"])
        ksm["prev_value"] = ksm.groupby(["pod_uid", "container"])["value"].shift(1)
        transitions = ksm[(ksm["value"] >= 1) & (ksm["prev_value"].fillna(0) < 1)]
        for _, row in transitions.iterrows():
            events.append(
                OomEvent(
                    pod_uid=row["pod_uid"],
                    container=row["container"],
                    timestamp=row["scrape_timestamp"],
                    source="ksm_last_terminated_reason",
                )
            )

    # De-dup: same (uid, container, timestamp) reported by both signals → keep cAdvisor (more precise).
    seen: set[tuple[str, str, pd.Timestamp]] = set()
    deduped: list[OomEvent] = []
    # cAdvisor entries are first in `events` because we collected them first.
    for evt in events:
        key = (evt.pod_uid, evt.container, evt.timestamp)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(evt)

    deduped.sort(key=lambda e: (e.pod_uid, e.container, e.timestamp))
    log.info(
        "label.events.detected",
        n_events=len(deduped),
        n_pods=len({(e.pod_uid, e.container) for e in deduped}),
    )
    return deduped


def _union_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Union a list of [start, end) intervals. Returns sorted, non-overlapping list."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _label_intervals_for_events(
    events: list[OomEvent],
    *,
    lead: pd.Timedelta,
) -> dict[tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Per (pod_uid, container), union of `[T_oom - lead, T_oom)` windows."""
    raw: dict[tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for e in events:
        key = (e.pod_uid, e.container)
        raw.setdefault(key, []).append((e.timestamp - lead, e.timestamp))
    return {k: _union_intervals(v) for k, v in raw.items()}


def _censored_intervals_for_events(
    events: list[OomEvent],
    *,
    cooldown: pd.Timedelta,
) -> dict[tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Per (pod_uid, container), union of `[T_oom, T_oom + cooldown]` windows.

    Rows inside these intervals are dropped — they leak the label via restart
    counters and stale post-event memory values.
    """
    raw: dict[tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for e in events:
        key = (e.pod_uid, e.container)
        raw.setdefault(key, []).append((e.timestamp, e.timestamp + cooldown))
    return {k: _union_intervals(v) for k, v in raw.items()}


def _row_in_intervals(
    ts: pd.Timestamp,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    """Closed-open membership: ts in any [a, b)."""
    for start, end in intervals:
        if start <= ts < end:
            return True
    return False


def _row_in_closed_intervals(
    ts: pd.Timestamp,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    """Closed-closed membership: ts in any [a, b]."""
    for start, end in intervals:
        if start <= ts <= end:
            return True
    return False


def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long → wide with one column per metric.

    Index becomes `(scrape_timestamp, pod_uid, container)` reset to columns.
    Metadata labels (`namespace`, `pod`, `node`) are carried via groupby-first.
    """
    _validate_long_frame(long_df)
    if long_df.empty:
        cols = ["scrape_timestamp", "pod_uid", "container", "namespace", "pod", "node"] + list(
            METRIC_COLUMNS
        )
        return pd.DataFrame(columns=cols)

    keys = ["scrape_timestamp", "pod_uid", "container"]
    wide = long_df.pivot_table(
        index=keys,
        columns="metric_name",
        values="value",
        aggfunc="first",
    )

    meta = (
        long_df.groupby(keys)[["namespace", "pod", "node"]].first().reset_index()
    )
    wide = wide.reset_index().merge(meta, on=keys, how="left")
    # Ensure all metric columns are present even if a metric was empty.
    for col in METRIC_COLUMNS:
        if col not in wide.columns:
            wide[col] = pd.Series(dtype="float64")
    return wide


def generate_labels(
    long_df: pd.DataFrame,
    *,
    lead_minutes: int = DEFAULT_LEAD_MINUTES,
    cooldown_minutes: int = DEFAULT_COOLDOWN_MINUTES,
) -> pd.DataFrame:
    """End-to-end: long → wide + `event_within_30min` label + censoring.

    Returns a wide DataFrame ready for feature engineering. Censored rows are
    DROPPED, not masked.
    """
    _validate_long_frame(long_df)
    if long_df.empty:
        wide = long_to_wide(long_df)
        wide[LABEL_COL] = pd.Series(dtype="int8")
        return wide

    lead = pd.Timedelta(minutes=lead_minutes)
    cooldown = pd.Timedelta(minutes=cooldown_minutes)
    events = detect_oom_events(long_df)
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
        return _row_in_closed_intervals(
            row["scrape_timestamp"],
            censored_intervals.get((row["pod_uid"], row["container"]), []),
        )

    wide[LABEL_COL] = wide.apply(_label, axis=1).astype("int8")
    wide["_censor"] = wide.apply(_censored, axis=1)

    n_in = len(wide)
    wide = wide[~wide["_censor"]].drop(columns=["_censor"]).reset_index(drop=True)
    log.info(
        "label.generation.done",
        n_rows_in=n_in,
        n_rows_out=len(wide),
        n_positive=int(wide[LABEL_COL].sum()),
        positive_pct=round(float(wide[LABEL_COL].mean()) * 100, 4) if len(wide) else 0.0,
        n_events=len(events),
        lead_min=lead_minutes,
        cooldown_min=cooldown_minutes,
    )
    return wide


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
        prog="cluster_canary.data.label_generator", description=__doc__
    )
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_LABELED_DIR)
    p.add_argument("--lead-minutes", type=int, default=DEFAULT_LEAD_MINUTES)
    p.add_argument("--cooldown-minutes", type=int, default=DEFAULT_COOLDOWN_MINUTES)
    return p


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)

    parquets = sorted(args.raw_dir.rglob("*.parquet"))
    if not parquets:
        log.error("label.no_input", raw_dir=str(args.raw_dir))
        return 1

    long_df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    log.info("label.read.done", n_files=len(parquets), n_rows=len(long_df))

    wide = generate_labels(
        long_df,
        lead_minutes=args.lead_minutes,
        cooldown_minutes=args.cooldown_minutes,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "labeled.parquet"
    wide.to_parquet(out, index=False)

    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M")
    metrics_dir = Path("reports/eval")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    n_positive = int(wide[LABEL_COL].sum())
    summary = {
        "ts_utc": ts,
        "n_files_in": len(parquets),
        "n_long_rows_in": int(len(long_df)),
        "n_wide_rows_out": int(len(wide)),
        "n_positive": n_positive,
        "positive_pct": round(float(wide[LABEL_COL].mean()) * 100, 4) if len(wide) else 0.0,
        "lead_minutes": args.lead_minutes,
        "cooldown_minutes": args.cooldown_minutes,
        "out_path": str(out),
    }
    (metrics_dir / "label_metrics.json").write_text(
        pd.Series(summary).to_json(indent=2)
    )
    log.info("label.complete", **summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
