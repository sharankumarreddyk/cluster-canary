"""Unit tests for the OOM event detector + window-labeler.

All tests use synthetic long-form DataFrames — no real Prometheus, no kind cluster.
Focus is on the algorithm: detection, union of overlapping windows, censoring,
and the long → wide pivot.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cluster_canary.data.label_generator import (
    LABEL_COL,
    detect_oom_events,
    generate_labels,
    long_to_wide,
)
from cluster_canary.data.schema import LONG_FRAME_DTYPES


def _row(
    *,
    ts: str,
    metric: str,
    value: float,
    pod_uid: str = "uid-1",
    container: str = "c",
    pod: str = "p",
    namespace: str = "workloads",
    node: str = "node-1",
) -> dict[str, object]:
    return {
        "scrape_timestamp": pd.Timestamp(ts, tz="UTC"),
        "metric_name": metric,
        "value": value,
        "namespace": namespace,
        "pod": pod,
        "pod_uid": pod_uid,
        "container": container,
        "node": node,
    }


def _df(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in LONG_FRAME_DTYPES.items()}
        )
    df = pd.DataFrame(rows)
    for col, dtype in LONG_FRAME_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df


def test_detect_events_from_oom_counter_delta() -> None:
    df = _df(
        [
            _row(ts="2026-05-18T12:00:00", metric="container_oom_events_total", value=0),
            _row(ts="2026-05-18T12:00:15", metric="container_oom_events_total", value=0),
            _row(ts="2026-05-18T12:00:30", metric="container_oom_events_total", value=1),
        ]
    )
    events = detect_oom_events(df)
    assert len(events) == 1
    assert events[0].timestamp == pd.Timestamp("2026-05-18T12:00:30", tz="UTC")
    assert events[0].source == "oom_events_total"


def test_detect_events_from_ksm_transition() -> None:
    df = _df(
        [
            _row(
                ts="2026-05-18T12:00:00",
                metric="kube_pod_container_status_last_terminated_reason_oomkilled",
                value=0,
            ),
            _row(
                ts="2026-05-18T12:00:30",
                metric="kube_pod_container_status_last_terminated_reason_oomkilled",
                value=1,
            ),
            _row(
                ts="2026-05-18T12:00:45",
                metric="kube_pod_container_status_last_terminated_reason_oomkilled",
                value=1,
            ),
        ]
    )
    events = detect_oom_events(df)
    assert len(events) == 1
    assert events[0].source == "ksm_last_terminated_reason"


def test_detect_events_dedup_when_both_signals_fire_at_same_time() -> None:
    ts = "2026-05-18T12:00:30"
    df = _df(
        [
            _row(ts="2026-05-18T12:00:15", metric="container_oom_events_total", value=0),
            _row(ts=ts, metric="container_oom_events_total", value=1),
            _row(
                ts="2026-05-18T12:00:15",
                metric="kube_pod_container_status_last_terminated_reason_oomkilled",
                value=0,
            ),
            _row(
                ts=ts,
                metric="kube_pod_container_status_last_terminated_reason_oomkilled",
                value=1,
            ),
        ]
    )
    events = detect_oom_events(df)
    assert len(events) == 1
    assert events[0].source == "oom_events_total"


def test_detect_events_per_pod_isolation() -> None:
    df = _df(
        [
            _row(
                ts="2026-05-18T12:00:00",
                metric="container_oom_events_total",
                value=0,
                pod_uid="uid-1",
            ),
            _row(
                ts="2026-05-18T12:00:15",
                metric="container_oom_events_total",
                value=1,
                pod_uid="uid-1",
            ),
            _row(
                ts="2026-05-18T12:00:00",
                metric="container_oom_events_total",
                value=5,
                pod_uid="uid-2",
            ),
            _row(
                ts="2026-05-18T12:00:15",
                metric="container_oom_events_total",
                value=5,
                pod_uid="uid-2",
            ),
        ]
    )
    events = detect_oom_events(df)
    assert len(events) == 1
    assert events[0].pod_uid == "uid-1"


def _ramp_rows(
    *,
    pod_uid: str = "uid-1",
    container: str = "c",
    base: str,
    minutes: int,
    every_seconds: int = 30,
) -> list[dict[str, object]]:
    """Build a feature time-series of `minutes` ramping memory values."""
    start = pd.Timestamp(base, tz="UTC")
    rows: list[dict[str, object]] = []
    n = (minutes * 60) // every_seconds
    for i in range(n):
        ts = start + pd.Timedelta(seconds=i * every_seconds)
        rows.append(
            _row(
                ts=ts.isoformat(),
                metric="container_memory_working_set_bytes",
                value=float(100 * i),
                pod_uid=pod_uid,
                container=container,
            )
        )
    return rows


def test_generate_labels_positive_window_30min_before_oom() -> None:
    # 60 min of feature samples; OOM happens at t=50min.
    feature_rows = _ramp_rows(base="2026-05-18T12:00:00", minutes=60)
    oom_rows = [
        _row(ts="2026-05-18T12:49:30", metric="container_oom_events_total", value=0),
        _row(ts="2026-05-18T12:50:00", metric="container_oom_events_total", value=1),
    ]
    df = _df(feature_rows + oom_rows)
    wide = generate_labels(df, lead_minutes=30, cooldown_minutes=5)

    # Row at t=19:30 should be NEGATIVE (>30 min before OOM; window is [12:20, 12:50)).
    early = wide[wide["scrape_timestamp"] == pd.Timestamp("2026-05-18T12:19:30", tz="UTC")]
    assert not early.empty, "expected a row at 12:19:30"
    assert (early[LABEL_COL] == 0).all()

    # Row at t=25min should be POSITIVE (within 30 min before OOM).
    late = wide[wide["scrape_timestamp"] == pd.Timestamp("2026-05-18T12:25:00", tz="UTC")]
    assert (late[LABEL_COL] == 1).all()


def test_generate_labels_censors_post_event_rows() -> None:
    # Pre-event rows + OOM at t=20min + post-event ramp continuing to t=30min.
    rows = _ramp_rows(base="2026-05-18T12:00:00", minutes=30)
    oom_rows = [
        _row(ts="2026-05-18T12:19:30", metric="container_oom_events_total", value=0),
        _row(ts="2026-05-18T12:20:00", metric="container_oom_events_total", value=1),
    ]
    df = _df(rows + oom_rows)
    wide = generate_labels(df, lead_minutes=30, cooldown_minutes=5)

    # Rows in [12:20, 12:25] should be censored (dropped).
    censored = wide[
        (wide["scrape_timestamp"] >= pd.Timestamp("2026-05-18T12:20:00", tz="UTC"))
        & (wide["scrape_timestamp"] <= pd.Timestamp("2026-05-18T12:25:00", tz="UTC"))
    ]
    assert censored.empty

    # Rows after cooldown should be present again.
    after = wide[wide["scrape_timestamp"] > pd.Timestamp("2026-05-18T12:25:00", tz="UTC")]
    assert len(after) > 0


def test_generate_labels_unions_overlapping_windows() -> None:
    rows = _ramp_rows(base="2026-05-18T12:00:00", minutes=90)
    # Two OOMs 15 min apart — their 30-min lead windows overlap.
    oom_rows = [
        _row(ts="2026-05-18T12:44:30", metric="container_oom_events_total", value=0),
        _row(ts="2026-05-18T12:45:00", metric="container_oom_events_total", value=1),
        _row(ts="2026-05-18T12:59:30", metric="container_oom_events_total", value=1),
        _row(ts="2026-05-18T13:00:00", metric="container_oom_events_total", value=2),
    ]
    df = _df(rows + oom_rows)
    wide = generate_labels(df, lead_minutes=30, cooldown_minutes=5)

    # Row at 12:30 should be positive (in lead window of first OOM, which is [12:15, 12:45)).
    r1 = wide[wide["scrape_timestamp"] == pd.Timestamp("2026-05-18T12:30:00", tz="UTC")]
    assert (r1[LABEL_COL] == 1).all()
    # Row at 12:55 should be positive (in lead window of second OOM after cooldown).
    r2 = wide[wide["scrape_timestamp"] == pd.Timestamp("2026-05-18T12:55:00", tz="UTC")]
    assert (r2[LABEL_COL] == 1).all()


def test_generate_labels_pod_with_no_oom_is_all_zero() -> None:
    rows = _ramp_rows(base="2026-05-18T12:00:00", minutes=10, pod_uid="happy-pod")
    df = _df(rows)
    wide = generate_labels(df, lead_minutes=30, cooldown_minutes=5)
    assert (wide[LABEL_COL] == 0).all()


def test_long_to_wide_pivots_metrics_to_columns() -> None:
    df = _df(
        [
            _row(ts="2026-05-18T12:00:00", metric="container_memory_working_set_bytes", value=100),
            _row(ts="2026-05-18T12:00:00", metric="container_memory_rss", value=80),
            _row(ts="2026-05-18T12:00:15", metric="container_memory_working_set_bytes", value=120),
            _row(ts="2026-05-18T12:00:15", metric="container_memory_rss", value=95),
        ]
    )
    wide = long_to_wide(df)
    assert "container_memory_working_set_bytes" in wide.columns
    assert "container_memory_rss" in wide.columns
    assert len(wide) == 2  # two timestamps for one (pod_uid, container)


def test_generate_labels_empty_input_returns_empty_with_label_column() -> None:
    df = _df([])
    wide = generate_labels(df)
    assert LABEL_COL in wide.columns
    assert wide.empty


def test_validate_long_frame_rejects_missing_column() -> None:
    df = pd.DataFrame({"scrape_timestamp": [pd.Timestamp("2026-05-18", tz="UTC")]})
    with pytest.raises(Exception, match="missing columns"):
        detect_oom_events(df)
