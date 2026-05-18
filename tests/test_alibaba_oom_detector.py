"""Unit tests for the Alibaba OOM detector.

Synthetic harmonized DataFrames; no real trace data. Focus is on the
algorithm: disappearance threshold, memory-saturation requirement, and the
end-to-end label generation contract.
"""

from __future__ import annotations

import pandas as pd

from cluster_canary.data.alibaba_oom_detector import (
    DetectorConfig,
    detect_oom_events,
    generate_labels,
)
from cluster_canary.data.alibaba_schema import NAMESPACE_TAG
from cluster_canary.data.label_generator import LABEL_COL
from cluster_canary.data.schema import LONG_FRAME_DTYPES


def _heartbeat_row(*, ts: str, pod_uid: str, mem_pct: float) -> dict[str, object]:
    return {
        "scrape_timestamp": pd.Timestamp(ts, tz="UTC"),
        "metric_name": "alibaba_container_mem_util_pct",
        "value": mem_pct,
        "namespace": NAMESPACE_TAG,
        "pod": pod_uid,
        "pod_uid": pod_uid,
        "container": pod_uid,
        "node": "m1",
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


def test_detector_flags_disappeared_pod_with_saturated_memory() -> None:
    # Pod c-dying: heartbeats for 10 min, then nothing for 30 min.
    # Pod c-alive: heartbeats throughout — represents the global trace end.
    dying = [
        _heartbeat_row(
            ts=f"2026-05-18T12:0{i // 6}:{(i * 10) % 60:02d}", pod_uid="c-dying", mem_pct=98.0
        )
        for i in range(10)
    ]
    alive = [
        _heartbeat_row(
            ts=(
                pd.Timestamp("2026-05-18T12:00:00", tz="UTC") + pd.Timedelta(seconds=i * 30)
            ).isoformat(),
            pod_uid="c-alive",
            mem_pct=40.0,
        )
        for i in range(80)  # 80 * 30s = 40 min after 12:00 → trace end ~12:40
    ]
    events = detect_oom_events(_df(dying + alive))
    assert len(events) == 1
    assert events[0].pod_uid == "c-dying"
    assert events[0].source.startswith("alibaba_disappearance+mem=98")


def test_detector_skips_disappeared_pod_with_low_memory() -> None:
    # c-clean: heartbeats with LOW memory then disappears — likely clean shutdown.
    clean = [
        _heartbeat_row(
            ts=f"2026-05-18T12:0{i // 6}:{(i * 10) % 60:02d}", pod_uid="c-clean", mem_pct=20.0
        )
        for i in range(10)
    ]
    alive = [
        _heartbeat_row(
            ts=(
                pd.Timestamp("2026-05-18T12:00:00", tz="UTC") + pd.Timedelta(seconds=i * 30)
            ).isoformat(),
            pod_uid="c-alive",
            mem_pct=40.0,
        )
        for i in range(80)
    ]
    events = detect_oom_events(_df(clean + alive))
    assert events == []


def test_detector_skips_pod_alive_at_trace_end() -> None:
    # c-alive sees heartbeats up to the global trace end — should NOT fire.
    alive = [
        _heartbeat_row(
            ts=(
                pd.Timestamp("2026-05-18T12:00:00", tz="UTC") + pd.Timedelta(seconds=i * 30)
            ).isoformat(),
            pod_uid="c-alive",
            mem_pct=99.0,  # saturated but still alive
        )
        for i in range(80)
    ]
    events = detect_oom_events(_df(alive))
    assert events == []


def test_detector_config_thresholds_are_honored() -> None:
    # Memory saturation threshold tuned higher than observed → no event.
    dying = [
        _heartbeat_row(
            ts=f"2026-05-18T12:0{i // 6}:{(i * 10) % 60:02d}", pod_uid="c-dying", mem_pct=80.0
        )
        for i in range(10)
    ]
    alive = [
        _heartbeat_row(
            ts=(
                pd.Timestamp("2026-05-18T12:00:00", tz="UTC") + pd.Timedelta(seconds=i * 30)
            ).isoformat(),
            pod_uid="c-alive",
            mem_pct=40.0,
        )
        for i in range(80)
    ]
    df = _df(dying + alive)

    default_events = detect_oom_events(df)
    assert len(default_events) == 0  # 80 < default threshold (95)

    strict_events = detect_oom_events(df, config=DetectorConfig(mem_saturation_threshold_pct=70.0))
    assert len(strict_events) == 1


def test_generate_labels_window_labels_alibaba_event() -> None:
    # 40 min of saturated-memory heartbeats for c-dying ending at 12:39:30.
    # Then 30 min of silence. c-alive runs through 13:30:00 to set trace end.
    dying = []
    for i in range(80):
        ts = pd.Timestamp("2026-05-18T12:00:00", tz="UTC") + pd.Timedelta(seconds=i * 30)
        dying.append(_heartbeat_row(ts=ts.isoformat(), pod_uid="c-dying", mem_pct=99.0))

    alive = []
    for i in range(180):
        ts = pd.Timestamp("2026-05-18T12:00:00", tz="UTC") + pd.Timedelta(seconds=i * 30)
        alive.append(_heartbeat_row(ts=ts.isoformat(), pod_uid="c-alive", mem_pct=30.0))

    wide = generate_labels(_df(dying + alive), lead_minutes=30, cooldown_minutes=5)

    # c-dying's last observation = ~12:39:30. Lead window = [12:09:30, 12:39:30).
    # Row at 12:20 should be positive.
    pos = wide[
        (wide["pod_uid"] == "c-dying")
        & (wide["scrape_timestamp"] == pd.Timestamp("2026-05-18T12:20:00", tz="UTC"))
    ]
    assert (pos[LABEL_COL] == 1).all()

    # c-alive should be all-zero — never an event for it.
    alive_rows = wide[wide["pod_uid"] == "c-alive"]
    assert (alive_rows[LABEL_COL] == 0).all()
