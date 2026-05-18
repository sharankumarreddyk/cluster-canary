"""Unit tests for the Phase 3 feature modules.

Synthetic-input only — these tests verify the algorithms, not real data
quality. The leakage audit in `tests/test_no_leakage.py` is the ground truth
for "the features don't peek into the future" once processed parquet exists.
"""

from __future__ import annotations

import pandas as pd

from cluster_canary.features.aggregations import (
    add_lineage_24h_oom_rate,
    add_node_cotenant_features,
)
from cluster_canary.features.build import build_features, select_model_columns
from cluster_canary.features.cpu import add_cpu_util
from cluster_canary.features.lifecycle import add_pod_age
from cluster_canary.features.memory import add_mem_pct_of_limit, add_mem_rolling
from cluster_canary.features.temporal import (
    TEMPORAL_FEATURE_COLS,
    add_temporal_features,
)


def _wide_row(
    *,
    ts: str,
    pod_uid: str = "uid-1",
    container: str = "c",
    node: str = "n1",
    namespace: str = "workloads",
    pod: str = "p",
    label: int = 0,
    **metric_cols: float,
) -> dict[str, object]:
    return {
        "scrape_timestamp": pd.Timestamp(ts, tz="UTC"),
        "namespace": namespace,
        "pod": pod,
        "pod_uid": pod_uid,
        "container": container,
        "node": node,
        "event_within_30min": label,
        **metric_cols,
    }


# --- temporal ------------------------------------------------------------- #


def test_temporal_features_extract_hour_dow_weekend_bizhours() -> None:
    df = pd.DataFrame(
        [
            _wide_row(ts="2026-05-18T14:00:00"),  # Mon 14:00 → biz hours
            _wide_row(ts="2026-05-23T03:00:00"),  # Sat 03:00 → weekend, not biz
        ]
    )
    out = add_temporal_features(df)
    for col in TEMPORAL_FEATURE_COLS:
        assert col in out.columns
    assert out.loc[0, "feat_hour_of_day"] == 14
    assert out.loc[0, "feat_day_of_week"] == 0  # Monday
    assert out.loc[0, "feat_is_weekend"] == 0
    assert out.loc[0, "feat_is_business_hours"] == 1
    assert out.loc[1, "feat_is_weekend"] == 1
    assert out.loc[1, "feat_is_business_hours"] == 0


# --- memory --------------------------------------------------------------- #


def test_mem_pct_of_limit_dispatches_to_alibaba_when_present() -> None:
    df = pd.DataFrame([_wide_row(ts="2026-05-18T12:00:00", alibaba_container_mem_util_pct=80.0)])
    out = add_mem_pct_of_limit(df)
    assert out["feat_mem_pct_of_limit"].iloc[0] == 0.8


def test_mem_pct_of_limit_dispatches_to_synthetic_when_no_alibaba_col() -> None:
    df = pd.DataFrame(
        [
            _wide_row(
                ts="2026-05-18T12:00:00",
                container_memory_working_set_bytes=400 * 1024 * 1024,
                kube_pod_container_resource_limits_memory_bytes=500 * 1024 * 1024,
            )
        ]
    )
    out = add_mem_pct_of_limit(df)
    assert out["feat_mem_pct_of_limit"].iloc[0] == 0.8


def test_mem_rolling_is_left_closed_no_future_peek() -> None:
    # 6 samples at 30s cadence; mem_pct_of_limit ramps 0.1, 0.2, ..., 0.6.
    rows = []
    for i in range(6):
        ts = pd.Timestamp("2026-05-18T12:00:00", tz="UTC") + pd.Timedelta(seconds=i * 30)
        rows.append(
            _wide_row(
                ts=ts.isoformat(),
                alibaba_container_mem_util_pct=10.0 * (i + 1),  # 10, 20, ..., 60
            )
        )
    df = pd.DataFrame(rows)
    out = add_mem_pct_of_limit(df)
    out = add_mem_rolling(out, windows=("90s",))
    # First row: no past in window → NaN (or 0 rows aggregated).
    first = out.sort_values("scrape_timestamp").iloc[0]
    assert pd.isna(first["feat_mem_pct_of_limit__90s_mean"])
    # Row 3 (t=12:01:00): window=[12:00:00, 12:01:00), 2 samples (i=0, i=1) = 0.10, 0.20 → mean 0.15.
    third = out.sort_values("scrape_timestamp").iloc[2]
    assert abs(third["feat_mem_pct_of_limit__90s_mean"] - 0.15) < 1e-9


# --- CPU ------------------------------------------------------------------ #


def test_cpu_util_dispatches_by_source() -> None:
    alibaba = pd.DataFrame(
        [_wide_row(ts="2026-05-18T12:00:00", alibaba_container_cpu_util_pct=50.0)]
    )
    synth = pd.DataFrame(
        [_wide_row(ts="2026-05-18T12:00:00", container_cpu_usage_seconds_rate=0.4)]
    )
    assert add_cpu_util(alibaba)["feat_cpu_util"].iloc[0] == 0.5
    assert add_cpu_util(synth)["feat_cpu_util"].iloc[0] == 0.4


# --- lifecycle ------------------------------------------------------------ #


def test_pod_age_sec_grows_with_first_seen() -> None:
    rows = []
    for sec in [0, 60, 300, 3600]:
        ts = pd.Timestamp("2026-05-18T12:00:00", tz="UTC") + pd.Timedelta(seconds=sec)
        rows.append(_wide_row(ts=ts.isoformat()))
    df = pd.DataFrame(rows)
    out = add_pod_age(df)
    expected = [0.0, 60.0, 300.0, 3600.0]
    assert out["feat_pod_age_sec"].tolist() == expected


# --- aggregations --------------------------------------------------------- #


def test_node_cotenant_features_sum_across_containers() -> None:
    ts = "2026-05-18T12:00:00"
    df = pd.DataFrame(
        [
            _wide_row(
                ts=ts, pod_uid="p1", container="c1", node="n1", alibaba_container_mem_util_pct=40.0
            ),
            _wide_row(
                ts=ts, pod_uid="p2", container="c2", node="n1", alibaba_container_mem_util_pct=30.0
            ),
            _wide_row(
                ts=ts, pod_uid="p3", container="c3", node="n2", alibaba_container_mem_util_pct=10.0
            ),
        ]
    )
    df = add_mem_pct_of_limit(df)
    out = add_node_cotenant_features(df)
    n1 = out[out["node"] == "n1"]
    assert n1["feat_node_n_containers"].iloc[0] == 2
    assert abs(n1["feat_node_mem_pct_sum"].iloc[0] - 0.7) < 1e-9
    assert abs(n1["feat_node_mem_pct_max"].iloc[0] - 0.4) < 1e-9


def test_lineage_24h_oom_rate_is_left_closed() -> None:
    # Two pods sharing a lineage (pod names "lf-a", "lf-b" → lineage "lf").
    # At t=0 one of them was labeled positive; at t=1h the OTHER pod should see
    # lineage_24h_oom_rate = 1/2 (one prior positive, one prior negative).
    rows = [
        _wide_row(
            ts="2026-05-18T11:00:00",
            pod_uid="u1",
            pod="lf-a",
            label=1,
            alibaba_container_mem_util_pct=99.0,
        ),
        _wide_row(
            ts="2026-05-18T11:00:00",
            pod_uid="u2",
            pod="lf-b",
            label=0,
            alibaba_container_mem_util_pct=20.0,
        ),
        _wide_row(
            ts="2026-05-18T12:00:00",
            pod_uid="u1",
            pod="lf-a",
            label=0,
            alibaba_container_mem_util_pct=20.0,
        ),
    ]
    df = pd.DataFrame(rows)
    out = add_lineage_24h_oom_rate(df)
    # The 12:00 row should see (1 prior positive + 1 prior negative) / 2 = 0.5.
    row_1200 = out[out["scrape_timestamp"] == pd.Timestamp("2026-05-18T12:00:00", tz="UTC")]
    assert abs(row_1200["feat_lineage_24h_oom_rate"].iloc[0] - 0.5) < 1e-9
    # The 11:00 rows have no PRIOR samples → NaN (left-closed).
    early = out[out["scrape_timestamp"] == pd.Timestamp("2026-05-18T11:00:00", tz="UTC")]
    assert early["feat_lineage_24h_oom_rate"].isna().all()


# --- build (end-to-end) --------------------------------------------------- #


def test_build_features_emits_all_feat_columns_and_label() -> None:
    rows = []
    for i in range(8):
        ts = pd.Timestamp("2026-05-18T12:00:00", tz="UTC") + pd.Timedelta(seconds=i * 30)
        rows.append(
            _wide_row(
                ts=ts.isoformat(),
                pod_uid=f"u{i % 2}",
                container=f"c{i % 2}",
                node="n1",
                alibaba_container_mem_util_pct=10.0 * (i + 1),
                alibaba_container_cpu_util_pct=5.0 * (i + 1),
                label=int(i == 7),
            )
        )
    df = pd.DataFrame(rows)
    built = build_features(df)
    model = select_model_columns(built)
    feat_cols = [c for c in model.columns if c.startswith("feat_")]
    assert len(feat_cols) > 0
    assert "event_within_30min" in model.columns
    assert "scrape_timestamp" in model.columns
    # No raw metric columns leak into model_df:
    assert "alibaba_container_mem_util_pct" not in model.columns
