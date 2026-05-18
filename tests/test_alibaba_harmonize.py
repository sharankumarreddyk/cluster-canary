"""Unit tests for the Alibaba CSV → long-form parquet harmonizer.

No real CSV files needed — we construct synthetic Alibaba-shape DataFrames in
memory and assert the long-form contract holds.
"""

from __future__ import annotations

import pandas as pd

from cluster_canary.data.alibaba_harmonize import (
    harmonize_container_meta,
    harmonize_container_usage,
)
from cluster_canary.data.alibaba_schema import (
    ALIGNED_DAY_RANGE,
    NAMESPACE_TAG,
    SECONDS_PER_DAY,
)
from cluster_canary.data.schema import LONG_FRAME_COLUMNS, LONG_FRAME_DTYPES


def _day_to_sec(day: int, second_in_day: int = 0) -> int:
    return day * SECONDS_PER_DAY + second_in_day


def test_harmonize_container_usage_produces_canonical_columns() -> None:
    df = pd.DataFrame(
        {
            "container_id": ["c1", "c1"],
            "machine_id": ["m1", "m1"],
            "time_stamp": [_day_to_sec(3), _day_to_sec(3, 60)],
            "cpu_util_percent": [50, 55],
            "mem_util_percent": [80, 82],
            "cpi": [1.0, 1.1],
            "mem_gps": [40, 42],
            "mpki": [10, 11],
            "net_in": [5, 6],
            "net_out": [4, 5],
            "disk_io_percent": [10, 12],
        }
    )
    long_df = harmonize_container_usage(df)
    assert list(long_df.columns) == list(LONG_FRAME_COLUMNS)
    for col, dtype in LONG_FRAME_DTYPES.items():
        assert str(long_df[col].dtype) == dtype


def test_harmonize_container_usage_emits_one_row_per_metric() -> None:
    df = pd.DataFrame(
        {
            "container_id": ["c1"],
            "machine_id": ["m1"],
            "time_stamp": [_day_to_sec(3)],
            "cpu_util_percent": [50],
            "mem_util_percent": [80],
            "cpi": [1.0],
            "mem_gps": [40],
            "mpki": [10],
            "net_in": [5],
            "net_out": [4],
            "disk_io_percent": [10],
        }
    )
    long_df = harmonize_container_usage(df)
    # 8 metric_names in CONTAINER_USAGE_METRIC_MAP x 1 timestamp.
    assert len(long_df) == 8
    expected_metrics = {
        "alibaba_container_cpu_util_pct",
        "alibaba_container_mem_util_pct",
        "alibaba_container_cpi",
        "alibaba_container_mem_gps_pct",
        "alibaba_container_mpki",
        "alibaba_container_net_in_pct",
        "alibaba_container_net_out_pct",
        "alibaba_container_disk_io_pct",
    }
    assert set(long_df["metric_name"]) == expected_metrics


def test_harmonize_container_usage_drops_disk_io_sentinels() -> None:
    df = pd.DataFrame(
        {
            "container_id": ["c1", "c1", "c1"],
            "machine_id": ["m1", "m1", "m1"],
            "time_stamp": [_day_to_sec(3), _day_to_sec(3, 30), _day_to_sec(3, 60)],
            "cpu_util_percent": [50, 50, 50],
            "mem_util_percent": [80, 80, 80],
            "cpi": [1.0, 1.0, 1.0],
            "mem_gps": [40, 40, 40],
            "mpki": [10, 10, 10],
            "net_in": [5, 5, 5],
            "net_out": [4, 4, 4],
            "disk_io_percent": [-1, 50, 101],  # two sentinels, one valid
        }
    )
    long_df = harmonize_container_usage(df)
    disk = long_df[long_df["metric_name"] == "alibaba_container_disk_io_pct"]
    assert len(disk) == 1
    assert disk["value"].iloc[0] == 50.0


def test_harmonize_drops_rows_outside_aligned_day_window() -> None:
    """Rows on day 1 (before alignment window) and day 9 (after) must be dropped."""
    df = pd.DataFrame(
        {
            "container_id": ["c1", "c1", "c1"],
            "machine_id": ["m1", "m1", "m1"],
            "time_stamp": [
                _day_to_sec(1),  # day 1 → drop
                _day_to_sec(ALIGNED_DAY_RANGE[0]),  # day 2 → keep
                _day_to_sec(ALIGNED_DAY_RANGE[1] + 1),  # day 9 → drop
            ],
            "cpu_util_percent": [10, 20, 30],
            "mem_util_percent": [10, 20, 30],
            "cpi": [1.0, 1.0, 1.0],
            "mem_gps": [40, 40, 40],
            "mpki": [10, 10, 10],
            "net_in": [5, 5, 5],
            "net_out": [4, 4, 4],
            "disk_io_percent": [10, 20, 30],
        }
    )
    long_df = harmonize_container_usage(df)
    # 1 surviving row * 8 metrics
    assert len(long_df) == 8
    # All surviving rows come from the day-2 timestamp.
    assert (long_df["value"].isin([20.0, 1.0, 40.0, 10.0, 5.0, 4.0])).all()


def test_harmonize_uses_synthetic_namespace_tag() -> None:
    df = pd.DataFrame(
        {
            "container_id": ["c1"],
            "machine_id": ["m1"],
            "time_stamp": [_day_to_sec(3)],
            "cpu_util_percent": [50],
            "mem_util_percent": [80],
            "cpi": [1.0],
            "mem_gps": [40],
            "mpki": [10],
            "net_in": [5],
            "net_out": [4],
            "disk_io_percent": [10],
        }
    )
    long_df = harmonize_container_usage(df)
    assert (long_df["namespace"] == NAMESPACE_TAG).all()


def test_harmonize_container_meta_emits_numeric_and_status_metrics() -> None:
    df = pd.DataFrame(
        {
            "container_id": ["c1", "c1"],
            "machine_id": ["m1", "m1"],
            "time_stamp": [_day_to_sec(3), _day_to_sec(3, 60)],
            "app_du": ["app1", "app1"],
            "status": ["started", "stopped"],
            "cpu_request": [100, 100],
            "cpu_limit": [200, 200],
            "mem_size": [50.0, 50.0],
        }
    )
    long_df = harmonize_container_meta(df)
    metric_names = set(long_df["metric_name"])
    # Three numeric metrics.
    assert "alibaba_container_cpu_request" in metric_names
    assert "alibaba_container_cpu_limit" in metric_names
    assert "alibaba_container_mem_size_pct" in metric_names
    # Two one-hot status metrics (one per observed value).
    assert "alibaba_container_status__started" in metric_names
    assert "alibaba_container_status__stopped" in metric_names


def test_harmonize_empty_input_returns_empty_with_columns() -> None:
    df = pd.DataFrame(
        columns=[
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
        ]
    )
    long_df = harmonize_container_usage(df)
    assert long_df.empty
    assert list(long_df.columns) == list(LONG_FRAME_COLUMNS)
