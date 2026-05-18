"""Tests for the temporal train/val/test split + entity-overlap opt-in."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cluster_canary.features.split import (
    DEFAULT_ENTITY_OVERLAP_ALLOWED,
    TemporalSplitError,
    temporal_split,
    write_entity_overlap_opt_in,
)


def _df_with_n_rows(n: int) -> pd.DataFrame:
    base = pd.Timestamp("2026-05-18T00:00:00", tz="UTC")
    rows = []
    for i in range(n):
        rows.append(
            {
                "scrape_timestamp": base + pd.Timedelta(minutes=i),
                "pod_uid": f"u{i}",
                "feat_x": float(i),
                "event_within_30min": int(i % 10 == 0),
            }
        )
    return pd.DataFrame(rows)


def test_temporal_split_60_20_20_yields_strict_ordering() -> None:
    df = _df_with_n_rows(1000)
    train, val, test, metrics = temporal_split(df, train_frac=0.60, val_frac=0.20)
    assert metrics.n_train == 600
    assert metrics.n_val == 200
    assert metrics.n_test == 200
    assert train["scrape_timestamp"].max() < val["scrape_timestamp"].min()
    assert val["scrape_timestamp"].max() < test["scrape_timestamp"].min()


def test_temporal_split_rejects_invalid_fractions() -> None:
    df = _df_with_n_rows(100)
    with pytest.raises(TemporalSplitError):
        temporal_split(df, train_frac=0.80, val_frac=0.30)
    with pytest.raises(TemporalSplitError):
        temporal_split(df, train_frac=1.5, val_frac=0.20)
    with pytest.raises(TemporalSplitError):
        temporal_split(df, train_frac=0.0, val_frac=0.20)


def test_temporal_split_rejects_empty_frame() -> None:
    df = _df_with_n_rows(0)
    with pytest.raises(TemporalSplitError, match="empty"):
        temporal_split(df)


def test_temporal_split_handles_duplicate_boundary_timestamps() -> None:
    # 1000 spread-out rows with a small duplicate cluster spanning the train/val
    # boundary. The helper should absorb the duplicates into `train` so strict
    # ordering holds and no split collapses.
    rows = []
    base = pd.Timestamp("2026-05-18T00:00:00", tz="UTC")
    # 595 rows at distinct timestamps (one per minute) — below the train cut.
    for i in range(595):
        rows.append(
            {
                "scrape_timestamp": base + pd.Timedelta(minutes=i),
                "feat_x": float(i),
                "event_within_30min": 0,
            }
        )
    # 10 rows all at the SAME timestamp — straddles the train/val boundary
    # (since train_frac=0.60 of 1000 = 600).
    cluster_ts = base + pd.Timedelta(minutes=596)
    for i in range(10):
        rows.append({"scrape_timestamp": cluster_ts, "feat_x": 595.0 + i, "event_within_30min": 0})
    # 395 more rows at distinct timestamps after the cluster.
    for i in range(395):
        rows.append(
            {
                "scrape_timestamp": base + pd.Timedelta(minutes=600 + i),
                "feat_x": 1000.0 + i,
                "event_within_30min": 0,
            }
        )
    df = pd.DataFrame(rows)
    train, val, test, _ = temporal_split(df, train_frac=0.60, val_frac=0.20)
    assert train["scrape_timestamp"].max() < val["scrape_timestamp"].min()
    assert val["scrape_timestamp"].max() < test["scrape_timestamp"].min()
    # All 10 duplicate-cluster rows ended up in train (boundary absorbed them).
    cluster_in_train = train[train["scrape_timestamp"] == cluster_ts]
    assert len(cluster_in_train) == 10


def test_split_metrics_records_per_split_positive_rate() -> None:
    df = _df_with_n_rows(1000)
    _, _, _, metrics = temporal_split(df, train_frac=0.60, val_frac=0.20)
    # 1 in 10 rows is positive (i % 10 == 0); rough sanity check.
    for rate in (
        metrics.positive_rate_train,
        metrics.positive_rate_val,
        metrics.positive_rate_test,
    ):
        assert 0.05 < rate < 0.20


def test_write_entity_overlap_opt_in_default_contains_pod_uid(tmp_path: Path) -> None:
    path = write_entity_overlap_opt_in(tmp_path)
    data = json.loads(path.read_text())
    assert "pod_uid" in data
    assert set(data) == set(DEFAULT_ENTITY_OVERLAP_ALLOWED)
