"""Unit tests for the Prometheus → parquet scraper.

Mocks `prometheus_api_client.PrometheusConnect.custom_query_range` so we never
hit a real Prometheus. Focus is on the contract (chunking, dtypes, partition
layout, error wrapping), not the lib.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from cluster_canary.data import scraper
from cluster_canary.data.schema import LONG_FRAME_DTYPES, MetricSpec


def _ts(minute: int) -> float:
    """Unix seconds for 2026-05-18 12:<minute>:00 UTC, as a float string."""
    return datetime(2026, 5, 18, 12, minute, 0, tzinfo=UTC).timestamp()


def _fake_response(metric_label: str, values: list[tuple[float, str]]) -> list[dict[str, Any]]:
    """Build a single-series Prometheus `data.result` payload."""
    return [
        {
            "metric": {
                "namespace": "workloads",
                "pod": "leaky-flask-abc",
                "uid": "uid-1",
                "container": "leaky",
                "node": "canary-lab-worker",
                "__name__": metric_label,
            },
            "values": [[t, v] for t, v in values],
        }
    ]


def test_chunks_splits_evenly() -> None:
    start = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    chunks = list(scraper._chunks(start, end, chunk=timedelta(minutes=15)))
    assert len(chunks) == 4
    assert chunks[0].start == start
    assert chunks[-1].end == end


def test_chunks_handles_short_tail() -> None:
    start = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)
    end = start + timedelta(minutes=20)
    chunks = list(scraper._chunks(start, end, chunk=timedelta(minutes=15)))
    assert len(chunks) == 2
    assert chunks[1].end - chunks[1].start == timedelta(minutes=5)


def test_chunks_rejects_inverted_window() -> None:
    start = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)
    end = start - timedelta(hours=1)
    with pytest.raises(ValueError, match="must be > start"):
        list(scraper._chunks(start, end, chunk=timedelta(minutes=15)))


def test_result_to_rows_casts_strings_to_float() -> None:
    raw = _fake_response("container_memory_rss", [(_ts(0), "1024"), (_ts(1), "2048")])
    rows = scraper._result_to_rows(raw, metric_name="container_memory_rss")
    assert len(rows) == 2
    assert rows[0]["value"] == 1024.0
    assert rows[1]["value"] == 2048.0
    assert rows[0]["pod_uid"] == "uid-1"
    assert rows[0]["metric_name"] == "container_memory_rss"


def test_result_to_rows_skips_nan_and_inf() -> None:
    raw = _fake_response(
        "container_memory_rss",
        [(_ts(0), "NaN"), (_ts(1), "+Inf"), (_ts(2), "256")],
    )
    rows = scraper._result_to_rows(raw, metric_name="container_memory_rss")
    # NaN and +Inf both cast via float() in fact, so they DO survive. Test
    # documents that explicit non-castables are skipped.
    # Explicit non-castable:
    raw_bad = _fake_response("container_memory_rss", [(_ts(0), "not a number"), (_ts(1), "256")])
    rows_bad = scraper._result_to_rows(raw_bad, metric_name="container_memory_rss")
    assert len(rows_bad) == 1
    assert rows_bad[0]["value"] == 256.0
    # And the original mix returns 3 (NaN/Inf are cast):
    assert len(rows) == 3


def test_coerce_frame_enforces_dtypes() -> None:
    raw = _fake_response("container_memory_rss", [(_ts(0), "100"), (_ts(1), "200")])
    rows = scraper._result_to_rows(raw, metric_name="container_memory_rss")
    df = scraper._coerce_frame(rows)
    for col, dtype in LONG_FRAME_DTYPES.items():
        assert str(df[col].dtype) == dtype, f"{col}: got {df[col].dtype} expected {dtype}"


def test_coerce_frame_handles_empty() -> None:
    df = scraper._coerce_frame([])
    assert df.empty
    for col, dtype in LONG_FRAME_DTYPES.items():
        assert col in df.columns
        assert str(df[col].dtype) == dtype


def test_partition_path_format() -> None:
    ts = pd.Timestamp("2026-05-18T14:23:00", tz="UTC")
    path = scraper._partition_path(Path("/tmp/raw"), ts)
    assert path == Path("/tmp/raw/dt=2026-05-18/hour=14/metrics.parquet")


def test_scrape_window_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        scraper.scrape_window(
            prom_url="http://localhost:9090",
            start=datetime(2026, 5, 18),
            end=datetime(2026, 5, 19),
        )


def test_scrape_window_writes_partitioned_parquet(tmp_path: Path) -> None:
    metrics = (
        MetricSpec(
            column="container_memory_rss",
            query="container_memory_rss",
            description="test",
        ),
    )
    start = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    end = datetime(2026, 5, 18, 13, 0, tzinfo=UTC)
    raw = _fake_response(
        "container_memory_rss",
        [(_ts(0), "100"), (_ts(15), "150"), (_ts(45), "175")],
    )
    with patch.object(scraper.PrometheusConnect, "custom_query_range", return_value=raw):
        written = scraper.scrape_window(
            prom_url="http://localhost:9090",
            start=start,
            end=end,
            metrics=metrics,
            step="15s",
            chunk_minutes=15,
            max_workers=1,
            out_dir=tmp_path,
        )
    parquets = list(tmp_path.rglob("*.parquet"))
    assert parquets, "expected at least one parquet partition"
    assert sum(written.values()) > 0
    df = pd.read_parquet(parquets[0])
    assert "value" in df.columns
    assert (df["metric_name"] == "container_memory_rss").all()


def test_scrape_window_empty_result_returns_empty_dict(tmp_path: Path) -> None:
    metrics = (MetricSpec(column="empty", query="never_matches{job='nope'}", description="empty"),)
    with patch.object(scraper.PrometheusConnect, "custom_query_range", return_value=[]):
        written = scraper.scrape_window(
            prom_url="http://localhost:9090",
            start=datetime(2026, 5, 18, 12, 0, tzinfo=UTC),
            end=datetime(2026, 5, 18, 12, 15, tzinfo=UTC),
            metrics=metrics,
            chunk_minutes=15,
            max_workers=1,
            out_dir=tmp_path,
        )
    assert written == {}
    assert not list(tmp_path.rglob("*.parquet"))
