"""Alibaba CSV → cluster-canary long-form parquet.

Reads the 5 CSV files from `data/raw/alibaba/` (after `alibaba_ingest`),
applies the metric-name mapping in `alibaba_schema.py`, anchors the
"seconds since trace start" timestamps to a synthetic UTC origin, trims to
the day-2..day-8 alignment window, and writes one hour-partitioned parquet
tree per source table under `data/interim/alibaba/`.

The output parquet matches `schema.py`'s `LONG_FRAME_COLUMNS` contract so the
downstream Phase 3 feature engineering treats Alibaba and synthetic data
identically (subject to the `alibaba_*` metric-name prefix).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pandas as pd
import structlog

from cluster_canary.data.alibaba_schema import (
    ALIGNED_DAY_RANGE,
    CONTAINER_META_COLS,
    CONTAINER_META_METRIC_MAP,
    CONTAINER_META_STATUS_METRIC_PREFIX,
    CONTAINER_USAGE_COLS,
    CONTAINER_USAGE_METRIC_MAP,
    DISK_IO_INVALID_SENTINELS,
    NAMESPACE_TAG,
    SECONDS_PER_DAY,
    TRACE_ORIGIN_UTC,
)
from cluster_canary.data.schema import (
    LONG_FRAME_COLUMNS,
    LONG_FRAME_DTYPES,
)

log = structlog.get_logger(__name__)

DEFAULT_RAW_DIR = Path("data/raw/alibaba")
DEFAULT_INTERIM_DIR = Path("data/interim/alibaba")

_ALIGNED_START_SEC: Final[int] = ALIGNED_DAY_RANGE[0] * SECONDS_PER_DAY  # incl
_ALIGNED_END_SEC: Final[int] = (ALIGNED_DAY_RANGE[1] + 1) * SECONDS_PER_DAY  # excl
_TRACE_ORIGIN: Final[pd.Timestamp] = pd.Timestamp(TRACE_ORIGIN_UTC)


class HarmonizeError(RuntimeError):
    """Raised when an Alibaba CSV cannot be harmonized."""


def _ts_from_trace_seconds(ts_seconds: pd.Series) -> pd.Series:
    """Convert Alibaba's `time_stamp` (seconds since trace start) to UTC datetime."""
    return _TRACE_ORIGIN + pd.to_timedelta(ts_seconds, unit="s")


def _filter_aligned_days(df: pd.DataFrame, *, ts_col: str = "time_stamp") -> pd.DataFrame:
    """Keep only rows in the day-2..day-8 window where machine_* and container_* align."""
    mask = (df[ts_col] >= _ALIGNED_START_SEC) & (df[ts_col] < _ALIGNED_END_SEC)
    n_in = len(df)
    out = df.loc[mask].reset_index(drop=True)
    log.info(
        "harmonize.aligned_filter",
        n_in=n_in,
        n_out=len(out),
        dropped_pct=round((n_in - len(out)) / n_in * 100, 2) if n_in else 0.0,
    )
    return out


def _empty_long_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {col: pd.Series(dtype=dtype) for col, dtype in LONG_FRAME_DTYPES.items()}
    )


def _coerce(rows: pd.DataFrame) -> pd.DataFrame:
    """Force the contract: same columns + dtypes as the synthetic scraper output."""
    for col in LONG_FRAME_COLUMNS:
        if col not in rows.columns:
            rows[col] = pd.Series(dtype=LONG_FRAME_DTYPES[col])
    rows = rows[list(LONG_FRAME_COLUMNS)]
    for col, dtype in LONG_FRAME_DTYPES.items():
        rows[col] = rows[col].astype(dtype)  # type: ignore[call-overload]
    return rows


def harmonize_container_usage(df: pd.DataFrame) -> pd.DataFrame:
    """`container_usage.csv` → long-form."""
    if df.empty:
        return _empty_long_frame()

    df = _filter_aligned_days(df)
    if df.empty:
        return _empty_long_frame()

    # Sentinel: disk_io_percent in {-1, 101} = invalid → drop those rows for the disk metric only.
    long_frames: list[pd.DataFrame] = []
    base = pd.DataFrame(
        {
            "scrape_timestamp": _ts_from_trace_seconds(df["time_stamp"]),
            "namespace": NAMESPACE_TAG,
            "pod": df["container_id"].astype("string"),
            "pod_uid": df["container_id"].astype("string"),
            "container": df["container_id"].astype("string"),
            "node": df["machine_id"].astype("string"),
        }
    )
    for src_col, metric_name in CONTAINER_USAGE_METRIC_MAP.items():
        sub = base.copy()
        sub["metric_name"] = metric_name
        sub["value"] = pd.to_numeric(df[src_col], errors="coerce")
        if src_col == "disk_io_percent":
            sub = sub[~sub["value"].isin(DISK_IO_INVALID_SENTINELS)]
        sub = sub.dropna(subset=["value"])
        long_frames.append(sub)

    out = pd.concat(long_frames, ignore_index=True) if long_frames else _empty_long_frame()
    return _coerce(out)


def harmonize_container_meta(df: pd.DataFrame) -> pd.DataFrame:
    """`container_meta.csv` → long-form (numeric columns + one-hot status)."""
    if df.empty:
        return _empty_long_frame()

    df = _filter_aligned_days(df)
    if df.empty:
        return _empty_long_frame()

    base = pd.DataFrame(
        {
            "scrape_timestamp": _ts_from_trace_seconds(df["time_stamp"]),
            "namespace": NAMESPACE_TAG,
            "pod": df["container_id"].astype("string"),
            "pod_uid": df["container_id"].astype("string"),
            "container": df["container_id"].astype("string"),
            "node": df["machine_id"].astype("string"),
        }
    )

    long_frames: list[pd.DataFrame] = []
    for src_col, metric_name in CONTAINER_META_METRIC_MAP.items():
        sub = base.copy()
        sub["metric_name"] = metric_name
        sub["value"] = pd.to_numeric(df[src_col], errors="coerce")
        sub = sub.dropna(subset=["value"])
        long_frames.append(sub)

    # One-hot status — emit one metric per observed value. The actual enum is
    # undocumented (issue #223 open), so we log unknowns rather than rejecting.
    observed_statuses = (
        df["status"].dropna().astype("string").unique().tolist()
    )
    log.info(
        "harmonize.container_meta.status_values",
        observed=observed_statuses,
    )
    for status_value in observed_statuses:
        sub = base.copy()
        sub["metric_name"] = f"{CONTAINER_META_STATUS_METRIC_PREFIX}{status_value}"
        sub["value"] = (df["status"].astype("string") == status_value).astype("float64")
        long_frames.append(sub)

    out = pd.concat(long_frames, ignore_index=True) if long_frames else _empty_long_frame()
    return _coerce(out)


def _read_no_header(path: Path, names: Iterable[str]) -> pd.DataFrame:
    """Read an Alibaba CSV (no header row) with explicit column names."""
    if not path.exists():
        raise HarmonizeError(f"missing CSV: {path}")
    return pd.read_csv(path, header=None, names=list(names))


def harmonize_all(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_dir: Path = DEFAULT_INTERIM_DIR,
) -> dict[str, Path]:
    """End-to-end: read 2 CSVs from `raw_dir`, write 2 parquets to `out_dir`.

    Returns a dict of `{source_csv_name: parquet_path}` for downstream consumers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    cu_csv = raw_dir / "container_usage.csv"
    cm_csv = raw_dir / "container_meta.csv"

    if cu_csv.exists():
        log.info("harmonize.start", csv=cu_csv.name)
        cu_df = _read_no_header(cu_csv, CONTAINER_USAGE_COLS)
        long_cu = harmonize_container_usage(cu_df)
        out = out_dir / "container_usage.parquet"
        long_cu.to_parquet(out, index=False)
        result["container_usage"] = out
        log.info("harmonize.done", csv=cu_csv.name, rows_out=len(long_cu), out=str(out))

    if cm_csv.exists():
        log.info("harmonize.start", csv=cm_csv.name)
        cm_df = _read_no_header(cm_csv, CONTAINER_META_COLS)
        long_cm = harmonize_container_meta(cm_df)
        out = out_dir / "container_meta.parquet"
        long_cm.to_parquet(out, index=False)
        result["container_meta"] = out
        log.info("harmonize.done", csv=cm_csv.name, rows_out=len(long_cm), out=str(out))

    if not result:
        raise HarmonizeError(
            f"no Alibaba CSVs found under {raw_dir}; run `alibaba_ingest` first"
        )

    return result


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
        prog="cluster_canary.data.alibaba_harmonize", description=__doc__
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(os.environ.get("ALIBABA_RAW_DIR", str(DEFAULT_RAW_DIR))),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("ALIBABA_INTERIM_DIR", str(DEFAULT_INTERIM_DIR))),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)
    result = harmonize_all(raw_dir=args.raw_dir, out_dir=args.out_dir)
    log.info("harmonize.complete", n_outputs=len(result), out_dir=str(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
