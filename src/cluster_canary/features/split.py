"""Temporal train/val/test split + entity-overlap opt-in.

The leakage audit requires:
1. `train[time].max() < val[time].min() < val[time].max() < test[time].min()`.
2. Any `_id`-suffixed column that overlaps across splits must be explicitly
   opted into `data/processed/entity_overlap.json` — otherwise the audit fails.

For cluster-canary, `pod_uid` overlap IS expected by design (the same pods
appear across training windows; we're learning per-pod patterns over time, not
generalizing to unseen pods). We write that opt-in alongside the split.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# Default 60/20/20 split. Override via `temporal_split(... train_frac=0.7 ...)`.
DEFAULT_TRAIN_FRAC: float = 0.60
DEFAULT_VAL_FRAC: float = 0.20
# test_frac = 1 - train_frac - val_frac

# Columns expected to overlap across splits by design for cluster-canary.
DEFAULT_ENTITY_OVERLAP_ALLOWED: tuple[str, ...] = ("pod_uid",)


class TemporalSplitError(RuntimeError):
    """Raised when temporal-split inputs violate the contract."""


@dataclass(frozen=True)
class SplitMetrics:
    """Summary of a temporal split — written to reports/eval/split_metrics.json."""

    n_train: int
    n_val: int
    n_test: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    positive_rate_train: float
    positive_rate_val: float
    positive_rate_test: float


def temporal_split(
    df: pd.DataFrame,
    *,
    time_col: str = "scrape_timestamp",
    label_col: str = "event_within_30min",
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetrics]:
    """Return `(train, val, test, metrics)` with strict temporal ordering.

    Sorts by `time_col` then cuts at quantile boundaries — robust to non-uniform
    data density across the time axis.
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise TemporalSplitError(
            f"invalid fractions: train={train_frac}, val={val_frac} "
            f"(must satisfy 0 < train_frac, 0 < val_frac, train_frac + val_frac < 1)"
        )
    if df.empty:
        raise TemporalSplitError("cannot split empty frame")
    if time_col not in df.columns:
        raise TemporalSplitError(f"time_col {time_col!r} not in frame")

    sorted_df = df.sort_values(time_col).reset_index(drop=True)
    n = len(sorted_df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # Anything left after train + val is the test set.
    train = sorted_df.iloc[:n_train].copy()
    val = sorted_df.iloc[n_train : n_train + n_val].copy()
    test = sorted_df.iloc[n_train + n_val :].copy()

    if train.empty or val.empty or test.empty:
        raise TemporalSplitError(
            f"degenerate split: train={len(train)}, val={len(val)}, test={len(test)}"
        )

    # Enforce strict temporal ordering: bump the boundaries forward if rows
    # share a timestamp across the cut (rare but possible at coarse cadences).
    train, val = _shift_boundary_forward(train, val, time_col)
    val, test = _shift_boundary_forward(val, test, time_col)
    if train.empty or val.empty or test.empty:
        raise TemporalSplitError(
            "split collapsed after boundary shift — increase frac or dedupe ts"
        )

    metrics = SplitMetrics(
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        train_start=str(train[time_col].min()),
        train_end=str(train[time_col].max()),
        val_start=str(val[time_col].min()),
        val_end=str(val[time_col].max()),
        test_start=str(test[time_col].min()),
        test_end=str(test[time_col].max()),
        positive_rate_train=float(train[label_col].mean())
        if label_col in train.columns
        else float("nan"),
        positive_rate_val=float(val[label_col].mean())
        if label_col in val.columns
        else float("nan"),
        positive_rate_test=float(test[label_col].mean())
        if label_col in test.columns
        else float("nan"),
    )
    log.info("split.done", **metrics.__dict__)
    return train, val, test, metrics


def _shift_boundary_forward(
    left: pd.DataFrame, right: pd.DataFrame, time_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move rows whose timestamp equals `left.max()` from `right` into `left`.

    Without this, two splits can share a timestamp at the boundary, which makes
    `left.max() < right.min()` fail by an equality.
    """
    if left.empty or right.empty:
        return left, right
    boundary = left[time_col].max()
    shifted = right[right[time_col] <= boundary]
    if shifted.empty:
        return left, right
    return pd.concat([left, shifted], ignore_index=True), right[
        right[time_col] > boundary
    ].reset_index(drop=True)


def write_entity_overlap_opt_in(
    out_dir: Path,
    *,
    allowed: Iterable[str] = DEFAULT_ENTITY_OVERLAP_ALLOWED,
) -> Path:
    """Write `entity_overlap.json` so the leakage audit recognises intentional overlap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "entity_overlap.json"
    path.write_text(json.dumps(sorted(set(allowed)), indent=2))
    log.info("split.entity_overlap.written", path=str(path), allowed=sorted(set(allowed)))
    return path
