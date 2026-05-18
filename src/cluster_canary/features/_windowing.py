"""Shared rolling-window primitive used by every feature module.

The whole reason this lives in a single place: the leakage audit in
`tests/test_no_leakage.py::test_rolling_windows_are_left_closed` greps every
`.rolling(` in `src/` and requires `closed='left'`. Centralizing the call
makes that audit a 5-line check instead of a per-feature reviewer burden.

`grouped_time_rolling` is the canonical pattern:
- sort by `(group_keys..., time_col)` so per-group rolling sees ordered samples
- set `time_col` as the index so `closed='left'` is interpreted in time, not
  sample-position
- groupby + rolling with `closed='left'` so the value at time `t` is computed
  from samples in `[t - window, t)` — strictly past, never `t` itself
- merge back on `(group_keys..., time_col)` so the result aligns with the
  original wide frame's row identity
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# Type aliases for the common aggregations we use.
RollingAgg = Literal["mean", "max", "min", "std", "sum", "median", "count"]


def grouped_time_rolling(
    df: pd.DataFrame,
    *,
    by: list[str],
    time_col: str,
    value_col: str,
    window: str,
    agg: RollingAgg,
    new_col: str,
) -> pd.DataFrame:
    """Add `new_col` = time-based left-closed rolling `agg` of `value_col` per `by` groups.

    Returns a NEW DataFrame (input not mutated). `new_col` is NaN for rows where
    the rolling window is empty (e.g. the first samples of each group).
    """
    if value_col not in df.columns:
        raise KeyError(f"grouped_time_rolling: value_col {value_col!r} not in frame")
    if df.empty:
        out = df.copy()
        out[new_col] = pd.Series(dtype="float64")
        return out

    sorted_df = df.sort_values([*by, time_col]).copy()
    rolled = (
        sorted_df.set_index(time_col)
        .groupby(by, group_keys=False, sort=False)[value_col]
        .rolling(window, closed="left")
        .agg(agg)
        .reset_index()
        .rename(columns={value_col: new_col})
        # When multiple input rows share (by, time_col), the rolling output has
        # one entry per input row at that (by, time). Naively merging back would
        # explode the frame to N-by-N. Dedupe — every row at the same (by, time)
        # sees the same rolling value by construction.
        .drop_duplicates(subset=[*by, time_col], keep="first")
    )
    return sorted_df.merge(rolled, on=[*by, time_col], how="left")


def grouped_time_growth_rate(
    df: pd.DataFrame,
    *,
    by: list[str],
    time_col: str,
    value_col: str,
    window: str,
    new_col: str,
) -> pd.DataFrame:
    """Add `new_col` = (value(t) - value(t - window)) / value(t - window).

    Implemented as a left-closed `.shift`-like via rolling-mean over a tiny
    window at the lag boundary. Returns NaN at boundaries.
    """
    # Compute "value `window` ago" as the rolling-mean over a small window ending
    # exactly `window` before now. We approximate by taking the mean of the last
    # sample before [t-window, t) — i.e. `.rolling(window, closed='left').apply(lambda s: s.iloc[0])`
    # but `.iloc[0]` won't accept; using `.first()` on a resample is cleaner.
    # Simpler approach: use shift on a uniformly-sampled grid AFTER sort.
    if df.empty:
        out = df.copy()
        out[new_col] = pd.Series(dtype="float64")
        return out

    sorted_df = df.sort_values([*by, time_col]).copy()
    # `merge_asof` lets us look up "the most recent value at or before t - window"
    # exactly, regardless of sample cadence.
    lag = pd.Timedelta(window)
    sorted_df["_lag_target_time"] = sorted_df[time_col] - lag

    left = sorted_df[[*by, time_col, value_col, "_lag_target_time"]].rename(
        columns={time_col: "_now", value_col: "_now_val"}
    )
    right = sorted_df[[*by, time_col, value_col]].rename(
        columns={time_col: "_then", value_col: "_then_val"}
    )

    merged = pd.merge_asof(
        left.sort_values("_lag_target_time"),
        right.sort_values("_then"),
        by=by,
        left_on="_lag_target_time",
        right_on="_then",
        direction="backward",
        allow_exact_matches=True,
    )
    # Compute growth rate.
    denom = merged["_then_val"].where(merged["_then_val"].abs() > 1e-12, other=np.nan)
    merged[new_col] = (merged["_now_val"] - merged["_then_val"]) / denom

    # Align back to the input ordering on (by..., _now == time_col).
    keys = [*by, "_now"]
    out = sorted_df.merge(
        merged[[*keys, new_col]],
        left_on=[*by, time_col],
        right_on=keys,
        how="left",
        suffixes=("", "_dup"),
    ).drop(columns=["_lag_target_time", "_now"], errors="ignore")
    if f"{new_col}_dup" in out.columns:
        out = out.drop(columns=[f"{new_col}_dup"])
    return out
