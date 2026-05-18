"""CPU features — rolling utilization + throttle ratio.

Synthetic side supplies cAdvisor rates (`container_cpu_usage_seconds_rate`,
`container_cpu_throttled_periods_rate`, `container_cpu_periods_rate`).
Alibaba side supplies normalized `alibaba_container_cpu_util_pct` only —
no throttle signal at all, so the throttle ratio is NaN'd out for Alibaba rows.

Throttle ratio is `throttled_periods / total_periods`. Counterintuitively
*predictive* of OOM in JVM/Go GC-pressure workloads (per Phase 1 research notes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cluster_canary.features._windowing import grouped_time_rolling

_ALIBABA_CPU_COL = "alibaba_container_cpu_util_pct"
_SYNTH_CPU_COL = "container_cpu_usage_seconds_rate"
_SYNTH_THROTTLED_COL = "container_cpu_throttled_periods_rate"
_SYNTH_PERIODS_COL = "container_cpu_periods_rate"

ROLLING_WINDOWS: tuple[str, ...] = ("5min", "15min")
_BY = ["pod_uid", "container"]
_TIME = "scrape_timestamp"


def add_cpu_util(df: pd.DataFrame) -> pd.DataFrame:
    """Unified CPU utilization across both sources into `feat_cpu_util`.

    Range [0, 1] for Alibaba (divided by 100 from the percent), cores-equivalent
    for synthetic (raw rate). Not strictly comparable across sources — Phase 4
    can choose to z-score per-source if it matters; for tree models the absolute
    value is fine inside one source.
    """
    out = df.copy()
    if _ALIBABA_CPU_COL in out.columns:
        out["feat_cpu_util"] = out[_ALIBABA_CPU_COL].astype("float64") / 100.0
    elif _SYNTH_CPU_COL in out.columns:
        out["feat_cpu_util"] = out[_SYNTH_CPU_COL].astype("float64")
    else:
        out["feat_cpu_util"] = np.nan
    return out


def add_cpu_throttle_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic-only: throttled / total CFS periods. NaN for Alibaba rows."""
    out = df.copy()
    if _SYNTH_THROTTLED_COL in out.columns and _SYNTH_PERIODS_COL in out.columns:
        throttled = out[_SYNTH_THROTTLED_COL].astype("float64")
        total = out[_SYNTH_PERIODS_COL].astype("float64")
        out["feat_cpu_throttle_ratio"] = np.where(total > 0, throttled / total, np.nan)
    else:
        out["feat_cpu_throttle_ratio"] = np.nan
    return out


def add_cpu_rolling(
    df: pd.DataFrame,
    *,
    windows: tuple[str, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Rolling means of `feat_cpu_util` and `feat_cpu_throttle_ratio` per window."""
    if "feat_cpu_util" not in df.columns:
        raise KeyError("add_cpu_rolling: call add_cpu_util first")
    out = df
    for window in windows:
        out = grouped_time_rolling(
            out,
            by=_BY,
            time_col=_TIME,
            value_col="feat_cpu_util",
            window=window,
            agg="mean",
            new_col=f"feat_cpu_util__{window}_mean",
        )
        # Only roll the throttle ratio if it has any non-NaN values.
        if out["feat_cpu_throttle_ratio"].notna().any():
            out = grouped_time_rolling(
                out,
                by=_BY,
                time_col=_TIME,
                value_col="feat_cpu_throttle_ratio",
                window=window,
                agg="mean",
                new_col=f"feat_cpu_throttle_ratio__{window}_mean",
            )
        else:
            out[f"feat_cpu_throttle_ratio__{window}_mean"] = np.nan
    return out


def add_cpu_features(df: pd.DataFrame) -> pd.DataFrame:
    """Composition of every CPU-feature step in the canonical order."""
    out = add_cpu_util(df)
    out = add_cpu_throttle_ratio(out)
    out = add_cpu_rolling(out)
    return out
