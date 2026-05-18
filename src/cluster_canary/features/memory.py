"""Memory features — the OOM predictor's load-bearing feature group.

Three layers:

1. **Unification**: `add_mem_pct_of_limit` produces a single `feat_mem_pct_of_limit`
   column from whichever raw inputs the source provides:
   - synthetic: `working_set_bytes / limit_bytes`
   - Alibaba:  `alibaba_container_mem_util_pct / 100`

2. **Rolling aggregates** over the unified column at 5 / 15 / 30 min windows,
   all left-closed (no future peek; the leakage audit enforces this).

3. **Growth rate** — `(value(t) - value(t - window)) / value(t - window)` for
   the 5 / 15 / 30 min windows. The strongest leading indicator of impending
   OOM per the prior-art research (Microsoft Narya OSDI 2020).

Synthetic-only extras: working-set vs RSS gap (the OOM-killer evaluates
working-set; RSS lags), memory_failcnt rate (cgroup hit limit but didn't OOM
— strongest leading signal of all).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cluster_canary.features._windowing import (
    grouped_time_growth_rate,
    grouped_time_rolling,
)

# Source-detector column names (one signals each side).
_ALIBABA_MEM_COL = "alibaba_container_mem_util_pct"
_SYNTH_WS_COL = "container_memory_working_set_bytes"
_SYNTH_LIMIT_COL = "kube_pod_container_resource_limits_memory_bytes"
_SYNTH_RSS_COL = "container_memory_rss"
_SYNTH_FAILCNT_COL = "container_memory_failcnt"

ROLLING_WINDOWS: tuple[str, ...] = ("5min", "15min", "30min")
_BY = ["pod_uid", "container"]
_TIME = "scrape_timestamp"


def add_mem_pct_of_limit(df: pd.DataFrame) -> pd.DataFrame:
    """Unify memory utilization across synthetic + Alibaba into `feat_mem_pct_of_limit`.

    Range [0, 1] (fraction, not percent — easier to compose with downstream math).
    NaN where the source doesn't have the needed inputs (defensive — features
    later either drop or impute).
    """
    out = df.copy()
    if _ALIBABA_MEM_COL in out.columns:
        out["feat_mem_pct_of_limit"] = out[_ALIBABA_MEM_COL].astype("float64") / 100.0
        return out
    if _SYNTH_WS_COL in out.columns and _SYNTH_LIMIT_COL in out.columns:
        limit = out[_SYNTH_LIMIT_COL].astype("float64")
        ws = out[_SYNTH_WS_COL].astype("float64")
        out["feat_mem_pct_of_limit"] = np.where(limit > 0, ws / limit, np.nan)
        return out
    out["feat_mem_pct_of_limit"] = np.nan
    return out


def add_mem_rolling(
    df: pd.DataFrame,
    *,
    windows: tuple[str, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Append rolling mean / max for `feat_mem_pct_of_limit` at each window.

    Output columns: `feat_mem_pct_of_limit__<window>_<agg>`.
    """
    if "feat_mem_pct_of_limit" not in df.columns:
        raise KeyError("add_mem_rolling: call add_mem_pct_of_limit first")
    out = df
    for window in windows:
        for agg in ("mean", "max"):
            out = grouped_time_rolling(
                out,
                by=_BY,
                time_col=_TIME,
                value_col="feat_mem_pct_of_limit",
                window=window,
                agg=agg,
                new_col=f"feat_mem_pct_of_limit__{window}_{agg}",
            )
    return out


def add_mem_growth_rate(
    df: pd.DataFrame,
    *,
    windows: tuple[str, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Append growth-rate features (relative change over each window)."""
    if "feat_mem_pct_of_limit" not in df.columns:
        raise KeyError("add_mem_growth_rate: call add_mem_pct_of_limit first")
    out = df
    for window in windows:
        out = grouped_time_growth_rate(
            out,
            by=_BY,
            time_col=_TIME,
            value_col="feat_mem_pct_of_limit",
            window=window,
            new_col=f"feat_mem_growth_rate__{window}",
        )
    return out


def add_synth_only_features(df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic-only memory features: working-set/RSS gap + failcnt rate.

    No-op for Alibaba rows (the raw inputs aren't present). For synthetic rows,
    appends:
    - `feat_mem_ws_rss_gap_pct`: (working_set - rss) / limit
    - `feat_mem_failcnt__5min_rate`: positive deltas in failcnt over 5 min
    """
    out = df.copy()

    if (
        _SYNTH_WS_COL in out.columns
        and _SYNTH_RSS_COL in out.columns
        and _SYNTH_LIMIT_COL in out.columns
    ):
        limit = out[_SYNTH_LIMIT_COL].astype("float64")
        ws = out[_SYNTH_WS_COL].astype("float64")
        rss = out[_SYNTH_RSS_COL].astype("float64")
        out["feat_mem_ws_rss_gap_pct"] = np.where(limit > 0, (ws - rss) / limit, np.nan)
    else:
        out["feat_mem_ws_rss_gap_pct"] = np.nan

    if _SYNTH_FAILCNT_COL in out.columns:
        out = grouped_time_rolling(
            out,
            by=_BY,
            time_col=_TIME,
            value_col=_SYNTH_FAILCNT_COL,
            window="5min",
            agg="max",
            new_col="_failcnt_5min_max",
        )
        out = grouped_time_rolling(
            out,
            by=_BY,
            time_col=_TIME,
            value_col=_SYNTH_FAILCNT_COL,
            window="5min",
            agg="min",
            new_col="_failcnt_5min_min",
        )
        out["feat_mem_failcnt__5min_rate"] = out["_failcnt_5min_max"] - out["_failcnt_5min_min"]
        out = out.drop(columns=["_failcnt_5min_max", "_failcnt_5min_min"])
    else:
        out["feat_mem_failcnt__5min_rate"] = np.nan

    return out


def add_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Composition of every memory-feature step in the canonical order."""
    out = add_mem_pct_of_limit(df)
    out = add_mem_rolling(out)
    out = add_mem_growth_rate(out)
    out = add_synth_only_features(out)
    return out
