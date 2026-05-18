"""Lifecycle features: pod age + recent restart-count rate.

Pod age is the time since this `(pod_uid, container)` was first observed.
Recent restart rate uses the synthetic-side `kube_pod_container_status_restarts_total`
counter — positive deltas in a 1-hour window indicate restarts.

For Alibaba, restarts aren't directly observable (no equivalent counter); we
emit NaN. Phase 4 model imputes per-source.

WARNING: pod age is bimodal in published K8s OOM literature (very young pods
OOM from misconfig; very old pods from leaks). Encode non-linearly downstream
if the model is linear; tree models handle this natively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cluster_canary.features._windowing import grouped_time_rolling

_SYNTH_RESTART_COL = "kube_pod_container_status_restarts_total"
_BY = ["pod_uid", "container"]
_TIME = "scrape_timestamp"


def add_pod_age(df: pd.DataFrame) -> pd.DataFrame:
    """Append `feat_pod_age_sec` — seconds since first observation per (pod_uid, container)."""
    out = df.copy()
    if out.empty:
        out["feat_pod_age_sec"] = pd.Series(dtype="float64")
        return out
    out = out.sort_values([*_BY, _TIME])
    first_seen = out.groupby(_BY, sort=False)[_TIME].transform("min")
    out["feat_pod_age_sec"] = (out[_TIME] - first_seen).dt.total_seconds().astype("float64")
    return out


def add_recent_restart_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic-only: positive delta of the restart counter over the last 1 h.

    NaN for Alibaba rows.
    """
    out = df.copy()
    if _SYNTH_RESTART_COL not in out.columns:
        out["feat_restart_rate_1h"] = np.nan
        return out

    out = grouped_time_rolling(
        out,
        by=_BY,
        time_col=_TIME,
        value_col=_SYNTH_RESTART_COL,
        window="1h",
        agg="max",
        new_col="_restart_1h_max",
    )
    out = grouped_time_rolling(
        out,
        by=_BY,
        time_col=_TIME,
        value_col=_SYNTH_RESTART_COL,
        window="1h",
        agg="min",
        new_col="_restart_1h_min",
    )
    out["feat_restart_rate_1h"] = (out["_restart_1h_max"] - out["_restart_1h_min"]).clip(lower=0)
    return out.drop(columns=["_restart_1h_max", "_restart_1h_min"])


def add_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Composition of every lifecycle-feature step in the canonical order."""
    out = add_pod_age(df)
    out = add_recent_restart_rate(out)
    return out
