"""Cross-pod aggregation features.

Two families:

1. **Co-tenant pressure (per-node)**: at each timestamp, sum/avg of all
   containers' `feat_mem_pct_of_limit` on the same node, plus the count.
   Strong predictor when node-level memory exhaustion is what's actually
   driving the OOM (rather than per-pod runaway).

2. **Image / app-du lineage 24h OOM rate**: for each image (synthetic) or
   `app_du` (Alibaba — when present in the wide frame), what fraction of pods
   sharing that lineage had an OOM event in the past 24 hours. Strict
   left-closed: at time `t`, we only use OOMs observed in `[t - 24h, t)`.
"""

from __future__ import annotations

import pandas as pd

from cluster_canary.features._windowing import grouped_time_rolling

_TIME = "scrape_timestamp"


# --- 1. Co-tenant pressure (per-node) -------------------------------------- #


def add_node_cotenant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append `feat_node_*` features — per-node aggregates at each timestamp.

    For each `(scrape_timestamp, node)`:
    - `feat_node_mem_pct_sum`: sum of `feat_mem_pct_of_limit` across all containers.
    - `feat_node_mem_pct_max`: max across all containers.
    - `feat_node_n_containers`: count of distinct containers reporting.
    """
    if "feat_mem_pct_of_limit" not in df.columns:
        raise KeyError("add_node_cotenant_features: call add_mem_pct_of_limit first")
    if df.empty:
        out = df.copy()
        for col in (
            "feat_node_mem_pct_sum",
            "feat_node_mem_pct_max",
            "feat_node_n_containers",
        ):
            out[col] = pd.Series(dtype="float64")
        return out

    grouped = (
        df.groupby([_TIME, "node"], sort=False)["feat_mem_pct_of_limit"]
        .agg(
            feat_node_mem_pct_sum="sum",
            feat_node_mem_pct_max="max",
            feat_node_n_containers="count",
        )
        .reset_index()
    )
    return df.merge(grouped, on=[_TIME, "node"], how="left")


# --- 2. Image / app_du lineage 24h OOM rate -------------------------------- #
# `app_du` is the Alibaba "deploy unit" — closest analog to a K8s
# Deployment/ReplicaSet. The synthetic side doesn't carry image hashes through
# the long-form scraper today, so we fall back to `pod` (the deployment-name
# prefix is the closest lineage key available). Phase 4+ can promote a real
# image-hash feature once it's wired through scraper.py.


def _lineage_key(df: pd.DataFrame) -> pd.Series:
    """Return the per-row lineage identifier."""
    if "app_du" in df.columns and df["app_du"].notna().any():
        return df["app_du"].astype("string").fillna(df["pod"].astype("string"))
    # Synthetic fallback: strip the random suffix from pod names
    # (e.g. "leaky-flask-abc123" → "leaky-flask"). Approximation; refine later.
    return df["pod"].astype("string").str.rsplit("-", n=1).str[0]


def add_lineage_24h_oom_rate(
    df: pd.DataFrame, *, label_col: str = "event_within_30min"
) -> pd.DataFrame:
    """For each row, fraction of same-lineage rows in `[t - 24h, t)` with label==1.

    Strict left-closed. NaN if no same-lineage rows exist in the window.
    Implementation: groupby lineage_key, rolling 24h sum + count, divide.
    """
    if label_col not in df.columns:
        raise KeyError(f"add_lineage_24h_oom_rate: {label_col!r} not in frame")

    out = df.copy()
    out["_lineage_key"] = _lineage_key(out).astype("string")
    out["_label_for_rolling"] = out[label_col].astype("float64")

    out = grouped_time_rolling(
        out,
        by=["_lineage_key"],
        time_col=_TIME,
        value_col="_label_for_rolling",
        window="24h",
        agg="sum",
        new_col="_lineage_24h_oom_sum",
    )
    out = grouped_time_rolling(
        out,
        by=["_lineage_key"],
        time_col=_TIME,
        value_col="_label_for_rolling",
        window="24h",
        agg="count",
        new_col="_lineage_24h_oom_count",
    )
    out["feat_lineage_24h_oom_rate"] = (
        out["_lineage_24h_oom_sum"] / out["_lineage_24h_oom_count"]
    ).where(out["_lineage_24h_oom_count"] > 0)

    return out.drop(
        columns=[
            "_lineage_key",
            "_label_for_rolling",
            "_lineage_24h_oom_sum",
            "_lineage_24h_oom_count",
        ]
    )


def add_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Composition of every aggregation-feature step in the canonical order."""
    out = add_node_cotenant_features(df)
    out = add_lineage_24h_oom_rate(out)
    return out
