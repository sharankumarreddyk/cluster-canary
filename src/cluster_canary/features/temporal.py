"""Time-derived features: hour-of-day, day-of-week, weekend / business-hours flags.

These are the cheapest features in the pipeline and capture diurnal patterns
that show up in batch-heavy workloads (overnight cron jobs, weekend traffic
troughs). No rolling, no group-by — pure per-row derivatives of
`scrape_timestamp`.
"""

from __future__ import annotations

import pandas as pd

TEMPORAL_FEATURE_COLS: tuple[str, ...] = (
    "feat_hour_of_day",
    "feat_day_of_week",
    "feat_is_weekend",
    "feat_is_business_hours",
)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the four temporal features to `df`. Returns a new DataFrame."""
    if "scrape_timestamp" not in df.columns:
        raise KeyError("add_temporal_features: 'scrape_timestamp' column required")
    out = df.copy()
    ts = pd.to_datetime(out["scrape_timestamp"], utc=True)
    out["feat_hour_of_day"] = ts.dt.hour.astype("int8")
    out["feat_day_of_week"] = ts.dt.dayofweek.astype("int8")
    out["feat_is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    out["feat_is_business_hours"] = (
        (ts.dt.dayofweek < 5) & (ts.dt.hour >= 9) & (ts.dt.hour < 18)
    ).astype("int8")
    return out
