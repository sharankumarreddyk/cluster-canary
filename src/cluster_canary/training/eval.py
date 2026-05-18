"""Evaluation harness — Brier, PR-AUC, precision@recall, lead-time, slice metrics.

Built around the headline acceptance criteria from `docs/PLAN.md`:

    Brier score                ≤ 0.10
    Precision @ recall=0.8     ≥ 0.5
    Median lead-time           ≥ 10 min

Lead-time is the gap between the first `P > threshold` observed for a pod and
its actual OOM event. Reported at 5 / 10 / 15 / 30 min horizons so the
precision-vs-lead-time tradeoff is visible (per [[impl-conventions]]).

Slice metrics: same headline metrics by `namespace`, `node`, and (when present)
by Alibaba `app_du` / synthetic pod-name prefix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

log = structlog.get_logger(__name__)

DEFAULT_THRESHOLD: float = 0.7
LEAD_TIME_HORIZONS_MIN: tuple[int, ...] = (5, 10, 15, 30)


@dataclass(frozen=True)
class HeadlineMetrics:
    """The metrics PLAN.md gates against."""

    brier_score: float
    pr_auc: float
    roc_auc: float
    precision_at_recall_0_5: float
    precision_at_recall_0_8: float
    recall_at_threshold: float
    precision_at_threshold: float
    threshold_used: float


def compute_headline_metrics(
    y_true: npt.NDArray[Any],
    y_prob: npt.NDArray[Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> HeadlineMetrics:
    """Compute the PLAN.md headline metrics on a `(y_true, y_prob)` pair."""
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")

    brier = float(brier_score_loss(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    try:
        roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        # ROC-AUC undefined if only one class present in y_true.
        roc = float("nan")
    prec_at_r5 = _precision_at_recall(y_true, y_prob, 0.5)
    prec_at_r8 = _precision_at_recall(y_true, y_prob, 0.8)

    pred = (y_prob >= threshold).astype("int8")
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec_at_t = tp / (tp + fp) if (tp + fp) else float("nan")
    rec_at_t = tp / (tp + fn) if (tp + fn) else float("nan")

    return HeadlineMetrics(
        brier_score=brier,
        pr_auc=pr_auc,
        roc_auc=roc,
        precision_at_recall_0_5=prec_at_r5,
        precision_at_recall_0_8=prec_at_r8,
        recall_at_threshold=rec_at_t,
        precision_at_threshold=prec_at_t,
        threshold_used=threshold,
    )


def _precision_at_recall(
    y_true: npt.NDArray[Any], y_prob: npt.NDArray[Any], target_recall: float
) -> float:
    """Maximum precision achievable at recall ≥ `target_recall`."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    mask = recall >= target_recall
    if not mask.any():
        return float("nan")
    return float(precision[mask].max())


def compute_lead_time_metrics(
    df: pd.DataFrame,
    y_prob: npt.NDArray[Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    horizons_min: tuple[int, ...] = LEAD_TIME_HORIZONS_MIN,
) -> dict[str, float]:
    """For each horizon, what fraction of OOMs had `P > threshold` at least that far ahead?

    Required columns on `df`: `scrape_timestamp`, `pod_uid`, `container`,
    `event_within_30min`. The OOM event time for each `(pod_uid, container)` is
    inferred as the LAST positive scrape_timestamp in its trajectory.
    """
    required = {"scrape_timestamp", "pod_uid", "container", "event_within_30min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df.empty:
        return {f"lead_time_p50_min_at_{h}min": float("nan") for h in horizons_min}

    work = df[["scrape_timestamp", "pod_uid", "container", "event_within_30min"]].copy()
    work["y_prob"] = y_prob
    # Event time for each pod = the last positive scrape, treated as `T_oom - epsilon`.
    pos = work[work["event_within_30min"] == 1]
    if pos.empty:
        return {f"lead_time_p50_min_at_{h}min": float("nan") for h in horizons_min}

    event_times_raw = (
        pos.groupby(["pod_uid", "container"], sort=False)["scrape_timestamp"].max().to_dict()
    )
    # `groupby` keys are typed as Hashable by mypy; cast to tuple explicitly.
    event_times: dict[tuple[str, str], pd.Timestamp] = {}
    for k, v in event_times_raw.items():
        k_tuple = k if isinstance(k, tuple) else (k,)
        event_times[(str(k_tuple[0]), str(k_tuple[1]))] = v
    metrics: dict[str, float] = {}
    for horizon in horizons_min:
        lead_minutes: list[float] = []
        for (pod_uid, container), event_t in event_times.items():
            pod_rows = work[(work["pod_uid"] == pod_uid) & (work["container"] == container)]
            crosses = pod_rows[
                (pod_rows["y_prob"] >= threshold)
                & (pod_rows["scrape_timestamp"] <= event_t - pd.Timedelta(minutes=horizon))
            ]
            if not crosses.empty:
                first_alert = crosses["scrape_timestamp"].min()
                gap = (event_t - first_alert).total_seconds() / 60.0
                lead_minutes.append(gap)
        if lead_minutes:
            metrics[f"lead_time_p50_min_at_{horizon}min"] = float(np.median(lead_minutes))
            metrics[f"lead_time_p25_min_at_{horizon}min"] = float(np.percentile(lead_minutes, 25))
            metrics[f"detection_rate_at_{horizon}min"] = float(len(lead_minutes) / len(event_times))
        else:
            metrics[f"lead_time_p50_min_at_{horizon}min"] = float("nan")
            metrics[f"lead_time_p25_min_at_{horizon}min"] = float("nan")
            metrics[f"detection_rate_at_{horizon}min"] = 0.0
    return metrics


def compute_slice_metrics(
    df: pd.DataFrame,
    y_prob: npt.NDArray[Any],
    *,
    slice_cols: tuple[str, ...] = ("namespace", "node"),
    threshold: float = DEFAULT_THRESHOLD,
    min_rows_per_slice: int = 100,
) -> list[dict[str, object]]:
    """Compute headline metrics per slice. Returns a list of dicts (one per slice value).

    Slices with fewer than `min_rows_per_slice` rows are dropped — metrics on
    tiny slices are noise.
    """
    required = {"scrape_timestamp", "pod_uid", "container", "event_within_30min", *slice_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns for slice_cols: {sorted(missing)}")

    out: list[dict[str, object]] = []
    work = df.copy()
    work["_y_prob"] = y_prob
    for col in slice_cols:
        for value, group in work.groupby(col, sort=False):
            if len(group) < min_rows_per_slice:
                continue
            if group["event_within_30min"].sum() == 0:
                # No positives — metrics undefined; record n_positive=0 for visibility.
                out.append(
                    {
                        "slice_col": col,
                        "slice_value": str(value),
                        "n_rows": len(group),
                        "n_positive": 0,
                        "brier_score": float(
                            brier_score_loss(
                                group["event_within_30min"].to_numpy(),
                                group["_y_prob"].to_numpy(),
                            )
                        ),
                        "pr_auc": float("nan"),
                        "precision_at_recall_0_8": float("nan"),
                    }
                )
                continue
            headline = compute_headline_metrics(
                group["event_within_30min"].to_numpy(),
                group["_y_prob"].to_numpy(),
                threshold=threshold,
            )
            d = asdict(headline)
            d.update(
                {
                    "slice_col": col,
                    "slice_value": str(value),
                    "n_rows": len(group),
                    "n_positive": int(group["event_within_30min"].sum()),
                }
            )
            out.append(d)
    return out


def assert_phase4_acceptance(metrics: HeadlineMetrics) -> dict[str, bool]:
    """Apply the PLAN.md Phase-4 acceptance gates. Returns per-gate pass/fail.

    Doesn't raise — the orchestrator decides whether a failing run still gets
    logged (for diagnostics) or is rejected.
    """
    return {
        "brier_le_0_10": metrics.brier_score <= 0.10,
        "precision_at_recall_0_8_ge_0_5": metrics.precision_at_recall_0_8 >= 0.5,
    }
