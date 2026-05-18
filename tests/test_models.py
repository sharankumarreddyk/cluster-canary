"""Unit tests for Phase 4 model components.

Tiny synthetic inputs. The real acceptance gates (Brier ≤ 0.10, P @ R=0.8 ≥ 0.5)
are only checkable on real data — those run once `data/processed/` exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cluster_canary.models.baseline import BaselineRule
from cluster_canary.models.calibration import (
    CalibrationError,
    Calibrator,
)
from cluster_canary.training.eval import (
    HeadlineMetrics,
    compute_headline_metrics,
    compute_lead_time_metrics,
    compute_slice_metrics,
)
from cluster_canary.training.imbalance import (
    ImbalanceError,
    downsample_negatives,
    elkan_correct,
)

# --- baseline ------------------------------------------------------------- #


def test_baseline_rule_predicts_above_threshold() -> None:
    df = pd.DataFrame(
        {
            "feat_mem_pct_of_limit__5min_max": [0.5, 0.85, 0.91, 0.99],
        }
    )
    rule = BaselineRule(threshold=0.9)
    p = rule.predict_proba(df)
    assert p.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_baseline_rule_falls_back_to_unrolled_col() -> None:
    df = pd.DataFrame({"feat_mem_pct_of_limit": [0.95]})
    rule = BaselineRule(threshold=0.9)
    assert rule.predict_proba(df).tolist() == [1.0]


def test_baseline_rule_raises_when_neither_col_present() -> None:
    df = pd.DataFrame({"some_other_col": [1.0]})
    with pytest.raises(KeyError, match="neither"):
        BaselineRule().predict_proba(df)


# --- calibration ---------------------------------------------------------- #


def test_calibrator_isotonic_monotone() -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.0, 1.0, size=1000)
    # Label = 1 with probability proportional to raw — but skewed.
    y = (rng.uniform(0, 1, 1000) < (raw**2)).astype(int)
    cal = Calibrator.fit(raw, y, method="isotonic")
    grid = np.linspace(0.0, 1.0, 50)
    out = cal.transform(grid)
    # Monotone non-decreasing.
    diffs = np.diff(out)
    assert (diffs >= -1e-9).all()


def test_calibrator_platt_returns_probabilities() -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.0, 1.0, size=500)
    y = (raw > 0.6).astype(int)
    cal = Calibrator.fit(raw, y, method="platt")
    out = cal.transform(np.array([0.1, 0.5, 0.9]))
    assert ((out >= 0) & (out <= 1)).all()


def test_calibrator_rejects_no_positives() -> None:
    with pytest.raises(CalibrationError, match="no positives"):
        Calibrator.fit(np.array([0.1, 0.2]), np.array([0, 0]))


def test_calibrator_rejects_out_of_range() -> None:
    with pytest.raises(CalibrationError, match="\\[0, 1\\]"):
        Calibrator.fit(np.array([0.5, 1.5]), np.array([0, 1]))


# --- imbalance ------------------------------------------------------------ #


def test_downsample_brings_positives_to_target_rate() -> None:
    df = pd.DataFrame(
        {
            "event_within_30min": [1] * 10 + [0] * 990,
            "feat_x": list(range(1000)),
        }
    )
    res = downsample_negatives(df, target_positive_rate=0.1)
    assert len(res.df) == 100  # 10 positives + 90 negatives
    assert res.df["event_within_30min"].mean() == pytest.approx(0.1, abs=1e-9)


def test_downsample_preserves_temporal_order() -> None:
    df = pd.DataFrame(
        {
            "event_within_30min": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            "scrape_timestamp": pd.date_range("2026-05-18", periods=12, freq="1min"),
        }
    )
    res = downsample_negatives(df, target_positive_rate=0.5)
    # Surviving rows should still be ordered ascending by scrape_timestamp.
    ts = res.df["scrape_timestamp"].tolist()
    assert ts == sorted(ts)


def test_downsample_raises_with_zero_positives() -> None:
    df = pd.DataFrame({"event_within_30min": [0, 0, 0]})
    with pytest.raises(ImbalanceError, match="no positives"):
        downsample_negatives(df)


def test_elkan_correct_identity_at_ratio_1() -> None:
    p = np.array([0.1, 0.5, 0.9])
    out = elkan_correct(p, sample_ratio=1.0)
    np.testing.assert_allclose(out, p)


def test_elkan_correct_lowers_probabilities_at_small_ratio() -> None:
    # At sample_ratio=0.1, the model saw 10x the true positive rate during
    # training, so its raw probabilities should be CORRECTED DOWN.
    p_train = np.array([0.5])
    out = elkan_correct(p_train, sample_ratio=0.1)
    assert out[0] < p_train[0]


# --- eval ----------------------------------------------------------------- #


def test_compute_headline_metrics_perfect_predictions() -> None:
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    m = compute_headline_metrics(y, p)
    assert m.brier_score == 0.0
    assert m.pr_auc == 1.0
    assert m.precision_at_recall_0_8 == 1.0


def test_compute_headline_metrics_returns_dataclass() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = compute_headline_metrics(y, p)
    assert isinstance(m, HeadlineMetrics)
    assert 0.0 <= m.brier_score <= 1.0
    assert 0.0 <= m.pr_auc <= 1.0


def test_compute_lead_time_metrics_recovers_known_lead() -> None:
    base = pd.Timestamp("2026-05-18T00:00:00", tz="UTC")
    rows = []
    for i in range(60):
        # 30s cadence — 30 minutes of samples.
        ts = base + pd.Timedelta(seconds=i * 30)
        # Last sample is the OOM event (label=1); model predicts P=0.9 from sample 20 on.
        label = 1 if i == 59 else 0
        prob = 0.9 if i >= 20 else 0.1
        rows.append(
            {
                "scrape_timestamp": ts,
                "pod_uid": "u1",
                "container": "c",
                "event_within_30min": label,
                "_prob": prob,
            }
        )
    df = pd.DataFrame(rows)
    # The 5-min-ahead horizon: lead time should be (29:30 - 10:00) / 60 = ~19.5 min.
    metrics = compute_lead_time_metrics(df, df["_prob"].to_numpy(), threshold=0.7)
    # detection_rate=1.0 means every pod with OOM was alerted at the horizon.
    assert metrics["detection_rate_at_5min"] == 1.0
    assert metrics["lead_time_p50_min_at_5min"] > 0


def test_compute_slice_metrics_drops_tiny_slices() -> None:
    base = pd.Timestamp("2026-05-18T00:00:00", tz="UTC")
    rows = []
    # Big slice — 200 rows in namespace "big" with some positives.
    for i in range(200):
        rows.append(
            {
                "scrape_timestamp": base + pd.Timedelta(seconds=i),
                "pod_uid": f"u{i}",
                "container": "c",
                "namespace": "big",
                "node": "n1",
                "event_within_30min": 1 if i % 10 == 0 else 0,
            }
        )
    # Tiny slice — 5 rows in namespace "tiny".
    for i in range(5):
        rows.append(
            {
                "scrape_timestamp": base + pd.Timedelta(seconds=200 + i),
                "pod_uid": f"t{i}",
                "container": "c",
                "namespace": "tiny",
                "node": "n2",
                "event_within_30min": 0,
            }
        )
    df = pd.DataFrame(rows)
    probs = np.linspace(0.1, 0.9, len(df))
    slices = compute_slice_metrics(df, probs, slice_cols=("namespace",), min_rows_per_slice=100)
    namespaces = {s["slice_value"] for s in slices}
    assert "big" in namespaces
    assert "tiny" not in namespaces  # below min_rows_per_slice
