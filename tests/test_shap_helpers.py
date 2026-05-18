"""Tests for the SHAP top-k helpers.

Trains a tiny LightGBM model on synthetic data and asserts:
- top-3 extraction works
- JSON output schema matches the contract
- batch top-k matches per-row top-k
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

# LightGBM on macOS Apple Silicon needs `brew install libomp` to load the native
# `lib_lightgbm.dylib`. The wheel imports cleanly but the dlopen raises OSError,
# which `pytest.importorskip` does NOT catch. Wrap manually.
try:
    import lightgbm as lgb
except (ImportError, OSError) as exc:
    pytest.skip(f"lightgbm not loadable: {exc}", allow_module_level=True)

from cluster_canary.training.shap_helpers import (
    batch_top_k_features,
    predict_with_explanation,
)


@pytest.fixture(scope="module")
def tiny_booster() -> tuple[lgb.Booster, list[str]]:
    """Train a deterministic 5-feature LightGBM model on 500 rows."""
    rng = np.random.default_rng(0)
    n = 500
    X = rng.standard_normal((n, 5))
    feat_names = [f"feat_{i}" for i in range(5)]
    # y is correlated with X[:, 0] strongly, X[:, 2] weakly.
    logits = 1.5 * X[:, 0] + 0.3 * X[:, 2]
    y = (rng.uniform(0, 1, n) < (1.0 / (1.0 + np.exp(-logits)))).astype(int)
    dset = lgb.Dataset(X, label=y, feature_name=feat_names)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_leaves": 7, "seed": 0},
        dset,
        num_boost_round=20,
    )
    return booster, feat_names


def test_predict_with_explanation_schema(tiny_booster) -> None:  # type: ignore[no-untyped-def]
    booster, feat_names = tiny_booster
    x_row = np.array([[1.0, 0.0, -0.5, 0.2, 0.1]])
    result = predict_with_explanation(booster, x_row, feature_names=feat_names, k=3)

    # Required top-level keys.
    for key in (
        "prediction",
        "prediction_logodds",
        "base_value",
        "model_version",
        "top_contributors",
    ):
        assert key in result

    # Probabilities in [0, 1].
    assert 0.0 <= result["prediction"] <= 1.0

    # Exactly 3 contributors with the documented shape.
    assert len(result["top_contributors"]) == 3
    for entry in result["top_contributors"]:
        assert set(entry) >= {
            "feature",
            "contribution_logodds",
            "contribution_prob_delta",
            "direction",
            "rank",
        }
        assert entry["direction"] in {"up", "down"}
        assert entry["feature"] in feat_names


def test_top_contributor_ranks_are_1_indexed_and_sorted(tiny_booster) -> None:  # type: ignore[no-untyped-def]
    booster, feat_names = tiny_booster
    x_row = np.array([[2.0, -1.0, 0.5, 0.1, -0.3]])
    result = predict_with_explanation(booster, x_row, feature_names=feat_names, k=3)
    ranks = [c["rank"] for c in result["top_contributors"]]
    assert ranks == [1, 2, 3]
    # Absolute log-odds contributions are non-increasing across ranks.
    abs_contribs = [abs(c["contribution_logodds"]) for c in result["top_contributors"]]
    assert all(a >= b for a, b in itertools.pairwise(abs_contribs))


def test_top_feature_for_strong_signal_is_correct(tiny_booster) -> None:  # type: ignore[no-untyped-def]
    booster, feat_names = tiny_booster
    # A row where feat_0 is extreme should make feat_0 the top contributor.
    x_row = np.array([[3.0, 0.0, 0.0, 0.0, 0.0]])
    result = predict_with_explanation(booster, x_row, feature_names=feat_names, k=3)
    assert result["top_contributors"][0]["feature"] == "feat_0"


def test_batch_top_k_matches_per_row(tiny_booster) -> None:  # type: ignore[no-untyped-def]
    booster, feat_names = tiny_booster
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 5))
    batch = batch_top_k_features(booster, X, feature_names=feat_names, k=3)
    assert len(batch) == 10
    for i in range(10):
        per_row = predict_with_explanation(booster, X[i : i + 1], feature_names=feat_names, k=3)
        per_row_names = [c["feature"] for c in per_row["top_contributors"]]
        batch_names = [name for name, _ in batch[i]]
        assert per_row_names == batch_names


def test_predict_with_explanation_rejects_multi_row(tiny_booster) -> None:  # type: ignore[no-untyped-def]
    booster, feat_names = tiny_booster
    with pytest.raises(ValueError, match="exactly one row"):
        predict_with_explanation(booster, np.zeros((2, 5)), feature_names=feat_names, k=3)
