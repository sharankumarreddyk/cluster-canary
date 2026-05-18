"""Top-k SHAP contributors per prediction — for the inference path's explainability.

Uses LightGBM's native `booster.predict(X, pred_contrib=True)` rather than the
`shap` package. Same TreeSHAP algorithm, but:
- Faster (no Python explainer object overhead — direct C++ call).
- Stable API (pred_contrib has been stable since LightGBM ~2017; `shap` has
  had breaking changes between minor versions).
- No extra dependency.

Returns log-odds contributions (LightGBM's `binary` objective is log-odds);
the inference response should label them explicitly so consumers don't
misinterpret as probability-point changes. The probability-delta is also
computed and surfaced separately — see `predict_with_explanation`.

References:
- TreeSHAP: Lundberg et al., Nature Machine Intelligence 2020
- LightGBM pred_contrib docs:
    https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.Booster.predict
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt


def _sigmoid(x: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64] | float:
    return 1.0 / (1.0 + np.exp(-x))


def predict_with_explanation(
    booster: lgb.Booster,
    x_row: npt.NDArray[np.float64],
    *,
    feature_names: list[str] | None = None,
    k: int = 3,
    model_version: str = "unversioned",
) -> dict[str, Any]:
    """Predict + attach top-k SHAP contributors for a single instance.

    `x_row` shape: `(1, n_features)` or `(n_features,)` (we reshape).
    Returns the JSON-serialisable schema documented in
    `docs/phase4_08_shap.md` (the research file).
    """
    if x_row.ndim == 1:
        x_row = x_row.reshape(1, -1)
    if x_row.shape[0] != 1:
        raise ValueError(
            f"predict_with_explanation: x_row must contain exactly one row, got shape {x_row.shape}"
        )

    names = feature_names if feature_names is not None else list(booster.feature_name())
    if len(names) != x_row.shape[1]:
        raise ValueError(
            f"feature_names length {len(names)} != x_row column count {x_row.shape[1]}"
        )

    contribs = np.asarray(booster.predict(x_row, pred_contrib=True), dtype="float64")[0]
    base_value = float(contribs[-1])
    feat_contribs = contribs[:-1]

    pred_logodds = float(base_value + feat_contribs.sum())
    pred_prob = float(_sigmoid(pred_logodds))

    top_idx = np.argsort(-np.abs(feat_contribs))[:k]
    top: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_idx, start=1):
        contrib_lo = float(feat_contribs[idx])
        # Marginal effect of this feature on probability — approximate (sigmoid is non-linear,
        # so contributions don't compose additively in probability space).
        prob_without = float(_sigmoid(pred_logodds - contrib_lo))
        delta_prob = pred_prob - prob_without
        top.append(
            {
                "feature": names[int(idx)],
                "contribution_logodds": contrib_lo,
                "contribution_prob_delta": delta_prob,
                "direction": "up" if contrib_lo > 0 else "down",
                "rank": rank,
            }
        )

    return {
        "prediction": pred_prob,
        "prediction_logodds": pred_logodds,
        "base_value": base_value,
        "model_version": model_version,
        "top_contributors": top,
    }


def batch_top_k_features(
    booster: lgb.Booster,
    X: npt.NDArray[np.float64],
    *,
    feature_names: list[str] | None = None,
    k: int = 3,
) -> list[list[tuple[str, float]]]:
    """Top-k contributors for each row of `X`. Faster than per-row in a loop.

    Returns a list of length `n_rows`, each entry a list of `(feature_name, log_odds_contrib)`
    sorted by descending |contribution|.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    names = feature_names if feature_names is not None else list(booster.feature_name())
    if len(names) != X.shape[1]:
        raise ValueError(f"feature_names length {len(names)} != X column count {X.shape[1]}")

    contribs = np.asarray(booster.predict(X, pred_contrib=True), dtype="float64")
    feat_contribs = contribs[:, :-1]
    top_idx = np.argsort(-np.abs(feat_contribs), axis=1)[:, :k]

    out: list[list[tuple[str, float]]] = []
    for row_i, idx_row in enumerate(top_idx):
        out.append([(names[int(j)], float(feat_contribs[row_i, j])) for j in idx_row])
    return out
