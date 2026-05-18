"""Probability calibration wrappers — Platt and isotonic.

The action layer thresholds at `P > 0.7` (per docs/PLAN.md). Calibration matters:
LightGBM's raw probabilities for a class imbalance of ~0.1 % positives are
typically over-confident at the high end. We fit a calibrator on the val set
and apply it to test predictions.

Per Phase 1 research (`docs/PHASE_1_CONTEXT.md` § 5), negative downsampling at
training time is followed by a recalibration step that maps back to the
ORIGINAL prior — the calibrator absorbs both effects: the model's own
miscalibration AND the prior shift.

Platt scaling (`sklearn.linear_model.LogisticRegression` on raw logits) is
faster and dominates isotonic at small positive class sizes — fit one
parameter (the slope) on val, apply to test. Isotonic adapts better when the
model has structural miscalibration but needs more positives to fit reliably.
We default to isotonic but the `Method` literal lets the caller choose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

Method = Literal["platt", "isotonic"]


class CalibrationError(RuntimeError):
    """Raised when calibration inputs violate the contract."""


@dataclass
class Calibrator:
    """Sklearn-flavoured wrapper that fits on (raw_probs, y_true) then transforms.

    Use `Calibrator.fit(...)` to construct from data. The fitted instance is
    pickle-able and small (a single sklearn estimator).
    """

    method: Method
    _estimator: LogisticRegression | IsotonicRegression

    @classmethod
    def fit(
        cls,
        raw_probs: npt.NDArray[np.floating[Any]],
        y_true: npt.NDArray[Any],
        *,
        method: Method = "isotonic",
    ) -> Calibrator:
        """Fit a calibrator on val predictions.

        `raw_probs` shape: `(n,)` — the model's predicted P(class=1).
        `y_true`   shape: `(n,)` — 0/1 labels.
        """
        if raw_probs.ndim != 1 or y_true.ndim != 1:
            raise CalibrationError(
                f"expected 1-D arrays, got raw={raw_probs.shape}, y={y_true.shape}"
            )
        if raw_probs.shape != y_true.shape:
            raise CalibrationError(
                f"shape mismatch: raw={raw_probs.shape}, y={y_true.shape}"
            )
        if not np.all((raw_probs >= 0) & (raw_probs <= 1)):
            raise CalibrationError("raw_probs must be in [0, 1]")
        if y_true.sum() == 0:
            raise CalibrationError("calibration set has no positives")

        if method == "platt":
            est = LogisticRegression(C=1e6, solver="lbfgs")
            est.fit(raw_probs.reshape(-1, 1), y_true.astype("int"))
            return cls(method=method, _estimator=est)

        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw_probs, y_true.astype("float64"))
            return cls(method=method, _estimator=iso)

        raise CalibrationError(f"unknown calibration method: {method!r}")

    def transform(self, raw_probs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply the calibrator. Returns `(n,)` calibrated probabilities."""
        if raw_probs.ndim != 1:
            raise CalibrationError(f"expected 1-D raw_probs, got {raw_probs.shape}")

        if self.method == "platt":
            est = self._estimator
            assert isinstance(est, LogisticRegression)
            return np.asarray(
                est.predict_proba(raw_probs.reshape(-1, 1))[:, 1], dtype="float64"
            )

        # Isotonic
        est = self._estimator
        assert isinstance(est, IsotonicRegression)
        out = est.predict(raw_probs)
        return np.clip(np.asarray(out, dtype="float64"), 0.0, 1.0)
