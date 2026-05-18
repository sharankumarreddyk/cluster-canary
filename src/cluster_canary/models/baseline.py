"""Rule-based baseline for OOM-in-next-30-min prediction.

The rule is the simplest credible thing many real teams ship and never test
against ML: a pod is flagged if its memory utilization has stayed above a
threshold for at least N consecutive samples (where N approximates a sustained
window). This is the "is the model better than the rule?" reference.

For the cluster-canary feature contract, the rule reads `feat_mem_pct_of_limit`
directly. To make the comparison fair vs the trained classifier (which sees
rolling features), we use `feat_mem_pct_of_limit__5min_max` as the input — both
the rule and the model "look back 5 min". Threshold defaults to 0.90.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

# The feature column the rule reads. Falls back to `feat_mem_pct_of_limit`
# if the rolling-max isn't present (e.g. in unit tests with a tiny fixture).
PRIMARY_INPUT_COL: str = "feat_mem_pct_of_limit__5min_max"
FALLBACK_INPUT_COL: str = "feat_mem_pct_of_limit"

DEFAULT_THRESHOLD: float = 0.90


@dataclass(frozen=True)
class BaselineRule:
    """Sustained-memory threshold rule. Single hyperparameter: the threshold."""

    threshold: float = DEFAULT_THRESHOLD

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return P(OOM in next 30 min) ∈ {0.0, 1.0}.

        Returns a 1-D array aligned to `X.index`. The probability is binary —
        the rule has no calibration. For downstream code that expects a 2-col
        array `(P[0], P[1])` use `predict_proba_2d`.
        """
        col = self._input_col(X)
        return (X[col].to_numpy(dtype="float64") >= self.threshold).astype("float64")

    def predict_proba_2d(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Sklearn-style `[[P(0), P(1)], …]` shape for compatibility with eval harness."""
        p1 = self.predict_proba(X)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.int8]:
        """Hard 0/1 predictions."""
        return (self.predict_proba(X) >= 0.5).astype("int8")

    def _input_col(self, X: pd.DataFrame) -> str:
        if PRIMARY_INPUT_COL in X.columns:
            return PRIMARY_INPUT_COL
        if FALLBACK_INPUT_COL in X.columns:
            return FALLBACK_INPUT_COL
        raise KeyError(
            f"BaselineRule: neither {PRIMARY_INPUT_COL!r} nor {FALLBACK_INPUT_COL!r} in X"
        )
