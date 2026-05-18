"""Negative downsampling + Elkan probability correction.

At ~0.1 % positive (typical for cluster-canary), LightGBM's histogram-based
split finding sees mostly negatives in every leaf and gradient signals for
positives become diluted. Downsample negatives to make positives 5-10 % of
training data.

Probability correction after downsampling (Elkan, "The Foundations of
Cost-Sensitive Learning", IJCAI 2001 § 3):

    p_true = p_train / (p_train + (1 - p_train) / w)

where `w = sample_ratio = P(kept | negative)`. This maps the model's
training-prior probabilities back to the production prior so downstream
calibration sees uncorrupted inputs.

Important: only the TRAINING set is downsampled. Val and test stay at the
original prior — calibration and threshold-selection must reflect production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


class ImbalanceError(RuntimeError):
    """Raised when downsampling inputs violate the contract."""


@dataclass(frozen=True)
class DownsampleResult:
    """Output of `downsample_negatives` — DataFrame + the sample ratio for Elkan."""

    df: pd.DataFrame
    sample_ratio: float


def downsample_negatives(
    df: pd.DataFrame,
    *,
    label_col: str = "event_within_30min",
    target_positive_rate: float = 0.10,
    random_state: int = 42,
) -> DownsampleResult:
    """Downsample the negative class so positives make up `target_positive_rate` of the output.

    The original temporal ORDER of surviving rows is preserved (we mask then
    take, not shuffle) — important because Phase 3's temporal split has already
    ordered rows by `scrape_timestamp` and downstream training expects that.
    """
    if label_col not in df.columns:
        raise ImbalanceError(f"label_col {label_col!r} not in frame")
    if not 0.0 < target_positive_rate < 1.0:
        raise ImbalanceError(f"target_positive_rate must be in (0, 1); got {target_positive_rate}")
    if df.empty:
        raise ImbalanceError("cannot downsample empty frame")

    n_positive = int(df[label_col].sum())
    n_negative = int(len(df) - n_positive)
    if n_positive == 0:
        raise ImbalanceError("no positives in training frame — cannot downsample")
    if n_negative == 0:
        return DownsampleResult(df=df.copy(), sample_ratio=1.0)

    # n_keep_negative such that n_positive / (n_positive + n_keep_negative) == target_positive_rate
    desired_neg = round(n_positive * (1.0 - target_positive_rate) / target_positive_rate)
    if desired_neg >= n_negative:
        log.warning(
            "downsample.no_action",
            n_positive=n_positive,
            n_negative=n_negative,
            target_positive_rate=target_positive_rate,
            reason="already_above_target",
        )
        return DownsampleResult(df=df.copy(), sample_ratio=1.0)

    sample_ratio = desired_neg / n_negative
    rng = np.random.default_rng(random_state)
    neg_mask = df[label_col] == 0
    neg_keep = rng.choice(neg_mask.sum(), size=desired_neg, replace=False)
    neg_indices = df.index[neg_mask].to_numpy()
    keep_negs = pd.Index(neg_indices[np.sort(neg_keep)])

    keep_mask = (df[label_col] == 1) | df.index.isin(keep_negs)
    out = df.loc[keep_mask].copy()

    log.info(
        "downsample.done",
        n_in=len(df),
        n_out=len(out),
        n_positive=n_positive,
        n_negative_in=n_negative,
        n_negative_kept=desired_neg,
        sample_ratio=round(sample_ratio, 6),
        out_positive_rate=round(n_positive / len(out), 4) if len(out) else 0.0,
    )
    return DownsampleResult(df=out, sample_ratio=sample_ratio)


def elkan_correct(
    p_train: npt.NDArray[np.float64], *, sample_ratio: float
) -> npt.NDArray[np.float64]:
    """Map a downsampled-trained model's probabilities back to the original prior.

    `sample_ratio = P(kept | negative)`. Per Elkan 2001 § 3:

        p_true = p_train / (p_train + (1 - p_train) / w)

    Idempotent at `sample_ratio == 1.0` (no downsampling) — returns p_train.
    Clips to [eps, 1 - eps] for numerical stability.
    """
    if not 0.0 < sample_ratio <= 1.0:
        raise ImbalanceError(f"sample_ratio must be in (0, 1]; got {sample_ratio}")
    if sample_ratio == 1.0:
        return np.asarray(p_train, dtype="float64")
    eps = 1e-12
    p = np.clip(np.asarray(p_train, dtype="float64"), eps, 1.0 - eps)
    return np.asarray(p / (p + (1.0 - p) / sample_ratio), dtype="float64")
