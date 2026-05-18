"""LightGBM trainer with Optuna hyperparameter search.

Search budget: 50 trials of TPE + Hyperband pruning. Optimization target is
**average precision (AUC-PR)** — at ~0.1 % positive rate, ROC-AUC misleads
(the negative pool dominates the FPR) and a constant `p=0` baseline already
scores Brier ≈ 0.001 (useless target).

Imbalance handling: callers downsample negatives via
`cluster_canary.training.imbalance.downsample_negatives` BEFORE calling
`train_lgbm`. The model trains on the balanced subset; predictions on val/test
are mapped back to the production prior via `imbalance.elkan_correct`. Then
isotonic calibration (`models.calibration.Calibrator`) absorbs any residual
miscalibration.

References:
- LightGBM tuning: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
- Optuna TPE: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
- Probability correction after downsampling: Elkan IJCAI 2001 § 3
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import optuna
import structlog
from sklearn.metrics import average_precision_score

# `LightGBMPruningCallback` lives in `optuna-integration[lightgbm]` since
# Optuna 4 split integrations into a separate package (declared in pyproject).
try:
    from optuna.integration import LightGBMPruningCallback
except ImportError as exc:  # pragma: no cover — environment problem, not a code path
    raise ImportError(
        "LightGBMPruningCallback requires `optuna-integration[lightgbm]`. "
        "Install via `uv sync --extra dev` (pyproject already declares it)."
    ) from exc

log = structlog.get_logger(__name__)

DEFAULT_N_TRIALS: int = 50
DEFAULT_MAX_BOOST_ROUNDS: int = 3000
DEFAULT_EARLY_STOPPING_ROUNDS: int = 75
DEFAULT_RANDOM_STATE: int = 42


@dataclass
class TrainedModel:
    """A trained LightGBM model plus the metadata callers need to reproduce it."""

    booster: lgb.Booster
    best_params: dict[str, Any]
    best_iteration: int
    sample_ratio: float  # the downsampling ratio used (1.0 if none)
    feature_names: list[str]

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Raw model probabilities on the training prior. Apply Elkan + calibration upstream."""
        return np.asarray(
            self.booster.predict(X, num_iteration=self.best_iteration),
            dtype="float64",
        )


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Verified search ranges from the LightGBM tuning docs + research notes."""
    return {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        # CRITICAL at ~0.1 % positive — default 20 is too high; lets rare-class splits happen.
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
        # =1 because callers downsample negatives upstream; data is already balanced.
        "scale_pos_weight": 1.0,
        "seed": DEFAULT_RANDOM_STATE,
    }


def _build_objective(
    X_train_ds: npt.NDArray[np.floating[Any]],
    y_train_ds: npt.NDArray[Any],
    X_val: npt.NDArray[np.floating[Any]],
    y_val: npt.NDArray[Any],
    sample_ratio: float,
    feature_names: list[str],
    *,
    max_boost_rounds: int,
    early_stopping_rounds: int,
) -> Callable[[optuna.Trial], float]:
    """Curry an Optuna objective with the train/val arrays + sample ratio bound in."""

    def _objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        dtrain = lgb.Dataset(X_train_ds, label=y_train_ds, feature_name=feature_names)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_names)

        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            LightGBMPruningCallback(trial, "average_precision"),
        ]
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=max_boost_rounds,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=callbacks,
        )

        p_train_prior = model.predict(X_val, num_iteration=model.best_iteration)
        # Elkan correction back to the true val prior before scoring.
        from cluster_canary.training.imbalance import elkan_correct

        p_true_prior = elkan_correct(np.asarray(p_train_prior), sample_ratio=sample_ratio)
        score = float(average_precision_score(y_val, p_true_prior))
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return score

    return _objective


def train_lgbm(
    X_train_ds: npt.NDArray[np.floating[Any]],
    y_train_ds: npt.NDArray[Any],
    X_val: npt.NDArray[np.floating[Any]],
    y_val: npt.NDArray[Any],
    *,
    feature_names: list[str],
    sample_ratio: float,
    n_trials: int = DEFAULT_N_TRIALS,
    max_boost_rounds: int = DEFAULT_MAX_BOOST_ROUNDS,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TrainedModel:
    """End-to-end Optuna search + final fit on best params.

    `X_train_ds` / `y_train_ds` are the DOWNSAMPLED training arrays. Val stays
    at the true prior (no downsampling). `sample_ratio` is the ratio used by
    `downsample_negatives` so the Elkan correction can be applied inside the
    objective.
    """
    if X_train_ds.shape[0] != y_train_ds.shape[0]:
        raise ValueError(
            f"shape mismatch: X_train_ds={X_train_ds.shape}, y_train_ds={y_train_ds.shape}"
        )
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError(f"shape mismatch: X_val={X_val.shape}, y_val={y_val.shape}")
    if X_train_ds.shape[1] != X_val.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match X_train_ds + X_val column counts")

    sampler = optuna.samplers.TPESampler(
        multivariate=True, group=True, n_startup_trials=10, seed=random_state
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=100, max_resource=max_boost_rounds, reduction_factor=3
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    objective = _build_objective(
        X_train_ds,
        y_train_ds,
        X_val,
        y_val,
        sample_ratio,
        feature_names,
        max_boost_rounds=max_boost_rounds,
        early_stopping_rounds=early_stopping_rounds,
    )
    log.info(
        "lgbm.search.start",
        n_trials=n_trials,
        n_train_ds=len(X_train_ds),
        n_val=len(X_val),
        sample_ratio=sample_ratio,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    log.info(
        "lgbm.search.done",
        best_score=study.best_value,
        best_params=best.params,
        n_trials_completed=len(study.trials),
    )

    # Final fit on the best params over the FULL boost-round budget; use early
    # stopping again so we get the best_iteration for inference.
    final_params = _suggest_params_static(best.params)
    dtrain = lgb.Dataset(X_train_ds, label=y_train_ds, feature_name=feature_names)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_names)
    final = lgb.train(
        final_params,
        dtrain,
        num_boost_round=max_boost_rounds,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return TrainedModel(
        booster=final,
        best_params=best.params,
        best_iteration=int(final.best_iteration),
        sample_ratio=sample_ratio,
        feature_names=list(feature_names),
    )


def _suggest_params_static(best_params: dict[str, Any]) -> dict[str, Any]:
    """Compose final-fit params: best Optuna suggestions + the fixed booster settings."""
    return {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "bagging_freq": 5,
        "scale_pos_weight": 1.0,
        "seed": DEFAULT_RANDOM_STATE,
        **best_params,
    }
