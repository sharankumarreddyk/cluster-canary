"""Phase 4 train orchestrator.

Reads `data/processed/{train,val,test}.parquet`, trains both the baseline rule
AND a LightGBM with Optuna search, applies Elkan probability correction +
isotonic calibration, evaluates everything per the PLAN.md gates, and writes:

- `models/canary_model.txt`        — final LightGBM booster
- `models/calibrator.json`         — fitted isotonic calibrator state
- `models/MODEL_CARD.md`           — generated card (`model_card.render_model_card`)
- `reports/eval/test_metrics.json` — headline + lead-time metrics on test
- `reports/eval/slice_metrics.json` — sliced metrics on test

All of the above are MLflow-tracked per `mlflow_run.start_tracked_run`
(experiment `cluster_canary__oom_30min`, tagged contract enforced).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog

from cluster_canary.models.baseline import BaselineRule
from cluster_canary.models.calibration import Calibrator
from cluster_canary.models.lgbm import (
    DEFAULT_N_TRIALS,
    TrainedModel,
    train_lgbm,
)
from cluster_canary.training.eval import (
    DEFAULT_THRESHOLD,
    assert_phase4_acceptance,
    compute_headline_metrics,
    compute_lead_time_metrics,
    compute_slice_metrics,
)
from cluster_canary.training.imbalance import (
    DownsampleResult,
    downsample_negatives,
    elkan_correct,
)
from cluster_canary.training.mlflow_run import (
    DEFAULT_EXPERIMENT_NAME,
    start_tracked_run,
)
from cluster_canary.training.model_card import render_model_card

log = structlog.get_logger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_REPORTS_DIR = Path("reports/eval")
LABEL_COL = "event_within_30min"
IDENTITY_COLS: tuple[str, ...] = (
    "scrape_timestamp",
    "namespace",
    "pod",
    "pod_uid",
    "container",
    "node",
)


class TrainError(RuntimeError):
    """Raised when training inputs / outputs violate the contract."""


def _split_features_and_target(
    df: pd.DataFrame,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8], list[str]]:
    feat_cols = sorted(c for c in df.columns if c.startswith("feat_"))
    if not feat_cols:
        raise TrainError("no `feat_*` columns in input frame — Phase 3 not run?")
    if LABEL_COL not in df.columns:
        raise TrainError(f"label column {LABEL_COL!r} missing")
    X = df[feat_cols].to_numpy(dtype="float64")
    y = df[LABEL_COL].to_numpy(dtype="int8")
    return X, y, feat_cols


def _data_window(df: pd.DataFrame) -> tuple[str, str]:
    if "scrape_timestamp" not in df.columns or df.empty:
        return ("unknown", "unknown")
    return (str(df["scrape_timestamp"].min()), str(df["scrape_timestamp"].max()))


def _save_calibrator(calibrator: Calibrator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(calibrator))


def _maybe_register_baseline(
    _train_df: pd.DataFrame,  # accepted for symmetry; baseline rule has nothing to learn
    test_df: pd.DataFrame,
    *,
    dataset_version: str,
    data_window: tuple[str, str],
    threshold: float,
    reports_dir: Path,
) -> dict[str, Any]:
    """Train + evaluate the rule baseline as its own MLflow run."""
    rule = BaselineRule(threshold=0.90)
    p_test = rule.predict_proba(test_df)
    y_test = test_df[LABEL_COL].to_numpy(dtype="int8")
    metrics = compute_headline_metrics(y_test, p_test, threshold=threshold)
    lead = compute_lead_time_metrics(test_df, p_test, threshold=threshold)
    log.info("baseline.headline", **asdict(metrics))

    with start_tracked_run(
        model_family="rule_baseline",
        dataset_version=dataset_version,
        data_window=data_window,
        stage="experiment",
    ) as run:
        mlflow.log_params(
            {"threshold": rule.threshold, "rule": "sustained_mem_pct_of_limit_5min_max"}
        )
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in asdict(metrics).items() if isinstance(v, (int, float))}
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in lead.items()})
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "baseline_test_metrics.json").write_text(
            json.dumps({**asdict(metrics), **lead}, indent=2)
        )
        mlflow.log_artifact(str(reports_dir / "baseline_test_metrics.json"))
        return {"run_id": run.info.run_id, "metrics": asdict(metrics), "lead_time": lead}


def _train_and_register_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    dataset_version: str,
    data_window: tuple[str, str],
    n_trials: int,
    threshold: float,
    target_positive_rate: float,
    models_dir: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    """Full LightGBM training run: downsample -> Optuna -> Elkan -> calibrate -> eval -> card."""
    _, _, feat_cols = _split_features_and_target(train_df)
    X_val, y_val, val_feat_cols = _split_features_and_target(val_df)
    X_test, y_test, test_feat_cols = _split_features_and_target(test_df)
    if feat_cols != val_feat_cols or feat_cols != test_feat_cols:
        raise TrainError("feature columns differ across train/val/test splits")

    ds: DownsampleResult = downsample_negatives(
        train_df, label_col=LABEL_COL, target_positive_rate=target_positive_rate
    )
    X_train_ds, y_train_ds, _ = _split_features_and_target(ds.df)

    trained: TrainedModel = train_lgbm(
        X_train_ds,
        y_train_ds,
        X_val,
        y_val,
        feature_names=feat_cols,
        sample_ratio=ds.sample_ratio,
        n_trials=n_trials,
    )

    raw_val = trained.predict(X_val)
    corrected_val = elkan_correct(raw_val, sample_ratio=ds.sample_ratio)
    calibrator = Calibrator.fit(corrected_val, y_val, method="isotonic")

    raw_test = trained.predict(X_test)
    corrected_test = elkan_correct(raw_test, sample_ratio=ds.sample_ratio)
    p_test = calibrator.transform(corrected_test)

    headline_test = compute_headline_metrics(y_test, p_test, threshold=threshold)
    lead = compute_lead_time_metrics(test_df, p_test, threshold=threshold)
    slices = compute_slice_metrics(
        test_df, p_test, slice_cols=("namespace", "node"), threshold=threshold
    )
    acceptance = assert_phase4_acceptance(headline_test)
    log.info("lgbm.headline", **asdict(headline_test), acceptance=acceptance)

    with start_tracked_run(
        model_family="lgbm",
        dataset_version=dataset_version,
        data_window=data_window,
        stage="candidate" if all(acceptance.values()) else "experiment",
    ) as run:
        mlflow.log_params(trained.best_params)
        mlflow.log_param("sample_ratio", ds.sample_ratio)
        mlflow.log_param("target_positive_rate", target_positive_rate)
        mlflow.log_param("calibrator_method", calibrator.method)
        mlflow.log_param("n_optuna_trials", n_trials)
        mlflow.log_param("feature_count", len(feat_cols))
        mlflow.log_param("threshold", threshold)

        mlflow.log_metrics(
            {
                f"test_{k}": v
                for k, v in asdict(headline_test).items()
                if isinstance(v, (int, float))
            }
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in lead.items()})
        mlflow.log_metrics({f"acceptance_{k}": int(v) for k, v in acceptance.items()})

        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / "canary_model.txt"
        trained.booster.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))

        calibrator_path = models_dir / "calibrator.pkl"
        _save_calibrator(calibrator, calibrator_path)
        mlflow.log_artifact(str(calibrator_path))

        test_metrics_path = reports_dir / "test_metrics.json"
        test_metrics_path.write_text(
            json.dumps({**asdict(headline_test), **lead, "acceptance": acceptance}, indent=2)
        )
        mlflow.log_artifact(str(test_metrics_path))

        slice_path = reports_dir / "slice_metrics.json"
        slice_path.write_text(json.dumps(slices, indent=2))
        mlflow.log_artifact(str(slice_path))

        feat_list_path = models_dir / "feature_list.json"
        feat_list_path.write_text(json.dumps(feat_cols, indent=2))
        mlflow.log_artifact(str(feat_list_path))

        card_path = render_model_card(
            model_name="canary-lgbm",
            version="v0.1.0",
            git_sha=run.data.tags.get("git_sha", "nogit"),
            mlflow_run_id=run.info.run_id,
            dataset_version=dataset_version,
            data_window=data_window,
            feature_count=len(feat_cols),
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            headline_test=headline_test,
            lead_time=lead,
            slice_metrics=slices,
            sample_ratio=ds.sample_ratio,
            best_params=trained.best_params,
            out_path=models_dir / "MODEL_CARD.md",
        )
        mlflow.log_artifact(str(card_path))

        return {
            "run_id": run.info.run_id,
            "metrics": asdict(headline_test),
            "lead_time": lead,
            "acceptance": acceptance,
            "best_params": trained.best_params,
            "sample_ratio": ds.sample_ratio,
        }


def run_pipeline(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    threshold: float = DEFAULT_THRESHOLD,
    target_positive_rate: float = 0.10,
) -> dict[str, Any]:
    """End-to-end Phase 4 pipeline. Returns the LightGBM + baseline summary."""
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    log.info(
        "train.read",
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
    )

    feature_list_path = processed_dir / "feature_list.json"
    if feature_list_path.exists():
        feature_list = json.loads(feature_list_path.read_text())
        log.info("train.feature_list.loaded", n_features=len(feature_list))

    dataset_version = os.environ.get("DATASET_VERSION", "unset")
    data_window = _data_window(train_df)

    baseline_result = _maybe_register_baseline(
        train_df,
        test_df,
        dataset_version=dataset_version,
        data_window=data_window,
        threshold=threshold,
        reports_dir=reports_dir,
    )
    lgbm_result = _train_and_register_lgbm(
        train_df,
        val_df,
        test_df,
        dataset_version=dataset_version,
        data_window=data_window,
        n_trials=n_trials,
        threshold=threshold,
        target_positive_rate=target_positive_rate,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )
    return {"baseline": baseline_result, "lgbm": lgbm_result}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cluster_canary.training.train", description=__doc__)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--target-positive-rate", type=float, default=0.10)
    return p


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint — parse args and run Phase 4 pipeline."""
    _configure_logging()
    args = _build_parser().parse_args(argv)
    out = run_pipeline(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        n_trials=args.n_trials,
        threshold=args.threshold,
        target_positive_rate=args.target_positive_rate,
    )
    log.info(
        "train.complete",
        baseline_run_id=out["baseline"]["run_id"],
        lgbm_run_id=out["lgbm"]["run_id"],
        acceptance=out["lgbm"]["acceptance"],
        experiment=os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
