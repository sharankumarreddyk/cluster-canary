"""DVC entrypoint for the Phase 4 training stage."""

from __future__ import annotations

import os
from pathlib import Path

from cluster_canary.models.lgbm import DEFAULT_N_TRIALS
from cluster_canary.training.eval import DEFAULT_THRESHOLD
from cluster_canary.training.train import (
    DEFAULT_MODELS_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_REPORTS_DIR,
    run_pipeline,
)


def main() -> int:
    """Run the Phase 4 train pipeline end-to-end."""
    import structlog

    from cluster_canary.training.train import _configure_logging

    _configure_logging()
    log = structlog.get_logger(__name__)

    out = run_pipeline(
        processed_dir=Path(os.environ.get("PROCESSED_DIR", str(DEFAULT_PROCESSED_DIR))),
        models_dir=Path(os.environ.get("MODELS_DIR", str(DEFAULT_MODELS_DIR))),
        reports_dir=Path(os.environ.get("REPORTS_DIR", str(DEFAULT_REPORTS_DIR))),
        n_trials=int(os.environ.get("OPTUNA_N_TRIALS", str(DEFAULT_N_TRIALS))),
        threshold=float(os.environ.get("DECISION_THRESHOLD", str(DEFAULT_THRESHOLD))),
        target_positive_rate=float(os.environ.get("TARGET_POSITIVE_RATE", "0.10")),
    )
    log.info(
        "train.pipeline.done",
        baseline_run_id=out["baseline"]["run_id"],
        lgbm_run_id=out["lgbm"]["run_id"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
