"""DVC entrypoint for the Phase 3 features stage.

Env-driven so DVC parametrization works:
- FEATURES_PROCESSED_DIR    (default: data/processed)
- FEATURES_REPORTS_DIR      (default: reports/eval)
- TRAIN_FRAC                (default: 0.60)
- VAL_FRAC                  (default: 0.20)
"""

from __future__ import annotations

import os
from pathlib import Path

from cluster_canary.features.build import (
    DEFAULT_LABELED_PATHS,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_REPORTS_DIR,
    run_pipeline,
)
from cluster_canary.features.split import (
    DEFAULT_TRAIN_FRAC,
    DEFAULT_VAL_FRAC,
)


def main() -> int:
    """Run the Phase 3 feature engineering pipeline end-to-end."""
    import structlog

    from cluster_canary.features.build import _configure_logging

    _configure_logging()
    log = structlog.get_logger(__name__)

    processed_dir = Path(os.environ.get("FEATURES_PROCESSED_DIR", str(DEFAULT_PROCESSED_DIR)))
    reports_dir = Path(os.environ.get("FEATURES_REPORTS_DIR", str(DEFAULT_REPORTS_DIR)))
    train_frac = float(os.environ.get("TRAIN_FRAC", str(DEFAULT_TRAIN_FRAC)))
    val_frac = float(os.environ.get("VAL_FRAC", str(DEFAULT_VAL_FRAC)))

    out = run_pipeline(
        labeled_paths=DEFAULT_LABELED_PATHS,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    log.info("features.pipeline.done", outputs={k: str(v) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
