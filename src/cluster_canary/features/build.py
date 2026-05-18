"""Phase 3 orchestrator: labeled wide frame → features → train/val/test parquet.

Reads the wide labeled parquet emitted by Phase 1 (synthetic) and/or Phase 2
(Alibaba), runs every feature family in canonical order, applies the temporal
split, and writes:

    data/processed/
      ├── train.parquet
      ├── val.parquet
      ├── test.parquet
      ├── feature_list.json
      └── entity_overlap.json

Plus a metrics summary at `reports/eval/feature_metrics.json` and
`reports/eval/split_metrics.json`.

The wide input must conform to the contract emitted by `generate_labels`:
columns `(scrape_timestamp, pod_uid, container, namespace, pod, node)` +
metric columns + `event_within_30min` label.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import structlog

from cluster_canary.features.aggregations import add_aggregation_features
from cluster_canary.features.cpu import add_cpu_features
from cluster_canary.features.lifecycle import add_lifecycle_features
from cluster_canary.features.memory import add_memory_features
from cluster_canary.features.split import (
    DEFAULT_ENTITY_OVERLAP_ALLOWED,
    DEFAULT_TRAIN_FRAC,
    DEFAULT_VAL_FRAC,
    temporal_split,
    write_entity_overlap_opt_in,
)
from cluster_canary.features.temporal import add_temporal_features

log = structlog.get_logger(__name__)

DEFAULT_LABELED_PATHS: tuple[Path, ...] = (
    Path("data/interim/labeled/labeled.parquet"),  # Phase 1 synthetic
    Path("data/interim/alibaba_labeled/labeled.parquet"),  # Phase 2 Alibaba
)
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_REPORTS_DIR = Path("reports/eval")

# The canonical, ordered list of feature columns the model trains on. Anything
# starting with `feat_` produced by an `add_*_features` step is included. The
# build records the discovered list to `feature_list.json` for the leakage
# audit's deny-list check.
FEATURE_PREFIX: str = "feat_"

# Identity + label columns that must travel with features through the split.
# Anything else (raw metric columns) is dropped from the model-input parquet
# but preserved in `data/interim/` for debugging.
IDENTITY_COLS: tuple[str, ...] = (
    "scrape_timestamp",
    "namespace",
    "pod",
    "pod_uid",
    "container",
    "node",
)
LABEL_COL: str = "event_within_30min"


# Public for the package __init__ re-export.
FEATURE_COLUMNS: tuple[str, ...] = ()  # populated at runtime; documented here for type-hints


class FeatureBuildError(RuntimeError):
    """Raised when build inputs / outputs violate the contract."""


def _read_inputs(paths: Iterable[Path]) -> pd.DataFrame:
    """Concatenate every existing labeled parquet into one wide frame."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            log.info("build.input.skip", path=str(path), reason="missing")
            continue
        df = pd.read_parquet(path)
        log.info("build.input.read", path=str(path), n_rows=len(df))
        frames.append(df)
    if not frames:
        raise FeatureBuildError(
            f"no labeled inputs found in {[str(p) for p in paths]}; run Phase 1 + Phase 2 first"
        )
    return pd.concat(frames, ignore_index=True)


def _all_feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column added by an `add_*_features` step (anything `feat_*`-prefixed)."""
    return sorted(c for c in df.columns if c.startswith(FEATURE_PREFIX))


def build_features(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature family in canonical order. Returns the wide+features frame."""
    if wide_df.empty:
        raise FeatureBuildError("build_features: empty input frame")
    if LABEL_COL not in wide_df.columns:
        raise FeatureBuildError(f"build_features: required label {LABEL_COL!r} not in frame")

    log.info("build.features.start", n_rows=len(wide_df), n_cols=len(wide_df.columns))
    out = wide_df
    out = add_temporal_features(out)
    out = add_memory_features(out)
    out = add_cpu_features(out)
    out = add_lifecycle_features(out)
    out = add_aggregation_features(out)
    feat_cols = _all_feature_columns(out)
    log.info("build.features.done", n_rows=len(out), n_feat_cols=len(feat_cols))
    return out


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Slice to identity + features + label. Drops raw metric columns."""
    keep = list(IDENTITY_COLS) + _all_feature_columns(df) + [LABEL_COL]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise FeatureBuildError(f"select_model_columns: missing {missing}")
    return df.loc[:, keep].copy()


def _write_feature_list(feature_list: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_list.json"
    path.write_text(json.dumps(feature_list, indent=2))
    return path


def _write_metrics(*, feature_list: list[str], n_in: int, n_out: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_metrics.json"
    path.write_text(
        json.dumps(
            {"n_input_rows": n_in, "n_output_rows": n_out, "n_features": len(feature_list)},
            indent=2,
        )
    )
    return path


def run_pipeline(
    labeled_paths: Iterable[Path] = DEFAULT_LABELED_PATHS,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    *,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> dict[str, Path]:
    """End-to-end Phase 3 pipeline. Returns `{name: path}` of outputs written."""
    wide = _read_inputs(labeled_paths)
    n_in = len(wide)

    featured = build_features(wide)
    model_df = select_model_columns(featured)
    feat_cols = _all_feature_columns(model_df)

    train, val, test, split_metrics = temporal_split(
        model_df, train_frac=train_frac, val_frac=val_frac
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "train": processed_dir / "train.parquet",
        "val": processed_dir / "val.parquet",
        "test": processed_dir / "test.parquet",
    }
    train.to_parquet(out["train"], index=False)
    val.to_parquet(out["val"], index=False)
    test.to_parquet(out["test"], index=False)

    out["feature_list"] = _write_feature_list(feat_cols, processed_dir)
    out["entity_overlap"] = write_entity_overlap_opt_in(
        processed_dir, allowed=DEFAULT_ENTITY_OVERLAP_ALLOWED
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    out["feature_metrics"] = _write_metrics(
        feature_list=feat_cols, n_in=n_in, n_out=len(model_df), out_dir=reports_dir
    )
    (reports_dir / "split_metrics.json").write_text(json.dumps(asdict(split_metrics), indent=2))
    out["split_metrics"] = reports_dir / "split_metrics.json"

    log.info(
        "build.pipeline.done",
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        n_features=len(feat_cols),
    )
    return out


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cluster_canary.features.build", description=__doc__)
    p.add_argument(
        "--labeled-paths",
        nargs="+",
        type=Path,
        default=list(DEFAULT_LABELED_PATHS),
        help="Input labeled parquets (Phase 1 synthetic, Phase 2 Alibaba, …).",
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(os.environ.get("FEATURES_PROCESSED_DIR", str(DEFAULT_PROCESSED_DIR))),
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(os.environ.get("FEATURES_REPORTS_DIR", str(DEFAULT_REPORTS_DIR))),
    )
    p.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint — parse args and run the Phase 3 pipeline."""
    _configure_logging()
    args = _build_parser().parse_args(argv)
    out = run_pipeline(
        labeled_paths=args.labeled_paths,
        processed_dir=args.processed_dir,
        reports_dir=args.reports_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    log.info("build.complete", outputs={k: str(v) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
