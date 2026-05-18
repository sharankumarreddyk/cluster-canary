"""MLflow tracking-conventions enforcement.

Wraps `mlflow.start_run` so every cluster-canary training run satisfies the
contract in `.claude/skills/mlflow-tracking-conventions/SKILL.md`:

- Experiment name `cluster_canary__<task>` set via env or arg.
- Run name `<model>__<dataset>__<sha[:7]>__<ts>`.
- Required tags: `git_sha`, `git_dirty`, `git_branch`, `dataset_version`,
  `chaos_plan_version`, `author`, `model_family`, `stage`,
  `data_window_start`, `data_window_end`.
- No "untitled run #N" — every run has a deterministic, grep-able name.

Callers can override tags via the `extra_tags` argument; required tags
populate from env / git automatically.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

import mlflow
import structlog

log = structlog.get_logger(__name__)

DEFAULT_EXPERIMENT_NAME: str = "cluster_canary__oom_30min"


class MLflowConfigError(RuntimeError):
    """Raised when the MLflow tracking configuration violates the contract."""


def _git_meta() -> dict[str, str]:
    """`git_sha`, `git_branch`, `git_dirty` if we're in a git repo. Empty strings otherwise."""
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return {"git_sha": sha, "git_branch": branch, "git_dirty": str(dirty)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": "", "git_branch": "", "git_dirty": "unknown"}


def _run_name(model_family: str, dataset_version: str, git_sha: str) -> str:
    ts = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M")
    sha7 = (git_sha or "nogit")[:7]
    return f"{model_family}__{dataset_version}__{sha7}__{ts}"


@contextmanager
def start_tracked_run(
    *,
    model_family: str,
    dataset_version: str,
    data_window: tuple[str, str],
    stage: str = "experiment",
    task: str = "oom_30min",
    chaos_plan_version: str = "unknown",
    extra_tags: dict[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run with the cluster-canary tagging contract enforced.

    `model_family`: lgbm | rule_baseline | pytorch_mlp | …
    `dataset_version`: the DVC short hash of the input parquet
                       (`dvc status` on data/processed/).
    `data_window`: `(start_iso, end_iso)`.
    `stage`: experiment | candidate | challenger | champion.
    """
    if not model_family:
        raise MLflowConfigError("model_family must be non-empty")
    if not dataset_version:
        raise MLflowConfigError("dataset_version must be non-empty")
    if stage not in {"experiment", "candidate", "challenger", "champion"}:
        raise MLflowConfigError(f"invalid stage: {stage!r}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", f"cluster_canary__{task}")
    mlflow.set_experiment(experiment_name)

    git = _git_meta()
    run_name = _run_name(model_family, dataset_version, git["git_sha"])
    tags = {
        **git,
        "dataset_version": dataset_version,
        "author": os.environ.get("USER", "unknown"),
        "model_family": model_family,
        "stage": stage,
        "chaos_plan_version": chaos_plan_version,
        "data_window_start": data_window[0],
        "data_window_end": data_window[1],
    }
    if extra_tags:
        tags.update(extra_tags)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        log.info(
            "mlflow.run.start",
            run_id=run.info.run_id,
            run_name=run_name,
            experiment=experiment_name,
            tags=tags,
        )
        yield run
