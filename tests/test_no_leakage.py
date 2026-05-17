"""Leakage audit — enforced from .claude/skills/feature-leakage-detector.

cluster-canary is a temporal prediction problem (OOMKill / CrashLoopBackOff in
the next 30 minutes). The audit catches leakage that's easy to introduce when
features are derived from rolling windows or joins on post-event signals.

Each test skips cleanly when the processed splits or feature list are not yet
present. Once Phase 2 generates them, these all run on real data.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import pandas as pd

pytestmark = pytest.mark.needs_data

# Target-leak deny-list — any feature here is a function of the future event we
# are trying to predict and MUST be dropped at feature-engineering time.
# Anything you could only know AFTER the OOMKill happened belongs in here.
DENYLIST: set[str] = {
    "restart_count_post",  # restart counter measured AFTER the prediction window
    "future_oom_count",  # any forward-looking OOM aggregate
    "post_event_memory_mb",  # memory snapshot taken at/after the event
    "termination_reason",  # the label itself
    "exit_code",  # 137 reveals the outcome
    "last_terminated_reason",  # kube-state-metrics post-mortem field
    "container_oom_events_post",  # cAdvisor counter sampled AFTER the event
}

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FEATURE_LIST_PATH = PROCESSED_DIR / "feature_list.json"
TARGET_COL = "event_within_30min"  # binary label: OOMKill in [t, t+30min]
TIME_COL = "scrape_timestamp"  # Prometheus scrape timestamp, UTC seconds


def _load_splits() -> dict[str, pd.DataFrame]:
    pd = pytest.importorskip("pandas")
    paths = {name: PROCESSED_DIR / f"{name}.parquet" for name in ("train", "val", "test")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"Processed splits not yet generated (Phase 2+): {missing}")
    return {name: pd.read_parquet(p) for name, p in paths.items()}


def _load_feature_list() -> set[str]:
    if not FEATURE_LIST_PATH.exists():
        pytest.skip(f"Feature list not yet generated (Phase 2+): {FEATURE_LIST_PATH}")
    return set(json.loads(FEATURE_LIST_PATH.read_text()))


def test_temporal_ordering() -> None:
    """train_end < val_start; val_end < test_start (no temporal overlap)."""
    splits = _load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    if TIME_COL not in train.columns:
        pytest.skip(f"{TIME_COL!r} column missing — drop this test if your project is non-temporal")
    assert train[TIME_COL].max() < val[TIME_COL].min(), (
        f"train/val temporal overlap: train_max={train[TIME_COL].max()}, "
        f"val_min={val[TIME_COL].min()}"
    )
    assert val[TIME_COL].max() < test[TIME_COL].min(), (
        f"val/test temporal overlap: val_max={val[TIME_COL].max()}, test_min={test[TIME_COL].min()}"
    )


def test_no_high_corr_features() -> None:
    """No numeric feature correlates >0.95 with the target (target-leak proxy)."""
    splits = _load_splits()
    train = splits["train"]
    if TARGET_COL not in train.columns:
        pytest.skip(f"Target column {TARGET_COL!r} not present")
    num = train.select_dtypes("number").drop(columns=[TARGET_COL], errors="ignore")
    corr = num.corrwith(train[TARGET_COL]).abs()
    suspicious: dict[str, Any] = corr[corr > 0.95].to_dict()
    assert not suspicious, f"Likely target leak (|corr| > 0.95): {suspicious}"


def test_no_target_proxy_in_features() -> None:
    """None of the project's target-component columns appear in the feature list."""
    if not DENYLIST:
        pytest.skip(
            "DENYLIST is empty — customize this for your domain in tests/test_no_leakage.py"
        )
    feats = _load_feature_list()
    forbidden = feats & DENYLIST
    assert not forbidden, f"Target-proxy features present in feature list: {forbidden}"


def test_rolling_windows_are_left_closed() -> None:
    """All `.rolling(...)` calls in src/ are explicitly left-closed (no future peek)."""
    src = Path(__file__).resolve().parent.parent / "src"
    if not any(src.rglob("*.py")):
        pytest.skip("No source files yet")
    try:
        out = subprocess.check_output(
            ["grep", "-rn", "--include=*.py", r"\.rolling(", str(src)],
            stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        return  # no matches — fine, nothing rolls yet

    failures: list[str] = []
    for line in out.splitlines():
        if "closed=" not in line:
            failures.append(f"Rolling without explicit closed=: {line}")
        elif "'left'" not in line and '"left"' not in line:
            failures.append(f"Rolling not left-closed: {line}")
    assert not failures, "Rolling-window leakage:\n" + "\n".join(failures)


def test_entity_overlap_documented() -> None:
    """Train/test overlap of high-cardinality entity IDs must be a deliberate choice.

    Any column ending in `_id` is treated as a potential entity. To opt in to
    overlap (e.g. zone-style IDs that legitimately repeat), list the columns
    in `data/processed/entity_overlap.json`.
    """
    splits = _load_splits()
    train, test = splits["train"], splits["test"]
    id_cols = [c for c in train.columns if c.endswith("_id")]
    if not id_cols:
        pytest.skip("No entity ID columns to check")

    opt_in_path = PROCESSED_DIR / "entity_overlap.json"
    allowed: set[str] = set()
    if opt_in_path.exists():
        allowed = set(json.loads(opt_in_path.read_text()))

    leaks: list[str] = []
    for col in id_cols:
        if col in allowed:
            continue
        overlap = set(train[col]) & set(test[col])
        if overlap:
            leaks.append(f"{col}: {len(overlap)} overlapping entities")
    assert not leaks, (
        "Entity leakage (add to data/processed/entity_overlap.json to opt in):\n" + "\n".join(leaks)
    )
