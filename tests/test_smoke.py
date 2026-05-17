"""Smoke tests — always run, no data required."""

from __future__ import annotations

import cluster_canary


def test_package_importable() -> None:
    assert cluster_canary.__version__


def test_submodules_importable() -> None:
    """Every declared submodule must import without error."""
    from cluster_canary import (
        cli,
        data,
        features,
        models,
        monitoring,
        pipelines,
        serving,
        training,
    )

    for mod in (cli, data, features, models, monitoring, pipelines, serving, training):
        assert mod is not None


def test_version_format() -> None:
    """__version__ is a PEP 440-ish dotted string."""
    parts = cluster_canary.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() or any(c.isalpha() for c in p) for p in parts)
