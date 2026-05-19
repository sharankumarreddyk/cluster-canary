"""Unit tests for the BentoML inference service.

The service class is instantiated DIRECTLY (no `bentoml serve` HTTP server)
with a real tiny LightGBM booster + isotonic calibrator written to a
`tmp_path`. We assert:

- model + calibrator + feature_list load on `__init__`
- warmup completes
- predict() returns the documented schema with correct shape
- feature-matrix construction raises KeyError on missing features (mapped to 422 upstream)
- top-3 SHAP contributors come back with the right shape

Skipped on macOS Apple Silicon where libomp is absent (LightGBM dlopen fails).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

# LightGBM + libomp guard — same pattern as test_shap_helpers.
try:
    import lightgbm as lgb
except (ImportError, OSError) as exc:
    pytest.skip(f"lightgbm not loadable: {exc}", allow_module_level=True)

# BentoML can be imported even without a model; the service class itself loads
# real files on `__init__`, so we build a tiny model fixture first.
try:
    import bentoml  # noqa: F401
except ImportError as exc:
    pytest.skip(f"bentoml not importable: {exc}", allow_module_level=True)

from cluster_canary.models.calibration import Calibrator
from cluster_canary.serving import service as service_module
from cluster_canary.serving.schema import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
)


def _train_tiny_booster(n_features: int = 5, n_rows: int = 500) -> tuple[lgb.Booster, Calibrator]:
    """Train a deterministic LightGBM booster + isotonic calibrator on synthetic data."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_rows, n_features))
    logits = 1.5 * X[:, 0] + 0.5 * X[:, 1]
    y = (rng.uniform(0, 1, n_rows) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_leaves": 7, "seed": 0},
        lgb.Dataset(X, label=y, feature_name=[f"feat_{i}" for i in range(n_features)]),
        num_boost_round=20,
    )
    raw = np.asarray(booster.predict(X), dtype="float64")
    calibrator = Calibrator.fit(raw, y, method="isotonic")
    return booster, calibrator


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    """Materialise a real models/ directory with the artefacts the service loads."""
    booster, calibrator = _train_tiny_booster(n_features=5)
    feat_names = [f"feat_{i}" for i in range(5)]
    booster.save_model(str(tmp_path / "canary_model.txt"))
    with (tmp_path / "calibrator.pkl").open("wb") as fh:
        pickle.dump(calibrator, fh)
    (tmp_path / "feature_list.json").write_text(json.dumps(feat_names))
    return tmp_path


@pytest.fixture
def service_instance(monkeypatch: pytest.MonkeyPatch, models_dir: Path):  # type: ignore[no-untyped-def]
    """Construct the service with MODELS_DIR pointing at the test fixture."""
    # Override the module-level DEFAULT_MODELS_DIR before instantiation.
    monkeypatch.setattr(service_module, "DEFAULT_MODELS_DIR", models_dir)
    monkeypatch.setattr(service_module, "MODEL_VERSION", "test-0.0.1")
    # BentoML's `@bentoml.service` wraps the class — instantiate the underlying
    # type via its `inner` attribute when available; fall back to direct call.
    Service = service_module.ClusterCanaryService
    inner = getattr(Service, "inner", Service)
    return inner()


def test_service_loads_model_and_warms_up(service_instance) -> None:  # type: ignore[no-untyped-def]
    assert service_instance.booster is not None
    assert service_instance.calibrator is not None
    assert service_instance.feature_list == [f"feat_{i}" for i in range(5)]
    assert service_instance.model_version == "test-0.0.1"


def test_build_feature_matrix_orders_columns_correctly(service_instance) -> None:  # type: ignore[no-untyped-def]
    request = PredictRequest(
        pod_uid="u1",
        container="c",
        features={f"feat_{i}": float(i) for i in range(5)},
    )
    X = service_instance._build_feature_matrix([request])
    assert X.shape == (1, 5)
    # Column 0 must hold feat_0's value, regardless of insertion order in the dict.
    assert X[0, 0] == 0.0
    assert X[0, 4] == 4.0


def test_build_feature_matrix_raises_on_missing_features(service_instance) -> None:  # type: ignore[no-untyped-def]
    request = PredictRequest(
        pod_uid="u1",
        container="c",
        features={"feat_0": 1.0, "feat_1": 2.0},  # missing feat_2/3/4
    )
    with pytest.raises(KeyError, match="feat_2"):
        service_instance._build_feature_matrix([request])


def test_predict_response_schema(service_instance) -> None:  # type: ignore[no-untyped-def]
    """Predict path is async — drive it via asyncio.run with a fake ctx."""
    import asyncio

    class _FakeResponseCtx:
        def __init__(self) -> None:
            self.headers: list[tuple[str, str]] = []
            self.status_code = 200

    class _FakeCtx:
        def __init__(self) -> None:
            self.response = _FakeResponseCtx()
            self.response.headers = type(
                "H", (), {"append": lambda _self, k, v: self.response.headers.append((k, v))}
            )()  # type: ignore[attr-defined]

        @property
        def headers(self) -> list[tuple[str, str]]:
            return self.response.headers  # type: ignore[return-value]

    request = PredictRequest(
        pod_uid="u1",
        container="c",
        features={f"feat_{i}": float(i) * 0.1 for i in range(5)},
    )
    ctx = _FakeCtx()
    responses: list[PredictResponse] = asyncio.run(
        service_instance.predict([request, request], ctx)
    )
    assert len(responses) == 2
    for r in responses:
        assert 0.0 <= r.prediction <= 1.0
        assert isinstance(r.prediction_logodds, float)
        assert isinstance(r.base_value, float)
        assert r.model_version == "test-0.0.1"
        assert len(r.top_contributors) == 3
        ranks = [c.rank for c in r.top_contributors]
        assert ranks == [1, 2, 3]
        assert r.inference_latency_ms > 0


def test_healthz_and_readyz_return_canonical_payloads(service_instance) -> None:  # type: ignore[no-untyped-def]
    class _FakeResponseCtx:
        def __init__(self) -> None:
            self.headers = type("H", (), {"append": lambda _self, _k, _v: None})()

    class _FakeCtx:
        def __init__(self) -> None:
            self.response = _FakeResponseCtx()

    ctx = _FakeCtx()
    health = service_instance.healthz(ctx)
    ready = service_instance.readyz(ctx)
    assert isinstance(health, HealthResponse)
    assert health.status == "ok"
    assert ready.status == "ready"
    assert ready.model_version == "test-0.0.1"
