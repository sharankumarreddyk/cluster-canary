"""BentoML inference service for cluster-canary.

Architecture (per docs/PHASE_1_CONTEXT.md § "Architecture"):
- An upstream `canary-feature-extractor` DaemonSet computes per-pod features
  from Prometheus scrapes and POSTs them to this service.
- This service runs the LightGBM model + isotonic calibrator and returns the
  probability + top-3 SHAP contributors.
- A downstream `canary-action-router` reads the prediction and decides
  whether to alert PagerDuty / kubeai-ops / emit a VPA recommendation.

Why HTTP, not gRPC: as of BentoML 1.3 the gRPC server is still beta and does
NOT support the new class-based `@bentoml.service` SDK. HTTP overhead at
pod-loopback is 1-3 ms — well within our 50 ms p95 latency budget. Revisit
gRPC when BentoML announces stable class-based support.

Why direct-path model loading (not BentoML's model store): the model files
ship inside the container image as build artifacts. Direct loading is faster
to iterate on, has no coupling to BentoML's model registry, and stays
versioned alongside the source code in git.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from pathlib import Path

import bentoml
import lightgbm as lgb
import numpy as np
import numpy.typing as npt
from prometheus_client import Counter, Histogram

from cluster_canary.models.calibration import Calibrator
from cluster_canary.serving.schema import (
    MAX_BATCH_LATENCY_MS,
    MAX_BATCH_SIZE,
    ContributorEntry,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from cluster_canary.training.shap_helpers import predict_with_explanation

DEFAULT_MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
MODEL_NAME = "cluster_canary"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "0.1.0")

# Custom Prometheus metrics — BentoML's /metrics endpoint auto-includes them.
# `bentoml_service_*` namespace is BentoML's built-in; we add `canary_*` for
# the things specific to our problem (calibrated probability distribution,
# top-1 feature frequency).
_pred_value = Histogram(
    "canary_predicted_prob",
    "Calibrated P(OOM in next 30 min) distribution",
    ["model_version"],
    buckets=(0.01, 0.05, 0.10, 0.25, 0.50, 0.70, 0.90, 0.99),
)
_pred_count = Counter(
    "canary_predict_total",
    "Predictions served",
    ["model_version", "status"],
)
_top1_feature = Counter(
    "canary_top1_feature_total",
    "Most-attributed feature on each prediction — helps drift triage",
    ["model_version", "feature"],
)

log = logging.getLogger("cluster_canary.serving")
logging.basicConfig(
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":%(message)s}',
    level=os.environ.get("MLP1_LOG_LEVEL", "INFO"),
)


@bentoml.service(
    name="cluster-canary",
    resources={"cpu": "1", "memory": "512Mi"},
    traffic={"timeout": 5, "max_concurrency": 256},
    workers=2,
)
class ClusterCanaryService:
    """OOM-in-next-30-min prediction service.

    Loads the LightGBM booster + isotonic calibrator from `MODELS_DIR/` at
    startup. The model files (`canary_model.txt`, `calibrator.pkl`,
    `feature_list.json`) are produced by `make train` (Phase 4) and shipped in
    the container image.
    """

    def __init__(self) -> None:
        models_dir = DEFAULT_MODELS_DIR
        self.booster: lgb.Booster = lgb.Booster(model_file=str(models_dir / "canary_model.txt"))
        with (models_dir / "calibrator.pkl").open("rb") as fh:
            self.calibrator: Calibrator = pickle.load(fh)
        self.feature_list: list[str] = json.loads((models_dir / "feature_list.json").read_text())
        self.model_version: str = MODEL_VERSION
        self._warmup()
        self._ready = True
        log.info(
            json.dumps(
                {
                    "event": "service.started",
                    "model_version": self.model_version,
                    "n_features": len(self.feature_list),
                    "models_dir": str(models_dir),
                }
            )
        )

    def _warmup(self) -> None:
        """Run one synthetic prediction to JIT the LightGBM hot paths."""
        dummy = np.zeros((1, len(self.feature_list)), dtype="float64")
        self.booster.predict(dummy)
        log.info(json.dumps({"event": "service.warmup_done"}))

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=MAX_BATCH_SIZE,
        max_latency_ms=MAX_BATCH_LATENCY_MS,
    )
    async def predict(
        self, requests: list[PredictRequest], ctx: bentoml.Context
    ) -> list[PredictResponse]:
        """Batch predict OOM probability + top-3 SHAP for each request."""
        t0 = time.perf_counter()
        try:
            X = self._build_feature_matrix(requests)
        except KeyError as exc:
            _pred_count.labels(self.model_version, "missing_features").inc(len(requests))
            ctx.response.status_code = 422
            raise bentoml.exceptions.InvalidArgument(f"missing required features: {exc}") from exc

        raw_p = np.asarray(self.booster.predict(X), dtype="float64")
        calibrated = self.calibrator.transform(raw_p)

        responses: list[PredictResponse] = []
        for i, request in enumerate(requests):
            explanation = predict_with_explanation(
                self.booster,
                X[i : i + 1],
                feature_names=self.feature_list,
                k=3,
                model_version=self.model_version,
            )
            # Override the raw prediction with the calibrated probability —
            # action-router thresholds at P > 0.7, so calibration matters.
            calibrated_p = float(calibrated[i])
            contributors = [ContributorEntry(**entry) for entry in explanation["top_contributors"]]
            response = PredictResponse(
                pod_uid=request.pod_uid,
                container=request.container,
                prediction=calibrated_p,
                prediction_logodds=float(explanation["prediction_logodds"]),
                base_value=float(explanation["base_value"]),
                model_version=self.model_version,
                top_contributors=contributors,
                inference_latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )
            responses.append(response)

            _pred_value.labels(self.model_version).observe(calibrated_p)
            if contributors:
                _top1_feature.labels(self.model_version, contributors[0].feature).inc()

        _pred_count.labels(self.model_version, "ok").inc(len(requests))
        ctx.response.headers.append("X-Model-Version", self.model_version)
        log.info(
            json.dumps(
                {
                    "event": "predict.batch",
                    "n_requests": len(requests),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "model_version": self.model_version,
                }
            )
        )
        return responses

    @bentoml.api(route="/canary/healthz")
    def healthz(self, ctx: bentoml.Context) -> HealthResponse:
        """Liveness: process is up."""
        ctx.response.headers.append("X-Model-Version", self.model_version)
        return HealthResponse(status="ok", model_version=self.model_version)

    @bentoml.api(route="/canary/readyz")
    def readyz(self, ctx: bentoml.Context) -> HealthResponse:
        """Readiness: model loaded + warmup done."""
        ctx.response.headers.append("X-Model-Version", self.model_version)
        return HealthResponse(
            status="ready" if self._ready else "not_ready",
            model_version=self.model_version,
        )

    def _build_feature_matrix(self, requests: list[PredictRequest]) -> npt.NDArray[np.float64]:
        """Materialise the wide feature matrix in the EXACT column order the model was trained on.

        Raises `KeyError` listing the first missing feature so the caller can
        return a clean 422 to the upstream extractor.
        """
        rows: list[list[float]] = []
        for request in requests:
            row: list[float] = []
            for name in self.feature_list:
                if name not in request.features:
                    raise KeyError(f"{name!r} (pod_uid={request.pod_uid})")
                row.append(float(request.features[name]))
            rows.append(row)
        return np.asarray(rows, dtype="float64")
