---
name: bentoml-service-scaffolder
description: Scaffolds production-ready BentoML services with adaptive batching, Prometheus metrics, structured request logging, input validation, graceful shutdown, and a paired locust load test. Use whenever code defines a model-serving endpoint, builds a Bento, or writes a Dockerfile for a model service. Default "wrap fastapi around joblib.load" services are rejected.
---

# BentoML Service Scaffolder

## When this skill applies

Trigger when code:
- Imports `bentoml` or defines a `bentoml.Service` / `bentoml.legacy.Service` / `@bentoml.service`.
- Creates an inference endpoint for a trained model (FastAPI, Flask, BentoML, Ray Serve).
- Writes a Dockerfile for a model server.
- Builds a `bentofile.yaml` or `pyproject.toml` serving config.

If a serving endpoint is being created without the contract below, fix it.

## The non-negotiables every service must have

1. **Input validation via Pydantic.** Reject malformed payloads at the boundary with HTTP 422, not via Python tracebacks.
2. **Adaptive batching enabled** for the predict runner. Max batch size and latency tuned to the model.
3. **Prometheus metrics** exposed at `/metrics`: request count, latency histogram, prediction histogram, error count by reason, batch size histogram.
4. **Structured JSON request logs** to stdout with `request_id`, `model_version`, `latency_ms`, `n_inputs`, `error`. PII scrubbed.
5. **Health endpoints**: `/healthz` (liveness — process up), `/readyz` (readiness — model loaded + warm).
6. **Model version surfaced** in every response header (`X-Model-Version`) and log line.
7. **Graceful shutdown** — drain in-flight requests on SIGTERM, max 30s.
8. **A paired `tests/loadtest.py`** using locust with a target SLO of p95 < 100ms at sustainable RPS.

## Skeleton service (use as the starting point)

```python
# src/cluster_canary/serving/service.py
from __future__ import annotations
import os, time, uuid, logging, json
from typing import Annotated
import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, conlist
from prometheus_client import Histogram, Counter

MODEL_NAME = "main_model"
MODEL_TAG  = os.environ.get("MODEL_TAG", f"{MODEL_NAME}:latest")

_predict_latency = Histogram(
    "model_predict_latency_seconds", "Inference latency",
    ["model", "version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
_predict_count   = Counter("model_predict_total", "Predictions served", ["model", "version", "status"])
_pred_value      = Histogram("model_prediction_value", "Predicted value distribution",
                             ["model", "version"], buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 100))
_batch_size      = Histogram("model_predict_batch_size", "Batch size distribution",
                             ["model"], buckets=(1, 2, 4, 8, 16, 32, 64, 128))

log = logging.getLogger("serving")
logging.basicConfig(format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}', level=logging.INFO)


class PredictionRequest(BaseModel):
    # Replace these with your model's input fields. Validate at the boundary:
    # use ge/le bounds, regex patterns, enums — fail fast at HTTP 422.
    feature_a: float = Field(..., ge=-1e9, le=1e9)
    feature_b: float = Field(..., ge=-1e9, le=1e9)
    category:  str   = Field(..., min_length=1, max_length=64)


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    request_id: str


@bentoml.service(
    name="prediction_service",
    resources={"cpu": "2", "memory": "1Gi"},
    traffic={"timeout": 5, "max_concurrency": 256},
)
class ClusterCanaryService:
    def __init__(self) -> None:
        self.model    = bentoml.lightgbm.load_model(MODEL_TAG)
        self.version  = bentoml.models.get(MODEL_TAG).tag.version
        self._warmup()

    def _warmup(self) -> None:
        # Force JIT compilation paths to fire before serving traffic.
        warm = pd.DataFrame([{"feature_a": 0.0, "feature_b": 0.0, "category": "default"}])
        self.model.predict(warm)

    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=64, max_latency_ms=20)
    async def predict(self, items: list[PredictionRequest]) -> list[PredictionResponse]:
        req_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        try:
            features = _featurize(items)
            _batch_size.labels(MODEL_NAME).observe(len(items))
            preds = self.model.predict(features)
            # Apply any domain-specific safety bounds here (e.g. np.clip).
            out = [
                PredictionResponse(prediction=float(p), model_version=self.version, request_id=req_id)
                for p in preds
            ]
            for p in preds:
                _pred_value.labels(MODEL_NAME, self.version).observe(float(p))
            _predict_count.labels(MODEL_NAME, self.version, "ok").inc(len(items))
            return out
        except Exception as e:  # pragma: no cover
            _predict_count.labels(MODEL_NAME, self.version, "error").inc(len(items))
            log.exception(json.dumps({"event": "predict_error", "request_id": req_id, "error": str(e)}))
            raise
        finally:
            dt = time.perf_counter() - t0
            _predict_latency.labels(MODEL_NAME, self.version).observe(dt)
            log.info(json.dumps({
                "event": "predict", "request_id": req_id, "model_version": self.version,
                "n_inputs": len(items), "latency_ms": round(dt * 1000, 2),
            }))

    @bentoml.api
    def healthz(self) -> dict[str, str]:
        return {"status": "ok"}

    @bentoml.api
    def readyz(self) -> dict[str, str]:
        return {"status": "ready", "model_version": self.version}
```

## Batching parameters — how to tune

Defaults above (`max_batch_size=64`, `max_latency_ms=20`) suit a CPU LightGBM model serving moderate QPS. Tune as follows:

| Model profile | max_batch_size | max_latency_ms |
|---|---|---|
| CPU tree (LightGBM, XGBoost) | 64–128 | 10–20 |
| CPU sklearn linear | 256 | 10 |
| CPU PyTorch MLP (<10M params) | 32 | 25 |
| GPU PyTorch transformer | 8–32 | 50 |
| GPU LLM (vLLM, TGI) | use continuous batching | n/a |

Always re-benchmark after tuning. Don't set numbers from gut.

## The paired locust load test (required artifact)

```python
# tests/loadtest.py
from locust import HttpUser, task, between
import random

class PredictionLoadUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task
    def predict(self):
        # Replace fields to match your service's PredictionRequest schema.
        body = {
            "feature_a": random.uniform(-1.0, 1.0),
            "feature_b": random.uniform(-1.0, 1.0),
            "category":  random.choice(["A", "B", "C"]),
        }
        with self.client.post("/predict", json=[body], catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status={r.status_code}")
```

Run: `locust -f tests/loadtest.py --host http://localhost:3000 --users 200 --spawn-rate 20 --run-time 5m --html reports/loadtest.html`

**Success criteria for the README results table:**
- p95 latency < 100ms at sustained 200 RPS on a 2-CPU container.
- Error rate < 0.1%.
- Memory steady-state < 800MB.

## Dockerfile pattern

```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*

FROM base AS build
WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt

FROM base AS runtime
COPY --from=build /install /usr/local
WORKDIR /app
COPY src/ ./src/
COPY bentofile.yaml ./
RUN useradd -m -u 1000 bento && chown -R bento:bento /app
USER bento
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD curl -fs http://localhost:3000/healthz || exit 1
ENTRYPOINT ["bentoml", "serve", "src.cluster_canary.serving.service:ClusterCanaryService", "--port", "3000", "--host", "0.0.0.0"]
```

Target image size: <300MB (with LightGBM model artifact). If you exceed that, audit the image with `dive`.

## Anti-patterns to refuse

- Wrapping the model in FastAPI directly and skipping BentoML's batching. You lose the biggest serving win.
- Loading the model inside the request handler (cold start every request).
- Returning prediction without `model_version` — makes A/B testing and rollback impossible.
- Logging raw input payloads (lat/lon + passenger count is OK; user identifiers, never).
- No `/readyz` (just `/healthz`) — k8s will route traffic to a pod whose model is still loading.
- Allocating `4` CPU and `8Gi` "to be safe" — tune from load test data.
