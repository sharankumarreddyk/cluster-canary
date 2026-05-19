"""Locust load test for the cluster-canary inference service.

Targets the Phase 5 SLO from docs/PLAN.md:

    p95 latency < 50 ms at sustained 200 RPS per replica
    error rate  < 0.1 %

Run against a local `bentoml serve` or a port-forwarded in-cluster service:

    bentoml serve cluster_canary.serving.service:ClusterCanaryService --port 3000

    locust -f tests/loadtest.py \\
        --host http://localhost:3000 \\
        --users 200 --spawn-rate 20 --run-time 5m \\
        --html reports/loadtest.html
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

# The feature list the model was trained on. If it doesn't exist locally we
# fall back to a placeholder so this file imports cleanly in CI.
_FEATURE_LIST_PATH = Path("data/processed/feature_list.json")
if _FEATURE_LIST_PATH.exists():
    FEATURE_NAMES: list[str] = json.loads(_FEATURE_LIST_PATH.read_text())
else:
    # 30-feature placeholder so this module imports in CI without Phase 3 outputs.
    FEATURE_NAMES = [f"feat_synthetic_{i}" for i in range(30)]


def _synth_request(i: int) -> dict[str, object]:
    """Build one realistic-looking inference request.

    Memory-pressure features are sampled from a bimodal distribution so the
    model sees a mix of healthy and dangerous pods (similar to production).
    """
    is_dangerous = random.random() < 0.05  # 5% chance of high-pressure
    features: dict[str, float] = {}
    for name in FEATURE_NAMES:
        if "mem_pct" in name:
            features[name] = (
                random.uniform(0.75, 0.98) if is_dangerous else random.uniform(0.05, 0.65)
            )
        elif "cpu" in name:
            features[name] = random.uniform(0.1, 0.9)
        elif "pod_age" in name:
            features[name] = random.uniform(60, 86_400)
        elif "is_weekend" in name or "is_business_hours" in name:
            features[name] = float(random.randint(0, 1))
        elif "hour_of_day" in name:
            features[name] = float(random.randint(0, 23))
        elif "day_of_week" in name:
            features[name] = float(random.randint(0, 6))
        else:
            features[name] = random.uniform(-1.0, 1.0)
    return {
        "pod_uid": f"uid-{i % 1000:04d}",
        "container": "app",
        "features": features,
    }


class PredictionUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task(10)
    def predict(self) -> None:
        body = _synth_request(random.randint(0, 1_000_000))
        with self.client.post(
            "/predict",
            json=[body],
            catch_response=True,
            name="POST /predict",
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code} body={response.text[:200]}")
                return
            try:
                data = response.json()
            except ValueError:
                response.failure("non-JSON response")
                return
            if not isinstance(data, list) or not data:
                response.failure(f"bad shape: {type(data).__name__}")
                return
            if not 0.0 <= data[0]["prediction"] <= 1.0:
                response.failure(f"prediction out of [0,1]: {data[0]['prediction']}")
                return
            if data[0].get("model_version") in (None, ""):
                response.failure("missing model_version in response")
                return

    @task(1)
    def healthz(self) -> None:
        with self.client.get("/healthz", catch_response=True, name="GET /healthz") as r:
            if r.status_code != 200:
                r.failure(f"healthz status={r.status_code}")
