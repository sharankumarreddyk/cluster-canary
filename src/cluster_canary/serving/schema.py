"""Pydantic v2 request/response schemas for the cluster-canary inference service.

Two-way contract between the feature-extractor DaemonSet and the inference
sidecar. The schema is deliberately loose on the `features` map (accept any
`feat_*` keys) so the model's feature set can evolve without breaking the wire
format — the service validates against the trained model's `feature_list.json`
at request time and returns 422 for missing required features.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MAX_BATCH_SIZE: int = 64
MAX_BATCH_LATENCY_MS: int = 20

# Outer bounds on identifier sizes — defensive, matches the K8s spec
# (`labels` are <=63 chars, `uid` is a UUID).
_ID_MAX = 128


class PredictRequest(BaseModel):
    """One inference request for a single (pod_uid, container).

    `features` is a dict from feature_name → value, where the keys must cover
    every entry in the model's `feature_list.json`. Extra keys are silently
    ignored (forward-compatible — the extractor can ship more features than
    the current model uses without coordinated deploys).
    """

    pod_uid: str = Field(..., min_length=1, max_length=_ID_MAX)
    container: str = Field(..., min_length=1, max_length=_ID_MAX)
    features: dict[str, float] = Field(..., min_length=1)


class ContributorEntry(BaseModel):
    """One row of the top-k SHAP-attribution table on a response."""

    feature: str
    contribution_logodds: float
    contribution_prob_delta: float
    direction: Literal["up", "down"]
    rank: int = Field(..., ge=1, le=10)


class PredictResponse(BaseModel):
    """One inference response. Mirrors `predict_with_explanation` + identity."""

    pod_uid: str
    container: str
    prediction: float = Field(..., ge=0.0, le=1.0)
    prediction_logodds: float
    base_value: float
    model_version: str
    top_contributors: list[ContributorEntry]
    inference_latency_ms: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    """`/healthz` and `/readyz` payload."""

    status: Literal["ok", "ready", "not_ready"]
    model_version: str | None = None
