# Serving — Phase 5

A BentoML HTTP service that loads the Phase 4 LightGBM model + isotonic
calibrator + feature_list, batches incoming requests, and returns calibrated
probabilities with top-3 SHAP contributors. Deployed as an in-cluster sidecar
via Helm + ArgoCD.

## Why HTTP, not gRPC

BentoML 1.3+ ships gRPC as a beta feature that does NOT yet support the new
class-based `@bentoml.service` SDK (verified by the BentoML team in
[discussion #3635](https://github.com/orgs/bentoml/discussions/3635)). Using
gRPC would force us back to the deprecated 1.1-style function-services API.
The latency penalty of HTTP at pod-loopback is 1-3 ms — comfortably inside
our 50 ms p95 budget. We'll re-evaluate when BentoML announces stable
class-based gRPC.

## Architecture

```
upstream feature-extractor DaemonSet
            |
            v   HTTP POST /predict  (Pydantic v2 PredictRequest)
       cluster-canary inference svc
        +--+--+--+--+--+
        |  bentoml batches |  adaptive: max_batch_size=64, max_latency_ms=20
        +--+--+--+--+--+
            |
            v
       LightGBM booster (1 call for the batch)
            |
            v
       isotonic calibrator (vectorised)
            |
            v
       per-row TreeSHAP top-3      (booster.predict(pred_contrib=True))
            |
            v   HTTP 200  (Pydantic v2 PredictResponse)
            v   header: X-Model-Version: <model_version>
downstream action-router
```

## Wire schema

Request:

```json
{
  "pod_uid": "uid-1234",
  "container": "leaky-flask",
  "features": {
    "feat_mem_pct_of_limit": 0.82,
    "feat_mem_pct_of_limit__5min_max": 0.91,
    "feat_mem_growth_rate__30min": 0.18,
    // ... every feat_* the model was trained on
  }
}
```

Response:

```json
{
  "pod_uid": "uid-1234",
  "container": "leaky-flask",
  "prediction": 0.87,
  "prediction_logodds": 1.89,
  "base_value": -2.10,
  "model_version": "0.1.0",
  "top_contributors": [
    {
      "feature": "feat_mem_pct_of_limit__5min_max",
      "contribution_logodds": 0.42,
      "contribution_prob_delta": 0.08,
      "direction": "up",
      "rank": 1
    },
    { "feature": "feat_mem_growth_rate__30min", "contribution_logodds": 0.18, ... },
    { "feature": "feat_cpu_throttle_ratio__5min_mean", ... }
  ],
  "inference_latency_ms": 4.7
}
```

Missing-feature handling: if the request omits any feature in the model's
`feature_list.json`, the service returns HTTP 422 with the first missing
column. Extra features are ignored (forward-compatible — the extractor can
ship more features than the current model uses without a coordinated deploy).

## Endpoints

| Path | Purpose | Builtin / custom |
|---|---|---|
| `POST /predict` | Inference (adaptive-batched) | custom — `ClusterCanaryService.predict` |
| `GET /healthz` | Liveness (process is up) | built-in BentoML — also `/canary/healthz` returns the model_version |
| `GET /readyz` | Readiness (model loaded + warmup done) | built-in — also `/canary/readyz` |
| `GET /metrics` | Prometheus (BentoML's `bentoml_service_*` + our `canary_*` Counters/Histograms) | auto |
| `GET /livez` | Alias of `/healthz` | built-in |

## Metrics

In addition to BentoML's defaults (`bentoml_service_request_total`,
`bentoml_service_request_duration_seconds`, etc.) the service emits:

- `canary_predicted_prob{model_version}` — Histogram of calibrated probability output
- `canary_predict_total{model_version, status}` — Counter; status ∈ {ok, missing_features}
- `canary_top1_feature_total{model_version, feature}` — Counter — top-1 SHAP attribution per prediction; great input to a drift triage Grafana panel

## Performance targets

| Metric | Target | Notes |
|---|---|---|
| p95 latency | < 50 ms | Sub-50 ms gates the in-cluster sidecar pattern |
| p99 latency | < 100 ms | Tail-latency budget |
| Sustained RPS | 200 per replica | Scale horizontally past this |
| Error rate | < 0.1 % | Mostly missing-feature 422s |
| Memory footprint | < 300 MB resident | Sidecar overhead budget |
| Image size | < 300 MB | `bentoml containerize` defaults to 600-900 MB — we hand-roll |

## How to run

### Local (no Docker)

```bash
# After `make train` produces models/canary_model.txt + calibrator.pkl + feature_list.json:
make serve              # http://localhost:3000
# in another shell:
make loadtest           # 200 users, 5 min run, html report at reports/loadtest.html
```

### Local Docker

```bash
make serve-docker-build       # multi-stage build, target <300 MB
make serve-docker-run         # binds 3000
docker image ls cluster-canary:dev --format "{{.Size}}"   # verify <300 MB
```

### In-cluster (Helm)

```bash
make helm-lint
make helm-template       # sanity-check render
helm install canary infra/helm/cluster-canary -n cluster-canary --create-namespace
```

### In-cluster (ArgoCD)

```bash
kubectl apply -f infra/argocd/application.yaml -n argocd
# ArgoCD reconciles the chart from the main branch onwards.
```

## Probing

```bash
curl http://localhost:3000/healthz
curl http://localhost:3000/readyz
curl http://localhost:3000/metrics | grep canary_

curl -X POST http://localhost:3000/predict \
  -H 'Content-Type: application/json' \
  -d '[{"pod_uid":"u1","container":"c","features":{"feat_mem_pct_of_limit":0.9}}]'
```

(A real request requires every key in `feature_list.json`; the above will 422.)

## Out of session scope

- Real load-test numbers — populate the README's "Results" section after
  running `make loadtest` against a real `make train` model.
- Hot-swap from a mounted PVC — `values.yaml model.pathInImage` accommodates
  it but the chart doesn't mount one yet (in scope for Phase 7's retrain
  loop).
- gRPC — revisit when BentoML stabilises class-based gRPC.
