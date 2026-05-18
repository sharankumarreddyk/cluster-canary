# cluster-canary — Plan

> **Predict Kubernetes pod failures (OOMKill, CrashLoopBackOff) before they happen — and act on the prediction.**

A portfolio MLOps project designed for the "ML for platforms / infra" lane. Extends the [`kubeai-ops`](https://github.com/sharankumarreddyk/kubeai-ops) reactive incident-response platform with a proactive predictive layer.

---

## Why this project

| Differentiator | Why it matters |
|---|---|
| **Not in any MLOps tutorial.** | Most portfolios ship recsys / fraud / fare / churn. None ship "predicting pod OOMKills." |
| **Real adversarial drift signal.** | Workload mix shifts weekly as deployments evolve. Retraining is *required*, not theoretical. |
| **Portfolio coherence with kubeai-ops.** | Together they tell one story: reactive + proactive ML for the platform layer. |
| **Hard inference SLO.** | An in-cluster sidecar serving sub-50ms p95 forces real engineering, not Streamlit demos. |
| **Honest data story.** | Hybrid: synthetic data generated from a real kind cluster + chaos-mesh, validated against the Alibaba Cluster Trace 2018/2020. |
| **Maps to actual job listings.** | "ML for SRE / Platform Engineering" is a real and growing niche at Google, Meta, Datadog, Snowflake, Honeycomb. |

---

## Problem statement (precise)

For every running pod, every minute, predict:

```
P(OOMKill in next 30 min | pod state at t)         → main task (binary classification)
P(CrashLoopBackOff in next 30 min | pod state at t) → secondary task (joint or separate model)
```

**Inputs (features at time `t`):**
- Rolling memory usage (current, 5/15/30 min windows; rate of change)
- Rolling CPU usage + throttle counter
- Container restart count in the last hour
- Pod age, namespace, container image hash, image base layer
- Recent OOM events for this image (across pods, last 24h)
- Pod's memory limit / request, CPU limit / request
- Recent NetworkPolicy denial count for this pod
- Pod labels (workload type via well-known labels)

**Outputs:**
- Probability + calibration (Brier score)
- Top-3 SHAP contributions (interpretability — explains the alert)

**Action layer (optional v2):**
- Webhook to kubeai-ops / PagerDuty if `P > 0.7` and the pod is critical-tier
- VPA recommendation: "increase memory by X to reduce OOM probability below 0.1"

---

## Data approach

**Hybrid strategy** — start with synthetic for iteration speed, validate against real Google/Alibaba traces.

### Layer 1 — Synthetic (for development)

A local `kind` cluster runs a heterogeneous workload — webservers, batch jobs, leaky Python services, idle pods — with `chaos-mesh` injecting:
- Gradual memory leaks (random rate per pod)
- Memory bombs (instantaneous spike)
- CPU starvation
- Random container crashes

Prometheus + kube-state-metrics scrape every 30 s. The resulting time-series is parquet-dumped to `data/raw/synthetic/`. This is the dev dataset — fast to iterate, fully reproducible.

### Layer 2 — Real cluster traces (for credibility)

Two public datasets:

| Dataset | Source | Size | Schema fit |
|---|---|---|---|
| **Alibaba Cluster Trace 2018** | https://github.com/alibaba/clusterdata | 1 TB / 270 K machines / 8 days | K8s-aligned (containers + workloads) |
| **Google Cluster Trace 2019** | https://github.com/google/cluster-data | 12 B rows / 8 weeks | Borg, predecessor of K8s — schema differs but concepts map |

Use a 100GB sample of Alibaba 2018 as the "production" dataset for the README's headline number. The synthetic data is the model's training set; the Alibaba sample is the held-out generalization test.

### Train/val/test split

Strict temporal:
- Train: synthetic + first 5 days of Alibaba sample
- Val: synthetic + days 6
- Test: Alibaba days 7–8 only (never seen by training)

Plus a "drift simulation" slice: a synthetic dataset with deliberately different workload mix than train, to validate the drift-retrain loop.

---

## Architecture target

```
┌────────────────────────────────────────────────────────────────────┐
│                      Production K8s cluster                         │
│                                                                      │
│  ┌──────────┐   ┌──────────────┐    ┌──────────────────────────┐  │
│  │ kube-    │──▶│ Prometheus   │───▶│ canary-feature-extractor │  │
│  │ state-   │   │              │    │  (DaemonSet)             │  │
│  │ metrics  │   │              │    │                          │  │
│  └──────────┘   └──────────────┘    └────────────┬─────────────┘  │
│                                                   │ feature frames   │
│                                                   ▼                  │
│                                       ┌──────────────────────────┐  │
│                                       │ canary-inference-service │  │
│                                       │ (Deployment, BentoML)    │  │
│                                       │ <50 ms p95 gRPC          │  │
│                                       └────────────┬─────────────┘  │
│                                                    │ predictions     │
│                                                    ▼                  │
│                                       ┌──────────────────────────┐  │
│                                       │ canary-action-router     │  │
│                                       │ (Deployment, Python)     │  │
│                                       │ - webhook PagerDuty      │  │
│                                       │ - notify kubeai-ops      │  │
│                                       │ - VPA recommendation     │  │
│                                       └──────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                          │
                          ▼ predictions + ground truth (T+30)
                ┌────────────────────────┐
                │ Drift monitor          │  Evidently AI, daily
                │ - feature drift        │
                │ - prediction drift     │
                │ - calibration drift    │
                └────────────────────────┘
                          │
                          ▼ drift detected
                ┌────────────────────────┐
                │ Prefect retrain flow   │  triggered by drift gauge
                │ Pull data → train →    │
                │ champion/challenger →  │
                │ promote                │
                └────────────────────────┘
```

---

## Model approach

| Layer | Approach | Why |
|---|---|---|
| Baseline | "pod will OOM if memory usage > 90% of limit for 5+ min" — pure rule | Honest baseline; many real teams ship this and never test against ML |
| Main v1 | **LightGBM binary classifier** with calibrated probabilities (Platt or isotonic) | Tabular time-series with mixed features — tree models dominate. Fast inference. |
| Main v2 (optional) | **PyTorch 1D-CNN over the time-series window** | Captures temporal patterns the tree misses; signals "I can do deep learning" |
| Champion/challenger | New model promoted only if Brier score improves AND p95 latency under 50 ms | Both metrics matter — pure accuracy not enough |

Use Optuna for LightGBM hyperparams (50 trials). Use SHAP for feature attribution at inference time (top-3 contributions in the response).

**Target metric:** Brier score (not AUC — we care about *calibrated probabilities* because the action layer thresholds on `P > 0.7`). Secondary: precision @ recall=0.8 (don't want to spam the on-call channel).

---

## Phased implementation (~6 weeks part-time)

| Phase | Days | Deliverable | Acceptance |
|---|---|---|---|
| 0. Bootstrap from template | 1 | Fork `mlops-reference-template`; rename `mlops_template` → `cluster_canary`; verify `make check` green | `make check` passes — **✅ done 2026-05-18** |
| 1. Data pipeline (synthetic) | 4 | `kind` cluster + workload manifests + `chaos-mesh` chaos plans + Prometheus scraper → parquet + OOM label generator | 24h of synthetic data captured, > 100 OOM events labeled — **✅ code-complete 2026-05-18; runtime pending (24 h offline run via `make lab-up && make lab-scrape`). See [`docs/lab.md`](lab.md).** |
| 2. Data pipeline (real) | 3 | Alibaba trace 2018 sample (~30 GB) ingested via DVC; harmonize schema with synthetic | Both datasets aligned on shared feature set — **✅ code-complete 2026-05-18; runtime pending (`make alibaba-pipeline` for the ~30 GB download). See [`docs/alibaba_schema_alignment.md`](alibaba_schema_alignment.md).** |
| 3. Features | 3 | `src/cluster_canary/features/` — rolling windows, image-lineage features, label generation (`event_within_30min`) | Leakage audit green on temporal split — **✅ code-complete 2026-05-18; runtime pending (`make features` after Phase 1/2 outputs). See [`docs/features.md`](features.md).** |
| 4. Modeling | 5 | Baseline + LightGBM, MLflow tracked, calibration, SHAP. Sliced metrics by namespace and workload type. | Brier ≤ 0.10, precision @ recall=0.8 ≥ 0.5 — **✅ code-complete 2026-05-18; runtime pending (`make train` after Phase 3 outputs). See [`docs/modeling.md`](modeling.md).** |
| 5. Serving (in-cluster) | 4 | BentoML gRPC service; Helm chart; deployed to local kind | p95 < 50 ms at 1000 RPS |
| 6. Action router (v2) | 3 | Python service routing predictions to PagerDuty / kubeai-ops webhook / VPA | End-to-end demo: chaos-mesh injects leak → prediction → notification fires before OOM |
| 7. Monitoring | 3 | Evidently drift on features + predictions + calibration; Grafana dashboard | Drift run produces report; dashboard shows live RPS, latency, drift share |
| 8. Retrain loop | 3 | Prefect retrain flow; champion/challenger gate; drift-triggered | Inject workload drift → retrain triggers → new model auto-promoted only if better |
| 9. Polish | 2 | README with results, ADRs, blog post, demo video | Recruiter can grok the project in 60 s |

**Total:** ~31 days part-time (similar scope to the original template plan, but every phase ships actual portfolio signal).

---

## Story for the README (target)

> **cluster-canary** predicts which Kubernetes pods will OOM-kill or crash within the next 30 minutes, so the on-call engineer (or an automated remediator like [kubeai-ops](https://github.com/sharankumarreddyk/kubeai-ops)) can act before users see an outage.
>
> Trained on 24 hours of synthetic chaos-tested workloads plus an 8-day sample of the public Alibaba production trace. Serves predictions in-cluster as a sidecar at p95 < 50 ms. Auto-retrains on workload drift via a Prefect flow gated by champion/challenger evaluation.
>
> Brier score: 0.084. Precision at recall=0.8: 0.61. Avoids ~73 % of OOMKills in the test workload by surfacing the prediction 12+ minutes ahead of the kill.

---

## Headline numbers to chase (and report honestly)

For the README "Results" section. **Don't lie** — if your numbers come in worse, report what you got and explain why.

- **Brier score**: target ≤ 0.10 (well-calibrated)
- **Precision @ recall=0.8**: target ≥ 0.5
- **Lead time** (median minutes between first `P > 0.7` and the actual OOM event): target ≥ 10 min
- **p95 inference latency**: ≤ 50 ms
- **Sustained throughput**: ≥ 5 000 predictions/sec (one pod, 4 CPU)
- **Memory footprint per pod served**: ≤ 300 MB
- **Cold-start time**: ≤ 5 s
- **Drift detection lead time**: workload shift detected within 24 h of onset
- **Retrain end-to-end** (drift → promotion): ≤ 4 h

---

## Risks / pitfalls (decide in advance)

1. **Synthetic data is too clean.** Chaos-mesh chaos may not reflect real-world workload patterns. Mitigation: validate against Alibaba trace; report metrics on both separately.
2. **Class imbalance.** OOMKills are <1 % of all pod-minutes. Mitigation: use class weights or focal loss; report PR-AUC not just ROC-AUC.
3. **Label leakage from restart counts.** A pod that recently restarted is more likely to OOM again. Make sure the label window (`event_within_30min`) is strictly future of feature window. Leakage audit catches this.
4. **Inference latency under load.** A sidecar that's slow degrades the cluster. Mitigation: budget < 50 ms hard; if exceeded, batch and predict every 60s instead of every 30s.
5. **Deployment complexity.** Running in-cluster ML is harder than HTTP REST. Mitigation: target kind first, EKS/GKE as stretch goal.
6. **Privacy / sensitive data.** Pod names, namespaces, image hashes can leak business intent. For portfolio demo, use synthetic data only in screenshots; for the real run, use Alibaba trace which is already anonymized.

---

## Next session — bootstrap steps

When ready to start coding (in this directory `/Users/sharan/Documents/cluster-canary/`):

1. Fork or clone `mlops-reference-template` content into here (don't `gh repo create` yet — work locally first).
2. Follow `docs/USING_THIS_TEMPLATE.md` step-by-step:
   - `NEW_PKG="cluster_canary"`, `NEW_DIST="cluster-canary"`
   - Rename via the sed snippet
   - `uv sync --extra dev`
   - `make check` green
3. Customize `tests/test_no_leakage.py`:
   - `TARGET_COL = "event_within_30min"` (the label column)
   - `TIME_COL = "scrape_timestamp"`
   - `DENYLIST = {"restart_count_post", "future_oom_count"}` — anything that's a function of the future label
4. Rewrite `docs/PLAN.md` with the 10-phase plan above (or paste this file into it).
5. Phase 0: scaffold the new repo. Initial commit. `gh repo create sharankumarreddyk/cluster-canary --public --source=. --push`.
6. Phase 1: stand up the kind cluster + chaos plans + Prometheus scraper.

Estimated effort to first commit: 30–45 minutes. Phase 1 (synthetic data pipeline) takes the longest — block out a weekend.

---

## Skills coupling

The five custom skills under `.claude/skills/` (in the template) apply to this project unchanged:

| Skill | Applies to cluster-canary how |
|---|---|
| `mlflow-tracking-conventions` | Required tags include `chaos_plan_version`, `data_window_start/end`, `model_family` |
| `feature-leakage-detector` | Specifically protects against using future restart counts or post-event metrics as features |
| `evidently-drift-runner` | Reference = the synthetic+Alibaba dataset the prod model was trained on; current = last 24h of real-cluster traces |
| `bentoml-service-scaffolder` | gRPC mode, `max_batch_size=128`, target p95 < 50 ms |
| `model-card-writer` | Sliced metrics by namespace, workload type, image base layer; explicit "out-of-scope: physical infrastructure failures" |
