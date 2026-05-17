# Phase 1 Context — Data lab + traces + OOM signals

Synthesized from four background research passes. Use this as the working brief before writing Phase 1 code (the local kind+chaos-mesh data lab, the Alibaba trace ingest, and the OOM-label generator).

---

## TL;DR — three decisions locked in

1. **Phase 1 data layer = local `kind` + `chaos-mesh` + Prometheus → parquet (Python scraper, not `remote_write`)**. Phase 2 brings in the Alibaba 2018 trace for generalization.
2. **OOMKill ground truth = `container_oom_events_total` (cAdvisor, Prometheus) cross-checked with `kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}` (kube-state-metrics v2.13+)**. Label window = every row in `[T_oom - 30 min, T_oom)` is positive.
3. **Scrape interval = 15 s, not 30 s.** 30 s undersamples derivative trends; we need ~120 samples per 30-min window to recover useful rolling features.

---

## 1. The local data lab (Phase 1 primary)

### kind cluster (sizes for a 16 GB MacBook)
- 1 control-plane + 3 workers, K8s 1.30.x.
- Allocate ~8 GB to Docker Desktop. cgroup v2 required (`docker info | grep -i cgroup`).
- Enable `MemoryQoS` feature gate — without it, `StressChaos` memory pressure simulation is degraded (no `memory.high` throttling).
- `maxPods: 60` per worker (default 110 is overkill).
- `evictionHard: memory.available<200Mi` so node-pressure kicks in realistically.

### chaos-mesh
- Chart version 2.7.x (Helm: `chaos-mesh/chaos-mesh`).
- **Non-negotiable:** `chaosDaemon.runtime=containerd`, `socketPath=/run/containerd/containerd.sock`. kind uses containerd, not Docker — this is the #1 install failure mode.
- For PSA-restricted namespaces, label your lab namespace with `pod-security.kubernetes.io/enforce=privileged`.

Three chaos plans to drive Phase 1's labeled dataset:
- **Slow memory leak** (`StressChaos` with rotating `size: "50MB"` over 30 min) — the most useful signal for 30-min-ahead prediction.
- **Instant OOM** (`StressChaos` `size: "1GB"` exceeding container limit) — clean label generator.
- **Random crash + jitter** (`PodChaos` `pod-kill` at `random-max-percent: 20`, cron `@every 7m`) — secondary CrashLoopBackOff labels.

### Workload mix (realistic, not microservices-demo)
| Workload | Replicas | CPU r/l | Mem r/l | Purpose |
|---|---|---|---|---|
| nginx | 3 | 50m/200m | 64/128 MiB | Well-behaved baseline |
| PostgreSQL | 1 | 200m/500m | 256/512 MiB | Stateful, slow start, real disk |
| Redis | 1 | 100m/300m | 128/256 MiB | OOM-prone, good crash target |
| Flask + gunicorn (leaky) | 2 | 100m/400m | 128/384 MiB | Inject `tracemalloc` leak via env |
| Go batch worker | 2 | 200m/800m | 64/128 MiB | CPU spikes, GC pauses |
| fluent-bit | DaemonSet | 50m/100m | 64/128 MiB | Background log shipper |

Total request: ~2 CPU, ~1.5 GB — fits with Prometheus + chaos-mesh inside an 8 GB Docker VM.

### Prometheus + scrape strategy
- **15 s scrape interval** for `cadvisor`, `kubelet`, `kube-state-metrics`. Don't go below 10 s (cAdvisor samples at ~10 s; you get aliasing).
- Retention 6 h is enough; the offline parquet is the durable store.
- **Pin `kube-state-metrics` to v2.13+** — older versions emit `Error` instead of `OOMKilled` in `last_terminated_reason`.

### Metrics → parquet via Prometheus HTTP API
Use `prometheus-api-client` + `pyarrow`, not `remote_write`. Reasons:
- Simpler — no Thanos / Cortex / sidecar.
- Re-queryable when feature definitions change.
- Parquet partitioning by hour/day falls out naturally.

Pattern: every 10 min, query last 15 min (with 1-min overlap) for the metric set, append a parquet row group partitioned by `dt=YYYY-MM-DD/hour=HH`.

### Pitfalls
- kind LoadBalancer doesn't work on macOS — use NodePort + `kubectl port-forward`, or install `cloud-provider-kind` / MetalLB.
- Prometheus disk fills fast on kind ephemeral storage — 6 h retention or `hostPath` PV.
- macOS sleep mid-experiment = Prometheus gaps that look like outages. Disable sleep during data-gen runs.
- Container-ID reuse: kind worker nodes recycle container IDs across restarts — when joining usage to events, partition by `(container_id, node_id, first_seen_day)`.

---

## 2. OOMKill ground truth — pick by reliability

Ranked by signal fidelity:

| Rank | Signal | Latency | Notes |
|---|---|---|---|
| 1 | cgroup `memory.events` `oom_kill` counter (cgroup v2) | ms | Kernel's own record. Survives container restart. Highest fidelity but requires node-exporter textfile collector or direct cgroup read. |
| 2 | `container_oom_events_total` (cAdvisor → Prometheus) | one scrape (~15 s) | Built on cgroup events / eBPF tracepoint. Reliable on cAdvisor ≥ 0.39, K8s ≥ 1.21. **Primary choice for cluster-canary.** |
| 3 | `kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}` | ~10 s kubelet sync + ~30 s ksm scrape | Misses repeated OOMs in the same container instance (only LAST recorded). Pair with restart-count rate for CrashLoopBackOff. |
| 4 | K8s Events (`reason=OOMKilling`) | seconds | TTL ~1 h, can be lost under apiserver pressure. Use as enrichment, not primary. |
| 5 | Exit code 137 | same as ksm | SIGKILL only — NOT OOM-specific. Manual `kubectl delete`, liveness-probe kill, evictions all yield 137. Sanity tie-break only. |
| 6 | dmesg / kernel log (`Out of memory: Killed process …`) | real-time | Kernel speaks the truth, but not scraped by default. Requires textfile collector or log-agent tail. |

**Decision for cluster-canary:** primary label from a delta in `container_oom_events_total`; cross-check via `last_terminated_reason{reason="OOMKilled"}` and exit code 137 for sanity. Don't depend on K8s Events.

---

## 3. Label generation — `event_within_30min`

For a pod `p` with an observed OOM event at time `T_oom`:

- **Window-labeling (standard, what we'll use):** every row in `[T_oom - 30 min, T_oom)` is positive. Teaches the model that the whole lead-up is anomalous and lets us compute lead time as the gap between first `P > τ` and `T_oom`.
- **Censoring:** drop or mask rows in `[T_oom, T_oom + cooldown]` — restart counters there leak the label.
- **Repeated OOMs in 30 min:** take the union of windows.
- **Pod-restart caveat:** treat the pod by `pod_uid`, not name — name reuse across restarts is normal.

This convention is consistent with Tang et al. (BackBlaze 2018) and Borg failure-prediction papers (Rosa 2015, El-Sayed 2017). No canonical K8s-specific paper fixes a window — we adopt window-labeling by analogy.

---

## 4. Features that actually predict OOMs

Beyond the obvious (memory used / limit ratio, growth rate, restart count, image hash):

- **Working-set vs. RSS gap** + rate of growth — the OOM-killer evaluates working-set; RSS lags.
- **`container_memory_failcnt`** + major page-fault rate — strongest *leading* indicators (cgroup hit limit, hasn't OOM'd yet).
- **CPU throttle ratio** (`cfs_throttled / cfs_periods`) — protective for some workloads, predictive for JVM/Go GC pressure. Counterintuitive but consistently shows up.
- **Workload-relative features:** this image's OOM rate in the last 24 h; namespace OOM rate; node memory pressure (`node_memory_MemAvailable_bytes` + sum of working-set on the node).
- **Time-of-day / day-of-week** for batch-heavy clusters.
- **Co-tenant memory pressure** on the same node — OOM can be node-level, not pod-level.
- **JVM / Go runtime metrics** when available (heap, GC pause) — dominant when present, rarely cluster-wide.
- **Pod age** is bimodal — encode non-linearly (very young: misconfig OOM; very old: leak).

Weak / counterintuitive: network I/O and disk I/O are weak predictors of *memory* failures (despite being heavily used in disk-failure prediction).

---

## 5. Class imbalance

Expect **< 0.1 %** positive pod-minutes in production. Alibaba-trace derivatives report ~0.1–1 % at minute-resolution with 30-min positive windowing.

**Strategy for LightGBM/XGBoost at ~0.1 % positives:**
1. **Negative downsampling to 5–10 % positive** is the standard. Cheap, and tree models are not very sensitive to the ratio as long as you re-calibrate (Platt or isotonic on a holdout with the ORIGINAL prior).
2. `scale_pos_weight` alone underperforms downsampling — histogram split-finding still sees mostly negatives.
3. Focal loss is for NN, rarely beats downsample + calibrate in tabular GBM.
4. **Do NOT use SMOTE on time-series features** — it synthesizes points that violate temporal causality.

Primary metrics: **Brier score** (calibration matters because the action layer thresholds on `P > 0.7`) and **PR-AUC**. ROC-AUC is misleading at this class balance — relegate to secondary.

---

## 6. Lead time realism

30 min ahead is at the optimistic end of published prior work. The closest analog (Microsoft Narya, OSDI 2020 — VM failure prediction on Azure) reports AUC 0.85 at a 30-min horizon, production-deployed. Borg job-failure papers (Rosa 2015, El-Sayed 2017) usually predict 5–15 min ahead at precision ~0.7, recall ~0.6.

**Decision:** PLAN.md targets at 30 min (Brier ≤ 0.10, precision at recall=0.8 ≥ 0.5) are aggressive. Always also report numbers at 5 / 10 / 15 min so the precision-vs-lead-time tradeoff is visible. The MLflow logging convention in `CLAUDE.md` enforces this.

---

## 7. The traces (Phase 2 + stretch)

### Alibaba Cluster Trace 2018 (Phase 2 primary)

6 CSV files (no headers). Download from `http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/<file>.tar.gz` via `fetchData.sh`. For our use case, fetch:
- `machine_meta` (92 KB, event-driven)
- `container_meta` (2.4 MB, event-driven, **critical — contains the only container-level status signal**)
- `machine_usage` (1.7 GB, non-uniform 10–60 s sampling)
- `container_usage` (28 GB, ~10 s sampling — main feature source)
- `batch_task` (125 MB)

Total: ~30 GB compressed, ~180 GB extracted. Skip `batch_instance.tar.gz` (20 GB) unless we want batch crash labels.

Container in Alibaba's schema ≈ K8s pod-on-node (Sigma is the K8s precursor). `app_du` ≈ Deployment/ReplicaSet.

**No explicit OOMKill label** — closest proxies:
1. `container_meta.status` transitions (enum values undocumented — verify with `cut -d, -f5 container_meta.csv | sort -u`).
2. `batch_instance.status = "Failed"` (batch-side crash proxy).
3. **Container disappearance** (container_id stops emitting in `container_usage` while machine still emits) — strong kill proxy.
4. `mem_util_percent` saturating to 100 in the minutes before disappearance — **best feature for a 30-min-ahead OOM predictor against this dataset**.

### Critical Alibaba gotchas
- **Day-misalignment bug (issue #52):** `machine_*` covers days 1–8 but `container_*` covers days 2–9. Only **days 2–8 (7 days)** are aligned across tables. Filter to that window.
- `machine_usage.csv` has 9 columns, docs say 10 — `disk_usage_percent` does NOT exist. Trust `schema.txt`.
- Sampling interval is non-uniform — **resample explicitly before any rolling-window feature**.
- Memory / CPU normalization factor is undisclosed — features must stay relative; absolute MB / cores cannot be recovered.
- Container-ID reuse over 8 days is undocumented — partition by `(container_id, machine_id, first_seen_day)`.
- Time stamps are seconds-since-start (NO wall clock, NO day-of-week label).

### Google Cluster Trace 2019 (stretch / generalization)

Five tables, BigQuery-only, ~2.4 TiB. For a 100 GB cell-`a` sample on the free tier (1 TiB/month):
- Restrict columns (BigQuery is columnar — ~10× scan reduction).
- Filter to one cell (`clusterdata_2019_a`).
- Use `TABLESAMPLE SYSTEM (n PERCENT)` or `MOD(FARM_FINGERPRINT(...), 100) = 0`.

`instance_events.type` enum: `0 SUBMIT 1 QUEUE 2 ENABLE 3 SCHEDULE 4 EVICT 5 FAIL 6 FINISH 7 KILL 8 LOST 9/10 UPDATE`. **OOMKill analog = `FAIL` (5) where `maximum_usage.memory ≈ resource_request.memory`**. CrashLoopBackOff analog = repeated `FAIL` for the same `(collection_id, instance_index)` in a rolling window.

Borg → K8s mapping: `collection (JOB)` → Deployment/Job, `instance` → Pod, `machine` → Node. Do NOT conflate `collection` ≈ namespace; collection IS the workload.

Use the Google trace in **Phase 2 / Phase 4 generalization-test mode only** — Alibaba is closer to K8s mental model and has CSV access (no BigQuery friction).

---

## 8. Prior art to cite / benchmark against

| Paper / project | Year | Horizon | Performance | Why it matters |
|---|---|---|---|---|
| Rosa, Chen, Wang — "Predicting and Mitigating Jobs Failures…" | 2015 | 5–15 min | P ~0.7, R ~0.6 | Foundational, RF on Google trace |
| El-Sayed et al. — ICDCS 2017 | 2017 | mins | feature-importance ranking | Confirms Rosa across 3 Google clusters |
| Lin et al. — FSE 2018 (Microsoft) | 2018 | hours | learning-to-rank | Azure node failure, feature engineering reusable |
| **Levy et al. — Narya (OSDI 2020, Microsoft Azure)** | 2020 | **30 min** | **AUC 0.85** | Closest cluster-canary analog (GBT + RL action layer, production-deployed) |
| Du et al. — DeepLog (CCS 2017) | 2017 | 5–10 min | F1 0.7–0.9 (curated) | Reference for log-based anomaly detection |
| `alibaba/clusterdata` derivative notebooks | n/a | n/a | n/a | Practical baselines on the Phase 2 dataset |

For the cluster-canary README's "Results" section we will report against Narya's published numbers as the closest comparable.

---

## 9. Open questions to confirm during Phase 1 (don't skip)

1. **Verify `container_oom_events_total` actually increments** in the kind cluster on a `StressChaos` OOM — on very old kernels the metric exists but stays at 0.
2. **Enumerate `kube_pod_container_status_last_terminated_reason` values** observed in the lab to confirm `"OOMKilled"` shows up reliably.
3. **Measure the actual lag** between OOM event time (cAdvisor) and the kube-state-metrics gauge transition. If > 60 s, prefer cAdvisor as canonical event time.
4. **Confirm container-ID reuse behavior** in kind across pod restarts before relying on `container_id` as a key.
5. **Pin** all critical tool versions in the lab manifests: kind 0.24+, K8s 1.30.x, kube-state-metrics v2.13+, cAdvisor (bundled with kubelet 1.30) 0.49+, chaos-mesh 2.7.x, Prometheus 2.55+.

---

## Source attribution

This file is synthesized from four background research passes (kind+chaos-mesh setup; Google Cluster Trace 2019; Alibaba Cluster Trace 2018; OOM signals + labeling). Raw agent outputs preserved at `/tmp/cc_research/0{1,2,3,4}_*.md` for the session; will be discarded with the temp dir. If you need them long-term, copy into `docs/research/`.

Two of the four agents reported they had **no live web access in their session** (Google trace + OOM signals) and answered from training knowledge — treat exact precision/recall numbers and BigQuery row counts as order-of-magnitude. Verify against upstream schema docs (`google/cluster-data/ClusterData2019.md`, `alibaba/clusterdata/cluster-trace-v2018/schema.txt`) before relying on numeric mappings in code.
