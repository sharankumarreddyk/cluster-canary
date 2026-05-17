# The cluster-canary data lab

A local Kubernetes lab that injects realistic failures and scrapes Prometheus into parquet, producing the synthetic training data for the Phase 4 model.

## What it spins up

| Component | Where | Purpose |
|---|---|---|
| `kind` cluster `canary-lab` | local Docker | 1 control-plane + 3 workers, K8s 1.30, cgroup v2 |
| `chaos-mesh` 2.7 | `chaos-mesh/` ns | Failure injection (CRD-driven) |
| `kube-state-metrics` v2.13 | `observability/` ns | OOM ground truth via `last_terminated_reason` |
| Prometheus 2.55 | `observability/` ns | 15 s scrape, 6 h retention, NodePort 30090 → host 9090 |
| Workloads | `workloads/` ns | nginx · postgres · redis · leaky-flask · go-batch · fluent-bit |
| Chaos plans | `workloads/` ns | `instant-oom` (one-shot on redis), `random-crash` (Schedule on leaky-flask) |

## Prerequisites

- Docker Desktop with **cgroup v2** (default on recent versions; verify with `docker info | grep -i cgroup`) and ~8 GB allocated to the VM.
- [`kind`](https://kind.sigs.k8s.io/) ≥ 0.24.
- `kubectl` and `helm` on PATH.
- (Already wired) Python 3.12 via `uv`.

If anything is missing, install with Homebrew:
```bash
brew install kind kubectl helm
```

## Bring the lab up

```bash
make lab-up     # ~3-4 min on first run (kind boot + helm install + image build)
make lab-status # all pods should be Running
```

What that does, in order:

1. `make lab-up-kind` — creates the kind cluster from `infra/lab/kind-config.yaml` (idempotent — re-running is fine).
2. `make lab-up-chaos` — `helm install` chaos-mesh 2.7.2 with our `values.yaml` (pinned `chaosDaemon.runtime=containerd` — the #1 setup failure mode if missing).
3. `make lab-up-observability` — applies the `observability` namespace, kube-state-metrics, and Prometheus.
4. `make lab-up-workloads` — builds the leaky-flask image, loads it into kind via `kind load docker-image`, applies all six workloads.
5. `make lab-up-chaos-plans` — applies `instant-oom.yaml` and `random-crash.yaml`.

Prometheus is reachable at <http://localhost:9090>.

## Generate data

Once the lab has been running for some time (ideally several hours for a good OOM-event yield), run the scraper:

```bash
make lab-scrape   # default: scrape the last 24h into data/raw/scrape/
```

To scrape an explicit window:
```bash
SCRAPE_START=2026-05-18T00:00 \
SCRAPE_END=2026-05-19T00:00 \
make lab-scrape
```

Output layout:
```
data/raw/scrape/
└── dt=2026-05-18/
    ├── hour=00/metrics.parquet
    ├── hour=01/metrics.parquet
    └── ...
data/interim/labeled/
└── labeled.parquet            # wide-form + event_within_30min label
reports/eval/label_metrics.json # summary: n_rows, n_positive, positive_pct, …
```

The scraper is **idempotent** — re-running over the same window overwrites partitions deterministically. The label generator unions overlapping OOM windows and censors post-event rows automatically.

## Verify the lab is producing OOM signal

After ~30 min of runtime:

```bash
# Watch for OOMKilled containers:
kubectl -n workloads get pods -w

# Confirm cAdvisor records OOM events:
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=container_oom_events_total > 0' | jq .

# Confirm kube-state-metrics shows the OOMKilled reason:
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}' | jq .
```

Both queries should return non-empty results once the leaky-flask pods (or chaos-mesh) have produced their first OOMs.

## Tear down

```bash
make lab-down       # delete the kind cluster (preserves nothing — full reset)
```

For a clean rebuild without losing local images:
```bash
make lab-reset      # = lab-down + lab-up
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `chaos-daemon` pods crash-looping with "cannot connect to runtime" | Confirm `chaosDaemon.runtime=containerd` in `infra/lab/chaos-mesh/values.yaml` |
| All pods `Pending` immediately after `lab-up` | Docker Desktop is out of resources — raise the VM memory to ≥ 8 GB |
| Prometheus scrape targets `down` for cAdvisor | Wait — kubelet's `/metrics/cadvisor` endpoint takes ~60 s after node-ready |
| `last_terminated_reason` always says `Error`, never `OOMKilled` | Older kube-state-metrics — confirm v2.13+ image (`registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.13.0`) |
| `make lab-scrape` returns "no series found" | Prometheus needs a few minutes to populate; verify with the curl commands above first |
| MacBook fan never stops | Normal — the chaos-mesh-induced stress is real CPU/memory work. Disable sleep during long data runs. |

## What's NOT in this lab

- A real metrics-history backend — Prometheus retains 6 h; durable storage is the parquet at `data/raw/scrape/`.
- A LoadBalancer — Prometheus is exposed via `NodePort 30090` which kind maps to host `9090`. If you need other ports, edit `infra/lab/kind-config.yaml` `extraPortMappings`.
- Grafana — out of scope for Phase 1. The Phase 6 monitoring deliverable adds it.

See [`PHASE_1_CONTEXT.md`](PHASE_1_CONTEXT.md) for the design decisions behind every choice here.
