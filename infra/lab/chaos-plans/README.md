# Chaos plans

| File | Trigger | Target | Label produced |
|---|---|---|---|
| `instant-oom.yaml` | one-shot StressChaos | `chaos-target: instant-oom` (redis) | `OOMKilled` (clean positive) |
| `random-crash.yaml` | Schedule (every 7 min) | `canary.lab/profile: leaky-app` | non-OOM SIGKILL (CrashLoopBackOff signal) |

## Where's the slow-leak chaos plan?

The slow-leak signal is produced by the **leaky-flask workload itself**, not by
chaos-mesh. Rationale:

1. chaos-mesh 2.7 `StressChaos` has no built-in memory ramp — `stressors.memory.size`
   is a fixed allocation, not a growth rate. Producing a true ramp via chaos-mesh
   requires chaining 5–6 `StressChaos` steps in a `Workflow`, which is more
   plumbing for less realism.
2. `leaky-flask` allocates `LEAK_RATE_MB_PER_MIN` of memory per minute via a
   real-Python `bytearray` accumulator — exactly the shape of a production memory
   leak. Combined with the deployment's 384 MiB limit, this produces an
   `OOMKilled` event roughly every 90 minutes with a smooth pre-failure trajectory.

The 30-min lead-time signal we want the model to learn lives in that smooth
trajectory — `instant-oom` provides cleanly-labeled positive samples for
calibration, `random-crash` provides non-OOM-kill negative-but-failure signal,
and `leaky-flask`'s self-inflicted OOMs provide the realistic ramp that
matches what production memory leaks look like.

## Applying / removing

```bash
# Apply during data generation:
kubectl apply -f infra/lab/chaos-plans/

# Trigger an extra instant-OOM on demand:
kubectl delete -f infra/lab/chaos-plans/instant-oom.yaml --ignore-not-found
kubectl apply  -f infra/lab/chaos-plans/instant-oom.yaml

# Pause chaos (keep workloads running, stop kills):
kubectl delete -f infra/lab/chaos-plans/
```

## Known issues

- chaos-mesh 2.7 + containerd 1.7.x occasionally leaves `stress-ng` zombies
  after `duration` expires. Symptom: target pod stays at high RSS after the
  chaos window closes. Workaround: `kubectl -n chaos-mesh rollout restart ds/chaos-daemon`.
- `Schedule + concurrencyPolicy: Forbid` sometimes retains history entries past
  `historyLimit`. Harmless; clean with `kubectl delete podchaos -n workloads -l 'managed-by=chaos-mesh-schedule'`.
