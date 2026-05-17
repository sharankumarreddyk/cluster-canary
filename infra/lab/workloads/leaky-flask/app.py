"""Self-leaking Flask service for cluster-canary training data generation.

Behavior
--------
- Allocates `LEAK_RATE_MB_PER_MIN` MiB of memory per minute into a process-global
  buffer, simulating a real-world Python memory leak (cache miss, reference
  cycle, missing close, etc.).
- Exposes Prometheus metrics at /metrics so the model can see app-level signal
  alongside cgroup-level memory metrics.
- /healthz always returns 200 (independent of memory pressure) so we don't get
  a confounding liveness signal.
- When the container's cgroup limit is hit, the kernel OOM-killer fires and
  the pod restarts — the chaos-mesh-driven instant-OOM uses Redis, not this
  service. This service produces the SLOW-LEAK signal that's the model's main
  predictive target (30 min lead time).

Env vars (all optional):
- LEAK_RATE_MB_PER_MIN  (default: 4)
- LEAK_START_AFTER_SEC  (default: 60)   — delay before leak begins (avoid spurious early OOM)
- LEAK_MAX_MB           (default: 1024) — safety cap
- WORKER_ID             (default: hostname suffix)
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time

from flask import Flask, jsonify
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    generate_latest,
)

LEAK_RATE_MB_PER_MIN = int(os.environ.get("LEAK_RATE_MB_PER_MIN", "4"))
LEAK_START_AFTER_SEC = int(os.environ.get("LEAK_START_AFTER_SEC", "60"))
LEAK_MAX_MB = int(os.environ.get("LEAK_MAX_MB", "1024"))
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname())

_ONE_MIB = 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("leaky-flask")

app = Flask(__name__)

# In-memory accumulator — the leak.
_leak_buffer: list[bytearray] = []
_leak_buffer_lock = threading.Lock()

leak_size_mb = Gauge(
    "leakyflask_leak_size_mib",
    "Size of the in-memory leak buffer.",
    ["worker"],
)
requests_total = Counter(
    "leakyflask_requests_total",
    "Requests served.",
    ["worker", "endpoint", "status"],
)


def _leak_forever() -> None:
    log.info("leak thread starting, rate=%d MiB/min", LEAK_RATE_MB_PER_MIN)
    time.sleep(LEAK_START_AFTER_SEC)
    sleep_per_mib = max(60.0 / max(LEAK_RATE_MB_PER_MIN, 1), 0.01)
    while True:
        with _leak_buffer_lock:
            if sum(len(b) for b in _leak_buffer) // _ONE_MIB >= LEAK_MAX_MB:
                log.warning("leak cap reached at %d MiB; idling", LEAK_MAX_MB)
                time.sleep(60)
                continue
            _leak_buffer.append(bytearray(_ONE_MIB))
            total_mib = sum(len(b) for b in _leak_buffer) // _ONE_MIB
            leak_size_mb.labels(worker=WORKER_ID).set(total_mib)
        time.sleep(sleep_per_mib)


@app.route("/")
def index() -> tuple[str, int]:
    requests_total.labels(worker=WORKER_ID, endpoint="/", status="200").inc()
    return jsonify({"worker": WORKER_ID, "leak_mib": _current_mib()}), 200


@app.route("/healthz")
def healthz() -> tuple[str, int]:
    return jsonify({"status": "ok"}), 200


@app.route("/readyz")
def readyz() -> tuple[str, int]:
    return jsonify({"status": "ready", "worker": WORKER_ID}), 200


@app.route("/metrics")
def metrics() -> tuple[bytes, int, dict[str, str]]:
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


def _current_mib() -> int:
    with _leak_buffer_lock:
        return sum(len(b) for b in _leak_buffer) // _ONE_MIB


def _start_leaker() -> None:
    t = threading.Thread(target=_leak_forever, name="leaker", daemon=True)
    t.start()


_start_leaker()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # noqa: S104 — pod-local container
