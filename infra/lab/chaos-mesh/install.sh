#!/usr/bin/env bash
# Idempotent chaos-mesh install for the kind canary-lab cluster.
#
# Prerequisites:
#   - kind cluster `canary-lab` is running (`make lab-up-kind`).
#   - kubectl context points at it.
#   - helm 3.x on PATH.
#
# Re-runnable: if chaos-mesh is already installed, this upgrades to the pinned
# version with the same values; if not, it installs fresh.

set -euo pipefail

CHAOS_NS="${CHAOS_NS:-chaos-mesh}"
CHAOS_VERSION="${CHAOS_VERSION:-2.7.2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALUES="${SCRIPT_DIR}/values.yaml"

echo "[chaos-mesh] target ns=${CHAOS_NS}, chart version=${CHAOS_VERSION}"

if ! helm repo list 2>/dev/null | grep -q '^chaos-mesh\s'; then
  helm repo add chaos-mesh https://charts.chaos-mesh.org
fi
helm repo update chaos-mesh >/dev/null

kubectl get ns "${CHAOS_NS}" >/dev/null 2>&1 || kubectl create ns "${CHAOS_NS}"

# PSA label — chaos-daemon runs privileged (cgroup + netns manipulation).
kubectl label ns "${CHAOS_NS}" \
  pod-security.kubernetes.io/enforce=privileged \
  --overwrite

helm upgrade --install chaos-mesh chaos-mesh/chaos-mesh \
  --namespace "${CHAOS_NS}" \
  --version "${CHAOS_VERSION}" \
  --values "${VALUES}" \
  --wait \
  --timeout 5m

echo "[chaos-mesh] waiting for chaos-daemon DaemonSet to be ready..."
kubectl -n "${CHAOS_NS}" rollout status ds/chaos-daemon --timeout=3m

echo "[chaos-mesh] installed."
kubectl -n "${CHAOS_NS}" get pods
