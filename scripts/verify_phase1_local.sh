#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
source .venv/bin/activate

pick_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

wait_for_http() {
  local url="$1"
  local max_attempts="${2:-180}"
  local attempt=1
  while (( attempt <= max_attempts )); do
    if curl --noproxy "*" -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    ((attempt++))
  done
  return 1
}

VLLM_PORT="${VLLM_PORT:-$(pick_port)}"
GATEWAY_PORT="${GATEWAY_PORT:-$(pick_port)}"
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}"
GATEWAY_BASE_URL="http://127.0.0.1:${GATEWAY_PORT}"
GATEWAY_TOKEN="${GATEWAY_TOKEN:-local-dev-token}"

mkdir -p logs results

cleanup() {
  set +e
  if [[ -n "${GATEWAY_PID:-}" ]]; then
    kill "${GATEWAY_PID}" 2>/dev/null || true
    wait "${GATEWAY_PID}" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[verify] starting vLLM on ${VLLM_PORT}"
VLLM_PORT="${VLLM_PORT}" ./scripts/run_vllm_local.sh > logs/vllm.stdout.log 2>&1 &
VLLM_PID=$!

echo "[verify] waiting for vLLM health"
wait_for_http "${VLLM_BASE_URL}/health"

echo "[verify] starting gateway on ${GATEWAY_PORT}"
VLLM_BASE_URL="${VLLM_BASE_URL}" GATEWAY_PORT="${GATEWAY_PORT}" \
  GATEWAY_TOKEN="${GATEWAY_TOKEN}" ./scripts/run_gateway_local.sh \
  > logs/gateway.stdout.log 2>&1 &
GATEWAY_PID=$!

echo "[verify] waiting for gateway healthz"
wait_for_http "${GATEWAY_BASE_URL}/healthz"

echo "[verify] running smoke tests"
VLLM_BASE_URL="${VLLM_BASE_URL}" \
GATEWAY_BASE_URL="${GATEWAY_BASE_URL}" \
GATEWAY_TOKEN="${GATEWAY_TOKEN}" \
./scripts/smoke_test_phase1.sh

echo "[verify] phase1 verification complete"
