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

read_gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 \
    | tr -d '[:space:]'
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

wait_for_vllm_health() {
  local url="$1"
  local pid="$2"
  local max_attempts="${3:-180}"
  local attempt=1
  while (( attempt <= max_attempts )); do
    if curl --noproxy "*" -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[verify] vLLM exited before becoming healthy" >&2
      echo "[verify] last vLLM log lines:" >&2
      tail -n 80 logs/vllm.stdout.log >&2 || true
      return 1
    fi
    sleep 2
    ((attempt++))
  done
  echo "[verify] timed out waiting for vLLM health" >&2
  echo "[verify] last vLLM log lines:" >&2
  tail -n 80 logs/vllm.stdout.log >&2 || true
  return 1
}

VLLM_PORT="${VLLM_PORT:-$(pick_port)}"
GATEWAY_PORT="${GATEWAY_PORT:-$(pick_port)}"
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}"
GATEWAY_BASE_URL="http://127.0.0.1:${GATEWAY_PORT}"
GATEWAY_TOKEN="${GATEWAY_TOKEN:-local-dev-token}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
export LOW_VRAM_MODE="${LOW_VRAM_MODE:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
export DTYPE="${DTYPE:-half}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-2}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-256}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-4096}"

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

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[verify] gpu snapshot before launch"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader \
    | tee "logs/phase1_nvidia_smi_before.txt"

  GPU_FREE_MB="$(read_gpu_free_mb)"
  if [[ -n "${GPU_FREE_MB}" ]] && [[ "${GPU_FREE_MB}" =~ ^[0-9]+$ ]] && (( GPU_FREE_MB < MIN_GPU_FREE_MB )); then
    echo "[verify] insufficient free GPU memory: ${GPU_FREE_MB} MiB < ${MIN_GPU_FREE_MB} MiB" >&2
    echo "[verify] release GPU memory or lower MAX_MODEL_LEN / GPU_MEMORY_UTILIZATION before retrying" >&2
    exit 2
  fi
fi

echo "[verify] starting vLLM on ${VLLM_PORT}"
VLLM_PORT="${VLLM_PORT}" ./scripts/run_vllm_local.sh > logs/vllm.stdout.log 2>&1 &
VLLM_PID=$!

echo "[verify] waiting for vLLM health"
wait_for_vllm_health "${VLLM_BASE_URL}/health" "${VLLM_PID}"

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
