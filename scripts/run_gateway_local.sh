#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "[run_gateway_local] missing virtual environment at ${REPO_ROOT}/.venv" >&2
  exit 1
fi

source .venv/bin/activate

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:19100}"
export GATEWAY_TOKEN="${GATEWAY_TOKEN:-local-dev-token}"
export GATEWAY_LOG_PATH="${GATEWAY_LOG_PATH:-./logs/gateway.jsonl}"
export GATEWAY_PORT="${GATEWAY_PORT:-18080}"

mkdir -p logs results

echo "[run_gateway_local] repo root: ${REPO_ROOT}"
echo "[run_gateway_local] vllm base url: ${VLLM_BASE_URL}"
echo "[run_gateway_local] gateway port: ${GATEWAY_PORT}"
echo "[run_gateway_local] gateway log path: ${GATEWAY_LOG_PATH}"

exec uvicorn gateway.main:app --host 0.0.0.0 --port "${GATEWAY_PORT}"
