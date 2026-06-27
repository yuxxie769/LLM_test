#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "[run_vllm_local] missing virtual environment at ${REPO_ROOT}/.venv" >&2
  exit 1
fi

source .venv/bin/activate

export PATH=/usr/lib/wsl/lib:${PATH}
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

DEFAULT_CUDA_HOME="${REPO_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="${CUDA_HOME:-${DEFAULT_CUDA_HOME}}"
if [[ -d "${CUDA_HOME}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export CUDACXX="${CUDA_HOME}/bin/nvcc"
fi

MODEL_DIR="${MODEL_DIR:-/mnt/d/models/qwen2.5-7b-awq}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-7b-awq-local}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-19100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[run_vllm_local] model directory not found: ${MODEL_DIR}" >&2
  exit 1
fi

mkdir -p logs results

echo "[run_vllm_local] repo root: ${REPO_ROOT}"
echo "[run_vllm_local] model dir: ${MODEL_DIR}"
echo "[run_vllm_local] served model name: ${SERVED_MODEL_NAME}"
echo "[run_vllm_local] host: ${VLLM_HOST}"
echo "[run_vllm_local] port: ${VLLM_PORT}"
echo "[run_vllm_local] max model len: ${MAX_MODEL_LEN}"
echo "[run_vllm_local] gpu memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "[run_vllm_local] HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "[run_vllm_local] TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "[run_vllm_local] VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
echo "[run_vllm_local] CUDA_HOME=${CUDA_HOME}"
echo "[run_vllm_local] CUDACXX=${CUDACXX:-unset}"

exec vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
