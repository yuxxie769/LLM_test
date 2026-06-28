#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "[run_vllm_local] missing virtual environment at ${REPO_ROOT}/.venv" >&2
  exit 1
fi

if [[ ! -x ".venv/bin/python3" ]]; then
  echo "[run_vllm_local] broken virtual environment: ${REPO_ROOT}/.venv/bin/python3 is missing or not executable" >&2
  echo "[run_vllm_local] rebuild .venv and reinstall dependencies before starting vLLM" >&2
  exit 1
fi

source .venv/bin/activate

if ! command -v vllm >/dev/null 2>&1; then
  echo "[run_vllm_local] vllm is not installed in ${REPO_ROOT}/.venv" >&2
  exit 1
fi

VLLM_SERVE_HELP="$(vllm serve --help 2>/dev/null || true)"
VLLM_SUPPORTS_OFFLOAD_BACKEND=0
if printf '%s' "${VLLM_SERVE_HELP}" | rg -q -- '--offload-backend'; then
  VLLM_SUPPORTS_OFFLOAD_BACKEND=1
fi

export PATH=/usr/lib/wsl/lib:${PATH}
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEFAULT_CUDA_HOME="${REPO_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="${CUDA_HOME:-${DEFAULT_CUDA_HOME}}"
if [[ -d "${CUDA_HOME}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export CUDACXX="${CUDA_HOME}/bin/nvcc"
fi

HAS_MAX_MODEL_LEN="${MAX_MODEL_LEN+1}"
HAS_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION+1}"
HAS_DTYPE="${DTYPE+1}"
HAS_VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER+1}"
HAS_VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB+1}"
HAS_VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS+1}"
HAS_VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS+1}"

DEFAULT_MODEL_DIR="/root/models/qwen2.5-0.5b"
if [[ -d "/root/autodl-tmp/qwen2.5-0.5b" ]]; then
  DEFAULT_MODEL_DIR="/root/autodl-tmp/qwen2.5-0.5b"
fi
MODEL_DIR="${MODEL_DIR:-${DEFAULT_MODEL_DIR}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-05b-local}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-19100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
DTYPE="${DTYPE:-half}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_DISABLE_LOG_STATS="${VLLM_DISABLE_LOG_STATS:-0}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
VLLM_OFFLOAD_BACKEND="${VLLM_OFFLOAD_BACKEND:-auto}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"

if [[ "${LOW_VRAM_MODE:-0}" == "1" ]]; then
  if [[ -z "${HAS_MAX_MODEL_LEN}" ]]; then
    MAX_MODEL_LEN="256"
  fi
  if [[ -z "${HAS_GPU_MEMORY_UTILIZATION}" ]]; then
    GPU_MEMORY_UTILIZATION="0.45"
  fi
  if [[ -z "${HAS_DTYPE}" ]]; then
    DTYPE="half"
  fi
  if [[ -z "${HAS_VLLM_ENFORCE_EAGER}" ]]; then
    VLLM_ENFORCE_EAGER="1"
  fi
  if [[ -z "${HAS_VLLM_CPU_OFFLOAD_GB}" ]]; then
    VLLM_CPU_OFFLOAD_GB="4"
  fi
  if [[ -z "${HAS_VLLM_MAX_NUM_SEQS}" ]]; then
    VLLM_MAX_NUM_SEQS="1"
  fi
  if [[ -z "${HAS_VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
    VLLM_MAX_NUM_BATCHED_TOKENS="256"
  fi
fi

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
echo "[run_vllm_local] dtype: ${DTYPE}"
echo "[run_vllm_local] low_vram_mode: ${LOW_VRAM_MODE:-0}"
echo "[run_vllm_local] vllm_enforce_eager: ${VLLM_ENFORCE_EAGER}"
echo "[run_vllm_local] vllm_cpu_offload_gb: ${VLLM_CPU_OFFLOAD_GB}"
echo "[run_vllm_local] vllm_offload_backend: ${VLLM_OFFLOAD_BACKEND}"
echo "[run_vllm_local] vllm_max_num_seqs: ${VLLM_MAX_NUM_SEQS:-unset}"
echo "[run_vllm_local] vllm_max_num_batched_tokens: ${VLLM_MAX_NUM_BATCHED_TOKENS:-unset}"
echo "[run_vllm_local] vllm_disable_log_stats: ${VLLM_DISABLE_LOG_STATS}"
echo "[run_vllm_local] HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "[run_vllm_local] TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "[run_vllm_local] VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
echo "[run_vllm_local] VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}"
echo "[run_vllm_local] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[run_vllm_local] CUDA_HOME=${CUDA_HOME}"
echo "[run_vllm_local] CUDACXX=${CUDACXX:-unset}"

COMMAND=(
  vllm
  serve
  "${MODEL_DIR}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
)

if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
  COMMAND+=(--enforce-eager)
fi

if [[ "${VLLM_DISABLE_LOG_STATS}" == "1" ]]; then
  COMMAND+=(--disable-log-stats)
fi

if [[ "${VLLM_CPU_OFFLOAD_GB}" != "0" ]]; then
  if [[ "${VLLM_SUPPORTS_OFFLOAD_BACKEND}" == "1" ]]; then
    COMMAND+=(--offload-backend "${VLLM_OFFLOAD_BACKEND}")
  fi
  COMMAND+=(--cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB}")
fi

if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
  COMMAND+=(--max-num-seqs "${VLLM_MAX_NUM_SEQS}")
fi

if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
  COMMAND+=(--max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}")
fi

echo "[run_vllm_local] command: ${COMMAND[*]}"

exec "${COMMAND[@]}"
