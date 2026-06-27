#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
source .venv/bin/activate

pick_port() {
  python - <<'PY' 2>/dev/null || true
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
  local timeout_seconds="$2"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if curl --noproxy "*" -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start_ts >= timeout_seconds )); then
      return 1
    fi
    sleep 2
  done
}

has_vllm_startup_failure() {
  local log_path="$1"
  [[ -f "${log_path}" ]] || return 1

  rg -q "EngineCore failed to start|OutOfMemoryError" "${log_path}"
}

wait_for_vllm_health() {
  local url="$1"
  local pid="$2"
  local timeout_seconds="$3"
  local log_path="$4"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if curl --noproxy "*" -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if has_vllm_startup_failure "${log_path}"; then
      echo "[phase2-smoke] detected fatal vLLM startup error in log" >&2
      echo "[phase2-smoke] last vLLM log lines:" >&2
      tail -n 80 "${log_path}" >&2 || true
      return 1
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[phase2-smoke] vLLM exited before becoming healthy" >&2
      echo "[phase2-smoke] last vLLM log lines:" >&2
      tail -n 80 "${log_path}" >&2 || true
      return 1
    fi
    if (( "$(date +%s)" - start_ts >= timeout_seconds )); then
      echo "[phase2-smoke] timed out waiting for vLLM health" >&2
      echo "[phase2-smoke] last vLLM log lines:" >&2
      tail -n 80 "${log_path}" >&2 || true
      return 1
    fi
    sleep 2
  done
}

AUTO_PORT="$(pick_port)"
VLLM_PORT="${VLLM_PORT:-${AUTO_PORT:-19100}}"
export VLLM_PORT
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
export BENCHMARK_NUM_PROMPTS="${BENCHMARK_NUM_PROMPTS:-4}"
export BENCHMARK_STREAM_NUM_PROMPTS="${BENCHMARK_STREAM_NUM_PROMPTS:-2}"
export LOW_VRAM_MODE="${LOW_VRAM_MODE:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-256}"
export DTYPE="${DTYPE:-half}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-4}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-256}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-4096}"
READINESS_TIMEOUT_SECONDS="${READINESS_TIMEOUT_SECONDS:-600}"
BATCH_RUN_ID="${BATCH_RUN_ID:-phase2smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/batches/${BATCH_RUN_ID}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_DIR}/baseline_summary.md}"
PLOTS_DIR="${PLOTS_DIR:-${OUTPUT_DIR}/plots}"

mkdir -p logs results

cleanup() {
  set +e
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[phase2-smoke] gpu snapshot before launch"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader \
    | tee "logs/phase2_nvidia_smi_before.txt"

  GPU_FREE_MB="$(read_gpu_free_mb)"
  if [[ -n "${GPU_FREE_MB}" ]] && [[ "${GPU_FREE_MB}" =~ ^[0-9]+$ ]] && (( GPU_FREE_MB < MIN_GPU_FREE_MB )); then
    echo "[phase2-smoke] insufficient free GPU memory: ${GPU_FREE_MB} MiB < ${MIN_GPU_FREE_MB} MiB" >&2
    echo "[phase2-smoke] release GPU memory or lower the smoke model before retrying" >&2
    exit 2
  fi
fi

echo "[phase2-smoke] starting vLLM on ${VLLM_BASE_URL}"
echo "[phase2-smoke] low_vram_mode=${LOW_VRAM_MODE}"
echo "[phase2-smoke] max_model_len=${MAX_MODEL_LEN}"
echo "[phase2-smoke] gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "[phase2-smoke] dtype=${DTYPE}"
echo "[phase2-smoke] cpu_offload_gb=${VLLM_CPU_OFFLOAD_GB}"
echo "[phase2-smoke] max_num_seqs=${VLLM_MAX_NUM_SEQS}"
echo "[phase2-smoke] max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}"
VLLM_LOG_PATH="logs/phase2_vllm_smoke.stdout.log"
./scripts/run_vllm_local.sh >"${VLLM_LOG_PATH}" 2>&1 &
VLLM_PID=$!

echo "[phase2-smoke] waiting for vLLM health (timeout=${READINESS_TIMEOUT_SECONDS}s)"
if ! wait_for_vllm_health "${VLLM_BASE_URL}/health" "${VLLM_PID}" "${READINESS_TIMEOUT_SECONDS}" "${VLLM_LOG_PATH}"; then
  exit 1
fi

echo "[phase2-smoke] checking /health"
curl --noproxy "*" -fsS "${VLLM_BASE_URL}/health"
echo

echo "[phase2-smoke] checking /metrics"
curl --noproxy "*" -fsS "${VLLM_BASE_URL}/metrics" >"logs/phase2_metrics_snapshot.txt"
sed -n '1,20p' logs/phase2_metrics_snapshot.txt

echo "[phase2-smoke] running single benchmark case"
VLLM_BASE_URL="${VLLM_BASE_URL}" python bench/run_single_case.py \
  --batch-run-id "${BATCH_RUN_ID}" \
  --suite baseline \
  --mode non_stream \
  --concurrency 1 \
  --input-tokens 128 \
  --output-tokens 32 \
  --repeat-index 1 \
  --num-prompts "${BENCHMARK_NUM_PROMPTS}" \
  >"logs/phase2_smoke_case.json"

echo "[phase2-smoke] aggregating results"
python analysis/aggregate_results.py --batch-run-id "${BATCH_RUN_ID}" --output-dir "${OUTPUT_DIR}"
python analysis/render_baseline_summary.py \
  --benchmark-csv "${OUTPUT_DIR}/baseline_metrics.csv" \
  --service-csv "${OUTPUT_DIR}/baseline_service_metrics.csv" \
  --output-path "${SUMMARY_PATH}"

if python - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("matplotlib") else 1)
PY
then
  echo "[phase2-smoke] generating plots"
  python analysis/plot_baseline.py \
    --benchmark-csv "${OUTPUT_DIR}/baseline_metrics.csv" \
    --service-csv "${OUTPUT_DIR}/baseline_service_metrics.csv" \
    --output-dir "${PLOTS_DIR}"
else
  echo "[phase2-smoke] matplotlib not installed, skipping plots"
fi

echo "[phase2-smoke] validating batch outputs"
python analysis/validate_batch.py \
  --batch-run-id "${BATCH_RUN_ID}" \
  --output-dir "${OUTPUT_DIR}"

echo "[phase2-smoke] artifacts"
find "results/raw/benchmark/${BATCH_RUN_ID}" "results/raw/prometheus/${BATCH_RUN_ID}" -maxdepth 2 -type f | sort
find "${OUTPUT_DIR}" -maxdepth 2 -type f | sort

echo "[phase2-smoke] complete"
