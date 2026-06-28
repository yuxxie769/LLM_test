#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
source .venv/bin/activate

SUITE="${1:-baseline}"
BATCH_RUN_ID="${BATCH_RUN_ID:-phase2-${SUITE}-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/batches/${BATCH_RUN_ID}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_DIR}/baseline_summary.md}"
PLOTS_DIR="${PLOTS_DIR:-${OUTPUT_DIR}/plots}"

echo "[phase2-suite] suite=${SUITE}"
echo "[phase2-suite] batch_run_id=${BATCH_RUN_ID}"
echo "[phase2-suite] vllm_base_url=${VLLM_BASE_URL:-http://127.0.0.1:19100}"
echo "[phase2-suite] model_dir=${MODEL_DIR:-/root/models/qwen2.5-0.5b}"
echo "[phase2-suite] output_dir=${OUTPUT_DIR}"

echo "[phase2-suite] checking vLLM health"
curl --noproxy "*" -fsS "${VLLM_BASE_URL:-http://127.0.0.1:19100}/health" >/dev/null

echo "[phase2-suite] running matrix"
if [[ -n "${MATRIX_LIMIT:-}" ]]; then
  echo "[phase2-suite] matrix_limit=${MATRIX_LIMIT}"
  python bench/run_matrix.py --suite "${SUITE}" --limit "${MATRIX_LIMIT}" --batch-run-id "${BATCH_RUN_ID}"
else
  python bench/run_matrix.py --suite "${SUITE}" --batch-run-id "${BATCH_RUN_ID}"
fi

echo "[phase2-suite] aggregating"
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
  echo "[phase2-suite] generating plots"
  python analysis/plot_baseline.py \
    --benchmark-csv "${OUTPUT_DIR}/baseline_metrics.csv" \
    --service-csv "${OUTPUT_DIR}/baseline_service_metrics.csv" \
    --output-dir "${PLOTS_DIR}"
else
  echo "[phase2-suite] matplotlib not installed, skipping plots"
fi

echo "[phase2-suite] validating batch outputs"
python analysis/validate_batch.py \
  --batch-run-id "${BATCH_RUN_ID}" \
  --output-dir "${OUTPUT_DIR}"

echo "[phase2-suite] complete"
echo "batch_run_id=${BATCH_RUN_ID}"
echo "output_dir=${OUTPUT_DIR}"
