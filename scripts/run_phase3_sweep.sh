#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
source .venv/bin/activate

EXTRA_ARGS=("$@")
TARGET_ARGS=()
if [[ ${#EXTRA_ARGS[@]} -gt 0 && "${EXTRA_ARGS[0]}" != -* ]]; then
  TARGET_ARGS+=(--target "${EXTRA_ARGS[0]}")
  EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
fi

SCRIPT_DRY_RUN="${DRY_RUN:-0}"
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    SCRIPT_DRY_RUN=1
    break
  fi
done

SWEEP_RUN_ID="${SWEEP_RUN_ID:-phase3-sweep-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/param_tuning}"
RUN_ARGS=(--sweep-run-id "${SWEEP_RUN_ID}" --output-dir "${OUTPUT_DIR}")
if [[ "${SCRIPT_DRY_RUN}" == "1" ]]; then
  RUN_ARGS+=(--dry-run)
fi

python bench/run_phase3_sweep.py "${RUN_ARGS[@]}" "${TARGET_ARGS[@]}" "${EXTRA_ARGS[@]}"

if [[ "${SCRIPT_DRY_RUN}" == "1" ]]; then
  exit 0
fi

MANIFEST_PATH="${OUTPUT_DIR}/raw/${SWEEP_RUN_ID}/manifest.json"
python analysis/aggregate_param_tuning.py           --manifest "${MANIFEST_PATH}"           --output-dir "${OUTPUT_DIR}"
python analysis/render_param_tuning_summary.py           --csv "${OUTPUT_DIR}/param_tuning.csv"           --output-path "${OUTPUT_DIR}/param_tuning_summary.md"

if python - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('matplotlib') else 1)
PY
then
  python analysis/plot_param_tuning.py             --csv "${OUTPUT_DIR}/param_tuning.csv"             --output-dir "${OUTPUT_DIR}/plots"
else
  echo "[phase3-sweep] matplotlib not installed, skipping plots"
fi

python analysis/validate_param_tuning.py           --manifest "${MANIFEST_PATH}"           --output-dir "${OUTPUT_DIR}"

echo "manifest_path=${MANIFEST_PATH}"
echo "output_dir=${OUTPUT_DIR}"
