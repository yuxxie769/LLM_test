#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
source .venv/bin/activate

REFERENCE_BATCH="${REFERENCE_BATCH:?set REFERENCE_BATCH to a batch id or results/batches path}"
CANDIDATE_BATCH="${CANDIDATE_BATCH:?set CANDIDATE_BATCH to a batch id or results/batches path}"
REFERENCE_LABEL="${REFERENCE_LABEL:-reference-model}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:-candidate-model}"
OUTPUT_DIR="${OUTPUT_DIR:-results/model_compare}"

python analysis/aggregate_model_compare.py           --batch "${REFERENCE_LABEL}=${REFERENCE_BATCH}"           --batch "${CANDIDATE_LABEL}=${CANDIDATE_BATCH}"           --output-dir "${OUTPUT_DIR}"

python analysis/render_model_compare_summary.py           --compare-csv "${OUTPUT_DIR}/model_compare.csv"           --output-path "${OUTPUT_DIR}/model_compare_summary.md"

if python - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('matplotlib') else 1)
PY
then
  python analysis/plot_model_compare.py             --long-csv "${OUTPUT_DIR}/model_compare_long.csv"             --output-dir "${OUTPUT_DIR}/plots"
else
  echo "[phase3-compare] matplotlib not installed, skipping plots"
fi

python analysis/validate_model_compare.py           --manifest "${OUTPUT_DIR}/manifest.json"           --output-dir "${OUTPUT_DIR}"

echo "output_dir=${OUTPUT_DIR}"
