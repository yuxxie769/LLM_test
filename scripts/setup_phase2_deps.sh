#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "[setup_phase2_deps] missing virtual environment at ${REPO_ROOT}/.venv" >&2
  exit 1
fi

source .venv/bin/activate
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

echo "[setup_phase2_deps] installing analysis dependencies"
python -m pip install pandas matplotlib seaborn

echo "[setup_phase2_deps] current package versions"
python - <<'PY'
from importlib import metadata

for name in ("pandas", "matplotlib", "seaborn"):
    version = metadata.version(name)
    dist = metadata.distribution(name)
    location = dist.locate_file("")
    print(f"{name}=={version} @ {location}")
PY
