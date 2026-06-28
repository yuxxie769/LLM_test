#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PATH=/usr/lib/wsl/lib:${PATH}

WORKSPACE_ROOT="${WSL_WORKSPACE_ROOT:-${HOME}/workspace}"
if [[ ! -w "${HOME}" ]]; then
  WORKSPACE_ROOT="${WSL_WORKSPACE_ROOT:-/tmp/workspace}"
fi

RUNTIME_DIR="${RUNTIME_DIR:-${WORKSPACE_ROOT}/LLM_test}"
if [[ ! -w "${HOME}" ]]; then
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
  XDG_DATA_HOME="${XDG_DATA_HOME:-/tmp/xdg-data}"
else
  UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
  XDG_DATA_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}"
fi

export UV_CACHE_DIR
export XDG_DATA_HOME

echo "[setup] repo root: ${REPO_ROOT}"
echo "[setup] runtime dir: ${RUNTIME_DIR}"
echo "[setup] uv cache: ${UV_CACHE_DIR}"
echo "[setup] xdg data: ${XDG_DATA_HOME}"

mkdir -p "${WORKSPACE_ROOT}" "${UV_CACHE_DIR}" "${XDG_DATA_HOME}"
rm -rf "${RUNTIME_DIR}"
mkdir -p "${RUNTIME_DIR}"
cp -a "${REPO_ROOT}/." "${RUNTIME_DIR}/"

cd "${RUNTIME_DIR}"

echo "[setup] running preflight"
"${RUNTIME_DIR}/scripts/wsl_preflight.sh"

echo "[setup] creating Python 3.12 environment"
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

echo "[setup] installing runtime dependencies"
uv pip install vllm --torch-backend=auto
uv pip install fastapi "uvicorn[standard]" httpx pydantic-settings orjson

echo "[setup] validating imports"
python - <<'PY'
import fastapi
import httpx
import orjson
import pydantic_settings
import vllm

print("fastapi", fastapi.__version__)
print("httpx", httpx.__version__)
print("orjson", orjson.__version__)
print("pydantic_settings", pydantic_settings.__version__)
print("vllm", getattr(vllm, "__version__", "unknown"))
PY

echo "[setup] complete"
echo "[setup] activate with: source ${RUNTIME_DIR}/.venv/bin/activate"
