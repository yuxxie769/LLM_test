#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/lib/wsl/lib:${PATH}

echo "[preflight] kernel:"
uname -a
echo

echo "[preflight] distro status:"
if command -v wsl.exe >/dev/null 2>&1; then
  wsl.exe -l -v | tr -d '\000\r' || true
else
  echo "wsl.exe not available in this shell"
fi
echo

echo "[preflight] workspace path:"
pwd
echo

echo "[preflight] /dev/dxg:"
ls -l /dev/dxg
echo

echo "[preflight] wsl nvidia-smi:"
nvidia-smi
echo

echo "[preflight] python toolchain:"
python3 --version
uv --version
git --version
curl --version | head -n 1
echo

echo "[preflight] filesystem types:"
df -T .
df -T /tmp
echo

if [[ "$(pwd)" == /mnt/* ]]; then
  echo "[warn] current directory is on a Windows-mounted filesystem. Move the runtime copy to Linux FS before installing."
fi

echo "[preflight] done"
