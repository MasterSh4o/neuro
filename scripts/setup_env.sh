#!/usr/bin/env bash
set -euo pipefail
echo "[setup_env] Updating pip/build tooling..."
pip install -U pip setuptools wheel
echo "[setup_env] Clean install of NumPy 1.26.4..."
pip uninstall -y numpy || true
pip cache remove numpy || true
pip install --no-cache-dir numpy==1.26.4
echo "[setup_env] Installing project requirements (no torch)..."
pip install --no-cache-dir -r requirements.txt
python - << 'PY'
import numpy, sys
print("Python:", sys.version)
print("NumPy :", numpy.__version__)
print("Path  :", numpy.__file__)
PY
echo "[setup_env] Done."
