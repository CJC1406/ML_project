#!/usr/bin/env bash
# Laptop 1: start the PathSnap FastAPI backend, reachable from other machines on the LAN.
# Override any of these by exporting them before running the script.
set -euo pipefail
cd "$(dirname "$0")/.."

export PATHSNAP_EVALUATOR_ROOT="${PATHSNAP_EVALUATOR_ROOT:-${PATHSNAP_TRAIN_ROOT:-data/raw}}"
export PATHSNAP_MODEL_BUNDLE="${PATHSNAP_MODEL_BUNDLE:-models/pathsnap_v1}"
PORT="${PORT:-8000}"
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"; elif [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi

LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || true)"
echo "Trips root   : $PATHSNAP_EVALUATOR_ROOT"
echo "Model bundle : $PATHSNAP_MODEL_BUNDLE"
echo "Laptop 2 should set VITE_API_BASE_URL=http://${LAN_IP:-<this-laptop-ip>}:${PORT}"

exec "$PY" -m uvicorn pathsnap.api:app --app-dir src --host 0.0.0.0 --port "$PORT"
