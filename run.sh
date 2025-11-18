#!/usr/bin/env bash
set -euo pipefail

# run.sh - Quick runner for Linux
# Creates a virtual environment in `.venv`, installs Python requirements (unless
# SKIP_DEPS=1), and runs `app.py` with any provided arguments.
#
# Usage examples:
#   ./run.sh                 # Run app.py with default args
#   ./run.sh --mode web --config config/config.yaml
#   SKIP_DEPS=1 ./run.sh      # Skip installing requirements (useful on constrained systems)

VENV_DIR=".venv"

echo "[run.sh] Starting quick runner"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found. Install Python 3 and try again." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[run.sh] Creating virtualenv in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "[run.sh] Activating virtualenv"
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

if [ "${SKIP_DEPS:-0}" != "1" ]; then
  echo "[run.sh] Upgrading pip and installing requirements (may take time)"
  python -m pip install --upgrade pip setuptools wheel
  if [ -f requirements.txt ]; then
    # Try to install requirements but don't fail the script completely if a heavy package fails
    pip install -r requirements.txt || echo "[run.sh] Warning: pip install had errors; you may need to install some packages manually (torch/ultralytics)"
  else
    echo "[run.sh] No requirements.txt found; skipping pip install"
  fi
else
  echo "[run.sh] SKIP_DEPS=1 set; skipping dependency installation"
fi

echo "[run.sh] Note: Installing PyTorch on Raspberry Pi often requires a platform-specific wheel. If you need torch, install it before running the app."

if [ "$#" -eq 0 ]; then
  # Default: run web server on port 8000 using config/config.yaml
  DEFAULT_CONFIG="config/config.yaml"
  if [ -f "$DEFAULT_CONFIG" ]; then
    echo "[run.sh] No args provided — launching web mode on port 8000 with config $DEFAULT_CONFIG"
    exec python3 app.py --mode web --config "$DEFAULT_CONFIG" --port 8000
  else
    echo "[run.sh] No args provided and no $DEFAULT_CONFIG found — launching web mode on port 8000"
    exec python3 app.py --mode web --port 8000
  fi
else
  echo "[run.sh] Running app.py with args: $*"
  exec python3 app.py "$@"
fi
