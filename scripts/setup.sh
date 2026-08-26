#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON_BIN="$VENV_PATH/bin/python"

if command -v python3 >/dev/null 2>&1; then
  SYSTEM_PYTHON="$(command -v python3)"
else
  echo "Python 3.11 or newer is required and was not found on PATH." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js and npm are required. Install a supported Node.js release and retry." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Creating Python virtual environment at $VENV_PATH ..."
  "$SYSTEM_PYTHON" -m venv "$VENV_PATH"
fi

echo "Installing FloodSight backend dependencies ..."
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -e "$PROJECT_ROOT/backend[dev]"

echo "Installing FloodSight frontend dependencies ..."
(cd "$PROJECT_ROOT/frontend" && npm install)

echo "FloodSight Phase 0 setup completed successfully."
echo "Run ./scripts/dev.sh to start the development services."

