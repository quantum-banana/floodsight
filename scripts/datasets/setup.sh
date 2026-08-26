#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv-datasets"
PYTHON_BIN="$VENV_PATH/bin/python"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.11 or newer is required and was not found on PATH." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Creating dataset-tooling environment at $VENV_PATH ..."
  python3 -m venv "$VENV_PATH"
fi

echo "Installing FloodSight dataset tooling ..."
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -e "$PROJECT_ROOT/ml[dev]"

echo "FloodSight Phase 3 dataset tooling is installed."
echo "Run ./scripts/datasets/doctor.sh before importing data."
