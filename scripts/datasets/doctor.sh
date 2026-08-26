#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv-datasets/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Dataset tooling is missing. Run ./scripts/datasets/setup.sh first." >&2
  exit 1
fi

exec "$PYTHON_BIN" -m floodsight_data.cli doctor "$@"
