#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" || ! -d "$PROJECT_ROOT/frontend/node_modules" ]]; then
  echo "Dependencies are missing. Run ./scripts/setup.sh first." >&2
  exit 1
fi

echo "Checking backend formatting and imports ..."
"$PYTHON_BIN" -m ruff check "$PROJECT_ROOT/backend/app" "$PROJECT_ROOT/backend/tests"

echo "Running backend tests and shared-schema validation ..."
"$PYTHON_BIN" -m pytest "$PROJECT_ROOT/backend/tests"

echo "Linting frontend ..."
(cd "$PROJECT_ROOT/frontend" && npm run lint)

echo "Running frontend tests ..."
(cd "$PROJECT_ROOT/frontend" && npm run test)

echo "Building frontend production bundle ..."
(cd "$PROJECT_ROOT/frontend" && npm run build)

echo "All FloodSight Phase 2 checks passed."
