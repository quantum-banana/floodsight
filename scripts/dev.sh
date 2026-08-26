#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" || ! -d "$PROJECT_ROOT/frontend/node_modules" ]]; then
  echo "Dependencies are missing. Run ./scripts/setup.sh first." >&2
  exit 1
fi

cleanup() {
  trap - EXIT INT TERM
  kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
  wait "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting FloodSight API at http://127.0.0.1:8000 ..."
(cd "$PROJECT_ROOT/backend" && "$PYTHON_BIN" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000) &
BACKEND_PID=$!

echo "Starting FloodSight frontend at http://127.0.0.1:5173 ..."
(cd "$PROJECT_ROOT/frontend" && npm run dev -- --host 127.0.0.1) &
FRONTEND_PID=$!

echo "Both services are running. Press Ctrl+C to stop them."
wait "$BACKEND_PID" "$FRONTEND_PID"

