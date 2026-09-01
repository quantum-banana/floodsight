#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd)"
venv_path="${FLOODSIGHT_TRAINING_VENV:-/data/floodsight-workspace/floodsight-cache/envs/floodsight-ml-py312-cu130-locked-v1}"
venv_python="${venv_path}/bin/python"
ruff_bin="${venv_path}/bin/ruff"
pytest_bin="${venv_path}/bin/pytest"

if [[ ! -x "${venv_python}" ]]; then
  echo "Locked training environment is missing. Run: bash scripts/training/setup-locked.sh" >&2
  exit 2
fi
if [[ ! -f "${venv_path}/FLOODSIGHT_ENVIRONMENT_ACCEPTED.json" ]]; then
  echo "Training environment has no accepted-environment marker: ${venv_path}" >&2
  exit 2
fi

export PYTHONPATH="${repo_root}/ml/segmentation:${repo_root}/ml/detection${PYTHONPATH:+:${PYTHONPATH}}"
export FLOODSIGHT_ALLOW_TRAINING=NO
runtime_dir="$(mktemp -d /tmp/floodsight-training-check.XXXXXX)"
trap 'rm -rf -- "${runtime_dir}"' EXIT
export FLOODSIGHT_ML_RUNTIME_CACHE="${runtime_dir}"
# shellcheck source=runtime-offline.sh
source "${script_dir}/runtime-offline.sh"
# The host inherits an unrelated cuDNN path from another project. The pinned
# Torch wheel must resolve only its own audited CUDA runtime.
unset LD_LIBRARY_PATH

lock_requirements="${repo_root}/ml/training/requirements-py312-cu130.lock"
acceptance_marker="${venv_path}/FLOODSIGHT_ENVIRONMENT_ACCEPTED.json"
accepted_marker_sha256="11ec5e2bc107465862ab04f8a01d58719c5012356489168cf28387f6848f96bd"
lock_snapshot="$(${venv_python} -m pip freeze --all | LC_ALL=C sort)"
expected_snapshot="$(sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "${lock_requirements}" | LC_ALL=C sort)"
if [[ "${lock_snapshot}" != "${expected_snapshot}" ]]; then
  echo "Training environment differs from the resolved FloodSight lock." >&2
  exit 3
fi
"${venv_python}" -m pip check

ACCEPTANCE_MARKER="${acceptance_marker}" \
ACCEPTED_MARKER_SHA256="${accepted_marker_sha256}" \
VENV_PATH="${venv_path}" \
LOCK_PATH="${lock_requirements}" \
  "${venv_python}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

marker = json.loads(Path(os.environ["ACCEPTANCE_MARKER"]).read_text(encoding="utf-8"))
marker_path = Path(os.environ["ACCEPTANCE_MARKER"])
lock = Path(os.environ["LOCK_PATH"]).resolve()
if marker_path.is_symlink() or not marker_path.is_file():
    raise SystemExit("Accepted-environment marker must be a regular non-symlink file.")
if hashlib.sha256(marker_path.read_bytes()).hexdigest() != os.environ["ACCEPTED_MARKER_SHA256"]:
    raise SystemExit("Accepted-environment marker hash drifted.")
if marker.get("environment") != str(Path(os.environ["VENV_PATH"]).resolve()):
    raise SystemExit("Accepted-environment marker points at a different environment.")
if marker.get("resolved_lock") != str(lock):
    raise SystemExit("Accepted-environment marker points at a different lock.")
if marker.get("resolved_lock_sha256") != hashlib.sha256(lock.read_bytes()).hexdigest():
    raise SystemExit("Accepted-environment lock hash drifted.")
if marker.get("pip_audit") != "PASS_WITH_EXACT_TORCH_CANONICAL_FALLBACK":
    raise SystemExit("Accepted-environment vulnerability-audit mode is not approved.")
PY

# The accepted marker is emitted only after setup-locked.sh completes both
# vulnerability audits. Its exact bytes are frozen above. Re-running pip-audit
# here would make this validation depend on mutable HTTP-cache freshness and
# could silently reintroduce an internet dependency.
torch_lock="$({ grep -E '^torch==|^torchvision==' "${lock_requirements}" || true; } | LC_ALL=C sort)"
expected_torch_lock=$'torch==2.13.0+cu130\ntorchvision==0.28.0+cu130'
if [[ "${torch_lock}" != "${expected_torch_lock}" ]]; then
  echo "Locked Torch/Torchvision versions drifted from the canonical audit exception." >&2
  exit 4
fi
canonical_audit="${repo_root}/ml/training/audit-canonical-requirements.txt"
canonical_snapshot="$(
  sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "${canonical_audit}" | LC_ALL=C sort
)"
expected_canonical=$'torch==2.13.0\ntorchvision==0.28.0'
if [[ "${canonical_snapshot}" != "${expected_canonical}" ]]; then
  echo "Canonical Torch/Torchvision vulnerability versions drifted." >&2
  exit 4
fi

"${ruff_bin}" check \
  "${repo_root}/ml/segmentation" \
  "${repo_root}/ml/detection"
"${pytest_bin}" -q \
  "${repo_root}/ml/segmentation/tests" \
  "${repo_root}/ml/detection/tests"

echo "FloodSight training infrastructure checks passed; real training was not authorized or started."
