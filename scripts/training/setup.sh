#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd)"
venv_path="${FLOODSIGHT_TRAINING_VENV:-/data/floodsight-workspace/floodsight-cache/envs/floodsight-ml-py312-cu130-v1}"
python_bin="${FLOODSIGHT_TRAINING_PYTHON:-/usr/bin/python3.12}"
torch_requirements="${repo_root}/ml/training/requirements-torch-cu130.txt"
pypi_requirements="${repo_root}/ml/training/requirements-pypi.txt"
lock_requirements="${repo_root}/ml/training/requirements-py312-cu130.lock"

resolved_venv_path="$(realpath -m -- "${venv_path}")"
case "${resolved_venv_path}" in
  "${repo_root}/.venv-training"|/data/floodsight-workspace/floodsight-cache/envs/*) ;;
  *)
    echo "Training venv must be repo-local or below the FloodSight cache env root: ${resolved_venv_path}" >&2
    exit 2
    ;;
esac

for direct_requirements in "${torch_requirements}" "${pypi_requirements}"; do
  while IFS= read -r requirement; do
    [[ -z "${requirement}" || "${requirement}" == \#* ]] && continue
    if ! grep -Fqx -- "${requirement}" "${lock_requirements}"; then
      echo "Direct requirement is absent from the resolved lock: ${requirement}" >&2
      exit 2
    fi
  done < "${direct_requirements}"
done

if [[ ! -x "${python_bin}" ]]; then
  echo "Required Python interpreter is unavailable: ${python_bin}" >&2
  exit 2
fi

if [[ "$(${python_bin} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')" != "3.12.3" ]]; then
  echo "FloodSight training requires CPython 3.12.3: ${python_bin}" >&2
  exit 2
fi

if [[ ! -d "${venv_path}" ]]; then
  "${python_bin}" -m venv "${venv_path}"
fi

venv_python="${venv_path}/bin/python"
if [[ ! -x "${venv_python}" ]] || [[ "$(${venv_python} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')" != "3.12.3" ]]; then
  echo "Existing FloodSight environment is not the required CPython 3.12.3 venv: ${venv_path}" >&2
  exit 2
fi
VENV_EXPECTED="${resolved_venv_path}" "${venv_python}" - <<'PY'
from __future__ import annotations

import os
import site
import sys
from pathlib import Path

expected = Path(os.environ["VENV_EXPECTED"]).resolve()
if Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
    raise SystemExit("Configured interpreter is not the expected isolated venv.")
if site.ENABLE_USER_SITE:
    raise SystemExit("Accepted FloodSight venv must disable the user site.")
PY
"${venv_python}" -m pip --isolated install --upgrade \
  --no-deps \
  --no-cache-dir \
  --only-binary=:all: \
  --index-url https://pypi.org/simple \
  pip==26.2.1 \
  setuptools==84.0.0 \
  packaging==26.3 \
  wheel==0.48.0
# Every resolved package is source-routed explicitly. Installing with --no-deps
# prevents either index from being consulted for a dependency owned by the other.
torch_source_pattern='^(torch|torchvision|triton|cuda-bindings|cuda-toolkit|nvidia-[A-Za-z0-9_.-]+)=='
"${venv_python}" -m pip --isolated install \
  --no-deps \
  --no-cache-dir \
  --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cu130 \
  --requirement <(grep -E "${torch_source_pattern}" "${lock_requirements}")
"${venv_python}" -m pip --isolated install \
  --no-deps \
  --no-cache-dir \
  --only-binary=:all: \
  --index-url https://pypi.org/simple \
  --requirement <(
    sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "${lock_requirements}" |
      grep -Ev "${torch_source_pattern}"
  )
"${venv_python}" -m pip check
lock_snapshot="$(${venv_python} -m pip freeze --all | LC_ALL=C sort)"
expected_snapshot="$(sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "${lock_requirements}" | LC_ALL=C sort)"
if [[ "${lock_snapshot}" != "${expected_snapshot}" ]]; then
  echo "Installed environment does not match the resolved FloodSight lock." >&2
  diff -u <(printf '%s\n' "${expected_snapshot}") <(printf '%s\n' "${lock_snapshot}") >&2 || true
  exit 3
fi
"${venv_python}" -m pip_audit --progress-spinner off
"${venv_python}" -m pip_audit \
  --progress-spinner off \
  --no-deps \
  --requirement "${repo_root}/ml/training/audit-canonical-requirements.txt"

audit_json="$(${venv_python} -m pip_audit --progress-spinner off --format json 2>/dev/null)"
AUDIT_JSON="${audit_json}" "${venv_python}" - <<'PY'
from __future__ import annotations

import json
import os

payload = json.loads(os.environ["AUDIT_JSON"])
skipped = sorted(
    item["name"]
    for item in payload.get("dependencies", [])
    if item.get("skip_reason") is not None
)
if skipped != ["torch", "torchvision"]:
    raise SystemExit(f"Unexpected pip-audit skip set: {skipped}")
PY

"${venv_python}" - <<'PY'
from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
import sys

import torch

if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("FloodSight accepted ML environment requires Linux x86_64.")
if torch.version.cuda != "13.0":
    raise SystemExit(f"Expected a CUDA 13.0 Torch build, found {torch.version.cuda!r}.")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("No CUDA device is visible to the accepted ML environment.")
if not any("H100" in torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())):
    raise SystemExit("No NVIDIA H100 is visible to the accepted ML environment.")

packages = [
    "accelerate",
    "numpy",
    "opencv-python",
    "Pillow",
    "PyYAML",
    "safetensors",
    "tensorboard",
    "timm",
    "torch",
    "torchvision",
    "transformers",
    "ultralytics",
]
payload = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "packages": {name: metadata.version(name) for name in packages},
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
