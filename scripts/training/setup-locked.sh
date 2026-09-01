#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd)"
python_bin="${FLOODSIGHT_TRAINING_PYTHON:-/usr/bin/python3.12}"
target="${FLOODSIGHT_TRAINING_VENV:-/data/floodsight-workspace/floodsight-cache/envs/floodsight-ml-py312-cu130-locked-v1}"
wheelhouse="${FLOODSIGHT_TRAINING_WHEELHOUSE:-/data/floodsight-workspace/floodsight-cache/ml/wheelhouse/py312-cu130-v1}"
lock="${repo_root}/ml/training/requirements-py312-cu130.lock"
acceptance="${repo_root}/ml/training/accepted-wheelhouse.sha256"

if [[ "$(${python_bin} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')" != "3.12.3" ]]; then
  echo "Locked FloodSight build requires CPython 3.12.3: ${python_bin}" >&2
  exit 2
fi
target="$(realpath -m -- "${target}")"
case "${target}" in
  /data/floodsight-workspace/floodsight-cache/envs/*) ;;
  *) echo "Locked environment must be below the FloodSight cache env root." >&2; exit 2 ;;
esac
if [[ -e "${target}" || -L "${target}" ]]; then
  echo "Refusing to reconcile or overwrite an existing locked environment: ${target}" >&2
  exit 3
fi
if [[ ! -d "${wheelhouse}" || -L "${wheelhouse}" ]]; then
  echo "Accepted wheelhouse is missing or unsafe: ${wheelhouse}" >&2
  exit 3
fi

declare -A expected
while IFS='=' read -r key value; do
  [[ -z "${key}" || "${key}" == \#* ]] && continue
  expected["${key}"]="${value}"
done < "${acceptance}"
if [[ "$(realpath -- "${wheelhouse}")" != "${expected[wheelhouse_path]}" ]]; then
  echo "Wheelhouse path does not match the accepted provenance record." >&2
  exit 3
fi
if [[ "$(sha256sum "${wheelhouse}/SHA256SUMS" | cut -d' ' -f1)" != "${expected[sha256sums_sha256]}" ]] ||
   [[ "$(sha256sum "${wheelhouse}/wheelhouse-inventory.json" | cut -d' ' -f1)" != "${expected[inventory_sha256]}" ]] ||
   [[ "$(sha256sum "${lock}" | cut -d' ' -f1)" != "${expected[resolved_lock_sha256]}" ]]; then
  echo "Locked environment inputs failed their top-level SHA-256 bindings." >&2
  exit 3
fi
(
  cd "${wheelhouse}"
  sha256sum --check --strict SHA256SUMS
) >/dev/null

LOCK_PATH="${lock}" INVENTORY_PATH="${wheelhouse}/wheelhouse-inventory.json" EXPECTED_COUNT="${expected[wheel_count]}" "${python_bin}" - <<'PY'
from __future__ import annotations

import json
import os
import re
from pathlib import Path


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


locked: dict[str, str] = {}
for raw in Path(os.environ["LOCK_PATH"]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    name, separator, version = line.partition("==")
    if separator != "==" or canonical(name) in locked:
        raise SystemExit(f"Invalid resolved-lock line: {line!r}")
    locked[canonical(name)] = version
inventory = json.loads(Path(os.environ["INVENTORY_PATH"]).read_text(encoding="utf-8"))
wheels = inventory.get("wheels", [])
observed = {canonical(row["name"]): row["version"] for row in wheels}
expected_count = int(os.environ["EXPECTED_COUNT"])
if len(wheels) != expected_count or len(observed) != expected_count or observed != locked:
    raise SystemExit("Wheel inventory does not exactly match the resolved package lock.")
PY

hashed_requirements="$(mktemp /tmp/floodsight-locked-requirements.XXXXXX)"
trap 'rm -f -- "${hashed_requirements}"' EXIT
INVENTORY_PATH="${wheelhouse}/wheelhouse-inventory.json" HASHED_REQUIREMENTS="${hashed_requirements}" "${python_bin}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

inventory = json.loads(Path(os.environ["INVENTORY_PATH"]).read_text(encoding="utf-8"))
lines = [
    f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}"
    for row in sorted(inventory["wheels"], key=lambda item: item["name"])
]
Path(os.environ["HASHED_REQUIREMENTS"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

mkdir -p "$(dirname -- "${target}")"
"${python_bin}" -m venv "${target}"
venv_python="${target}/bin/python"
"${venv_python}" -m pip --isolated install \
  --no-cache-dir \
  --no-index \
  --no-deps \
  --only-binary=:all: \
  --require-hashes \
  --find-links "${wheelhouse}" \
  --requirement "${hashed_requirements}"

installed="$(${venv_python} -m pip freeze --all | LC_ALL=C sort)"
locked="$(sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "${lock}" | LC_ALL=C sort)"
if [[ "${installed}" != "${locked}" ]]; then
  echo "Offline environment does not exactly match the accepted lock." >&2
  exit 4
fi
"${venv_python}" -m pip check
"${venv_python}" -m pip_audit --progress-spinner off
"${venv_python}" -m pip_audit --progress-spinner off --no-deps \
  --requirement "${repo_root}/ml/training/audit-canonical-requirements.txt"

unset LD_LIBRARY_PATH
TARGET_PATH="${target}" LOCK_PATH="${lock}" WHEELHOUSE_PATH="${wheelhouse}" "${venv_python}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import platform
import site
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

target = Path(os.environ["TARGET_PATH"]).resolve()
if Path(sys.prefix).resolve() != target or sys.prefix == sys.base_prefix or site.ENABLE_USER_SITE:
    raise SystemExit("Accepted interpreter is not the intended isolated venv.")
if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("Accepted environment requires Linux x86_64.")
if sys.version_info[:3] != (3, 12, 3):
    raise SystemExit("Accepted environment requires CPython 3.12.3.")
if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
    raise SystemExit("Accepted environment requires Torch 2.13.0+cu130.")
gpus = [
    {
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "capability": list(torch.cuda.get_device_capability(index)),
    }
    for index in range(torch.cuda.device_count())
]
if not torch.cuda.is_available() or not any(
    "H100" in item["name"] and item["capability"] == [9, 0] for item in gpus
):
    raise SystemExit("No NVIDIA H100 (compute capability 9.0) is visible.")

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

marker = {
    "schema_version": "floodsight-locked-ml-environment-v1",
    "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
    "environment": str(target),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "gpus": gpus,
    "resolved_lock": str(Path(os.environ["LOCK_PATH"]).resolve()),
    "resolved_lock_sha256": digest(Path(os.environ["LOCK_PATH"])),
    "wheelhouse": str(Path(os.environ["WHEELHOUSE_PATH"]).resolve()),
    "wheelhouse_sha256s_sha256": digest(Path(os.environ["WHEELHOUSE_PATH"]) / "SHA256SUMS"),
    "wheelhouse_inventory_sha256": digest(
        Path(os.environ["WHEELHOUSE_PATH"]) / "wheelhouse-inventory.json"
    ),
    "pip_check": "PASS",
    "pip_audit": "PASS_WITH_EXACT_TORCH_CANONICAL_FALLBACK",
}
path = target / "FLOODSIGHT_ENVIRONMENT_ACCEPTED.json"
with path.open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(marker, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps(marker, indent=2, sort_keys=True))
PY
