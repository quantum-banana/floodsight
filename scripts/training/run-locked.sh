#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd)"
venv_path="${FLOODSIGHT_TRAINING_VENV:-/data/floodsight-workspace/floodsight-cache/envs/floodsight-ml-py312-cu130-locked-v1}"
venv_python="${venv_path}/bin/python"
canonical_venv="/data/floodsight-workspace/floodsight-cache/envs/floodsight-ml-py312-cu130-locked-v1"

if [[ "${venv_path}" != "${canonical_venv}" || -L "${venv_path}" || ! -x "${venv_python}" ]]; then
  echo "Accepted locked training environment is unavailable: ${venv_path}" >&2
  exit 2
fi
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {segmentation|detection} <command> [arguments...]" >&2
  exit 2
fi
if ! "${venv_python}" "${repo_root}/ml/training/verify-locked-environment.py" >/dev/null; then
  echo "Accepted locked training environment failed full identity verification." >&2
  exit 2
fi

stack="$1"
shift
case "${stack}" in
  segmentation)
    module="floodsight_segmentation"
    package_path="${repo_root}/ml/segmentation"
    ;;
  detection)
    module="floodsight_detection"
    package_path="${repo_root}/ml/detection"
    pinned_hash_seed="20260831"
    if [[ -n "${PYTHONHASHSEED+x}" && "${PYTHONHASHSEED}" != "${pinned_hash_seed}" ]]; then
      echo "Detection requires PYTHONHASHSEED=${pinned_hash_seed} before Python starts." >&2
      exit 2
    fi
    export PYTHONHASHSEED="${pinned_hash_seed}"
    ;;
  *)
    echo "Unknown training stack: ${stack}" >&2
    exit 2
    ;;
esac

export PYTHONPATH="${package_path}"
# shellcheck source=runtime-offline.sh
source "${script_dir}/runtime-offline.sh"

# This host exports an unrelated project's cuDNN directory globally. The
# accepted Torch wheel carries its own audited CUDA libraries; allowing the
# inherited path to win makes even a forward pass ABI-incompatible.
unset LD_LIBRARY_PATH

exec "${venv_python}" -m "${module}" "$@"
