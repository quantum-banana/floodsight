"""Exact runtime checks for real SegFormer operations."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import stat
import sys
from pathlib import Path

from .errors import DependencyError

REQUIRED_PYTHON = (3, 12, 3)
REQUIRED_DISTRIBUTIONS = {
    "numpy": "2.2.6",
    "Pillow": "12.3.0",
    "PyYAML": "6.0.3",
    "safetensors": "0.8.0",
    "torch": "2.13.0+cu130",
    "transformers": "5.15.1",
}
REQUIRED_TORCH_CUDA = "13.0"
ACCEPTED_ENVIRONMENT_ROOT = Path(
    "/data/floodsight-workspace/floodsight-cache/envs/floodsight-ml-py312-cu130-locked-v1"
)
ACCEPTED_ENVIRONMENT_MARKER = ACCEPTED_ENVIRONMENT_ROOT / "FLOODSIGHT_ENVIRONMENT_ACCEPTED.json"
ACCEPTED_ENVIRONMENT_MARKER_SHA256 = (
    "11ec5e2bc107465862ab04f8a01d58719c5012356489168cf28387f6848f96bd"
)
ACCEPTED_RESOLVED_LOCK = Path(
    "/data/floodsight-workspace/floodsight/ml/training/requirements-py312-cu130.lock"
)
ACCEPTED_RESOLVED_LOCK_SHA256 = "33e7ca74a272659827d10c3bc882de1aa6e39b871c36435eb52279bd88eb58e1"
ACCEPTED_INSTALLED_DISTRIBUTIONS_SHA256 = (
    "ee7f9ce2704ddaea38312d0e11dacb8d01270f8be04ac1aad7e31095878ce775"
)
ACCEPTED_INSTALLED_DISTRIBUTION_COUNT = 103


def _validate_accepted_environment() -> dict[str, str]:
    if (
        Path(sys.prefix) != ACCEPTED_ENVIRONMENT_ROOT
        or Path(sys.prefix).is_symlink()
        or Path(sys.executable).parent.parent != ACCEPTED_ENVIRONMENT_ROOT
    ):
        raise DependencyError("SegFormer is not using the accepted locked environment.")
    for path, expected in (
        (ACCEPTED_ENVIRONMENT_MARKER, ACCEPTED_ENVIRONMENT_MARKER_SHA256),
        (ACCEPTED_RESOLVED_LOCK, ACCEPTED_RESOLVED_LOCK_SHA256),
    ):
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise DependencyError(f"Accepted runtime identity is missing: {path}") from exc
        if (
            not stat.S_ISREG(mode)
            or path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise DependencyError(f"Accepted runtime identity is unsafe or drifted: {path}")
    try:
        marker = json.loads(ACCEPTED_ENVIRONMENT_MARKER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DependencyError("Accepted environment marker is unreadable.") from exc
    required_marker = {
        "schema_version": "floodsight-locked-ml-environment-v1",
        "environment": str(ACCEPTED_ENVIRONMENT_ROOT),
        "python": "3.12.3",
        "torch": "2.13.0+cu130",
        "torch_cuda_build": "13.0",
        "resolved_lock": str(ACCEPTED_RESOLVED_LOCK),
        "resolved_lock_sha256": ACCEPTED_RESOLVED_LOCK_SHA256,
        "pip_check": "PASS",
        "pip_audit": "PASS_WITH_EXACT_TORCH_CANONICAL_FALLBACK",
    }
    if not isinstance(marker, dict) or any(
        marker.get(key) != value for key, value in required_marker.items()
    ):
        raise DependencyError("Accepted environment marker content drifted.")
    locked: dict[str, str] = {}
    for raw in ACCEPTED_RESOLVED_LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if separator != "==" or not name or not version or canonical in locked:
            raise DependencyError("Accepted environment lock is malformed.")
        locked[canonical] = version
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name:
            raise DependencyError("Installed distribution has no project name.")
        canonical = re.sub(r"[-_.]+", "-", raw_name).lower()
        if canonical in installed:
            raise DependencyError("Accepted environment has duplicate distributions.")
        installed[canonical] = distribution.version
    snapshot = "".join(f"{name}=={installed[name]}\n" for name in sorted(installed))
    snapshot_sha256 = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
    if (
        installed != locked
        or len(installed) != ACCEPTED_INSTALLED_DISTRIBUTION_COUNT
        or snapshot_sha256 != ACCEPTED_INSTALLED_DISTRIBUTIONS_SHA256
    ):
        raise DependencyError("Installed distributions differ from the accepted lock snapshot.")
    return {
        "environment_marker_sha256": ACCEPTED_ENVIRONMENT_MARKER_SHA256,
        "resolved_lock_sha256": ACCEPTED_RESOLVED_LOCK_SHA256,
        "installed_distributions_sha256": snapshot_sha256,
        "installed_distribution_count": str(len(installed)),
    }


def validate_runtime_versions() -> dict[str, str]:
    """Require the accepted interpreter and exact direct ML dependencies."""

    accepted = _validate_accepted_environment()
    if sys.version_info[:3] != REQUIRED_PYTHON:
        found = ".".join(str(item) for item in sys.version_info[:3])
        required = ".".join(str(item) for item in REQUIRED_PYTHON)
        raise DependencyError(f"SegFormer requires CPython {required}; found {found}.")
    observed: dict[str, str] = {}
    for distribution, required in REQUIRED_DISTRIBUTIONS.items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DependencyError(f"Required distribution is missing: {distribution}") from exc
        if installed != required:
            raise DependencyError(
                f"SegFormer requires {distribution}=={required}; found {installed}."
            )
        observed[distribution] = installed

    import torch

    if torch.version.cuda != REQUIRED_TORCH_CUDA:
        raise DependencyError(
            f"SegFormer requires the CUDA {REQUIRED_TORCH_CUDA} Torch build; "
            f"found {torch.version.cuda}."
        )
    return (
        accepted
        | observed
        | {
            "python": ".".join(map(str, REQUIRED_PYTHON)),
            "torch_cuda": "13.0",
        }
    )


def require_h100(device: object) -> None:
    """Reject a full run unless its selected CUDA device is an NVIDIA H100."""

    import torch

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise DependencyError("Full SegFormer training requires an NVIDIA H100 CUDA device.")
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    if "H100" not in name or capability != (9, 0):
        raise DependencyError(
            f"Full SegFormer training requires H100 capability 9.0; found {name} {capability}."
        )
