"""Exact offline H100 runtime gate for full detector training."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file

REQUIRED_PYTHON = (3, 12, 3)
REQUIRED_VERSIONS = {
    "ultralytics": "8.3.222",
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
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
RUNTIME_ROOT = Path("/data/floodsight-workspace/floodsight-cache/ml/runtime/locked-v1")
RUNTIME_ASSET_ROOT = Path(
    "/data/floodsight-workspace/floodsight-cache/ml/runtime-assets/ultralytics-v8.3.222-v1"
)
RUNTIME_FONT = RUNTIME_ROOT / "yolo-config/Ultralytics/Arial.ttf"
RUNTIME_AUX_WEIGHT = RUNTIME_ASSET_ROOT / "yolo11n.pt"
RUNTIME_ASSET_HASHES = {
    RUNTIME_FONT: "525979822591a3447cfc49d943d6f7683508e25543407871c0ed8fed05fd2bd9",
    RUNTIME_AUX_WEIGHT: "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
}


def _fail(message: str, code: str = "ml_runtime_mismatch") -> None:
    raise DetectionInfrastructureError(message, code=code)


def bound_training_device(configured: str, override: str | None) -> str:
    """Reject a CLI device override that is not bound by the frozen config hash."""

    if override is not None and override != configured:
        _fail(
            "The full-training device override differs from the frozen configuration.",
            "training_device_drift",
        )
    return configured


def bound_training_output_root(configured: str | Path, supplied: str | Path) -> Path:
    """Require the CLI root to resolve to the exact config-hashed run root."""

    configured_root = Path(configured).expanduser().resolve()
    supplied_root = Path(supplied).expanduser().resolve()
    if supplied_root != configured_root:
        _fail(
            "The full-training output root differs from the frozen configuration.",
            "training_output_root_drift",
        )
    return configured_root


def bound_real_smoke_output_directory(configured_root: str | Path, supplied: str | Path) -> Path:
    """Require a real-smoke output to be one new direct child of its frozen root."""

    root = Path(configured_root).expanduser().resolve()
    output = Path(supplied).expanduser().resolve()
    if output == root or output.parent != root:
        _fail(
            "Real-smoke output must be one direct child of the frozen smoke root.",
            "real_smoke_output_policy_drift",
        )
    return output


def validate_accepted_environment() -> dict[str, str]:
    """Bind execution to the exact audited venv marker and resolved lock bytes."""

    try:
        prefix = Path(sys.prefix)
        executable = Path(sys.executable)
        root = ACCEPTED_ENVIRONMENT_ROOT.resolve(strict=True)
    except OSError as exc:
        raise DetectionInfrastructureError(
            "The canonical accepted detector environment cannot be resolved.",
            code="accepted_environment_mismatch",
        ) from exc
    if (
        prefix != ACCEPTED_ENVIRONMENT_ROOT
        or prefix.is_symlink()
        or prefix.resolve(strict=True) != root
        or executable.parent.parent != ACCEPTED_ENVIRONMENT_ROOT
    ):
        _fail(
            "Detector execution is not using the exact accepted locked environment.",
            "accepted_environment_mismatch",
        )
    for path, expected_sha256, label in (
        (
            ACCEPTED_ENVIRONMENT_MARKER,
            ACCEPTED_ENVIRONMENT_MARKER_SHA256,
            "environment acceptance marker",
        ),
        (ACCEPTED_RESOLVED_LOCK, ACCEPTED_RESOLVED_LOCK_SHA256, "resolved lock"),
    ):
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise DetectionInfrastructureError(
                f"The accepted detector {label} is unavailable: {path}.",
                code="accepted_environment_mismatch",
            ) from exc
        if not stat.S_ISREG(mode) or path.is_symlink() or sha256_file(path) != expected_sha256:
            _fail(
                f"The accepted detector {label} is unsafe or hash-drifted: {path}.",
                "accepted_environment_mismatch",
            )
    try:
        marker = json.loads(ACCEPTED_ENVIRONMENT_MARKER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionInfrastructureError(
            "The accepted detector environment marker is unreadable.",
            code="accepted_environment_mismatch",
        ) from exc
    expected_marker_fields = {
        "schema_version": "floodsight-locked-ml-environment-v1",
        "environment": str(ACCEPTED_ENVIRONMENT_ROOT),
        "python": ".".join(map(str, REQUIRED_PYTHON)),
        "torch": REQUIRED_VERSIONS["torch"],
        "torch_cuda_build": REQUIRED_TORCH_CUDA,
        "resolved_lock": str(ACCEPTED_RESOLVED_LOCK),
        "resolved_lock_sha256": ACCEPTED_RESOLVED_LOCK_SHA256,
        "pip_check": "PASS",
        "pip_audit": "PASS_WITH_EXACT_TORCH_CANONICAL_FALLBACK",
    }
    if not isinstance(marker, dict) or any(
        marker.get(key) != value for key, value in expected_marker_fields.items()
    ):
        _fail(
            "The accepted detector environment marker content does not match policy.",
            "accepted_environment_mismatch",
        )
    locked: dict[str, str] = {}
    for raw in ACCEPTED_RESOLVED_LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if separator != "==" or not name or not version or canonical in locked:
            _fail(
                "The accepted detector resolved lock is malformed.", "accepted_environment_mismatch"
            )
        locked[canonical] = version
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name:
            _fail("An installed distribution has no project name.", "accepted_environment_mismatch")
        canonical = re.sub(r"[-_.]+", "-", raw_name).lower()
        if canonical in installed:
            _fail(
                "The accepted detector environment has duplicate distributions.",
                "accepted_environment_mismatch",
            )
        installed[canonical] = distribution.version
    snapshot = "".join(f"{name}=={installed[name]}\n" for name in sorted(installed))
    import hashlib

    snapshot_sha256 = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
    if (
        installed != locked
        or len(installed) != ACCEPTED_INSTALLED_DISTRIBUTION_COUNT
        or snapshot_sha256 != ACCEPTED_INSTALLED_DISTRIBUTIONS_SHA256
    ):
        _fail(
            "Installed detector distributions differ from the accepted full snapshot.",
            "accepted_environment_mismatch",
        )
    return {
        "environment_root": str(root),
        "environment_marker_path": str(ACCEPTED_ENVIRONMENT_MARKER),
        "environment_marker_sha256": ACCEPTED_ENVIRONMENT_MARKER_SHA256,
        "resolved_lock_path": str(ACCEPTED_RESOLVED_LOCK),
        "resolved_lock_sha256": ACCEPTED_RESOLVED_LOCK_SHA256,
        "installed_distributions_sha256": snapshot_sha256,
        "installed_distribution_count": str(len(installed)),
    }


def validate_full_training_runtime(device: str) -> dict[str, Any]:
    """Require exact dependencies, offline assets, CUDA 13, and an H100."""

    accepted_environment = validate_accepted_environment()
    if sys.version_info[:3] != REQUIRED_PYTHON:
        found = ".".join(str(item) for item in sys.version_info[:3])
        _fail(f"Detector training requires CPython 3.12.3; found {found}.")
    versions: dict[str, str] = {}
    for distribution, required in REQUIRED_VERSIONS.items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DetectionInfrastructureError(
                f"Required detector distribution is missing: {distribution}.",
                code="ml_dependency_missing",
            ) from exc
        if installed != required:
            _fail(f"Detector requires {distribution}=={required}; found {installed}.")
        versions[distribution] = installed

    if os.environ.get("YOLO_OFFLINE", "").lower() != "true":
        _fail("Full detector training requires YOLO_OFFLINE=true.", "offline_runtime_required")
    expected_config_parent = RUNTIME_ROOT / "yolo-config"
    configured_parent = Path(os.environ.get("YOLO_CONFIG_DIR", "")).expanduser()
    if not configured_parent.is_absolute() or configured_parent.resolve() != expected_config_parent:
        _fail(
            "YOLO_CONFIG_DIR does not point at the audited FloodSight runtime.",
            "offline_runtime_required",
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        _fail("Deterministic CUDA workspace configuration is missing.")
    for path, expected_hash in RUNTIME_ASSET_HASHES.items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hash:
            _fail(f"Offline Ultralytics runtime asset is missing or drifted: {path}")

    import torch  # imported only after the explicit full-training authorization path

    if torch.__version__ != REQUIRED_VERSIONS["torch"] or torch.version.cuda != REQUIRED_TORCH_CUDA:
        _fail(
            f"Detector requires Torch {REQUIRED_VERSIONS['torch']} with CUDA {REQUIRED_TORCH_CUDA}."
        )
    if not device.isdigit():
        _fail("The frozen full-training device must be one CUDA index.", "training_device_drift")
    index = int(device)
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        _fail(f"Configured CUDA device {index} is unavailable.", "h100_required")
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    memory_bytes = torch.cuda.get_device_properties(index).total_memory
    if "H100" not in name or capability != (9, 0):
        _fail(
            f"Full detector training requires H100 capability 9.0; found {name} {capability}.",
            "h100_required",
        )
    return {
        **accepted_environment,
        "python": ".".join(map(str, REQUIRED_PYTHON)),
        **versions,
        "torch_cuda": REQUIRED_TORCH_CUDA,
        "device": index,
        "gpu_name": name,
        "gpu_capability": list(capability),
        "gpu_memory_bytes": memory_bytes,
        "offline_assets_verified": True,
    }
