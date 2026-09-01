#!/usr/bin/env python3
"""Fail-closed verifier for the shared accepted FloodSight ML environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import stat
import sys
from pathlib import Path

ENVIRONMENT_ROOT = Path(
    "/data/floodsight-workspace/floodsight-cache/envs/floodsight-ml-py312-cu130-locked-v1"
)
MARKER = ENVIRONMENT_ROOT / "FLOODSIGHT_ENVIRONMENT_ACCEPTED.json"
MARKER_SHA256 = "11ec5e2bc107465862ab04f8a01d58719c5012356489168cf28387f6848f96bd"
LOCK = Path("/data/floodsight-workspace/floodsight/ml/training/requirements-py312-cu130.lock")
LOCK_SHA256 = "33e7ca74a272659827d10c3bc882de1aa6e39b871c36435eb52279bd88eb58e1"
INSTALLED_SNAPSHOT_SHA256 = "ee7f9ce2704ddaea38312d0e11dacb8d01270f8be04ac1aad7e31095878ce775"
INSTALLED_DISTRIBUTION_COUNT = 103


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked() -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        canonical = _canonical_name(name)
        if separator != "==" or not name or not version or canonical in result:
            raise SystemExit(f"Invalid or duplicate resolved-lock entry: {line!r}")
        result[canonical] = version
    return result


def main() -> None:
    if (
        Path(sys.prefix) != ENVIRONMENT_ROOT
        or Path(sys.prefix).is_symlink()
        or Path(sys.executable).parent.parent != ENVIRONMENT_ROOT
    ):
        raise SystemExit("Execution is not inside the canonical accepted FloodSight venv.")
    for path, expected, label in (
        (MARKER, MARKER_SHA256, "acceptance marker"),
        (LOCK, LOCK_SHA256, "resolved lock"),
    ):
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise SystemExit(f"Missing {label}: {path}") from exc
        if not stat.S_ISREG(mode) or path.is_symlink() or _digest(path) != expected:
            raise SystemExit(f"Unsafe or hash-drifted {label}: {path}")
    marker = json.loads(MARKER.read_text(encoding="utf-8"))
    required_marker = {
        "schema_version": "floodsight-locked-ml-environment-v1",
        "environment": str(ENVIRONMENT_ROOT),
        "python": "3.12.3",
        "torch": "2.13.0+cu130",
        "torch_cuda_build": "13.0",
        "resolved_lock": str(LOCK),
        "resolved_lock_sha256": LOCK_SHA256,
        "pip_check": "PASS",
        "pip_audit": "PASS_WITH_EXACT_TORCH_CANONICAL_FALLBACK",
    }
    if not isinstance(marker, dict) or any(
        marker.get(key) != value for key, value in required_marker.items()
    ):
        raise SystemExit("Accepted-environment marker content drifted from policy.")
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name:
            raise SystemExit("Installed distribution has no canonical project name.")
        name = _canonical_name(raw_name)
        if name in installed:
            raise SystemExit(f"Duplicate installed distribution identity: {name}")
        installed[name] = distribution.version
    locked = _locked()
    snapshot = "".join(f"{name}=={installed[name]}\n" for name in sorted(installed))
    snapshot_sha256 = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
    if (
        installed != locked
        or len(installed) != INSTALLED_DISTRIBUTION_COUNT
        or snapshot_sha256 != INSTALLED_SNAPSHOT_SHA256
    ):
        raise SystemExit("Installed distributions differ from the exact accepted lock snapshot.")
    print(
        json.dumps(
            {
                "status": "PASS",
                "environment_marker_sha256": MARKER_SHA256,
                "resolved_lock_sha256": LOCK_SHA256,
                "installed_distributions_sha256": snapshot_sha256,
                "installed_distribution_count": len(installed),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
