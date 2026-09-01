"""Deterministic fingerprint for the executable SegFormer package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def training_source_sha256() -> str:
    """Hash SegFormer Python plus the shared locked-runtime launch policy."""

    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parents[2]
    records: list[dict[str, object]] = []
    source_paths = list(package_root.rglob("*.py"))
    source_paths.extend(
        repository_root / relative
        for relative in (
            "scripts/training/run-locked.sh",
            "scripts/training/runtime-offline.sh",
            "ml/training/verify-locked-environment.py",
            "ml/training/requirements-py312-cu130.lock",
            "ml/training/accepted-wheelhouse.sha256",
            "ml/training/ultralytics-runtime-assets-v1.json",
            "ml/training/ultralytics-runtime-assets-v1.sha256",
            "ml/training/ultralytics-settings-v1.json",
        )
    )
    for path in sorted(source_paths):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Training source must be a regular non-symlink file: {path}")
        payload = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(repository_root).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not records:
        raise RuntimeError(f"No SegFormer training source found under {package_root}")
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
