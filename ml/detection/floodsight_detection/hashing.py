"""Small deterministic hashing helpers with no Phase 3 package dependency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_source_sha256() -> str:
    """Hash detector code plus every launcher and locked-runtime policy input."""

    package_root = Path(__file__).resolve().parent
    repo_root = package_root.parents[2]
    runtime_policy_paths = (
        repo_root / "scripts/training/run-locked.sh",
        repo_root / "scripts/training/runtime-offline.sh",
        repo_root / "scripts/training/check.sh",
        repo_root / "ml/training/verify-locked-environment.py",
        repo_root / "ml/training/requirements-py312-cu130.lock",
        repo_root / "ml/training/ultralytics-runtime-assets-v1.sha256",
        repo_root / "ml/training/ultralytics-settings-v1.json",
        repo_root / "ml/training/audit-canonical-requirements.txt",
    )
    records: list[dict[str, Any]] = []
    source_paths = [*package_root.rglob("*.py"), *runtime_policy_paths]
    for path in sorted(source_paths):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Training source must be a regular non-symlink file: {path}")
        payload = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not records:
        raise RuntimeError(f"No detector execution source found under {repo_root}")
    return stable_sha256(records)
