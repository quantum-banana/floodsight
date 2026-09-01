"""Collision-safe run reservation and checkpoint-resume validation."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from floodsight_detection.checkpointing import (
    TrustedCheckpoint,
    load_trusted_checkpoint,
    write_json_exclusive,
)
from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import stable_sha256

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_METADATA = ".floodsight-run.json"
_PROCESS_LOCK = ".floodsight-training.lock"
_SCHEMA = "floodsight-detection-run-v4"


@dataclass(frozen=True, slots=True)
class RunReservation:
    run_directory: Path
    metadata_path: Path
    resumed: bool
    checkpoint: Path | None
    trusted_checkpoint: TrustedCheckpoint | None = None


@contextmanager
def exclusive_run_lock(run: RunReservation) -> Iterator[Path]:
    """Hold a nonblocking OS lock for the complete train/resume operation."""

    path = run.run_directory / _PROCESS_LOCK
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise DetectionInfrastructureError(
            "Detection run lock path is unsafe.", code="run_lock_invalid"
        )
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise DetectionInfrastructureError(
            "Unable to open the detection run lock.", code="run_lock_invalid"
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise DetectionInfrastructureError(
                "Detection run lock is not a regular file.", code="run_lock_invalid"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DetectionInfrastructureError(
                "Another process already owns this detection run.",
                code="run_already_active",
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        yield path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_exclusive(path, payload, mode=0o440)


def _metadata_payload(
    root: Path,
    run_name: str,
    *,
    config_sha256: str,
    training_code_sha256: str,
    manifest_id: str,
    dataset_id: str,
    preparation_version: str,
    manifest_sha256: str,
    dataset_fingerprint: str,
    taxonomy_version: str,
    taxonomy_sha256: str,
    mapping_version: str,
    mapping_sha256: str,
    weights_path: str,
    weights_sha256: str,
    weight_audit_path: str,
    weight_audit_sha256: str,
    approval_sha256: str,
    approval_id: str,
    real_smoke_report_path: str,
    real_smoke_report_sha256: str,
    device: str,
) -> dict[str, Any]:
    if _RUN_NAME.fullmatch(run_name) is None:
        raise DetectionInfrastructureError(
            "Run name must be 1-96 safe filename characters.", code="run_name_invalid"
        )
    hashes = {
        "config_sha256": config_sha256,
        "training_code_sha256": training_code_sha256,
        "manifest_sha256": manifest_sha256,
        "dataset_fingerprint": dataset_fingerprint,
        "taxonomy_sha256": taxonomy_sha256,
        "mapping_sha256": mapping_sha256,
        "weights_sha256": weights_sha256,
        "weight_audit_sha256": weight_audit_sha256,
        "approval_sha256": approval_sha256,
        "real_smoke_report_sha256": real_smoke_report_sha256,
    }
    invalid_hash = any(
        not isinstance(value, str) or _HEX64.fullmatch(value) is None for value in hashes.values()
    )
    if invalid_hash:
        raise DetectionInfrastructureError(
            "Run identity contains an invalid SHA-256.", code="run_identity_invalid"
        )
    strings = {
        "manifest_id": manifest_id,
        "dataset_id": dataset_id,
        "preparation_version": preparation_version,
        "taxonomy_version": taxonomy_version,
        "mapping_version": mapping_version,
        "approval_id": approval_id,
        "real_smoke_report_path": real_smoke_report_path,
        "device": device,
        "weights_path": weights_path,
        "weight_audit_path": weight_audit_path,
    }
    if any(not isinstance(value, str) or not value for value in strings.values()):
        raise DetectionInfrastructureError(
            "Run identity contains an empty or non-string field.", code="run_identity_invalid"
        )
    if (
        not Path(weights_path).is_absolute()
        or not Path(weight_audit_path).is_absolute()
        or not Path(real_smoke_report_path).is_absolute()
    ):
        raise DetectionInfrastructureError(
            "Run model, audit, and real-smoke identities must use absolute paths.",
            code="run_identity_invalid",
        )
    return {
        "schema_version": _SCHEMA,
        "status": "RESERVED",
        "run_name": run_name,
        "output_root": str(root),
        **hashes,
        **strings,
    }


def reserve_new_run(
    output_root: str | Path,
    run_name: str,
    *,
    config_sha256: str,
    training_code_sha256: str,
    manifest_id: str,
    dataset_id: str,
    preparation_version: str,
    manifest_sha256: str,
    dataset_fingerprint: str,
    taxonomy_version: str,
    taxonomy_sha256: str,
    mapping_version: str,
    mapping_sha256: str,
    weights_path: str,
    weights_sha256: str,
    weight_audit_path: str,
    weight_audit_sha256: str,
    approval_sha256: str,
    approval_id: str,
    real_smoke_report_path: str,
    real_smoke_report_sha256: str,
    device: str,
) -> RunReservation:
    root = Path(output_root).expanduser().resolve()
    metadata = _metadata_payload(
        root,
        run_name,
        config_sha256=config_sha256,
        training_code_sha256=training_code_sha256,
        manifest_id=manifest_id,
        dataset_id=dataset_id,
        preparation_version=preparation_version,
        manifest_sha256=manifest_sha256,
        dataset_fingerprint=dataset_fingerprint,
        taxonomy_version=taxonomy_version,
        taxonomy_sha256=taxonomy_sha256,
        mapping_version=mapping_version,
        mapping_sha256=mapping_sha256,
        weights_path=weights_path,
        weights_sha256=weights_sha256,
        weight_audit_path=weight_audit_path,
        weight_audit_sha256=weight_audit_sha256,
        approval_sha256=approval_sha256,
        approval_id=approval_id,
        real_smoke_report_path=real_smoke_report_path,
        real_smoke_report_sha256=real_smoke_report_sha256,
        device=device,
    )
    root.mkdir(parents=True, exist_ok=True)
    run = root / run_name
    try:
        run.mkdir(mode=0o750, exist_ok=False)
    except FileExistsError as exc:
        raise DetectionInfrastructureError(
            f"Refusing to overwrite or reuse run directory: {run}", code="run_collision"
        ) from exc
    metadata["reservation_sha256"] = stable_sha256(metadata)
    metadata_path = run / _METADATA
    _write_exclusive_json(metadata_path, metadata)
    return RunReservation(run, metadata_path, False, None)


def validate_resume(
    checkpoint: str | Path,
    output_root: str | Path,
    *,
    run_name: str,
    config_sha256: str,
    training_code_sha256: str,
    manifest_id: str,
    dataset_id: str,
    preparation_version: str,
    manifest_sha256: str,
    dataset_fingerprint: str,
    taxonomy_version: str,
    taxonomy_sha256: str,
    mapping_version: str,
    mapping_sha256: str,
    weights_path: str,
    weights_sha256: str,
    weight_audit_path: str,
    weight_audit_sha256: str,
    approval_sha256: str,
    approval_id: str,
    real_smoke_report_path: str,
    real_smoke_report_sha256: str,
    device: str,
    checkpoint_trainer_arguments: dict[str, Any],
    last_checkpoint_filename: str = "last.pt",
) -> RunReservation:
    root = Path(output_root).expanduser().resolve(strict=True)
    declared_candidate = Path(checkpoint).expanduser()
    if declared_candidate.is_symlink():
        raise DetectionInfrastructureError(
            "Resume checkpoint must not be a symbolic link.",
            code="unsafe_resume_checkpoint",
        )
    candidate = declared_candidate.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DetectionInfrastructureError(
            "Resume checkpoint must be contained by the configured output root.",
            code="unsafe_resume_checkpoint",
        ) from exc
    if (
        last_checkpoint_filename != "last.pt"
        or not candidate.is_file()
        or candidate.suffix != ".pt"
        or candidate.name != last_checkpoint_filename
    ):
        raise DetectionInfrastructureError(
            "Resume requires the exact run weights/last.pt checkpoint.",
            code="resume_checkpoint_invalid",
        )
    if candidate.parent.name != "weights":
        raise DetectionInfrastructureError(
            "Resume checkpoint must live in a weights directory.", code="resume_checkpoint_invalid"
        )
    run = candidate.parent.parent
    if run.parent != root or run.name != run_name:
        raise DetectionInfrastructureError(
            "Resume checkpoint is not in the exact approved run directory.",
            code="resume_contract_mismatch",
        )
    metadata_path = run / _METADATA
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise DetectionInfrastructureError(
            "Resume run metadata is missing or unsafe.", code="resume_metadata_invalid"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionInfrastructureError(
            "Resume run metadata is missing or invalid.", code="resume_metadata_invalid"
        ) from exc
    if not isinstance(metadata, dict):
        raise DetectionInfrastructureError(
            "Resume run metadata must be an object.", code="resume_metadata_invalid"
        )
    expected = _metadata_payload(
        root,
        run_name,
        config_sha256=config_sha256,
        training_code_sha256=training_code_sha256,
        manifest_id=manifest_id,
        dataset_id=dataset_id,
        preparation_version=preparation_version,
        manifest_sha256=manifest_sha256,
        dataset_fingerprint=dataset_fingerprint,
        taxonomy_version=taxonomy_version,
        taxonomy_sha256=taxonomy_sha256,
        mapping_version=mapping_version,
        mapping_sha256=mapping_sha256,
        weights_path=weights_path,
        weights_sha256=weights_sha256,
        weight_audit_path=weight_audit_path,
        weight_audit_sha256=weight_audit_sha256,
        approval_sha256=approval_sha256,
        approval_id=approval_id,
        real_smoke_report_path=real_smoke_report_path,
        real_smoke_report_sha256=real_smoke_report_sha256,
        device=device,
    )
    required_fields = set(expected) | {"reservation_sha256"}
    if set(metadata) != required_fields:
        raise DetectionInfrastructureError(
            "Resume metadata fields do not match the frozen schema.",
            code="resume_metadata_invalid",
        )
    reservation_sha256 = metadata.get("reservation_sha256")
    if (
        not isinstance(reservation_sha256, str)
        or _HEX64.fullmatch(reservation_sha256) is None
        or stable_sha256(
            {key: value for key, value in metadata.items() if key != "reservation_sha256"}
        )
        != reservation_sha256
    ):
        raise DetectionInfrastructureError(
            "Resume reservation metadata failed its integrity check.",
            code="resume_metadata_invalid",
        )
    drift = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if drift:
        raise DetectionInfrastructureError(
            "Resume source, approval, runtime, or dataset identity drifted.",
            code="resume_contract_mismatch",
            details=[drift],
        )
    trusted = load_trusted_checkpoint(
        run_directory=run,
        run_metadata_path=metadata_path,
        declared_live_checkpoint=candidate,
        expected_trainer_arguments=checkpoint_trainer_arguments,
    )
    return RunReservation(run, metadata_path, True, trusted.path, trusted)
