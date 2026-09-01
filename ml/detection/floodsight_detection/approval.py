"""Machine-readable human authorization bound to one full training run."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file, training_source_sha256

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_ACKNOWLEDGEMENTS = frozenset(
    {
        "FULL_TRAINING_EXPLICITLY_AUTHORIZED",
        "DATASET_AND_LABEL_REVIEW_COMPLETE",
        "LICENSE_REVIEW_COMPLETE",
        "HUMAN_DECISION_SUPPORT_ONLY",
        "REAL_SMOKE_REVIEW_COMPLETE",
    }
)
_DEFERRED_REVIEW_ACKNOWLEDGEMENTS = frozenset(
    {
        "FULL_TRAINING_EXPLICITLY_AUTHORIZED",
        "DATASET_AND_LABEL_REVIEW_DEFERRED_BY_USER",
        "LICENSE_REVIEW_DEFERRED_BY_USER",
        "HUMAN_DECISION_SUPPORT_ONLY",
        "REAL_SMOKE_TECHNICALLY_VERIFIED",
        "USER_OVERRIDE_HASH_BOUND",
    }
)
_STANDARD_APPROVAL_SCHEMA = "floodsight-full-training-approval-v4"
_DEFERRED_APPROVAL_SCHEMA = "floodsight-full-training-approval-v5"
_USER_LAUNCH_OVERRIDE_SCHEMA = "floodsight-user-training-launch-override-v1"
_USER_LAUNCH_OVERRIDE_DECISION = (
    "AUTHORIZE_PRODUCTION_TRAINING_WHEN_EXISTING_TECHNICAL_REQUIREMENTS_PASS_"
    "AND_A_SAFE_H100_IS_AVAILABLE"
)


@dataclass(frozen=True, slots=True)
class TrainingApproval:
    path: Path
    sha256: str
    approval_id: str
    approved_by: str
    training_code_sha256: str
    output_root: Path
    device: str
    manifest_id: str
    dataset_id: str
    preparation_version: str
    taxonomy_version: str
    taxonomy_sha256: str
    mapping_version: str
    mapping_sha256: str
    weights_path: Path
    weight_audit_path: Path
    weight_audit_sha256: str
    real_smoke_report_path: Path
    real_smoke_report_sha256: str
    human_review_path: Path
    human_review_sha256: str
    review_disposition: str


def _validate_deferred_user_override(
    path: Path,
    *,
    expected_sha256: str,
    approved_by: str,
    approved_at: str,
) -> None:
    if sha256_file(path) != expected_sha256:
        raise DetectionInfrastructureError(
            "User launch override SHA-256 does not match.",
            code="training_approval_mismatch",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionInfrastructureError(
            "Unable to read the bound user launch override.",
            code="training_approval_invalid",
        ) from exc
    required = {
        "schema_version",
        "instruction_accepted_utc",
        "target_training_launch_utc",
        "instruction_source_path",
        "instruction_source_sha256",
        "authorized_by",
        "decision",
        "authorized_models",
        "human_review_status",
        "provenance_review_status",
        "human_review_completed",
        "provenance_review_completed",
        "full_training_explicitly_authorized",
        "additional_broad_audits_authorized",
        "persistent_tmux_required",
        "source_freeze_required_before_each_launch",
        "notes",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise DetectionInfrastructureError(
            "User launch override fields do not match the frozen schema.",
            code="training_approval_invalid",
        )
    if (
        payload["schema_version"] != _USER_LAUNCH_OVERRIDE_SCHEMA
        or payload["decision"] != _USER_LAUNCH_OVERRIDE_DECISION
        or payload["authorized_by"] != approved_by
    ):
        raise DetectionInfrastructureError(
            "User launch override identity or authorization is invalid.",
            code="training_not_authorized",
        )
    timestamp_fields = {
        "instruction_accepted_utc": payload["instruction_accepted_utc"],
        "target_training_launch_utc": payload["target_training_launch_utc"],
        "approved_at": approved_at,
    }
    if any(
        not isinstance(value, str) or not value.strip()
        for value in timestamp_fields.values()
    ):
        raise DetectionInfrastructureError(
            "User launch override timestamps are invalid.",
            code="training_approval_invalid",
        )
    try:
        accepted_time = datetime.fromisoformat(
            payload["instruction_accepted_utc"].replace("Z", "+00:00")
        )
        target_time = datetime.fromisoformat(
            payload["target_training_launch_utc"].replace("Z", "+00:00")
        )
        approval_time = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DetectionInfrastructureError(
            "User launch override timestamps are invalid.",
            code="training_approval_invalid",
        ) from exc
    if (
        accepted_time.tzinfo is None
        or target_time.tzinfo is None
        or approval_time.tzinfo is None
        or accepted_time > approval_time
        or target_time < accepted_time
    ):
        raise DetectionInfrastructureError(
            "User launch override timestamps are inconsistent.",
            code="training_approval_invalid",
        )
    instruction_value = payload["instruction_source_path"]
    instruction_hash = payload["instruction_source_sha256"]
    if (
        not isinstance(instruction_value, str)
        or not Path(instruction_value).is_absolute()
        or not isinstance(instruction_hash, str)
        or _HEX64.fullmatch(instruction_hash) is None
    ):
        raise DetectionInfrastructureError(
            "User launch instruction binding is invalid.",
            code="training_approval_invalid",
        )
    instruction_path = Path(instruction_value)
    if (
        instruction_path.is_symlink()
        or not instruction_path.is_file()
        or sha256_file(instruction_path.resolve()) != instruction_hash
    ):
        raise DetectionInfrastructureError(
            "User launch instruction binding does not match current bytes.",
            code="training_approval_mismatch",
        )
    models = payload["authorized_models"]
    expected_strings = {
        "human_review_status": "DEFERRED_BY_USER",
        "provenance_review_status": "DEFERRED_BY_USER",
    }
    expected_booleans = {
        "human_review_completed": False,
        "provenance_review_completed": False,
        "full_training_explicitly_authorized": True,
        "additional_broad_audits_authorized": False,
        "persistent_tmux_required": True,
        "source_freeze_required_before_each_launch": True,
    }
    if (
        not isinstance(models, list)
        or set(models) != {"SEGFORMER", "YOLO"}
        or len(models) != 2
        or any(payload[key] != value for key, value in expected_strings.items())
        or any(
            type(payload[key]) is not bool or payload[key] is not value
            for key, value in expected_booleans.items()
        )
    ):
        raise DetectionInfrastructureError(
            "User launch override does not preserve deferred-review safety boundaries.",
            code="training_approval_invalid",
        )
    notes = payload["notes"]
    if (
        not isinstance(notes, list)
        or not notes
        or any(not isinstance(note, str) or not note.strip() for note in notes)
    ):
        raise DetectionInfrastructureError(
            "User launch override notes are invalid.",
            code="training_approval_invalid",
        )


def load_training_approval(
    path: str | Path,
    *,
    config_sha256: str,
    manifest_sha256: str,
    dataset_fingerprint: str,
    weights_sha256: str,
    weights_path: str | Path,
    weight_audit_path: str | Path,
    weight_audit_sha256: str,
    run_name: str,
    output_root: str | Path,
    device: str,
    manifest_id: str,
    dataset_id: str,
    preparation_version: str,
    taxonomy_version: str,
    taxonomy_sha256: str,
    mapping_version: str,
    mapping_sha256: str,
    real_smoke_report_path: str | Path,
    real_smoke_report_sha256: str,
) -> TrainingApproval:
    declared_path = Path(path).expanduser()
    if declared_path.is_symlink():
        raise DetectionInfrastructureError(
            "Human training approval must not be a symbolic link.",
            code="training_approval_invalid",
        )
    approval_path = declared_path.resolve(strict=True)
    try:
        payload = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionInfrastructureError(
            f"Unable to read human training approval: {approval_path}",
            code="training_approval_invalid",
        ) from exc
    required = {
        "schema_version",
        "approval_id",
        "decision",
        "approved_by",
        "approved_at",
        "run_name",
        "output_root",
        "device",
        "training_code_sha256",
        "config_sha256",
        "manifest_id",
        "dataset_id",
        "preparation_version",
        "manifest_sha256",
        "dataset_fingerprint",
        "taxonomy_version",
        "taxonomy_sha256",
        "mapping_version",
        "mapping_sha256",
        "weights_path",
        "weights_sha256",
        "weight_audit_path",
        "weight_audit_sha256",
        "real_smoke_report_path",
        "real_smoke_report_sha256",
        "human_review_path",
        "human_review_sha256",
        "acknowledgements",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise DetectionInfrastructureError(
            "Human approval fields do not match the frozen schema.",
            code="training_approval_invalid",
        )
    approval_schema = payload["schema_version"]
    if (
        approval_schema not in {_STANDARD_APPROVAL_SCHEMA, _DEFERRED_APPROVAL_SCHEMA}
        or payload["decision"] != "APPROVE_FULL_TRAINING"
    ):
        raise DetectionInfrastructureError(
            "Human approval does not explicitly authorize full training.",
            code="training_not_authorized",
        )
    canonical_output_root = Path(output_root).expanduser().resolve()
    canonical_weights_path = Path(weights_path).expanduser().resolve()
    canonical_weight_audit_path = Path(weight_audit_path).expanduser().resolve()
    declared_real_smoke_path = Path(real_smoke_report_path).expanduser()
    if (
        declared_real_smoke_path.is_symlink()
        or not declared_real_smoke_path.is_absolute()
        or not declared_real_smoke_path.is_file()
    ):
        raise DetectionInfrastructureError(
            "Expected real-smoke report is unsafe.", code="training_approval_invalid"
        )
    canonical_real_smoke_path = declared_real_smoke_path.resolve(strict=True)
    expected: dict[str, Any] = {
        "run_name": run_name,
        "output_root": str(canonical_output_root),
        "device": device,
        "training_code_sha256": training_source_sha256(),
        "config_sha256": config_sha256,
        "manifest_id": manifest_id,
        "dataset_id": dataset_id,
        "preparation_version": preparation_version,
        "manifest_sha256": manifest_sha256,
        "dataset_fingerprint": dataset_fingerprint,
        "taxonomy_version": taxonomy_version,
        "taxonomy_sha256": taxonomy_sha256,
        "mapping_version": mapping_version,
        "mapping_sha256": mapping_sha256,
        "weights_path": str(canonical_weights_path),
        "weights_sha256": weights_sha256,
        "weight_audit_path": str(canonical_weight_audit_path),
        "weight_audit_sha256": weight_audit_sha256,
        "real_smoke_report_path": str(canonical_real_smoke_path),
        "real_smoke_report_sha256": real_smoke_report_sha256,
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if drift:
        raise DetectionInfrastructureError(
            "Human approval is not bound to this exact frozen run.",
            code="training_approval_mismatch",
            details=[drift],
        )
    hash_fields = {
        "training_code_sha256",
        "config_sha256",
        "manifest_sha256",
        "dataset_fingerprint",
        "taxonomy_sha256",
        "mapping_sha256",
        "weights_sha256",
        "weight_audit_sha256",
        "real_smoke_report_sha256",
    }
    if any(
        not isinstance(payload[key], str) or _HEX64.fullmatch(payload[key]) is None
        for key in hash_fields
    ):
        raise DetectionInfrastructureError(
            "Human approval contains an invalid SHA-256.", code="training_approval_invalid"
        )
    acknowledgements = payload["acknowledgements"]
    expected_acknowledgements = (
        _DEFERRED_REVIEW_ACKNOWLEDGEMENTS
        if approval_schema == _DEFERRED_APPROVAL_SCHEMA
        else _REQUIRED_ACKNOWLEDGEMENTS
    )
    if (
        not isinstance(acknowledgements, list)
        or frozenset(acknowledgements) != expected_acknowledgements
    ):
        raise DetectionInfrastructureError(
            "Human approval acknowledgements are incomplete.",
            code="training_approval_invalid",
        )
    for field in ("approval_id", "approved_by", "approved_at"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise DetectionInfrastructureError(
                f"Human approval field {field} is missing.", code="training_approval_invalid"
            )
    for field in ("weights_path", "weight_audit_path", "real_smoke_report_path"):
        if not isinstance(payload[field], str) or not Path(payload[field]).is_absolute():
            raise DetectionInfrastructureError(
                f"Human approval field {field} must be an absolute path.",
                code="training_approval_invalid",
            )
    if canonical_real_smoke_path.is_symlink() or not canonical_real_smoke_path.is_file():
        raise DetectionInfrastructureError(
            "Human approval real-smoke report is unsafe.",
            code="training_approval_invalid",
        )
    if sha256_file(canonical_real_smoke_path) != real_smoke_report_sha256:
        raise DetectionInfrastructureError(
            "Human approval real-smoke report hash does not match current bytes.",
            code="training_approval_mismatch",
        )
    review_value = payload["human_review_path"]
    if not isinstance(review_value, str) or not Path(review_value).is_absolute():
        raise DetectionInfrastructureError(
            "Human approval requires an absolute review-report path.",
            code="training_approval_invalid",
        )
    review_declared = Path(review_value)
    if review_declared.is_symlink():
        raise DetectionInfrastructureError(
            "Human review report must not be a symbolic link.",
            code="training_approval_invalid",
        )
    review_path = review_declared.resolve(strict=True)
    review_hash = payload["human_review_sha256"]
    if not isinstance(review_hash, str) or _HEX64.fullmatch(review_hash) is None:
        raise DetectionInfrastructureError(
            "Human review report SHA-256 is invalid.", code="training_approval_invalid"
        )
    if sha256_file(review_path) != review_hash:
        raise DetectionInfrastructureError(
            "Human review report SHA-256 does not match.", code="training_approval_mismatch"
        )
    review_disposition = "COMPLETED"
    if approval_schema == _DEFERRED_APPROVAL_SCHEMA:
        _validate_deferred_user_override(
            review_path,
            expected_sha256=review_hash,
            approved_by=payload["approved_by"],
            approved_at=payload["approved_at"],
        )
        review_disposition = "DEFERRED_BY_USER"
    return TrainingApproval(
        path=approval_path,
        sha256=sha256_file(approval_path),
        approval_id=payload["approval_id"],
        approved_by=payload["approved_by"],
        training_code_sha256=payload["training_code_sha256"],
        output_root=canonical_output_root,
        device=payload["device"],
        manifest_id=payload["manifest_id"],
        dataset_id=payload["dataset_id"],
        preparation_version=payload["preparation_version"],
        taxonomy_version=payload["taxonomy_version"],
        taxonomy_sha256=payload["taxonomy_sha256"],
        mapping_version=payload["mapping_version"],
        mapping_sha256=payload["mapping_sha256"],
        weights_path=canonical_weights_path,
        weight_audit_path=canonical_weight_audit_path,
        weight_audit_sha256=payload["weight_audit_sha256"],
        real_smoke_report_path=canonical_real_smoke_path,
        real_smoke_report_sha256=payload["real_smoke_report_sha256"],
        human_review_path=review_path,
        human_review_sha256=review_hash,
        review_disposition=review_disposition,
    )
