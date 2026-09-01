"""Audit-bound local pretrained weight artifacts with no network fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LICENSE_REVIEW_STATES = frozenset({"PENDING_HUMAN_SIGNOFF", "APPROVED_FOR_RESEARCH_DEMO"})


@dataclass(frozen=True, slots=True)
class WeightArtifact:
    audit_path: Path
    audit_sha256: str
    path: Path
    sha256: str
    architecture: str
    source_url: str
    source_release: str
    reviewed_by: str
    license_review_status: str


def load_weight_audit(
    path: str | Path,
    *,
    expected_filename: str,
    expected_weight_path: str | Path,
    expected_weight_sha256: str,
    expected_audit_path: str | Path,
    expected_audit_sha256: str,
    require_license_approval: bool = True,
) -> WeightArtifact:
    if (
        _HEX64.fullmatch(expected_weight_sha256) is None
        or _HEX64.fullmatch(expected_audit_sha256) is None
    ):
        raise DetectionInfrastructureError(
            "Expected pretrained artifact identity has an invalid SHA-256.",
            code="weight_audit_invalid",
        )
    declared_audit_path = Path(path).expanduser()
    expected_audit_declared = Path(expected_audit_path).expanduser()
    if declared_audit_path.is_symlink() or expected_audit_declared.is_symlink():
        raise DetectionInfrastructureError(
            "Pretrained-weight audit must not be a symbolic link.",
            code="weight_audit_identity_mismatch",
        )
    audit_path = declared_audit_path.resolve(strict=True)
    canonical_audit_path = expected_audit_declared.resolve(strict=True)
    if audit_path != canonical_audit_path:
        raise DetectionInfrastructureError(
            "Pretrained-weight audit path differs from the frozen configuration.",
            code="weight_audit_identity_mismatch",
        )
    audit_sha256 = sha256_file(audit_path)
    if audit_sha256 != expected_audit_sha256:
        raise DetectionInfrastructureError(
            "Pretrained-weight audit SHA-256 differs from the frozen configuration.",
            code="weight_audit_identity_mismatch",
        )
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionInfrastructureError(
            f"Unable to read pretrained-weight audit: {audit_path}",
            code="weight_audit_invalid",
        ) from exc
    required = {
        "schema_version",
        "architecture",
        "local_path",
        "sha256",
        "source_url",
        "source_release",
        "license_review_status",
        "reviewed_by",
        "reviewed_at",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise DetectionInfrastructureError(
            "Pretrained-weight audit fields do not match the frozen schema.",
            code="weight_audit_invalid",
        )
    if payload["schema_version"] != "floodsight-yolo-weight-audit-v1":
        raise DetectionInfrastructureError(
            "Unsupported pretrained-weight audit schema.", code="weight_audit_invalid"
        )
    local_value = payload["local_path"]
    if not isinstance(local_value, str) or not Path(local_value).is_absolute():
        raise DetectionInfrastructureError(
            "Pretrained weights require an explicit absolute local path.",
            code="weight_path_invalid",
        )
    declared_weight_path = Path(local_value)
    expected_weight_declared = Path(expected_weight_path).expanduser()
    if declared_weight_path.is_symlink() or expected_weight_declared.is_symlink():
        raise DetectionInfrastructureError(
            "Pretrained weights must not be a symbolic link.", code="weight_path_invalid"
        )
    weight_path = declared_weight_path.resolve(strict=True)
    canonical_weight_path = expected_weight_declared.resolve(strict=True)
    if (
        not weight_path.is_file()
        or weight_path.name != expected_filename
        or weight_path != canonical_weight_path
    ):
        raise DetectionInfrastructureError(
            f"Expected the exact config-bound local {expected_filename} file.",
            code="weight_path_invalid",
        )
    expected_hash = payload["sha256"]
    if not isinstance(expected_hash, str) or _HEX64.fullmatch(expected_hash) is None:
        raise DetectionInfrastructureError(
            "Pretrained-weight audit has an invalid SHA-256.", code="weight_audit_invalid"
        )
    actual_hash = sha256_file(weight_path)
    if expected_hash != expected_weight_sha256 or actual_hash != expected_weight_sha256:
        raise DetectionInfrastructureError(
            "Pretrained-weight SHA-256 does not match its config and audit record.",
            code="weight_hash_mismatch",
        )
    if payload["architecture"] != "yolo11l":
        raise DetectionInfrastructureError(
            "Pretrained-weight architecture must be yolo11l.", code="weight_audit_invalid"
        )
    for field in ("source_url", "source_release", "reviewed_by", "reviewed_at"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise DetectionInfrastructureError(
                f"Pretrained-weight audit field {field} is missing.",
                code="weight_audit_invalid",
            )
    if not str(payload["source_url"]).startswith("https://"):
        raise DetectionInfrastructureError(
            "Pretrained-weight source_url must be HTTPS.", code="weight_audit_invalid"
        )
    license_review_status = payload["license_review_status"]
    if license_review_status not in _LICENSE_REVIEW_STATES:
        raise DetectionInfrastructureError(
            "Pretrained-weight license review has an unsupported state.",
            code="weight_audit_invalid",
        )
    if require_license_approval and license_review_status != "APPROVED_FOR_RESEARCH_DEMO":
        raise DetectionInfrastructureError(
            "Full training requires approved pretrained-weight license review.",
            code="weight_license_not_approved",
        )
    return WeightArtifact(
        audit_path=audit_path,
        audit_sha256=audit_sha256,
        path=weight_path,
        sha256=actual_hash,
        architecture="yolo11l",
        source_url=payload["source_url"],
        source_release=payload["source_release"],
        reviewed_by=payload["reviewed_by"],
        license_review_status=license_review_status,
    )


def validate_training_license_disposition(
    weights: WeightArtifact,
    *,
    review_disposition: str,
) -> None:
    """Accept completed review or an exact, separately validated user deferral."""

    if weights.license_review_status == "APPROVED_FOR_RESEARCH_DEMO":
        return
    if (
        weights.license_review_status == "PENDING_HUMAN_SIGNOFF"
        and review_disposition == "DEFERRED_BY_USER"
    ):
        return
    raise DetectionInfrastructureError(
        "Full training requires approved license review or an exact hash-bound "
        "DEFERRED_BY_USER production override.",
        code="weight_license_not_approved",
    )
