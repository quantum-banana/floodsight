"""Machine-readable human approval gate bound to every immutable run input."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifact import ModelArtifactSpec
from .errors import TrainingAuthorizationError
from .integrity import training_source_sha256
from .manifest import (
    CANONICAL_MANIFEST_LOCKS,
    SHA256_PATTERN,
    ManifestSpec,
    sha256_file,
)

APPROVAL_SCHEMA = "floodsight-training-approval-v3"
HUMAN_REVIEW_SCHEMA = "floodsight-segmentation-human-review-v1"
USER_LAUNCH_OVERRIDE_SCHEMA = "floodsight-user-training-launch-override-v1"
USER_LAUNCH_OVERRIDE_DECISION = (
    "AUTHORIZE_PRODUCTION_TRAINING_WHEN_EXISTING_TECHNICAL_REQUIREMENTS_PASS_"
    "AND_A_SAFE_H100_IS_AVAILABLE"
)
REQUIRED_REVIEW_ACKNOWLEDGEMENTS = frozenset(
    {
        "segmentation_taxonomy_mapping_reviewed",
        "pool_separate_from_water",
        "partial_supervision_reviewed",
        "cross_dataset_leakage_reviewed",
        "real_smoke_reviewed",
        "dataset_provenance_reviewed",
        "license_discrepancy_acknowledged",
        "training_not_yet_started",
    }
)
ALLOWED_OPERATIONS = frozenset({"TRAIN", "VALIDATE"})


@dataclass(frozen=True, slots=True)
class HumanApproval:
    approval_id: str
    approved_by: str
    approved_at: str
    operations: frozenset[str]
    record_path: Path
    record_sha256: str
    human_review_path: Path
    human_review_sha256: str
    real_smoke_path: Path
    real_smoke_sha256: str
    training_code_sha256: str
    manifest_fingerprints: Mapping[str, str]
    taxonomy_sha256: Mapping[str, str]
    run_directory: Path


def _fail(message: str) -> TrainingAuthorizationError:
    return TrainingAuthorizationError(message)


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"Expected a non-empty string at approval.{location}.")
    return value


def _sha(value: Any, location: str) -> str:
    digest = _string(value, location).lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise _fail(f"Expected a lowercase SHA-256 at approval.{location}.")
    return digest


def _absolute_path(value: Any, location: str) -> Path:
    raw = Path(_string(value, location)).expanduser()
    if not raw.is_absolute():
        raise _fail(f"Approval path must be absolute at {location}.")
    if raw.is_symlink():
        raise _fail(f"Approval path must not be a symbolic link at {location}.")
    return raw.resolve()


def _normalized_manifest_hashes(specs: Sequence[ManifestSpec]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path, digest, _fingerprint in specs:
        normalized_digest = digest.lower()
        if not SHA256_PATTERN.fullmatch(normalized_digest):
            raise _fail(f"Invalid declared manifest SHA-256 for {path}.")
        normalized_path = str(path.expanduser().resolve())
        if normalized_path in hashes and hashes[normalized_path] != normalized_digest:
            raise _fail(f"Conflicting hashes declared for manifest {normalized_path}.")
        hashes[normalized_path] = normalized_digest
    if not hashes:
        raise _fail("At least one frozen manifest must be bound to the approval.")
    return dict(sorted(hashes.items()))


def _normalized_manifest_fingerprints(specs: Sequence[ManifestSpec]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for path, _digest, fingerprint in specs:
        normalized = fingerprint.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise _fail(f"Invalid declared manifest fingerprint for {path}.")
        normalized_path = str(path.expanduser().resolve())
        if normalized_path in fingerprints and fingerprints[normalized_path] != normalized:
            raise _fail(f"Conflicting fingerprints declared for manifest {normalized_path}.")
        fingerprints[normalized_path] = normalized
    if not fingerprints:
        raise _fail("At least one manifest fingerprint must be bound to the approval.")
    return dict(sorted(fingerprints.items()))


def _normalized_path_hashes(values: Mapping[str, str], *, location: str) -> dict[str, str]:
    if not values:
        raise _fail(f"At least one path/hash is required at approval.{location}.")
    normalized: dict[str, str] = {}
    for raw_path, raw_digest in values.items():
        path = str(_absolute_path(raw_path, f"{location} path"))
        digest = _sha(raw_digest, f"{location}.{raw_path}")
        if path in normalized and normalized[path] != digest:
            raise _fail(f"Conflicting hashes declared at approval.{location} for {path}.")
        normalized[path] = digest
    return dict(sorted(normalized.items()))


def _validate_human_review(
    path: Path,
    *,
    approved_by: str,
    approval_time: datetime,
) -> None:
    """Require a substantive, blocker-free review record rather than an opaque file hash."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail(f"Unable to parse bound human-review report: {path}") from exc
    if not isinstance(raw, Mapping):
        raise _fail("Bound human-review report must be a JSON object.")
    if raw.get("schema_version") == USER_LAUNCH_OVERRIDE_SCHEMA:
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
        if set(raw) != required:
            raise _fail(
                "Invalid user-launch-override keys: "
                f"missing={sorted(required - set(raw))}, "
                f"extra={sorted(set(raw) - required)}."
            )
        if raw["decision"] != USER_LAUNCH_OVERRIDE_DECISION:
            raise _fail("User launch override does not explicitly authorize production training.")
        if _string(raw["authorized_by"], "user_override.authorized_by") != approved_by:
            raise _fail("User launch override author must match approval.approved_by.")
        accepted_at = _string(
            raw["instruction_accepted_utc"], "user_override.instruction_accepted_utc"
        )
        target_at = _string(
            raw["target_training_launch_utc"],
            "user_override.target_training_launch_utc",
        )
        try:
            accepted_time = datetime.fromisoformat(accepted_at.replace("Z", "+00:00"))
            target_time = datetime.fromisoformat(target_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise _fail("User launch override timestamps must be ISO-8601 values.") from exc
        if accepted_time.tzinfo is None or target_time.tzinfo is None:
            raise _fail("User launch override timestamps must include timezones.")
        if accepted_time > approval_time or target_time < accepted_time:
            raise _fail("User launch override timestamps are inconsistent with the approval.")
        instruction_path = _absolute_path(
            raw["instruction_source_path"], "user_override.instruction_source_path"
        )
        if not instruction_path.is_file():
            raise _fail("The bound user launch instruction is missing.")
        instruction_sha256 = _sha(
            raw["instruction_source_sha256"],
            "user_override.instruction_source_sha256",
        )
        if sha256_file(instruction_path) != instruction_sha256:
            raise _fail("The bound user launch instruction SHA-256 does not match.")
        models = raw["authorized_models"]
        if (
            not isinstance(models, list)
            or set(models) != {"SEGFORMER", "YOLO"}
            or len(models) != 2
        ):
            raise _fail("User launch override must authorize the exact two production models.")
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
        if any(raw[key] != value for key, value in expected_strings.items()) or any(
            type(raw[key]) is not bool or raw[key] is not value
            for key, value in expected_booleans.items()
        ):
            raise _fail(
                "User launch override must preserve deferred review, explicit training "
                "authorization, persistence, and source-freeze boundaries."
            )
        notes = raw["notes"]
        if (
            not isinstance(notes, list)
            or not notes
            or any(not isinstance(note, str) or not note.strip() for note in notes)
        ):
            raise _fail("User launch override notes must be a non-empty string array.")
        return
    required = {
        "schema_version",
        "review_kind",
        "decision",
        "reviewed_by",
        "reviewed_at",
        "acknowledgements",
        "open_blockers",
    }
    if set(raw) != required:
        raise _fail(
            "Invalid human-review keys: "
            f"missing={sorted(required - set(raw))}, extra={sorted(set(raw) - required)}."
        )
    if raw["schema_version"] != HUMAN_REVIEW_SCHEMA:
        raise _fail("Unsupported human-review schema.")
    if raw["review_kind"] != "FLOODSIGHT_FINAL_PRETRAINING_REVIEW":
        raise _fail("Human review has the wrong review_kind.")
    if raw["decision"] != "APPROVED":
        raise _fail("Human review decision must be APPROVED.")
    if _string(raw["reviewed_by"], "human_review.reviewed_by") != approved_by:
        raise _fail("Human-review reviewer must match approval.approved_by.")
    reviewed_at = _string(raw["reviewed_at"], "human_review.reviewed_at")
    try:
        review_time = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _fail("human_review.reviewed_at must be an ISO-8601 timestamp.") from exc
    if review_time.tzinfo is None:
        raise _fail("human_review.reviewed_at must include a timezone.")
    if review_time > approval_time:
        raise _fail("Human review cannot occur after its approval record.")
    acknowledgements = raw["acknowledgements"]
    if not isinstance(acknowledgements, Mapping) or set(acknowledgements) != set(
        REQUIRED_REVIEW_ACKNOWLEDGEMENTS
    ):
        raise _fail("Human review has missing or extra required acknowledgements.")
    if any(value is not True for value in acknowledgements.values()):
        raise _fail("Every required human-review acknowledgement must be true.")
    blockers = raw["open_blockers"]
    if not isinstance(blockers, list) or blockers:
        raise _fail("Human review must contain an empty open_blockers array.")


def _validate_real_smoke(
    path: Path,
    *,
    expected_sha256: str,
    config_sha256: str,
    taxonomy_sha256: Mapping[str, str],
    model_artifact: ModelArtifactSpec,
    real_smoke_root: Path,
) -> None:
    """Require the exact bounded production smoke before human approval can unlock a run."""

    if not path.is_file():
        raise _fail(f"Bound SegFormer real-smoke report is missing: {path}")
    if sha256_file(path) != expected_sha256:
        raise _fail("Bound SegFormer real-smoke report SHA-256 does not match.")
    root = real_smoke_root.expanduser().resolve()
    if path.name != "real-manifest-smoke.json" or path.parent.parent != root:
        raise _fail("SegFormer real-smoke report is outside the frozen smoke root.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail(f"Unable to parse bound SegFormer real-smoke report: {path}") from exc
    if not isinstance(raw, Mapping):
        raise _fail("Bound SegFormer real-smoke report must be a JSON object.")
    required_values = {
        "status": "PASS",
        "artifact_type": "BOUNDED_REAL_MANIFEST_SMOKE",
        "provenance": "REAL_ML_OUTPUT",
        "full_training": False,
        "training_authorized": False,
        "epochs": 0,
        "optimizer_steps": 1,
        "training_transform": "PASS",
        "validation_transform": "PASS",
        "parameter_changed": True,
        "checkpoint_reload": "FRESH_PYTHON_PROCESS_PASS",
        "config_sha256": config_sha256,
    }
    if any(raw.get(key) != value for key, value in required_values.items()):
        raise _fail("Bound SegFormer real-smoke report does not contain an exact PASS envelope.")
    if raw.get("training_source_sha256") != training_source_sha256():
        raise _fail("SegFormer real smoke was not produced by the current executable source.")
    expected_manifest_hashes = {
        str(manifest_path.resolve()): digest
        for (_dataset_id, split), (
            manifest_path,
            digest,
            _fingerprint,
        ) in CANONICAL_MANIFEST_LOCKS.items()
        if split == "train"
    }
    expected_manifest_fingerprints = {
        str(manifest_path.resolve()): fingerprint
        for (_dataset_id, split), (
            manifest_path,
            _digest,
            fingerprint,
        ) in CANONICAL_MANIFEST_LOCKS.items()
        if split == "train"
    }
    if raw.get("manifest_sha256") != dict(sorted(expected_manifest_hashes.items())):
        raise _fail("SegFormer real smoke is not bound to both canonical training manifests.")
    if raw.get("manifest_fingerprint") != dict(
        sorted(expected_manifest_fingerprints.items())
    ):
        raise _fail("SegFormer real-smoke manifest fingerprints do not match.")
    expected_taxonomy = _normalized_path_hashes(
        taxonomy_sha256, location="expected_real_smoke_taxonomy_sha256"
    )
    if raw.get("taxonomy_sha256") != expected_taxonomy:
        raise _fail("SegFormer real-smoke taxonomy/mapping hashes do not match.")
    normalized_model = model_artifact.normalized()
    expected_model = {
        "model_safetensors_path": str(normalized_model.safetensors_path),
        "model_safetensors_sha256": normalized_model.safetensors_sha256,
        "model_provenance_path": str(normalized_model.provenance_path),
        "model_provenance_sha256": normalized_model.provenance_sha256,
    }
    if any(raw.get(key) != value for key, value in expected_model.items()):
        raise _fail("SegFormer real smoke used a different local model artifact.")
    sample_ids = raw.get("sample_ids")
    samples_per_dataset = raw.get("samples_per_dataset")
    if (
        not isinstance(sample_ids, list)
        or not 2 <= len(sample_ids) <= 4
        or len(sample_ids) != len(set(sample_ids))
        or not isinstance(samples_per_dataset, Mapping)
        or set(samples_per_dataset) != {"floodnet", "rescuenet"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in samples_per_dataset.values()
        )
        or sum(samples_per_dataset.values()) != len(sample_ids)
    ):
        raise _fail("SegFormer real-smoke sample envelope is invalid.")
    for field in ("loss", "gradient_norm"):
        value = raw.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise _fail(f"SegFormer real-smoke {field} must be positive and finite.")
    validation = raw.get("validation")
    if not isinstance(validation, Mapping):
        raise _fail("SegFormer real-smoke validation evidence is missing.")
    for field in ("loss", "mean_iou"):
        value = validation.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise _fail(f"SegFormer real-smoke validation {field} is not finite.")
    checkpoint = _absolute_path(raw.get("checkpoint"), "real_smoke.checkpoint")
    checkpoint_sha256 = _sha(
        raw.get("checkpoint_sha256"), "real_smoke.checkpoint_sha256"
    )
    if (
        checkpoint.parent != path.parent
        or checkpoint.name != "real-manifest-smoke.pt"
        or not checkpoint.is_file()
        or sha256_file(checkpoint) != checkpoint_sha256
    ):
        raise _fail("SegFormer real-smoke checkpoint identity does not match its report.")
    probe = raw.get("fresh_process_checkpoint_proof")
    if not isinstance(probe, Mapping):
        raise _fail("SegFormer real-smoke fresh-process proof is missing.")
    expected_probe_values = {
        "status": "PASS",
        "provenance": "REAL_ML_OUTPUT",
        "fresh_python_process": True,
        "training_state": "PASS",
        "model_state": "PASS",
        "optimizer_state": "PASS",
        "scheduler_state": "PASS",
        "scaler_state": "PASS",
        "python_numpy_torch_cpu_cuda_generator_rng": "PASS",
        "child_executable_source_rehash": "PASS",
        "training_source_sha256": training_source_sha256(),
        "checkpoint": str(checkpoint),
    }
    creator_pid = probe.get("creator_pid")
    resumer_pid = probe.get("resumer_pid")
    if (
        any(probe.get(key) != value for key, value in expected_probe_values.items())
        or isinstance(creator_pid, bool)
        or not isinstance(creator_pid, int)
        or isinstance(resumer_pid, bool)
        or not isinstance(resumer_pid, int)
        or creator_pid <= 0
        or resumer_pid <= 0
        or creator_pid == resumer_pid
    ):
        raise _fail("SegFormer real-smoke fresh-process checkpoint proof is invalid.")


def validate_human_approval(
    path: Path,
    *,
    expected_record_sha256: str,
    operation: str,
    config_sha256: str,
    manifest_specs: Sequence[ManifestSpec],
    taxonomy_sha256: Mapping[str, str],
    run_directory: Path,
    model_id: str,
    model_revision: str,
    model_artifact: ModelArtifactSpec,
    real_smoke_root: Path,
) -> HumanApproval:
    """Validate human approval before any manifest, model, or dataset is opened."""

    if operation not in ALLOWED_OPERATIONS:
        raise _fail(f"Unsupported approval operation {operation!r}.")
    expected_record_sha256 = _sha(expected_record_sha256, "record_sha256")
    if path.is_symlink():
        raise _fail("The approval record must not be a symbolic link.")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise _fail(f"Human approval record is missing: {path}")
    actual_record_sha256 = sha256_file(path)
    if actual_record_sha256 != expected_record_sha256:
        raise _fail(
            f"Human approval record SHA-256 mismatch: expected {expected_record_sha256}, "
            f"found {actual_record_sha256}."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail(f"Unable to parse human approval record: {path}") from exc
    if not isinstance(raw, Mapping):
        raise _fail("Human approval record must be a JSON object.")
    required = {
        "schema_version",
        "approval_kind",
        "approval_id",
        "decision",
        "approved_by",
        "approved_at",
        "operations",
        "training_code_sha256",
        "config_sha256",
        "manifest_sha256",
        "manifest_fingerprint",
        "taxonomy_sha256",
        "run_directory",
        "model",
        "real_smoke",
        "human_review",
    }
    if set(raw) != required:
        raise _fail(
            f"Invalid approval keys: missing={sorted(required - set(raw))}, "
            f"extra={sorted(set(raw) - required)}."
        )
    if raw["schema_version"] != APPROVAL_SCHEMA:
        raise _fail("Unsupported human approval schema.")
    if raw["approval_kind"] != "HUMAN" or raw["decision"] != "APPROVED":
        raise _fail("A HUMAN record with decision APPROVED is required.")
    approval_id = _string(raw["approval_id"], "approval_id")
    approved_by = _string(raw["approved_by"], "approved_by")
    approved_at = _string(raw["approved_at"], "approved_at")
    try:
        parsed_time = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _fail("approval.approved_at must be an ISO-8601 timestamp.") from exc
    if parsed_time.tzinfo is None:
        raise _fail("approval.approved_at must include a timezone.")
    operations_raw = raw["operations"]
    if not isinstance(operations_raw, list) or not operations_raw:
        raise _fail("approval.operations must be a non-empty array.")
    operations = frozenset(_string(item, "operations[]") for item in operations_raw)
    if not operations <= ALLOWED_OPERATIONS or operation not in operations:
        raise _fail(f"Human approval does not authorize operation {operation}.")
    declared_code_sha256 = _sha(raw["training_code_sha256"], "training_code_sha256")
    actual_code_sha256 = training_source_sha256()
    if declared_code_sha256 != actual_code_sha256:
        raise _fail("Human approval is not bound to the current training-source SHA-256.")
    if _sha(raw["config_sha256"], "config_sha256") != config_sha256:
        raise _fail("Human approval is not bound to this configuration SHA-256.")
    declared_manifests = raw["manifest_sha256"]
    if not isinstance(declared_manifests, Mapping):
        raise _fail("approval.manifest_sha256 must be an object.")
    parsed_manifests = {
        str(_absolute_path(_string(key, "manifest_sha256 path"), "manifest_sha256 path")): _sha(
            value, f"manifest_sha256.{key}"
        )
        for key, value in declared_manifests.items()
    }
    expected_manifests = _normalized_manifest_hashes(manifest_specs)
    if dict(sorted(parsed_manifests.items())) != expected_manifests:
        raise _fail("Human approval is not bound to the exact manifest path/hash set.")
    declared_fingerprints = raw["manifest_fingerprint"]
    if not isinstance(declared_fingerprints, Mapping):
        raise _fail("approval.manifest_fingerprint must be an object.")
    parsed_fingerprints = {
        str(
            _absolute_path(
                _string(key, "manifest_fingerprint path"),
                "manifest_fingerprint path",
            )
        ): _sha(value, f"manifest_fingerprint.{key}")
        for key, value in declared_fingerprints.items()
    }
    expected_fingerprints = _normalized_manifest_fingerprints(manifest_specs)
    if dict(sorted(parsed_fingerprints.items())) != expected_fingerprints:
        raise _fail("Human approval is not bound to the exact manifest fingerprint set.")
    declared_taxonomy = raw["taxonomy_sha256"]
    if not isinstance(declared_taxonomy, Mapping):
        raise _fail("approval.taxonomy_sha256 must be an object.")
    parsed_taxonomy = _normalized_path_hashes(
        declared_taxonomy, location="taxonomy_sha256"
    )
    expected_taxonomy = _normalized_path_hashes(
        taxonomy_sha256, location="expected_taxonomy_sha256"
    )
    if parsed_taxonomy != expected_taxonomy:
        raise _fail("Human approval is not bound to the exact taxonomy/mapping path/hash set.")
    approved_run_directory = _absolute_path(raw["run_directory"], "run_directory")
    expected_run_directory = run_directory.expanduser().resolve()
    if approved_run_directory != expected_run_directory:
        raise _fail("Human approval is not bound to the exact output run directory.")
    model = raw["model"]
    if not isinstance(model, Mapping) or set(model) != {
        "model_id",
        "revision",
        "safetensors_path",
        "safetensors_sha256",
        "provenance_path",
        "provenance_sha256",
    }:
        raise _fail("approval.model has invalid keys.")
    normalized_artifact = model_artifact.normalized()
    expected_model = {
        "model_id": model_id,
        "revision": model_revision,
        "safetensors_path": str(normalized_artifact.safetensors_path),
        "safetensors_sha256": normalized_artifact.safetensors_sha256,
        "provenance_path": str(normalized_artifact.provenance_path),
        "provenance_sha256": normalized_artifact.provenance_sha256,
    }
    parsed_model = {
        "model_id": _string(model["model_id"], "model.model_id"),
        "revision": _string(model["revision"], "model.revision"),
        "safetensors_path": str(
            _absolute_path(model["safetensors_path"], "model.safetensors_path")
        ),
        "safetensors_sha256": _sha(
            model["safetensors_sha256"], "model.safetensors_sha256"
        ),
        "provenance_path": str(
            _absolute_path(model["provenance_path"], "model.provenance_path")
        ),
        "provenance_sha256": _sha(
            model["provenance_sha256"], "model.provenance_sha256"
        ),
    }
    if parsed_model != expected_model:
        raise _fail("Human approval is not bound to the exact local model artifact.")
    smoke = raw["real_smoke"]
    if not isinstance(smoke, Mapping) or set(smoke) != {"report_path", "report_sha256"}:
        raise _fail("approval.real_smoke has invalid keys.")
    smoke_path = _absolute_path(smoke["report_path"], "real_smoke.report_path")
    smoke_sha256 = _sha(smoke["report_sha256"], "real_smoke.report_sha256")
    _validate_real_smoke(
        smoke_path,
        expected_sha256=smoke_sha256,
        config_sha256=config_sha256,
        taxonomy_sha256=taxonomy_sha256,
        model_artifact=model_artifact,
        real_smoke_root=real_smoke_root,
    )
    review = raw["human_review"]
    if not isinstance(review, Mapping) or set(review) != {"report_path", "report_sha256"}:
        raise _fail("approval.human_review has invalid keys.")
    review_path = _absolute_path(review["report_path"], "human_review.report_path")
    review_sha256 = _sha(review["report_sha256"], "human_review.report_sha256")
    if not review_path.is_file():
        raise _fail(f"Bound human-review report is missing: {review_path}")
    actual_review_sha256 = sha256_file(review_path)
    if actual_review_sha256 != review_sha256:
        raise _fail("Bound human-review report SHA-256 does not match.")
    _validate_human_review(
        review_path,
        approved_by=approved_by,
        approval_time=parsed_time,
    )
    return HumanApproval(
        approval_id=approval_id,
        approved_by=approved_by,
        approved_at=approved_at,
        operations=operations,
        record_path=path,
        record_sha256=actual_record_sha256,
        human_review_path=review_path,
        human_review_sha256=actual_review_sha256,
        real_smoke_path=smoke_path,
        real_smoke_sha256=smoke_sha256,
        training_code_sha256=actual_code_sha256,
        manifest_fingerprints=expected_fingerprints,
        taxonomy_sha256=expected_taxonomy,
        run_directory=approved_run_directory,
    )
