"""Offline-only, content-addressed SegFormer weight artifact validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .errors import ArtifactError
from .manifest import SHA256_PATTERN, sha256_file

MODEL_ARTIFACT_SCHEMA = "floodsight-model-artifact-v1"


@dataclass(frozen=True, slots=True)
class ModelArtifactSpec:
    """Operator-declared immutable paths and hashes, before either file is opened."""

    safetensors_path: Path
    safetensors_sha256: str
    provenance_path: Path
    provenance_sha256: str

    def normalized(self) -> ModelArtifactSpec:
        return ModelArtifactSpec(
            safetensors_path=self.safetensors_path.expanduser().resolve(),
            safetensors_sha256=self.safetensors_sha256.lower(),
            provenance_path=self.provenance_path.expanduser().resolve(),
            provenance_sha256=self.provenance_sha256.lower(),
        )


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    safetensors_path: Path
    safetensors_sha256: str
    provenance_path: Path
    provenance_sha256: str
    source_revision: str
    source_sha256: str
    human_review_status: str


def _mapping(value: Any, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"Expected an object at {location}.")
    return value


def _string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"Expected a non-empty string at {location}.")
    return value


def _sha(value: Any, *, location: str) -> str:
    digest = _string(value, location=location).lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise ArtifactError(f"Expected a lowercase SHA-256 at {location}.")
    return digest


def _timestamp(value: Any, *, location: str) -> str:
    text = _string(value, location=location)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactError(f"Expected an ISO-8601 timestamp at {location}.") from exc
    if parsed.tzinfo is None:
        raise ArtifactError(f"Timestamp must include a timezone at {location}.")
    return text


def _validate_safetensors_container(path: Path) -> None:
    """Validate the non-executable safetensors envelope before Transformers opens it."""

    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            header_size_bytes = stream.read(8)
            if len(header_size_bytes) != 8:
                raise ArtifactError("Safetensors artifact has a truncated header length.")
            header_size = int.from_bytes(header_size_bytes, byteorder="little", signed=False)
            if header_size < 2 or header_size > min(size - 8, 100 * 1024 * 1024):
                raise ArtifactError("Safetensors artifact declares an invalid header size.")
            header = json.loads(stream.read(header_size).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("Safetensors artifact has an invalid JSON envelope.") from exc
    if not isinstance(header, Mapping):
        raise ArtifactError("Safetensors header must be a JSON object.")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise ArtifactError("Safetensors artifact contains no tensors.")
    data_size = size - 8 - header_size
    for name, descriptor in tensors.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            raise ArtifactError("Safetensors tensor descriptors are invalid.")
        offsets = descriptor.get("data_offsets")
        shape = descriptor.get("shape")
        dtype = descriptor.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_size
            or not isinstance(shape, list)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in shape
            )
            or not isinstance(dtype, str)
            or not dtype
        ):
            raise ArtifactError(f"Invalid safetensors descriptor for tensor {name!r}.")


def validate_model_artifact(spec: ModelArtifactSpec, config: ModelConfig) -> ModelArtifact:
    """Validate exact local safetensors bytes and their conversion provenance."""

    declared = spec.normalized()
    expected_identity = (
        (declared.safetensors_path, config.safetensors_path, "safetensors path"),
        (declared.safetensors_sha256, config.safetensors_sha256, "safetensors SHA-256"),
        (declared.provenance_path, config.provenance_path, "provenance path"),
        (declared.provenance_sha256, config.provenance_sha256, "provenance SHA-256"),
    )
    for actual, expected, label in expected_identity:
        if actual != expected:
            raise ArtifactError(
                f"Operator-declared {label} does not match the exact frozen model identity."
            )
    for digest, label in (
        (declared.safetensors_sha256, "safetensors_sha256"),
        (declared.provenance_sha256, "provenance_sha256"),
    ):
        if not SHA256_PATTERN.fullmatch(digest):
            raise ArtifactError(f"Expected a lowercase SHA-256 for {label}.")
    if spec.safetensors_path.is_symlink() or spec.provenance_path.is_symlink():
        raise ArtifactError("Model artifacts and provenance records must not be symbolic links.")
    if not declared.safetensors_path.is_file():
        raise ArtifactError(f"Local safetensors artifact is missing: {declared.safetensors_path}")
    if declared.safetensors_path.name != "model.safetensors":
        raise ArtifactError("The audited weight filename must be exactly model.safetensors.")
    if not declared.provenance_path.is_file():
        raise ArtifactError(f"Model provenance record is missing: {declared.provenance_path}")
    if declared.provenance_path.parent != declared.safetensors_path.parent:
        raise ArtifactError("Weights and their provenance record must share one audited directory.")
    competing_weights = {
        candidate.resolve()
        for pattern in ("*.safetensors", "*.bin", "*.safetensors.index.json", "*.bin.index.json")
        for candidate in declared.safetensors_path.parent.glob(pattern)
    }
    if competing_weights != {declared.safetensors_path}:
        raise ArtifactError(
            "The audited model directory must contain exactly one weight artifact: "
            f"{declared.safetensors_path}."
        )
    actual_weights_sha = sha256_file(declared.safetensors_path)
    if actual_weights_sha != declared.safetensors_sha256:
        raise ArtifactError(
            f"Safetensors SHA-256 mismatch: expected {declared.safetensors_sha256}, "
            f"found {actual_weights_sha}."
        )
    _validate_safetensors_container(declared.safetensors_path)
    actual_provenance_sha = sha256_file(declared.provenance_path)
    if actual_provenance_sha != declared.provenance_sha256:
        raise ArtifactError(
            f"Model provenance SHA-256 mismatch: expected {declared.provenance_sha256}, "
            f"found {actual_provenance_sha}."
        )
    try:
        raw = json.loads(declared.provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("Unable to parse model provenance JSON.") from exc
    payload = _mapping(raw, location="<root>")
    required = {
        "schema_version",
        "artifact_kind",
        "model_id",
        "source_revision",
        "source_filename",
        "source_sha256",
        "conversion_tool",
        "converted_at",
        "safetensors_filename",
        "safetensors_sha256",
        "audit_status",
        "human_review_status",
    }
    if set(payload) != required:
        raise ArtifactError(
            f"Invalid model provenance keys: missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - required)}."
        )
    expected_values = {
        "schema_version": MODEL_ARTIFACT_SCHEMA,
        "artifact_kind": "SEGFORMER_B2_SAFETENSORS",
        "model_id": config.pretrained_model_name_or_path,
        "source_revision": config.revision,
        "source_filename": "pytorch_model.bin",
        "source_sha256": config.upstream_pytorch_model_sha256,
        "safetensors_filename": declared.safetensors_path.name,
        "safetensors_sha256": actual_weights_sha,
        "audit_status": "PASS",
        "human_review_status": "PENDING_HUMAN_SIGNOFF",
    }
    for key, expected in expected_values.items():
        if payload.get(key) != expected:
            raise ArtifactError(
                f"Model provenance mismatch at {key}: expected {expected!r}, "
                f"found {payload.get(key)!r}."
            )
    _string(payload["conversion_tool"], location="conversion_tool")
    _timestamp(payload["converted_at"], location="converted_at")
    return ModelArtifact(
        safetensors_path=declared.safetensors_path,
        safetensors_sha256=actual_weights_sha,
        provenance_path=declared.provenance_path,
        provenance_sha256=actual_provenance_sha,
        source_revision=config.revision,
        source_sha256=config.upstream_pytorch_model_sha256,
        human_review_status="PENDING_HUMAN_SIGNOFF",
    )
