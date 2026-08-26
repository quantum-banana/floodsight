from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from floodsight_data.config import DATASET_CONFIG_ROOT, load_yaml
from floodsight_data.errors import DatasetToolError


class TaskType(StrEnum):
    SEMANTIC_SEGMENTATION = "SEMANTIC_SEGMENTATION"
    AERIAL_DETECTION = "AERIAL_DETECTION"


class LicenseReviewState(StrEnum):
    VERIFIED_RESEARCH_USE = "VERIFIED_RESEARCH_USE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    USER_ACCEPTANCE_REQUIRED = "USER_ACCEPTANCE_REQUIRED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    canonical_id: str
    display_name: str
    task_type: TaskType
    official_reference: str
    citation: str
    license_reference: str
    license_review_state: LicenseReviewState
    expected_splits: tuple[str, ...]
    annotation_type: str
    acquisition_methods: tuple[str, ...]
    expected_source_structure: tuple[str, ...]
    source_version_notes: str
    preparation_version: str
    required_manual_steps: tuple[str, ...]
    commercial_use_requires_review: bool
    source_config: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_type"] = self.task_type.value
        payload["license_review_state"] = self.license_review_state.value
        payload["source_config"] = self.source_config.as_posix()
        for key in (
            "expected_splits",
            "acquisition_methods",
            "expected_source_structure",
            "required_manual_steps",
        ):
            payload[key] = list(payload[key])
        return payload


REQUIRED_FIELDS = {
    "canonical_id",
    "display_name",
    "task_type",
    "official_reference",
    "citation",
    "license_reference",
    "license_review_state",
    "expected_splits",
    "annotation_type",
    "acquisition_methods",
    "expected_source_structure",
    "source_version_notes",
    "preparation_version",
    "required_manual_steps",
    "commercial_use_requires_review",
}


def _record(path: Path) -> DatasetRecord:
    payload = load_yaml(path)
    missing = sorted(REQUIRED_FIELDS - payload.keys())
    if missing:
        raise DatasetToolError(
            f"Dataset config {path.name} is missing: {', '.join(missing)}",
            code="registry_invalid",
        )
    try:
        record = DatasetRecord(
            canonical_id=str(payload["canonical_id"]),
            display_name=str(payload["display_name"]),
            task_type=TaskType(payload["task_type"]),
            official_reference=str(payload["official_reference"]),
            citation=str(payload["citation"]),
            license_reference=str(payload["license_reference"]),
            license_review_state=LicenseReviewState(payload["license_review_state"]),
            expected_splits=tuple(str(item) for item in payload["expected_splits"]),
            annotation_type=str(payload["annotation_type"]),
            acquisition_methods=tuple(str(item) for item in payload["acquisition_methods"]),
            expected_source_structure=tuple(
                str(item) for item in payload["expected_source_structure"]
            ),
            source_version_notes=str(payload["source_version_notes"]),
            preparation_version=str(payload["preparation_version"]),
            required_manual_steps=tuple(str(item) for item in payload["required_manual_steps"]),
            commercial_use_requires_review=bool(payload["commercial_use_requires_review"]),
            source_config=path,
        )
    except (TypeError, ValueError) as exc:
        raise DatasetToolError(
            f"Dataset config {path.name} contains an invalid registry value: {exc}",
            code="registry_invalid",
        ) from exc
    if path.stem != record.canonical_id.replace("_", "-"):
        raise DatasetToolError(
            f"Dataset config filename does not match canonical ID: {path.name}",
            code="registry_invalid",
        )
    if not record.expected_splits or not record.acquisition_methods:
        raise DatasetToolError(
            f"Dataset config {path.name} must define splits and acquisition methods.",
            code="registry_invalid",
        )
    return record


def load_registry(config_root: Path = DATASET_CONFIG_ROOT) -> dict[str, DatasetRecord]:
    records: dict[str, DatasetRecord] = {}
    for path in sorted(config_root.glob("*.yaml")):
        record = _record(path)
        if record.canonical_id in records:
            raise DatasetToolError(
                f"Duplicate dataset ID: {record.canonical_id}",
                code="registry_invalid",
            )
        records[record.canonical_id] = record
    if not records:
        raise DatasetToolError("No dataset registry files were found.", code="registry_empty")
    return records


def get_dataset(dataset_id: str) -> DatasetRecord:
    registry = load_registry()
    try:
        return registry[dataset_id]
    except KeyError as exc:
        raise DatasetToolError(
            f"Unknown dataset '{dataset_id}'. Choose: {', '.join(sorted(registry))}.",
            code="dataset_unknown",
        ) from exc
