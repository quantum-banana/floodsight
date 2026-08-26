from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from floodsight_data import __version__
from floodsight_data.common.atomic import atomic_write_json, atomic_write_text
from floodsight_data.config import SCHEMA_ROOT
from floodsight_data.errors import DatasetToolError
from floodsight_data.hashing import dataset_fingerprint


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetToolError(f"Unable to read JSON: {path}", code="json_invalid") from exc
    if not isinstance(payload, dict):
        raise DatasetToolError(f"Expected a JSON object: {path}", code="json_invalid")
    return payload


def validate_schema(payload: dict[str, Any], schema_name: str) -> None:
    schema = read_json(SCHEMA_ROOT / schema_name)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.path) or "<root>"
        raise DatasetToolError(
            f"{schema_name} validation failed at {location}: {first.message}",
            code="schema_validation_failed",
        )


def build_manifest(
    *,
    dataset_id: str,
    task_type: str,
    source_version: str,
    preparation_version: str,
    taxonomy_version: str,
    integrity_mode: str,
    samples: Iterable[dict[str, Any]],
    source_records: Iterable[dict[str, Any]],
    mapping_hashes: dict[str, str],
    preparation: dict[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    ordered_samples = sorted(samples, key=lambda item: item["sample_id"])
    fingerprint = dataset_fingerprint(
        source_records,
        taxonomy_version=taxonomy_version,
        mapping_hashes=mapping_hashes,
        preparation=preparation,
        tool_version=__version__,
    )
    manifest = {
        "schema_version": "dataset-manifest-v1",
        "manifest_id": f"{dataset_id}-{preparation_version}",
        "dataset_id": dataset_id,
        "task_type": task_type,
        "source_version": source_version,
        "preparation_version": preparation_version,
        "taxonomy_version": taxonomy_version,
        "integrity_mode": integrity_mode,
        "created_at": created_at or _utc_now(),
        "tool_version": __version__,
        "fingerprint": fingerprint,
        "samples": ordered_samples,
    }
    validate_schema(manifest, "dataset-manifest.schema.json")
    return manifest


def write_manifest(manifest: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    dataset_id = str(manifest["dataset_id"])
    version = str(manifest["preparation_version"])
    json_path = directory / f"{dataset_id}-{version}.json"
    jsonl_path = directory / f"{dataset_id}-{version}.jsonl"
    atomic_write_json(json_path, manifest)
    lines = [
        json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for sample in manifest["samples"]
    ]
    atomic_write_text(jsonl_path, "\n".join(lines) + ("\n" if lines else ""))
    return json_path, jsonl_path
