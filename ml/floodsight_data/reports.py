from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from floodsight_data.acquisition import read_acquisition_metadata
from floodsight_data.common.atomic import atomic_write_json, atomic_write_text
from floodsight_data.errors import DatasetToolError
from floodsight_data.manifests import read_json, validate_schema
from floodsight_data.paths import DataPaths
from floodsight_data.registry import get_dataset, load_registry
from floodsight_data.validation import validate_dataset


def derive_readiness(
    report: dict[str, Any],
    *,
    required_splits: tuple[str, ...],
    license_reviewed: bool,
    mapping_reviewed: bool,
    visual_reviewed: bool,
) -> bool:
    present_splits = {split for split, count in report.get("split_counts", {}).items() if count}
    expected = {"val" if split == "validation" else split for split in required_splits}
    return bool(
        report.get("acquisition_status") in {"IMPORTED", "VERIFIED"}
        and not report.get("blocking_errors")
        and not report.get("unknown_labels")
        and report.get("image_count", 0) > 0
        and report.get("image_count") == report.get("annotation_count")
        and report.get("conversion_count") == report.get("image_count")
        and report.get("failed_conversion_count") == 0
        and expected.issubset(present_splits)
        and license_reviewed
        and mapping_reviewed
        and visual_reviewed
    )


def _review_path(paths: DataPaths, dataset_id: str) -> Path:
    return paths.locks / f"{dataset_id}-human-review.json"


def read_review(paths: DataPaths, dataset_id: str, fingerprint: str | None) -> dict[str, Any]:
    path = _review_path(paths, dataset_id)
    default = {
        "license_reviewed": False,
        "mapping_reviewed": False,
        "visual_reviewed": False,
        "reviewer": None,
        "fingerprint_matches": False,
    }
    if not path.is_file() or fingerprint is None:
        return default
    payload = read_json(path)
    if payload.get("dataset_fingerprint") != fingerprint:
        return default
    return {
        "license_reviewed": bool(payload.get("license_reviewed")),
        "mapping_reviewed": bool(payload.get("mapping_reviewed")),
        "visual_reviewed": bool(payload.get("visual_reviewed")),
        "reviewer": payload.get("reviewer"),
        "fingerprint_matches": True,
    }


def record_review(
    paths: DataPaths,
    dataset_id: str,
    *,
    reviewer: str,
    license_reviewed: bool,
    mapping_reviewed: bool,
    visual_reviewed: bool,
) -> Path:
    manifest = _manifest_for(paths, dataset_id)
    if manifest is None:
        raise DatasetToolError(
            f"Prepared manifest is missing for {dataset_id}.", code="manifest_missing"
        )
    payload = {
        "schema_version": "dataset-human-review-v1",
        "dataset_id": dataset_id,
        "dataset_fingerprint": manifest["fingerprint"],
        "reviewer": reviewer,
        "reviewed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "license_reviewed": license_reviewed,
        "mapping_reviewed": mapping_reviewed,
        "visual_reviewed": visual_reviewed,
    }
    path = _review_path(paths, dataset_id)
    atomic_write_json(path, payload)
    return path


def _manifest_for(paths: DataPaths, dataset_id: str) -> dict[str, Any] | None:
    version = "detection_v1" if dataset_id == "visdrone_det" else "segmentation_v1"
    path = paths.manifests / f"{dataset_id}-{version}.json"
    return read_json(path) if path.is_file() else None


def build_dataset_report(paths: DataPaths, dataset_id: str) -> dict[str, Any]:
    record = get_dataset(dataset_id)
    acquisition = read_acquisition_metadata(paths, dataset_id)
    try:
        validation = validate_dataset(paths, dataset_id)
    except Exception as exc:  # Converted into report state by this reporting boundary.
        validation = {
            "split_counts": {},
            "image_count": 0,
            "annotation_count": 0,
            "missing_images": [],
            "missing_annotations": [],
            "corrupt_files": [],
            "dimensions": {},
            "source_labels": {},
            "ignored_source_labels": [],
            "unknown_labels": [],
            "duplicates": {
                "duplicate_images": [],
                "duplicate_annotations": [],
                "cross_split_leakage": [],
                "conflicting_annotations": [],
            },
            "blocking_errors": [str(exc)],
        }
    manifest = _manifest_for(paths, dataset_id)
    samples = manifest["samples"] if manifest else []
    review = read_review(paths, dataset_id, None if manifest is None else manifest["fingerprint"])
    class_distribution: Counter[str] = Counter()
    for sample in samples:
        class_distribution.update(sample["class_counts"])
    duplicates = validation.get("duplicates", {})
    missing_pairs = [
        *validation.get("missing_images", []),
        *validation.get("missing_annotations", []),
    ]
    blocking = list(validation.get("blocking_errors", []))
    warnings = ["Real source palette, mappings, and visual samples still require human review."]
    report = {
        "schema_version": "dataset-report-v1",
        "dataset_id": dataset_id,
        "source_version": "imported-source-unverified",
        "acquisition_status": "IMPORTED" if acquisition else "MISSING",
        "license_review_status": record.license_review_state.value,
        "review_status": review,
        "split_counts": validation.get("split_counts", {}),
        "image_count": validation.get("image_count", 0),
        "annotation_count": validation.get("annotation_count", 0),
        "missing_pairs": missing_pairs,
        "corrupt_files": validation.get("corrupt_files", []),
        "dimensions": validation.get("dimensions", {}),
        "class_distribution": dict(sorted(class_distribution.items())),
        "ignored_labels": validation.get("ignored_source_labels", []),
        "unknown_labels": validation.get("unknown_labels", []),
        "duplicate_count": len(duplicates.get("duplicate_images", [])),
        "cross_split_leakage_count": len(duplicates.get("cross_split_leakage", [])),
        "conversion_count": len(samples),
        "failed_conversion_count": max(0, validation.get("image_count", 0) - len(samples)),
        "output_paths": [] if manifest is None else [str(_manifest_for_path(paths, dataset_id))],
        "fingerprint": None if manifest is None else manifest["fingerprint"],
        "warnings": warnings,
        "blocking_errors": blocking,
        "recommended_next_action": (
            "Import the official source through a supported command."
            if acquisition is None
            else "Resolve blocking errors, review mappings, and complete visual inspection."
        ),
        "ready_for_next_phase": False,
    }
    report["ready_for_next_phase"] = derive_readiness(
        report,
        required_splits=record.expected_splits,
        license_reviewed=review["license_reviewed"],
        mapping_reviewed=review["mapping_reviewed"],
        visual_reviewed=review["visual_reviewed"],
    )
    validate_schema(report, "dataset-report.schema.json")
    return report


def _manifest_for_path(paths: DataPaths, dataset_id: str) -> Path:
    version = "detection_v1" if dataset_id == "visdrone_det" else "segmentation_v1"
    return paths.manifests / f"{dataset_id}-{version}.json"


def report_markdown(report: dict[str, Any]) -> str:
    splits = (
        ", ".join(f"{name}: {count}" for name, count in sorted(report["split_counts"].items()))
        or "none"
    )
    blocking = "\n".join(f"- {item}" for item in report["blocking_errors"]) or "- None"
    warnings = "\n".join(f"- {item}" for item in report["warnings"]) or "- None"
    return (
        f"# {report['dataset_id']} dataset health\n\n"
        f"- Acquisition: **{report['acquisition_status']}**\n"
        f"- License review: **{report['license_review_status']}**\n"
        f"- Splits: {splits}\n"
        f"- Images / annotations / converted: {report['image_count']} / "
        f"{report['annotation_count']} / {report['conversion_count']}\n"
        f"- Duplicate groups: {report['duplicate_count']}\n"
        f"- Cross-split leaks: {report['cross_split_leakage_count']}\n"
        f"- Ready for next phase: **{report['ready_for_next_phase']}**\n\n"
        f"## Blocking errors\n\n{blocking}\n\n"
        f"## Warnings\n\n{warnings}\n\n"
        f"## Recommended next action\n\n{report['recommended_next_action']}\n"
    )


def write_dataset_report(paths: DataPaths, dataset_id: str) -> tuple[Path, Path, dict[str, Any]]:
    report = build_dataset_report(paths, dataset_id)
    json_path = paths.reports / f"{dataset_id}-health.json"
    markdown_path = paths.reports / f"{dataset_id}-health.md"
    atomic_write_json(json_path, report)
    atomic_write_text(markdown_path, report_markdown(report))
    return json_path, markdown_path, report


def write_combined_report(paths: DataPaths) -> tuple[Path, Path, dict[str, Any]]:
    reports = {
        dataset_id: build_dataset_report(paths, dataset_id) for dataset_id in load_registry()
    }
    payload = {
        "status": "DATA_VERIFIED"
        if all(item["ready_for_next_phase"] for item in reports.values())
        else "NOT_READY",
        "datasets": reports,
        "segmentation_taxonomy_support": {
            "candidate": True,
            "real_data_reviewed": False,
            "unsupported_product_concepts": ["debris_landslide"],
        },
        "detection_taxonomy_support": {"candidate": True, "real_data_reviewed": False},
        "phase_4_may_start": reports["floodnet"]["ready_for_next_phase"]
        and reports["rescuenet"]["ready_for_next_phase"],
        "phase_5_may_start": reports["visdrone_det"]["ready_for_next_phase"],
    }
    json_path = paths.reports / "floodsight-dataset-readiness.json"
    markdown_path = paths.reports / "floodsight-dataset-readiness.md"
    atomic_write_json(json_path, payload)
    markdown = [
        "# FloodSight dataset readiness",
        "",
        f"Overall status: **{payload['status']}**",
        "",
    ]
    for dataset_id, report in reports.items():
        markdown.append(
            f"- {dataset_id}: acquisition={report['acquisition_status']}, "
            f"ready={report['ready_for_next_phase']}, blocking={len(report['blocking_errors'])}"
        )
    markdown.extend(
        (
            "",
            f"Phase 4 segmentation training permitted: **{payload['phase_4_may_start']}**",
            f"Phase 5 detection training permitted: **{payload['phase_5_may_start']}**",
            "",
        )
    )
    atomic_write_text(markdown_path, "\n".join(markdown))
    return json_path, markdown_path, payload
