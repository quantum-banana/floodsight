from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from floodsight_data.common.atomic import atomic_write_json, atomic_write_text
from floodsight_data.common.discovery import discover_visdrone_pairs
from floodsight_data.common.images import image_dimensions
from floodsight_data.common.materialize import MaterializationStrategy, materialize_image
from floodsight_data.config import TAXONOMY_ROOT, stable_relative
from floodsight_data.errors import BlockingValidationError, DatasetToolError
from floodsight_data.hashing import (
    IntegrityMode,
    file_integrity,
    sha256_file,
    stable_digest,
    stable_sample_id,
)
from floodsight_data.manifests import build_manifest, read_json, write_manifest
from floodsight_data.paths import DataPaths
from floodsight_data.registry import get_dataset
from floodsight_data.taxonomy import (
    MappingAction,
    load_mapping,
    load_taxonomy,
    validate_mapping_targets,
)
from floodsight_data.visdrone.parser import VisDroneObject, parse_annotation


def _normalize_box(
    item: VisDroneObject,
    image_width: int,
    image_height: int,
    *,
    annotation: Path,
) -> tuple[float, float, float, float, bool]:
    if item.width <= 0 or item.height <= 0:
        raise BlockingValidationError(
            f"Invalid non-positive box at {annotation}:{item.line_number}.",
            code="invalid_bounding_box",
        )
    left = max(0.0, item.left)
    top = max(0.0, item.top)
    right = min(float(image_width), item.left + item.width)
    bottom = min(float(image_height), item.top + item.height)
    if right <= left or bottom <= top:
        raise BlockingValidationError(
            f"Bounding box is outside the image at {annotation}:{item.line_number}.",
            code="invalid_bounding_box",
        )
    clamped = (left, top, right, bottom) != (
        item.left,
        item.top,
        item.left + item.width,
        item.top + item.height,
    )
    width = right - left
    height = bottom - top
    center_x = (left + right) / 2 / image_width
    center_y = (top + bottom) / 2 / image_height
    normalized = (center_x, center_y, width / image_width, height / image_height)
    if any(value < 0 or value > 1 for value in normalized):
        raise BlockingValidationError(
            f"Normalized box is outside [0,1] at {annotation}:{item.line_number}.",
            code="invalid_bounding_box",
        )
    return (*normalized, clamped)


def convert_visdrone_dataset(
    paths: DataPaths,
    *,
    integrity: IntegrityMode,
    materialization: MaterializationStrategy,
    dry_run: bool = False,
) -> dict[str, Any]:
    dataset_id = "visdrone_det"
    record = get_dataset(dataset_id)
    source_root = paths.dataset_raw(dataset_id)
    if not source_root.is_dir():
        raise DatasetToolError(
            f"Dataset is missing at {source_root}. Import it first.", code="dataset_missing"
        )
    discovery = discover_visdrone_pairs(source_root)
    if discovery.missing_images or discovery.missing_annotations:
        raise BlockingValidationError(
            "VisDrone has missing image/annotation pairs.",
            code="pairing_failed",
            details=[
                {
                    "missing_images": [str(path) for path in discovery.missing_images],
                    "missing_annotations": [str(path) for path in discovery.missing_annotations],
                }
            ],
        )
    if not discovery.pairs:
        raise DatasetToolError(
            f"No VisDrone DET pairs were discovered under {source_root}.",
            code="dataset_structure_unrecognized",
        )
    mapping = load_mapping(dataset_id)
    validate_mapping_targets(mapping, "detection-taxonomy-v1.yaml")
    taxonomy_version, target_classes = load_taxonomy("detection-taxonomy-v1.yaml")
    output_root = paths.processed / "detection_v1"
    samples: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    split_images: dict[str, list[str]] = {}
    resumed = 0
    for pair in discovery.pairs:
        width, height = image_dimensions(pair.image)
        objects = parse_annotation(pair.annotation)
        target_lines: list[str] = []
        class_counts: Counter[int] = Counter()
        object_metadata: list[dict[str, Any]] = []
        ignored = 0
        for item in objects:
            entry = mapping.by_source_id.get(item.class_id)
            if entry is None:
                raise BlockingValidationError(
                    f"Unsupported VisDrone class {item.class_id} at "
                    f"{pair.annotation}:{item.line_number}.",
                    code="unsupported_detection_class",
                )
            if item.score == 0 or entry.action is MappingAction.IGNORE:
                ignored += 1
                continue
            if entry.action is MappingAction.ERROR or entry.target_id is None:
                raise BlockingValidationError(
                    f"Rejected VisDrone class {item.class_id} at "
                    f"{pair.annotation}:{item.line_number}.",
                    code="unsupported_detection_class",
                )
            center_x, center_y, box_width, box_height, clamped = _normalize_box(
                item,
                width,
                height,
                annotation=pair.annotation,
            )
            target_lines.append(
                f"{entry.target_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
            )
            class_counts[entry.target_id] += 1
            object_metadata.append(
                {
                    "source_class_id": item.class_id,
                    "target_class_id": entry.target_id,
                    "truncation": item.truncation,
                    "occlusion": item.occlusion,
                    "clamped": clamped,
                }
            )
        target_annotation = output_root / "labels" / pair.split / f"{pair.image.stem}.txt"
        target_text = "\n".join(target_lines) + ("\n" if target_lines else "")
        target_image = output_root / "images" / pair.split / pair.image.name
        if not dry_run:
            if (
                target_annotation.is_file()
                and target_annotation.read_text(encoding="utf-8") == target_text
            ):
                resumed += 1
            else:
                atomic_write_text(target_annotation, target_text)
            materialized = materialize_image(
                pair.image,
                target_image,
                strategy=materialization,
            )
        else:
            materialized = (
                pair.image
                if materialization is MaterializationStrategy.MANIFEST_ONLY
                else target_image
            )
        image_relative = stable_relative(materialized, paths.root)
        split_images.setdefault(pair.split, []).append(image_relative)
        annotation_relative = stable_relative(pair.annotation, paths.root)
        samples.append(
            {
                "sample_id": stable_sample_id(
                    dataset_id, pair.split, stable_relative(pair.image, paths.root)
                ),
                "source_dataset": dataset_id,
                "source_split": pair.split,
                "target_split": pair.split,
                "image_path": image_relative,
                "source_annotation_path": annotation_relative,
                "target_annotation_path": stable_relative(target_annotation, paths.root),
                "width": width,
                "height": height,
                "image_hash": sha256_file(pair.image),
                "annotation_hash": sha256_file(pair.annotation),
                "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
                "ignored_count": ignored,
                "invalid_count": 0,
                "preparation_version": "detection_v1",
                "taxonomy_version": taxonomy_version,
                "objects": object_metadata,
            }
        )
        source_records.extend(
            (
                file_integrity(
                    pair.image,
                    relative_path=stable_relative(pair.image, paths.root),
                    mode=integrity,
                ),
                file_integrity(
                    pair.annotation,
                    relative_path=annotation_relative,
                    mode=integrity,
                    annotation=True,
                ),
            )
        )
    mapping_path = TAXONOMY_ROOT / "visdrone-mapping-v1.yaml"
    manifest_path = paths.manifests / "visdrone_det-detection_v1.json"
    created_at = read_json(manifest_path).get("created_at") if manifest_path.is_file() else None
    manifest = build_manifest(
        dataset_id=dataset_id,
        task_type=record.task_type.value,
        source_version="VisDrone2019-DET-imported-source-unverified",
        preparation_version="detection_v1",
        taxonomy_version=taxonomy_version,
        integrity_mode=integrity.value,
        samples=samples,
        source_records=source_records,
        mapping_hashes={mapping.mapping_version: sha256_file(mapping_path)},
        preparation={"materialization": materialization.value, "splits_preserved": True},
        created_at=created_at,
    )
    if not dry_run:
        names = {item.class_id: item.name for item in target_classes}
        dataset_yaml = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test-dev" if "test-dev" in split_images else "images/test",
            "names": names,
            "floodsight_manifest": "../../manifests/visdrone_det-detection_v1.json",
            "materialization": materialization.value,
        }
        atomic_write_text(
            output_root / "dataset.yaml",
            yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        )
        if materialization is MaterializationStrategy.MANIFEST_ONLY:
            for split, images in split_images.items():
                relative_lines = [
                    Path(os.path.relpath(paths.root / image, output_root)).as_posix()
                    for image in sorted(images)
                ]
                atomic_write_text(output_root / f"{split}.txt", "\n".join(relative_lines) + "\n")
        paths.manifests.mkdir(parents=True, exist_ok=True)
        manifest_paths = write_manifest(manifest, paths.manifests)
        lock = {
            "schema_version": "dataset-lock-v1",
            "dataset_id": dataset_id,
            "source_reference": record.official_reference,
            "acquisition_method": "imported-source",
            "acquired_at": manifest["created_at"],
            "source_fingerprint": stable_digest(source_records),
            "preparation_version": "detection_v1",
            "taxonomy_version": taxonomy_version,
            "mapping_hash": sha256_file(mapping_path),
            "integrity_mode": integrity.value,
            "dataset_fingerprint": manifest["fingerprint"],
            "complete": True,
        }
        atomic_write_json(paths.locks / "visdrone_det-detection_v1.json", lock)
    else:
        manifest_paths = (paths.manifests / "dry-run.json", paths.manifests / "dry-run.jsonl")
    return {
        "dataset_id": dataset_id,
        "status": "DRY_RUN" if dry_run else "CONVERTED",
        "sample_count": len(samples),
        "resumed_count": resumed,
        "fingerprint": manifest["fingerprint"],
        "manifest": str(manifest_paths[0]),
        "output": str(output_root),
    }
