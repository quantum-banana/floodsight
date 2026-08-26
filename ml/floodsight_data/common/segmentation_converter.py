from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from floodsight_data.common.atomic import atomic_build, atomic_write_json
from floodsight_data.common.discovery import discover_segmentation_pairs
from floodsight_data.common.images import convert_source_mask, image_dimensions, read_source_mask
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
from floodsight_data.taxonomy import load_mapping, load_taxonomy, validate_mapping_targets


def _save_mask(path: Path, mask: np.ndarray) -> None:
    def build(temporary: Path) -> None:
        Image.fromarray(mask, mode="L").save(temporary, format="PNG", compress_level=9)

    atomic_build(path, build)


def _same_mask(path: Path, expected: np.ndarray) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            actual = np.asarray(image, dtype=np.uint8)
        return actual.shape == expected.shape and bool(np.array_equal(actual, expected))
    except OSError:
        return False


def convert_segmentation_dataset(
    paths: DataPaths,
    dataset_id: str,
    *,
    integrity: IntegrityMode,
    materialization: MaterializationStrategy,
    dry_run: bool = False,
) -> dict[str, Any]:
    record = get_dataset(dataset_id)
    if record.task_type.value != "SEMANTIC_SEGMENTATION":
        raise DatasetToolError(f"{dataset_id} is not a segmentation dataset.", code="task_mismatch")
    source_root = paths.dataset_raw(dataset_id)
    if not source_root.is_dir():
        raise DatasetToolError(
            f"Dataset is missing at {source_root}. Import it first.", code="dataset_missing"
        )
    discovery = discover_segmentation_pairs(source_root)
    if (
        discovery.missing_images
        or discovery.missing_annotations
        or discovery.conflicting_annotations
    ):
        raise BlockingValidationError(
            f"{dataset_id} has missing or conflicting image/mask pairs.",
            code="pairing_failed",
            details=[
                {
                    "missing_images": [str(path) for path in discovery.missing_images],
                    "missing_annotations": [str(path) for path in discovery.missing_annotations],
                    "conflicting_annotations": [
                        [str(first), str(second)]
                        for first, second in discovery.conflicting_annotations
                    ],
                }
            ],
        )
    if not discovery.pairs:
        raise DatasetToolError(
            f"No paired segmentation samples were discovered under {source_root}.",
            code="dataset_structure_unrecognized",
        )
    mapping = load_mapping(dataset_id)
    validate_mapping_targets(mapping, "segmentation-taxonomy-v1.yaml")
    taxonomy_version, classes = load_taxonomy("segmentation-taxonomy-v1.yaml")
    valid_ids = {item.class_id for item in classes}
    output_root = paths.processed / "segmentation_v1" / dataset_id
    samples: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    resumed = 0
    for pair in discovery.pairs:
        width, height = image_dimensions(pair.image)
        source_mask = read_source_mask(pair.annotation, mapping)
        if source_mask.shape != (height, width):
            raise BlockingValidationError(
                f"Image/mask dimensions differ for {pair.image.name}: "
                f"image={width}x{height}, mask={source_mask.shape[1]}x{source_mask.shape[0]}",
                code="dimension_mismatch",
                details=[{"image": str(pair.image), "annotation": str(pair.annotation)}],
            )
        target, class_counts, ignored = convert_source_mask(
            source_mask,
            mapping,
            path=pair.annotation,
            valid_target_ids=valid_ids,
        )
        target_mask = output_root / pair.split / "masks" / f"{pair.image.stem}.png"
        target_image = output_root / pair.split / "images" / pair.image.name
        if not dry_run:
            if _same_mask(target_mask, target):
                resumed += 1
            else:
                _save_mask(target_mask, target)
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
        annotation_relative = stable_relative(pair.annotation, paths.root)
        target_relative = stable_relative(target_mask, paths.root)
        image_hash = sha256_file(pair.image)
        annotation_hash = sha256_file(pair.annotation)
        sample = {
            "sample_id": stable_sample_id(
                dataset_id, pair.split, stable_relative(pair.image, paths.root)
            ),
            "source_dataset": dataset_id,
            "source_split": pair.split,
            "target_split": pair.split,
            "image_path": image_relative,
            "source_annotation_path": annotation_relative,
            "target_annotation_path": target_relative,
            "width": width,
            "height": height,
            "image_hash": image_hash,
            "annotation_hash": annotation_hash,
            "class_counts": {str(key): value for key, value in class_counts.items()},
            "ignored_count": ignored,
            "invalid_count": 0,
            "preparation_version": "segmentation_v1",
            "taxonomy_version": taxonomy_version,
            "objects": [],
        }
        samples.append(sample)
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
    mapping_path = TAXONOMY_ROOT / f"{dataset_id}-mapping-v1.yaml"
    manifest_path = paths.manifests / f"{dataset_id}-segmentation_v1.json"
    created_at = read_json(manifest_path).get("created_at") if manifest_path.is_file() else None
    manifest = build_manifest(
        dataset_id=dataset_id,
        task_type=record.task_type.value,
        source_version="imported-source-unverified",
        preparation_version="segmentation_v1",
        taxonomy_version=taxonomy_version,
        integrity_mode=integrity.value,
        samples=samples,
        source_records=source_records,
        mapping_hashes={mapping.mapping_version: sha256_file(mapping_path)},
        preparation={"materialization": materialization.value, "splits_preserved": True},
        created_at=created_at,
    )
    if not dry_run:
        paths.manifests.mkdir(parents=True, exist_ok=True)
        manifest_paths = write_manifest(manifest, paths.manifests)
        lock = {
            "schema_version": "dataset-lock-v1",
            "dataset_id": dataset_id,
            "source_reference": record.official_reference,
            "acquisition_method": "imported-source",
            "acquired_at": manifest["created_at"],
            "source_fingerprint": stable_digest(source_records),
            "preparation_version": "segmentation_v1",
            "taxonomy_version": taxonomy_version,
            "mapping_hash": sha256_file(mapping_path),
            "integrity_mode": integrity.value,
            "dataset_fingerprint": manifest["fingerprint"],
            "complete": True,
        }
        atomic_write_json(paths.locks / f"{dataset_id}-segmentation_v1.json", lock)
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
