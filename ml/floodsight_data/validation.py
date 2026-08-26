from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from floodsight_data.common.discovery import (
    DiscoveryResult,
    SourcePair,
    discover_segmentation_pairs,
    discover_visdrone_pairs,
)
from floodsight_data.common.images import image_dimensions, source_label_inventory
from floodsight_data.errors import BlockingValidationError, DatasetToolError
from floodsight_data.hashing import sha256_file
from floodsight_data.paths import DataPaths
from floodsight_data.registry import TaskType, get_dataset
from floodsight_data.taxonomy import MappingAction, load_mapping
from floodsight_data.visdrone.parser import parse_annotation


@dataclass(frozen=True, slots=True)
class DuplicateAudit:
    duplicate_images: tuple[tuple[str, ...], ...]
    duplicate_annotations: tuple[tuple[str, ...], ...]
    cross_split_leakage: tuple[tuple[str, ...], ...]
    conflicting_annotations: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _group_hashes(items: Iterable[tuple[str, Path]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for label, path in items:
        groups[sha256_file(path)].append(label)
    return groups


def audit_duplicates(pairs: Iterable[SourcePair], root: Path) -> DuplicateAudit:
    pairs = tuple(pairs)
    image_groups = _group_hashes(
        (f"{pair.split}:{pair.image.relative_to(root).as_posix()}", pair.image) for pair in pairs
    )
    annotation_groups = _group_hashes(
        (
            f"{pair.split}:{pair.annotation.relative_to(root).as_posix()}",
            pair.annotation,
        )
        for pair in pairs
    )
    duplicate_images = tuple(
        tuple(sorted(items)) for items in image_groups.values() if len(items) > 1
    )
    duplicate_annotations = tuple(
        tuple(sorted(items)) for items in annotation_groups.values() if len(items) > 1
    )
    leakage = tuple(
        group for group in duplicate_images if len({item.split(":", 1)[0] for item in group}) > 1
    )
    annotation_by_image: dict[str, set[str]] = defaultdict(set)
    labels_by_image: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        image_hash = sha256_file(pair.image)
        annotation_by_image[image_hash].add(sha256_file(pair.annotation))
        labels_by_image[image_hash].append(pair.image.relative_to(root).as_posix())
    conflicts = tuple(
        tuple(sorted(labels_by_image[image_hash]))
        for image_hash, annotations in annotation_by_image.items()
        if len(annotations) > 1
    )
    return DuplicateAudit(
        tuple(sorted(duplicate_images)),
        tuple(sorted(duplicate_annotations)),
        tuple(sorted(leakage)),
        tuple(sorted(conflicts)),
    )


def _discovery(paths: DataPaths, dataset_id: str) -> tuple[Path, DiscoveryResult]:
    record = get_dataset(dataset_id)
    root = paths.dataset_raw(dataset_id)
    if not root.is_dir():
        raise DatasetToolError(f"Dataset is missing at {root}.", code="dataset_missing")
    if record.task_type is TaskType.SEMANTIC_SEGMENTATION:
        return root, discover_segmentation_pairs(root)
    return root, discover_visdrone_pairs(root)


def inspect_source(paths: DataPaths, dataset_id: str) -> dict[str, Any]:
    record = get_dataset(dataset_id)
    root, discovery = _discovery(paths, dataset_id)
    mapping = load_mapping(dataset_id)
    extensions: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    source_labels: Counter[int] = Counter()
    empty_annotations = 0
    corrupt_files: list[str] = []
    for pair in discovery.pairs:
        extensions[pair.image.suffix.lower()] += 1
        extensions[pair.annotation.suffix.lower()] += 1
        try:
            width, height = image_dimensions(pair.image)
            dimensions[f"{width}x{height}"] += 1
            if record.task_type is TaskType.SEMANTIC_SEGMENTATION:
                inventory = source_label_inventory(pair.annotation, mapping)
                for label in inventory["labels"]:
                    source_labels[int(label["source_id"])] += int(label["pixel_count"])
            else:
                objects = parse_annotation(pair.annotation)
                if not objects:
                    empty_annotations += 1
                for item in objects:
                    if item.class_id not in mapping.by_source_id:
                        raise BlockingValidationError(
                            f"Unsupported source class {item.class_id} in {pair.annotation}.",
                            code="unsupported_detection_class",
                        )
                    source_labels[item.class_id] += 1
        except BlockingValidationError as exc:
            if exc.code in {"image_corrupt", "annotation_corrupt"}:
                corrupt_files.append(str(pair.image))
            else:
                raise
    ignored_labels = [
        entry.source_id
        for entry in mapping.entries
        if entry.action in {MappingAction.IGNORE, MappingAction.ERROR}
    ]
    return {
        "dataset_id": dataset_id,
        "task_type": record.task_type.value,
        "source_root": str(root),
        "split_counts": dict(sorted(Counter(pair.split for pair in discovery.pairs).items())),
        "image_count": len(discovery.pairs) + len(discovery.missing_annotations),
        "annotation_count": len(discovery.pairs) + len(discovery.missing_images),
        "paired_count": len(discovery.pairs),
        "image_extensions": dict(sorted(extensions.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "source_labels": {
            str(source_id): {
                "name": mapping.by_source_id[source_id].source_name,
                "count": count,
            }
            for source_id, count in sorted(source_labels.items())
        },
        "mask_representation": (
            "indexed, palette, or exact RGB PNG (inspected per file)"
            if record.task_type is TaskType.SEMANTIC_SEGMENTATION
            else None
        ),
        "pairing_rule": "canonical split plus normalized filename stem",
        "missing_images": [str(path.relative_to(root)) for path in discovery.missing_images],
        "missing_annotations": [
            str(path.relative_to(root)) for path in discovery.missing_annotations
        ],
        "conflicting_annotations": [
            [str(first.relative_to(root)), str(second.relative_to(root))]
            for first, second in discovery.conflicting_annotations
        ],
        "corrupt_files": corrupt_files,
        "empty_annotations": empty_annotations,
        "ignored_source_labels": ignored_labels,
        "unknown_labels": [],
        "mapping_review_required": mapping.real_data_review_required,
    }


def validate_dataset(paths: DataPaths, dataset_id: str) -> dict[str, Any]:
    root, discovery = _discovery(paths, dataset_id)
    inventory = inspect_source(paths, dataset_id)
    duplicates = audit_duplicates(discovery.pairs, root)
    blocking: list[str] = []
    if discovery.missing_images:
        blocking.append(f"{len(discovery.missing_images)} annotations lack images")
    if discovery.missing_annotations:
        blocking.append(f"{len(discovery.missing_annotations)} images lack annotations")
    if discovery.conflicting_annotations:
        blocking.append(f"{len(discovery.conflicting_annotations)} conflicting annotation pairs")
    if duplicates.cross_split_leakage:
        blocking.append(f"{len(duplicates.cross_split_leakage)} cross-split image leaks")
    if duplicates.conflicting_annotations:
        blocking.append(
            f"{len(duplicates.conflicting_annotations)} identical images have "
            "conflicting annotations"
        )
    if inventory["corrupt_files"]:
        blocking.append(f"{len(inventory['corrupt_files'])} corrupt files")
    return {
        **inventory,
        "duplicates": duplicates.to_dict(),
        "blocking_errors": blocking,
        "valid": not blocking,
    }


FORBIDDEN_DATA_SUFFIXES = {
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".7z",
    ".rar",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".engine",
}
IGNORED_REPOSITORY_PARTS = {
    ".git",
    ".venv",
    ".venv-datasets",
    "node_modules",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}


def guard_repository_artifacts(repository_root: Path) -> list[str]:
    violations: list[str] = []
    for path in repository_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repository_root)
        if any(part in IGNORED_REPOSITORY_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in FORBIDDEN_DATA_SUFFIXES:
            violations.append(relative.as_posix())
            continue
        if relative.parts and relative.parts[0] == "datasets" and path.name != ".gitkeep":
            violations.append(relative.as_posix())
            continue
        if path.stat().st_size > 25 * 1024 * 1024:
            violations.append(f"{relative.as_posix()} (>25 MiB)")
    return sorted(violations)
