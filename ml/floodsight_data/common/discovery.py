from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from floodsight_data.common.images import ANNOTATION_EXTENSIONS, IMAGE_EXTENSIONS

ANNOTATION_TOKENS = ("mask", "label", "annotation", "groundtruth", "ground-truth", "gt")
IMAGE_TOKENS = ("image", "images", "org-img", "org_img", "rgb")


@dataclass(frozen=True, slots=True)
class SourcePair:
    split: str
    image: Path
    annotation: Path


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    pairs: tuple[SourcePair, ...]
    missing_images: tuple[Path, ...]
    missing_annotations: tuple[Path, ...]
    conflicting_annotations: tuple[tuple[Path, Path], ...]


def canonical_split(path: Path) -> str | None:
    lowered = "/".join(part.lower() for part in path.parts)
    tokens = re.split(r"[^a-z0-9]+", lowered)
    if any(token in {"train", "training", "trainset"} for token in tokens):
        return "train"
    if any(token in {"val", "valid", "validation", "valset"} for token in tokens):
        return "val"
    if "test-dev" in lowered or "test_dev" in lowered:
        return "test-dev"
    if any(token in {"test", "testing", "testset"} for token in tokens):
        return "test"
    return None


def normalized_stem(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_lab", "-lab", "_label", "-label", "_mask", "-mask", "_gt", "-gt"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _has_token(path: Path, tokens: tuple[str, ...]) -> bool:
    lowered_parts = [part.lower() for part in path.parts[:-1]]
    return any(any(token in part for token in tokens) for part in lowered_parts)


def discover_segmentation_pairs(root: Path) -> DiscoveryResult:
    images: dict[tuple[str, str], Path] = {}
    annotations: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        split = canonical_split(path.relative_to(root))
        if split is None:
            continue
        key = (split, normalized_stem(path))
        is_annotation = path.suffix.lower() in ANNOTATION_EXTENSIONS and _has_token(
            path.relative_to(root), ANNOTATION_TOKENS
        )
        is_image = path.suffix.lower() in IMAGE_EXTENSIONS and (
            _has_token(path.relative_to(root), IMAGE_TOKENS) or not is_annotation
        )
        if is_annotation:
            annotations.setdefault(key, []).append(path)
        elif is_image:
            images.setdefault(key, path)
    shared = sorted(images.keys() & annotations.keys())
    pairs: list[SourcePair] = []
    conflicts: list[tuple[Path, Path]] = []
    for split, stem in shared:
        image = images[(split, stem)]
        candidates = annotations[(split, stem)]
        scored = sorted(
            ((_shared_parent_depth(image, candidate), candidate) for candidate in candidates),
            key=lambda item: (-item[0], item[1].as_posix()),
        )
        best_score, selected = scored[0]
        conflicts.extend(
            (selected, candidate) for score, candidate in scored[1:] if score == best_score
        )
        pairs.append(SourcePair(split, image, selected))
    missing_images = tuple(
        annotation
        for key in sorted(annotations.keys() - images.keys())
        for annotation in annotations[key]
    )
    missing_annotations = tuple(images[key] for key in sorted(images.keys() - annotations.keys()))
    return DiscoveryResult(tuple(pairs), missing_images, missing_annotations, tuple(conflicts))


def _shared_parent_depth(left: Path, right: Path) -> int:
    depth = 0
    for left_part, right_part in zip(left.parent.parts, right.parent.parts, strict=False):
        if left_part != right_part:
            break
        depth += 1
    return depth


def discover_visdrone_pairs(root: Path) -> DiscoveryResult:
    images: dict[tuple[str, str], Path] = {}
    annotations: dict[tuple[str, str], Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        split = canonical_split(relative)
        if split is None:
            continue
        key = (split, path.stem.lower())
        parent = path.parent.name.lower()
        if parent == "images" and path.suffix.lower() in IMAGE_EXTENSIONS:
            images[key] = path
        elif parent == "annotations" and path.suffix.lower() == ".txt":
            annotations[key] = path
    shared = sorted(images.keys() & annotations.keys())
    pairs = tuple(
        SourcePair(split, images[(split, stem)], annotations[(split, stem)])
        for split, stem in shared
    )
    missing_images = tuple(annotations[key] for key in sorted(annotations.keys() - images.keys()))
    missing_annotations = tuple(images[key] for key in sorted(images.keys() - annotations.keys()))
    return DiscoveryResult(pairs, missing_images, missing_annotations, ())
