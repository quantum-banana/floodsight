from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from floodsight_data.errors import BlockingValidationError
from floodsight_data.taxonomy import IGNORE_INDEX, MappingAction, MappingTable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ANNOTATION_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp"}


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise BlockingValidationError(
            f"Corrupt or unsupported image: {path}",
            code="image_corrupt",
            details=[{"file": str(path)}],
        ) from exc


def _unknown_mask(path: Path, values: list[object], *, kind: str) -> BlockingValidationError:
    return BlockingValidationError(
        f"Unknown segmentation {kind} in {path}: {values[:10]}",
        code=f"unknown_mask_{kind}",
        details=[{"file": str(path), "values": values[:50]}],
    )


def _color_array_to_ids(
    rgb: NDArray[np.uint8],
    mapping: MappingTable,
    path: Path,
) -> NDArray[np.int32]:
    height, width, _ = rgb.shape
    source = np.full((height, width), -1, dtype=np.int32)
    known = mapping.by_color
    unique = np.unique(rgb.reshape(-1, 3), axis=0)
    unknown: list[tuple[int, int, int]] = []
    for raw_color in unique:
        color = tuple(int(channel) for channel in raw_color)
        entry = known.get(color)
        if entry is None:
            unknown.append(color)
            continue
        matches = np.all(rgb == raw_color, axis=2)
        source[matches] = entry.source_id
    if unknown:
        raise _unknown_mask(path, unknown, kind="colors")
    return source


def read_source_mask(path: Path, mapping: MappingTable) -> NDArray[np.int32]:
    try:
        with Image.open(path) as image:
            if image.mode == "P":
                indexed = np.asarray(image, dtype=np.int32)
                unknown_ids = sorted(set(np.unique(indexed).tolist()) - set(mapping.by_source_id))
                if not unknown_ids:
                    return indexed.copy()
                palette = image.getpalette()
                if palette is None:
                    raise _unknown_mask(path, unknown_ids, kind="ids")
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                return _color_array_to_ids(rgb, mapping, path)
            if image.mode in {"1", "L", "I", "I;16"}:
                indexed = np.asarray(image, dtype=np.int32)
                unknown_ids = sorted(set(np.unique(indexed).tolist()) - set(mapping.by_source_id))
                if unknown_ids:
                    raise _unknown_mask(path, unknown_ids, kind="ids")
                return indexed.copy()
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return _color_array_to_ids(rgb, mapping, path)
    except BlockingValidationError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise BlockingValidationError(
            f"Corrupt segmentation mask: {path}",
            code="annotation_corrupt",
            details=[{"file": str(path)}],
        ) from exc


def convert_source_mask(
    source: NDArray[np.int32],
    mapping: MappingTable,
    *,
    path: Path,
    valid_target_ids: set[int],
) -> tuple[NDArray[np.uint8], dict[int, int], int]:
    target = np.full(source.shape, IGNORE_INDEX, dtype=np.uint8)
    for source_id in np.unique(source).tolist():
        entry = mapping.by_source_id.get(int(source_id))
        if entry is None:
            raise _unknown_mask(path, [int(source_id)], kind="ids")
        if entry.action is MappingAction.ERROR:
            raise BlockingValidationError(
                f"Source label is explicitly rejected: {entry.source_name} in {path}",
                code="source_label_rejected",
            )
        if entry.action in {MappingAction.MAP, MappingAction.MERGE}:
            assert entry.target_id is not None
            target[source == source_id] = entry.target_id
    output_ids = set(np.unique(target).tolist())
    invalid = sorted(output_ids - valid_target_ids - {IGNORE_INDEX})
    if invalid:
        raise BlockingValidationError(
            f"Conversion produced invalid target IDs in {path}: {invalid}",
            code="target_label_invalid",
        )
    counts = Counter(int(value) for value in target.reshape(-1) if value != IGNORE_INDEX)
    ignored = int(np.count_nonzero(target == IGNORE_INDEX))
    return target, dict(sorted(counts.items())), ignored


def source_label_inventory(path: Path, mapping: MappingTable) -> dict[str, object]:
    source = read_source_mask(path, mapping)
    counts = Counter(int(value) for value in source.reshape(-1))
    return {
        "path": str(path),
        "representation": "indexed-or-resolved-color",
        "labels": [
            {
                "source_id": source_id,
                "source_name": mapping.by_source_id[source_id].source_name,
                "pixel_count": count,
            }
            for source_id, count in sorted(counts.items())
        ],
    }
