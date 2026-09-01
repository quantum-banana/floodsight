"""Strict manifest-to-YOLO dataset boundary.

Only the corrected Phase 3 ``visdrone_det-detection_v2`` manifest is accepted.  Every
referenced training label is parsed, every retained box is checked, and a
run-local immutable contract is generated without modifying the dataset root.
"""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import re
import shutil
import stat
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file, stable_sha256

DETECTION_CLASSES: dict[int, str] = {
    0: "person",
    1: "car",
    2: "van",
    3: "truck",
    4: "bus",
    5: "bicycle",
    6: "motorcycle",
    7: "tricycle",
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_INTEGER = re.compile(r"^(0|[1-9][0-9]*)$")
_ALLOWED_SPLITS = frozenset({"train", "val", "test", "test-dev"})
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
_VISDRONE_TO_TARGET: dict[int, int | None] = {
    0: None,
    1: 0,
    2: 0,
    3: 5,
    4: 1,
    5: 2,
    6: 3,
    7: 7,
    8: 7,
    9: 4,
    10: 6,
    11: None,
}
_CANONICAL_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_AUTHORIZED_INVALID_SOURCE_ROW = {
    "path": (
        "raw/visdrone_det/VisDrone2019-DET-train/annotations/"
        "9999985_00000_d_0000020.txt"
    ),
    "line": 12,
    "raw": "611,158,4,0,1,4,0,0",
    "values": (611.0, 158.0, 4.0, 0.0, 1, 4, 0, 0),
}
_FICLONE = 0x40049409
_REFLINK_FALLBACK_ERRNOS = {
    errno.EXDEV,
    errno.EINVAL,
    errno.ENOTTY,
    errno.EOPNOTSUPP,
    errno.ENOSYS,
}
_SAMPLE_KEYS = frozenset(
    {
        "annotation_hash",
        "class_counts",
        "height",
        "ignored_count",
        "image_hash",
        "image_path",
        "invalid_count",
        "objects",
        "preparation_version",
        "sample_id",
        "source_annotation_path",
        "source_dataset",
        "source_split",
        "target_annotation_hash",
        "target_annotation_path",
        "target_image_hash",
        "target_split",
        "taxonomy_version",
        "width",
    }
)


@dataclass(frozen=True, slots=True)
class ValidatedSample:
    sample_id: str
    split: str
    image_path: Path
    source_annotation_path: Path
    label_path: Path
    image_sha256: str
    source_annotation_sha256: str
    label_sha256: str
    width: int
    height: int
    box_count: int
    class_counts: dict[int, int]


@dataclass(frozen=True, slots=True)
class DatasetContract:
    manifest_path: Path
    manifest_sha256: str
    dataset_fingerprint: str
    data_root: Path
    samples: tuple[ValidatedSample, ...]
    split_counts: dict[str, int]
    class_counts: dict[int, int]
    total_boxes: int
    image_hashes_verified: bool

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "floodsight-detection-contract-audit-v1",
            "dataset_id": "visdrone_det",
            "preparation_version": "detection_v2",
            "taxonomy_version": "detection-taxonomy-v1",
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "dataset_fingerprint": self.dataset_fingerprint,
            "data_root": str(self.data_root),
            "sample_count": len(self.samples),
            "split_counts": dict(sorted(self.split_counts.items())),
            "class_counts": {
                str(class_id): self.class_counts.get(class_id, 0) for class_id in DETECTION_CLASSES
            },
            "total_boxes": self.total_boxes,
            "image_hashes_verified": self.image_hashes_verified,
            "labels_exhaustively_validated": True,
            "class_names": DETECTION_CLASSES,
        }


def _fail(message: str, code: str, **detail: Any) -> None:
    raise DetectionInfrastructureError(
        message,
        code=code,
        details=[detail] if detail else None,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionInfrastructureError(
            f"Unable to read detection manifest: {path}",
            code="manifest_unreadable",
        ) from exc
    if not isinstance(payload, dict):
        _fail("Detection manifest must be a JSON object.", "manifest_invalid")
    return payload


def _contained_file(root: Path, relative: Any, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\0" in relative:
        _fail(f"{field} is not a safe POSIX relative path.", "unsafe_dataset_path", field=field)
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(
            f"{field} is not a safe relative path: {relative!r}",
            "unsafe_dataset_path",
            field=field,
            path=relative,
        )
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise DetectionInfrastructureError(
            f"{field} is missing or escapes the data root: {relative}",
            code="unsafe_dataset_path",
            details=[{"field": field, "path": relative}],
        ) from exc
    if not resolved.is_file():
        _fail(
            f"{field} is not a regular file: {relative}",
            "dataset_file_invalid",
            field=field,
            path=relative,
        )
    return resolved


def _require_string(payload: dict[str, Any], key: str, expected: str) -> None:
    if payload.get(key) != expected:
        _fail(
            f"Manifest {key} must be {expected!r}; found {payload.get(key)!r}.",
            "manifest_contract_mismatch",
            field=key,
            expected=expected,
            actual=payload.get(key),
        )


def _require_hash(value: Any, *, field: str, sample_id: str | None = None) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(
            f"{field} must be a lowercase SHA-256 digest.",
            "manifest_hash_invalid",
            field=field,
            sample_id=sample_id,
        )
    return value


def _manifest_counts(value: Any, *, sample_id: str) -> dict[int, int]:
    if not isinstance(value, dict):
        _fail("class_counts must be an object.", "manifest_sample_invalid", sample_id=sample_id)
    counts: dict[int, int] = {}
    for raw_class, raw_count in value.items():
        if not isinstance(raw_class, str) or _INTEGER.fullmatch(raw_class) is None:
            _fail("class_counts contains a non-integer class key.", "class_contract_mismatch")
        class_id = int(raw_class)
        if class_id not in DETECTION_CLASSES:
            _fail(
                f"Manifest contains unsupported target class {class_id}.",
                "unsupported_target_class",
                sample_id=sample_id,
                class_id=class_id,
            )
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            _fail("class_counts contains an invalid count.", "manifest_sample_invalid")
        if raw_count:
            counts[class_id] = raw_count
    return counts


def _parse_yolo_label(path: Path, *, sample_id: str) -> tuple[int, dict[int, int]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DetectionInfrastructureError(
            f"Unable to read YOLO label: {path}",
            code="label_unreadable",
            details=[{"sample_id": sample_id, "path": str(path)}],
        ) from exc
    counts: Counter[int] = Counter()
    box_count = 0
    seen_boxes: set[tuple[int, float, float, float, float]] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            _fail(
                "Blank lines are not allowed inside YOLO label files.",
                "label_row_invalid",
                sample_id=sample_id,
                path=str(path),
                line=line_number,
            )
        fields = line.split()
        if len(fields) != 5 or _INTEGER.fullmatch(fields[0]) is None:
            _fail(
                "Each YOLO row must contain an integer class and four coordinates.",
                "label_row_invalid",
                sample_id=sample_id,
                path=str(path),
                line=line_number,
            )
        class_id = int(fields[0])
        if class_id not in DETECTION_CLASSES:
            _fail(
                f"YOLO label contains unsupported target class {class_id}.",
                "unsupported_target_class",
                sample_id=sample_id,
                path=str(path),
                line=line_number,
            )
        try:
            center_x, center_y, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise DetectionInfrastructureError(
                f"YOLO label contains a non-numeric coordinate at {path}:{line_number}.",
                code="label_row_invalid",
            ) from exc
        values = (center_x, center_y, width, height)
        if not all(math.isfinite(value) for value in values):
            _fail("YOLO coordinates must be finite.", "label_box_invalid")
        tolerance = 1e-6
        if not (0 <= center_x <= 1 and 0 <= center_y <= 1):
            _fail("YOLO box centre is outside [0,1].", "label_box_invalid")
        if not (0 < width <= 1 and 0 < height <= 1):
            _fail("YOLO box size must be in (0,1].", "label_box_invalid")
        if (
            center_x - width / 2 < -tolerance
            or center_y - height / 2 < -tolerance
            or center_x + width / 2 > 1 + tolerance
            or center_y + height / 2 > 1 + tolerance
        ):
            _fail(
                "YOLO box edges extend outside the normalized image bounds.",
                "label_box_invalid",
                sample_id=sample_id,
                path=str(path),
                line=line_number,
            )
        box = (class_id, center_x, center_y, width, height)
        if box in seen_boxes:
            _fail(
                "Duplicate semantic YOLO rows are forbidden because the runtime "
                "would silently remove them.",
                "duplicate_label_row",
                sample_id=sample_id,
                path=str(path),
                line=line_number,
            )
        seen_boxes.add(box)
        counts[class_id] += 1
        box_count += 1
    return box_count, dict(counts)


def _verify_jpeg_image(path: Path, *, width: int, height: int, sample_id: str) -> None:
    """Fully decode the exact JPEG policy used by the frozen VisDrone manifest."""

    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        _fail(
            "The frozen VisDrone detector contract accepts JPEG images only.",
            "image_format_mismatch",
            sample_id=sample_id,
        )
    try:
        with path.open("rb") as stream:
            stream.seek(-2, os.SEEK_END)
            if stream.read(2) != b"\xff\xd9":
                _fail(
                    "JPEG is missing its EOI marker; Ultralytics could rewrite it.",
                    "jpeg_eoi_missing",
                    sample_id=sample_id,
                    path=str(path),
                )
        from PIL import Image, ImageFile  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DetectionInfrastructureError(
            "Pillow is required for exhaustive detection image validation.",
            code="ml_dependency_missing",
        ) from exc
    except OSError as exc:
        raise DetectionInfrastructureError(
            f"Unable to read detector image: {path}",
            code="image_decode_failed",
        ) from exc
    previous_truncated_policy = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as image:
            if image.format != "JPEG" or getattr(image, "n_frames", 1) != 1:
                _fail(
                    "Detector image is not a single-frame JPEG.",
                    "image_format_mismatch",
                    sample_id=sample_id,
                )
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.format != "JPEG" or image.mode != "RGB" or image.size != (width, height):
                _fail(
                    "Decoded JPEG mode or dimensions differ from the frozen manifest.",
                    "image_dimension_mismatch",
                    sample_id=sample_id,
                    expected=[width, height, "RGB"],
                    actual=[*image.size, image.mode],
                )
    except DetectionInfrastructureError:
        raise
    except (OSError, SyntaxError, ValueError) as exc:
        raise DetectionInfrastructureError(
            f"Detector JPEG failed a complete decode: {path}",
            code="image_decode_failed",
            details=[{"sample_id": sample_id, "path": str(path)}],
        ) from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated_policy


def _source_rows(
    path: Path, *, sample_id: str
) -> list[tuple[str, tuple[float, float, float, float, int, int, int, int]]]:
    try:
        raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DetectionInfrastructureError(
            f"Unable to read source annotation: {path}",
            code="source_annotation_unreadable",
            details=[{"sample_id": sample_id}],
        ) from exc
    rows: list[
        tuple[str, tuple[float, float, float, float, int, int, int, int]]
    ] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            _fail(
                "Blank source annotation rows are forbidden.",
                "source_annotation_invalid",
                sample_id=sample_id,
                line=line_number,
            )
        fields = [field.strip() for field in raw_line.split(",")]
        if fields and fields[-1] == "":
            fields.pop()
        if len(fields) != 8:
            _fail(
                "VisDrone source rows must contain exactly eight fields.",
                "source_annotation_invalid",
                sample_id=sample_id,
                line=line_number,
            )
        try:
            geometry = tuple(float(value) for value in fields[:4])
            if not all(math.isfinite(value) for value in geometry):
                raise ValueError
            if any(_CANONICAL_INTEGER.fullmatch(value) is None for value in fields[4:]):
                raise ValueError
            metadata = tuple(int(value) for value in fields[4:])
        except ValueError as exc:
            raise DetectionInfrastructureError(
                f"Non-canonical VisDrone source row at {path}:{line_number}.",
                code="source_annotation_invalid",
            ) from exc
        score, source_class_id, truncation, occlusion = metadata
        if (
            score not in {0, 1}
            or source_class_id not in _VISDRONE_TO_TARGET
            or truncation not in {0, 1, 2}
            or occlusion not in {0, 1, 2}
        ):
            _fail(
                "VisDrone source metadata is outside the frozen ranges.",
                "source_annotation_invalid",
                sample_id=sample_id,
                line=line_number,
            )
        rows.append((raw_line, (*geometry, score, source_class_id, truncation, occlusion)))
    return rows


def _reconstruct_source_contract(
    path: Path,
    *,
    relative_path: str,
    width: int,
    height: int,
    sample_id: str,
) -> tuple[list[str], list[dict[str, Any]], int]:
    target_lines: list[str] = []
    objects: list[dict[str, Any]] = []
    ignored_count = 0
    seen: set[tuple[int, float, float, float, float]] = set()
    for line_number, (raw_line, row) in enumerate(
        _source_rows(path, sample_id=sample_id), start=1
    ):
        left, top, box_width, box_height, score, source_class_id, truncation, occlusion = row
        target_class_id = _VISDRONE_TO_TARGET[source_class_id]
        if score == 0 or target_class_id is None:
            ignored_count += 1
            continue
        invalid = (
            box_width <= 0
            or box_height <= 0
            or left + box_width <= 0
            or top + box_height <= 0
            or left >= width
            or top >= height
        )
        if invalid:
            authorized = (
                relative_path == _AUTHORIZED_INVALID_SOURCE_ROW["path"]
                and line_number == _AUTHORIZED_INVALID_SOURCE_ROW["line"]
                and raw_line == _AUTHORIZED_INVALID_SOURCE_ROW["raw"]
                and row == _AUTHORIZED_INVALID_SOURCE_ROW["values"]
            )
            if authorized:
                continue
            _fail(
                "Unapproved retained source geometry is invalid.",
                "source_geometry_invalid",
                sample_id=sample_id,
                line=line_number,
            )
        source_right = left + box_width
        source_bottom = top + box_height
        clamped_left = max(0.0, left)
        clamped_top = max(0.0, top)
        clamped_right = min(float(width), source_right)
        clamped_bottom = min(float(height), source_bottom)
        if clamped_right <= clamped_left or clamped_bottom <= clamped_top:
            _fail("Retained source geometry is outside the image.", "source_geometry_invalid")
        values = (
            ((clamped_left + clamped_right) / 2) / width,
            ((clamped_top + clamped_bottom) / 2) / height,
            (clamped_right - clamped_left) / width,
            (clamped_bottom - clamped_top) / height,
        )
        formatted = tuple(f"{value:.6f}" for value in values)
        semantic_key = (target_class_id, *(float(value) for value in formatted))
        if semantic_key in seen:
            continue
        seen.add(semantic_key)
        target_lines.append(f"{target_class_id} {' '.join(formatted)}")
        objects.append(
            {
                "source_class_id": source_class_id,
                "target_class_id": target_class_id,
                "truncation": truncation,
                "occlusion": occlusion,
                "clamped": (clamped_left, clamped_top, clamped_right, clamped_bottom)
                != (left, top, source_right, source_bottom),
            }
        )
    return target_lines, objects, ignored_count


def validate_dataset_contract(
    manifest_path: str | Path,
    data_root: str | Path,
    *,
    expected_manifest_path: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
    expected_dataset_fingerprint: str | None = None,
    expected_source_version: str | None = None,
    verify_image_hashes: bool = True,
    require_full_integrity: bool = True,
    required_splits: tuple[str, ...] = ("train", "val"),
    require_all_train_classes: bool = True,
    reject_duplicate_images: bool = True,
) -> DatasetContract:
    """Validate every sample and label before Ultralytics can see the dataset."""

    root = Path(data_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        _fail("The configured data root is not a directory.", "data_root_invalid")
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    if expected_manifest_path is not None:
        expected_path = Path(expected_manifest_path).expanduser()
        if (
            not expected_path.is_absolute()
            or expected_path.is_symlink()
            or expected_path.resolve(strict=True) != manifest
        ):
            _fail(
                "The detection manifest path differs from the config-bound canonical file.",
                "manifest_identity_mismatch",
            )
    actual_manifest_sha256 = sha256_file(manifest)
    if (
        expected_manifest_sha256 is not None
        and actual_manifest_sha256 != expected_manifest_sha256
    ):
        _fail(
            "The detection manifest SHA-256 differs from the frozen configuration.",
            "manifest_identity_mismatch",
        )
    try:
        manifest.relative_to(root)
    except ValueError as exc:
        raise DetectionInfrastructureError(
            "The detection manifest must be contained by the explicit data root.",
            code="unsafe_manifest_path",
        ) from exc
    payload = _read_manifest(manifest)
    _require_string(payload, "schema_version", "dataset-manifest-v1")
    _require_string(payload, "manifest_id", "visdrone_det-detection_v2")
    _require_string(payload, "dataset_id", "visdrone_det")
    _require_string(payload, "task_type", "AERIAL_DETECTION")
    source_version = payload.get("source_version")
    if not isinstance(source_version, str) or not source_version:
        _fail("Detection manifest source_version is missing.", "manifest_identity_mismatch")
    if expected_source_version is not None and source_version != expected_source_version:
        _fail(
            "Detection manifest source_version differs from the frozen configuration.",
            "manifest_identity_mismatch",
        )
    _require_string(payload, "preparation_version", "detection_v2")
    _require_string(payload, "taxonomy_version", "detection-taxonomy-v1")
    if require_full_integrity:
        _require_string(payload, "integrity_mode", "full")
    fingerprint = _require_hash(payload.get("fingerprint"), field="fingerprint")
    if (
        expected_dataset_fingerprint is not None
        and fingerprint != expected_dataset_fingerprint
    ):
        _fail(
            "Detection manifest fingerprint differs from the frozen configuration.",
            "manifest_identity_mismatch",
        )
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        _fail("Detection manifest has no samples.", "manifest_samples_missing")

    validated: list[ValidatedSample] = []
    split_counts: Counter[str] = Counter()
    all_class_counts: Counter[int] = Counter()
    train_class_counts: Counter[int] = Counter()
    sample_ids: set[str] = set()
    image_paths: set[Path] = set()
    label_paths: set[Path] = set()
    hashes: dict[str, tuple[str, str]] = {}
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            _fail("Manifest sample must be an object.", "manifest_sample_invalid", index=index)
        if set(raw) != _SAMPLE_KEYS:
            _fail(
                "Manifest sample fields differ from the frozen Stage13 schema.",
                "manifest_sample_invalid",
                index=index,
            )
        sample_id = raw.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            _fail("Manifest sample_id is missing.", "manifest_sample_invalid", index=index)
        if sample_id in sample_ids:
            _fail(f"Duplicate sample_id: {sample_id}", "duplicate_sample_id")
        sample_ids.add(sample_id)
        split = raw.get("target_split")
        if not isinstance(split, str) or split not in _ALLOWED_SPLITS:
            _fail(
                f"Unsupported target split {split!r} for {sample_id}.",
                "unsupported_target_split",
            )
        if raw.get("source_split") != split:
            _fail(
                f"Source/target split drift for {sample_id}.",
                "split_contract_mismatch",
            )
        if raw.get("source_dataset") != "visdrone_det":
            _fail("Sample source_dataset is not visdrone_det.", "manifest_sample_invalid")
        if raw.get("preparation_version") != "detection_v2":
            _fail("Sample preparation_version drifted.", "manifest_sample_invalid")
        if raw.get("taxonomy_version") != "detection-taxonomy-v1":
            _fail("Sample taxonomy_version drifted.", "manifest_sample_invalid")
        if raw.get("invalid_count") != 0:
            _fail(
                f"Sample {sample_id} records invalid retained objects.",
                "invalid_sample_objects",
            )
        width, height = raw.get("width"), raw.get("height")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height < 1
        ):
            _fail("Sample dimensions are invalid.", "manifest_sample_invalid")
        image_relative = raw.get("image_path")
        source_annotation_relative = raw.get("source_annotation_path")
        image = _contained_file(root, image_relative, field="image_path")
        source_annotation = _contained_file(
            root, source_annotation_relative, field="source_annotation_path"
        )
        label = _contained_file(
            root,
            raw.get("target_annotation_path"),
            field="target_annotation_path",
        )
        if (
            image.suffix.lower() not in _IMAGE_SUFFIXES
            or source_annotation.suffix.lower() != ".txt"
            or label.suffix.lower() != ".txt"
        ):
            _fail("Sample file suffixes are incompatible with YOLO.", "dataset_file_invalid")
        if image.stem != label.stem or source_annotation.stem != image.stem:
            _fail(
                f"Image/label stem mismatch for {sample_id}.",
                "image_label_pairing_failed",
            )
        if image in image_paths or label in label_paths:
            _fail("A dataset file is referenced by multiple samples.", "duplicate_dataset_path")
        image_paths.add(image)
        label_paths.add(label)
        expected_image_hash = _require_hash(
            raw.get("target_image_hash"), field="target_image_hash", sample_id=sample_id
        )
        expected_label_hash = _require_hash(
            raw.get("target_annotation_hash"),
            field="target_annotation_hash",
            sample_id=sample_id,
        )
        expected_source_annotation_hash = _require_hash(
            raw.get("annotation_hash"), field="annotation_hash", sample_id=sample_id
        )
        if expected_image_hash != _require_hash(
            raw.get("image_hash"), field="image_hash", sample_id=sample_id
        ):
            _fail("Materialized image hash differs from source image hash.", "image_hash_mismatch")
        actual_label_hash = sha256_file(label)
        if actual_label_hash != expected_label_hash:
            _fail(
                f"Target label hash mismatch for {sample_id}.",
                "label_hash_mismatch",
            )
        if sha256_file(source_annotation) != expected_source_annotation_hash:
            _fail(
                f"Source annotation hash mismatch for {sample_id}.",
                "source_annotation_hash_mismatch",
            )
        if verify_image_hashes and sha256_file(image) != expected_image_hash:
            _fail(
                f"Target image hash mismatch for {sample_id}.",
                "image_hash_mismatch",
            )
        _verify_jpeg_image(image, width=width, height=height, sample_id=sample_id)
        prior_hash = hashes.get(expected_image_hash)
        if reject_duplicate_images and prior_hash is not None:
            _fail(
                f"Duplicate image content: {prior_hash[0]} and {sample_id}.",
                "duplicate_image_content",
                first_sample_id=prior_hash[0],
                first_split=prior_hash[1],
                sample_id=sample_id,
                split=split,
            )
        hashes.setdefault(expected_image_hash, (sample_id, split))
        box_count, observed_counts = _parse_yolo_label(label, sample_id=sample_id)
        declared_counts = _manifest_counts(raw.get("class_counts"), sample_id=sample_id)
        if observed_counts != declared_counts:
            _fail(
                f"Label/manifest class count mismatch for {sample_id}.",
                "class_count_mismatch",
                sample_id=sample_id,
                manifest=declared_counts,
                observed=observed_counts,
            )
        reconstructed_lines, reconstructed_objects, reconstructed_ignored = (
            _reconstruct_source_contract(
                source_annotation,
                relative_path=str(source_annotation_relative),
                width=width,
                height=height,
                sample_id=sample_id,
            )
        )
        try:
            target_lines = label.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise DetectionInfrastructureError(
                f"Unable to re-read target annotation: {label}",
                code="label_unreadable",
            ) from exc
        if target_lines != reconstructed_lines:
            _fail(
                f"Source annotation does not reconstruct target labels for {sample_id}.",
                "source_target_geometry_mismatch",
            )
        objects = raw.get("objects")
        if not isinstance(objects, list) or len(objects) != box_count:
            _fail(
                f"Object provenance count mismatch for {sample_id}.",
                "object_provenance_mismatch",
            )
        if objects != reconstructed_objects:
            _fail(
                "Object provenance does not exactly match source mapping and geometry.",
                "object_provenance_mismatch",
            )
        if raw.get("ignored_count") != reconstructed_ignored:
            _fail(
                "Manifest ignored_count does not match source-row dispositions.",
                "ignored_count_mismatch",
            )
        split_counts[split] += 1
        all_class_counts.update(observed_counts)
        if split == "train":
            train_class_counts.update(observed_counts)
        validated.append(
            ValidatedSample(
                sample_id=sample_id,
                split=split,
                image_path=image,
                source_annotation_path=source_annotation,
                label_path=label,
                image_sha256=expected_image_hash,
                source_annotation_sha256=expected_source_annotation_hash,
                label_sha256=expected_label_hash,
                width=width,
                height=height,
                box_count=box_count,
                class_counts=observed_counts,
            )
        )

    missing_splits = sorted(set(required_splits) - split_counts.keys())
    if missing_splits:
        _fail(
            f"Required detection splits are absent: {', '.join(missing_splits)}.",
            "required_split_missing",
        )
    if require_all_train_classes:
        missing_classes = [
            class_id for class_id in DETECTION_CLASSES if train_class_counts[class_id] == 0
        ]
        if missing_classes:
            _fail(
                f"Training split lacks classes: {missing_classes}.",
                "train_class_coverage_incomplete",
                missing_classes=missing_classes,
            )
    return DatasetContract(
        manifest_path=manifest,
        manifest_sha256=actual_manifest_sha256,
        dataset_fingerprint=fingerprint,
        data_root=root,
        samples=tuple(sorted(validated, key=lambda item: item.sample_id)),
        split_counts=dict(split_counts),
        class_counts=dict(all_class_counts),
        total_boxes=sum(all_class_counts.values()),
        image_hashes_verified=verify_image_hashes,
    )


def _write_exclusive(path: Path, content: str) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise DetectionInfrastructureError(
            f"Refusing to overwrite frozen contract artifact: {path}",
            code="contract_collision",
        ) from exc


def source_integrity_snapshot(contract: DatasetContract) -> dict[str, Any]:
    """Rehash every immutable source/target input represented by a contract."""

    records: list[dict[str, Any]] = []
    for sample in contract.samples:
        identities = (
            ("image", sample.image_path, sample.image_sha256),
            (
                "source_annotation",
                sample.source_annotation_path,
                sample.source_annotation_sha256,
            ),
            ("target_label", sample.label_path, sample.label_sha256),
        )
        record: dict[str, Any] = {"sample_id": sample.sample_id}
        for label, path, expected in identities:
            if path.is_symlink() or not path.is_file():
                _fail("A source snapshot path became unsafe.", "source_integrity_drift")
            actual = sha256_file(path)
            if actual != expected:
                _fail(
                    f"Source integrity drifted for {sample.sample_id}: {label}.",
                    "source_integrity_drift",
                )
            record[f"{label}_path"] = str(path)
            record[f"{label}_sha256"] = actual
        records.append(record)
    return {
        "schema_version": "floodsight-detection-source-snapshot-v1",
        "sample_count": len(records),
        "records_sha256": stable_sha256(records),
    }


def _copy_independent_regular_file(source: Path, destination: Path) -> str:
    flags_source = os.O_RDONLY | os.O_CLOEXEC
    flags_destination = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags_source |= os.O_NOFOLLOW
        flags_destination |= os.O_NOFOLLOW
    source_fd = os.open(source, flags_source)
    try:
        destination_fd = os.open(destination, flags_destination, 0o600)
    except BaseException:
        os.close(source_fd)
        raise
    method = "reflink"
    try:
        if not stat.S_ISREG(os.fstat(source_fd).st_mode):
            _fail("Source image is not a regular file.", "dataset_file_invalid")
        try:
            fcntl.ioctl(destination_fd, _FICLONE, source_fd)
        except OSError as exc:
            if exc.errno not in _REFLINK_FALLBACK_ERRNOS:
                raise
            method = "byte_copy"
            os.ftruncate(destination_fd, 0)
            os.lseek(source_fd, 0, os.SEEK_SET)
            with os.fdopen(os.dup(source_fd), "rb") as source_stream, os.fdopen(
                os.dup(destination_fd), "wb"
            ) as destination_stream:
                shutil.copyfileobj(source_stream, destination_stream, 1024 * 1024)
                destination_stream.flush()
        os.fsync(destination_fd)
        source_stat = os.fstat(source_fd)
        destination_stat = os.fstat(destination_fd)
        if (
            not stat.S_ISREG(destination_stat.st_mode)
            or destination_stat.st_nlink != 1
            or (source_stat.st_dev, source_stat.st_ino)
            == (destination_stat.st_dev, destination_stat.st_ino)
        ):
            _fail(
                "Frozen image is not an independent regular file.",
                "image_materialization_failed",
            )
    finally:
        os.close(destination_fd)
        os.close(source_fd)
    destination.chmod(0o440)
    return method


def freeze_dataset_contract(contract: DatasetContract, output_directory: str | Path) -> Path:
    """Create one immutable, run-local YOLO data contract.

    The output directory must not already exist.  Split lists contain absolute
    paths to validated images, while the source dataset remains untouched.
    """

    output = Path(output_directory).expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise DetectionInfrastructureError(
            f"Refusing to reuse dataset contract directory: {output}",
            code="contract_collision",
        ) from exc
    grouped: dict[str, list[tuple[Path, ValidatedSample]]] = defaultdict(list)
    staged_samples: list[dict[str, Any]] = []
    for index, sample in enumerate(contract.samples):
        # Ultralytics may repair malformed JPEGs in place.  Every image is
        # therefore a private COW reflink or byte copy, never a symlink/hardlink.
        staged_stem = f"{index:08d}-{stable_sha256(sample.sample_id)[:12]}"
        staged_image = (
            output / "images" / sample.split / f"{staged_stem}{sample.image_path.suffix.lower()}"
        )
        staged_label = output / "labels" / sample.split / f"{staged_stem}.txt"
        staged_image.parent.mkdir(parents=True, exist_ok=True)
        staged_label.parent.mkdir(parents=True, exist_ok=True)
        image_materialization = _copy_independent_regular_file(
            sample.image_path, staged_image
        )
        try:
            with staged_label.open("xb") as target, sample.label_path.open("rb") as source:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
        except FileExistsError as exc:
            raise DetectionInfrastructureError(
                f"Frozen label view collision: {staged_label}", code="contract_collision"
            ) from exc
        staged_label.chmod(0o440)
        if staged_image.is_symlink() or sha256_file(staged_image) != sample.image_sha256:
            _fail(
                f"Frozen image copy changed unexpectedly: {staged_image}",
                "image_hash_mismatch",
            )
        if sha256_file(staged_label) != sample.label_sha256:
            _fail(
                f"Frozen label copy changed unexpectedly: {staged_label}",
                "label_hash_mismatch",
            )
        grouped[sample.split].append((staged_image, sample))
        staged_samples.append(
            {
                "sample_id": sample.sample_id,
                "source_image": str(sample.image_path),
                "source_annotation": str(sample.source_annotation_path),
                "source_label": str(sample.label_path),
                "staged_image": str(staged_image),
                "staged_label": str(staged_label),
                "image_sha256": sample.image_sha256,
                "source_annotation_sha256": sample.source_annotation_sha256,
                "label_sha256": sample.label_sha256,
                "image_materialization": image_materialization,
            }
        )
    split_files: dict[str, str] = {}
    for split, samples in sorted(grouped.items()):
        split_path = output / f"{split}.txt"
        content = "\n".join(str(staged_image) for staged_image, _sample in samples) + "\n"
        _write_exclusive(split_path, content)
        split_files[split] = str(split_path)
    data_payload: dict[str, Any] = {
        "path": str(output),
        "train": split_files["train"],
        "val": split_files["val"],
        "names": DETECTION_CLASSES,
        "floodsight_manifest": str(contract.manifest_path),
        "floodsight_manifest_sha256": contract.manifest_sha256,
        "floodsight_dataset_fingerprint": contract.dataset_fingerprint,
    }
    if "test-dev" in split_files:
        data_payload["test"] = split_files["test-dev"]
    elif "test" in split_files:
        data_payload["test"] = split_files["test"]
    data_yaml = output / "dataset.yaml"
    # JSON is valid YAML 1.2 and avoids adding YAML to the import-safe package.
    _write_exclusive(data_yaml, json.dumps(data_payload, indent=2, sort_keys=True) + "\n")
    samples_payload = [
        {
            **asdict(sample),
            "image_path": str(sample.image_path),
            "source_annotation_path": str(sample.source_annotation_path),
            "label_path": str(sample.label_path),
            "class_counts": {str(key): value for key, value in sample.class_counts.items()},
        }
        for sample in contract.samples
    ]
    snapshot = contract.summary() | {
        "split_files": split_files,
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": sha256_file(data_yaml),
        "samples": samples_payload,
        "staged_samples": staged_samples,
        "samples_digest": stable_sha256(samples_payload),
        "staged_samples_digest": stable_sha256(staged_samples),
        "image_materialization": "independent_reflink_or_byte_copy",
        "label_materialization": "read_only_copy",
    }
    snapshot["contract_sha256"] = stable_sha256(snapshot)
    _write_exclusive(
        output / "contract.json", json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    )
    return data_yaml
