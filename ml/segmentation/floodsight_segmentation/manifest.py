"""Strict readers for immutable Phase-3 segmentation manifests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .config import MAPPING_ASSETS, TARGET_DATASET_SUPPORT
from .errors import ManifestError

SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
TASK_TYPE = "SEMANTIC_SEGMENTATION"
SCHEMA_VERSION = "dataset-manifest-v1"
FINGERPRINT_ALGORITHM = "sha256-canonical-manifest-identity-v1"
MANIFEST_IDENTITY_FIELDS = (
    "schema_version",
    "manifest_id",
    "dataset_id",
    "task_type",
    "source_version",
    "preparation_version",
    "taxonomy_version",
    "integrity_mode",
    "tool_version",
    "git_commit",
    "samples",
)
MANIFEST_ROOT_FIELDS = frozenset((*MANIFEST_IDENTITY_FIELDS, "created_at", "fingerprint"))
MANIFEST_SAMPLE_FIELDS = frozenset(
    {
        "sample_id",
        "source_dataset",
        "source_split",
        "target_split",
        "image_path",
        "source_annotation_path",
        "target_annotation_path",
        "width",
        "height",
        "image_hash",
        "annotation_hash",
        "target_image_hash",
        "target_annotation_hash",
        "class_counts",
        "ignored_count",
        "invalid_count",
        "preparation_version",
        "taxonomy_version",
        "objects",
        "source_schema",
        "target_mapping_version",
        "target_mapping_sha256",
        "valid_supervision_classes",
        "ignore_index",
        "ignore_semantics",
        "exclusion_status",
        "exclusion_reason",
    }
)
PREPARATION_VERSION = "segmentation_v2_20260831T131322Z_v1"
TOOL_VERSION = "stage09-10-segmentation-finalize-20260831-v1"
SOURCE_VERSIONS = {
    "floodnet": "FloodNet-Supervised_v1.0",
    "rescuenet": "official_2023_figshare_v1",
}
SOURCE_SCHEMAS = {
    "floodnet": "floodnet-supervised-v1.0-indexed-mask-ids-0-9",
    "rescuenet": "rescuenet-official-2023-indexed-mask-ids-0-10",
}
MAPPING_VERSIONS = {
    "floodnet": "floodnet-mapping-v2",
    "rescuenet": "rescuenet-mapping-v2",
}
IGNORE_SEMANTICS = (
    "255_reserved_for_genuine_invalid_or_unlabelled_pixels;"
    "audited_remaps_emit_no_ignore_pixels;"
    "unsupported_classes_are_removed_from_dataset_aware_softmax_not_relabelled_as_background"
)
TARGET_SPLITS = frozenset({"train", "val", "holdout"})
SOURCE_SPLIT_FOR_TARGET = {"train": "train", "val": "val", "holdout": "test"}
EXPECTED_GIT_COMMIT = "f272a31ae05e9cb8e532e939e3ca02365755e6a9"
ManifestSpec = tuple[Path, str, str]
CANONICAL_MANIFEST_ROOT = Path(
    "/data/floodsight-workspace/floodsight-datasets/reports/pretraining_gate/"
    "segmentation_stages09_10_20260831T131322Z_v1/manifests"
)
CANONICAL_MANIFEST_LOCKS: Mapping[tuple[str, str], ManifestSpec] = {
    ("floodnet", "train"): (
        CANONICAL_MANIFEST_ROOT / "floodnet_train_loader_manifest.json",
        "edcf9f1549f1a3490b0f8c8ac4d66b0296e0d90c4425a1095b86237e6fc66dd2",
        "e954b083227577618dc5c18a36b9b8b5c6eaed2033aff2709d593a416cdf38a5",
    ),
    ("floodnet", "val"): (
        CANONICAL_MANIFEST_ROOT / "floodnet_val_loader_manifest.json",
        "368a63b5354c2f573f0e12c18b322b9a1ff08514954e09da406d36ce31e2fb61",
        "4bdd6282b2027a03139f320c1f960c4502c4cdacad18556e7e2e96a2618975d3",
    ),
    ("floodnet", "holdout"): (
        CANONICAL_MANIFEST_ROOT / "floodnet_holdout_loader_manifest.json",
        "847599365060a991422832225c4393eb574c1ec6a8f8a34b3a646ac602c72f6c",
        "ca31fdd56241f586e2ba5da2421561e51de13f48d98cf8b021d408f078c906e9",
    ),
    ("rescuenet", "train"): (
        CANONICAL_MANIFEST_ROOT / "rescuenet_train_loader_manifest.json",
        "d8a5e9a9d7cf2fd32ba36bcc5bec9e8048b9e6dfd90ea443816ae94a837151cc",
        "e72773a169b1edf1a7172046b8a80e28bd91c4284d627b5d4f4f637410f68466",
    ),
    ("rescuenet", "val"): (
        CANONICAL_MANIFEST_ROOT / "rescuenet_val_loader_manifest.json",
        "0b701b7f40c10e77579c6e61dd5f85ac2d36d33d1d720f9b845c49f252f31540",
        "2e2ce653901965df67859548d9919463e7379ba8731ac5ece67c1f50b03a0638",
    ),
    ("rescuenet", "holdout"): (
        CANONICAL_MANIFEST_ROOT / "rescuenet_holdout_loader_manifest.json",
        "e8b761fba47b50fbb67d1f7e9cfdae9f7703993b73163dcc036a45416a580ed0",
        "b9f72f80c1135ea2fbfb2b52c8c505f3ea1cccc6785c5288f6cc94d2489411b4",
    ),
}


@dataclass(frozen=True, slots=True)
class ManifestSample:
    """One immutable image/mask pair from a Phase-3 manifest."""

    sample_id: str
    source_dataset: str
    source_split: str
    target_split: str
    image_path: str
    source_annotation_path: str
    target_annotation_path: str
    width: int
    height: int
    image_hash: str
    annotation_hash: str
    target_image_hash: str
    target_annotation_hash: str
    class_counts: Mapping[int, int]
    ignored_count: int
    invalid_count: int
    preparation_version: str
    taxonomy_version: str
    source_schema: str
    target_mapping_version: str
    target_mapping_sha256: str
    valid_supervision_classes: tuple[int, ...]
    ignore_index: int
    ignore_semantics: str
    exclusion_status: str
    exclusion_reason: str


@dataclass(frozen=True, slots=True)
class FrozenManifest:
    """Validated manifest metadata and selected samples."""

    path: Path
    sha256: str
    manifest_id: str
    dataset_id: str
    taxonomy_version: str
    integrity_mode: str
    fingerprint: str
    samples: tuple[ManifestSample, ...]


@dataclass(frozen=True, slots=True)
class ManifestCollection:
    """A deterministic, duplicate-free collection of frozen manifests."""

    manifests: tuple[FrozenManifest, ...]
    samples: tuple[ManifestSample, ...]

    @property
    def hashes(self) -> dict[str, str]:
        return {str(item.path): item.sha256 for item in self.manifests}

    @property
    def fingerprints(self) -> dict[str, str]:
        return {str(item.path): item.fingerprint for item in self.manifests}

    @property
    def set_fingerprint(self) -> str:
        """Bind exact paths, file hashes, and recomputed dataset fingerprints."""

        records = [
            {
                "path": str(item.path),
                "manifest_sha256": item.sha256,
                "dataset_fingerprint": item.fingerprint,
            }
            for item in sorted(self.manifests, key=lambda manifest: str(manifest.path))
        ]
        canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of a regular file without loading it into memory."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise ManifestError(f"Unable to hash file: {path}") from exc
    return digest.hexdigest()


def canonical_manifest_fingerprint(payload: Mapping[str, Any]) -> str:
    """Recompute the v1 manifest fingerprint from persisted immutable identity.

    Formula ``sha256(canonical_json(identity))`` where canonical JSON uses
    UTF-8, sorted object keys, no insignificant whitespace, and samples sorted
    by ``sample_id``. ``created_at`` and the stored ``fingerprint`` are excluded.
    No converter state, source inventory, or network resource is consulted.
    """

    missing = set(MANIFEST_IDENTITY_FIELDS) - set(payload)
    if missing:
        raise ManifestError(
            f"Manifest fingerprint identity fields are missing: {sorted(missing)}."
        )
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ManifestError("Expected an array at samples for fingerprint recomputation.")
    samples: list[Mapping[str, Any]] = []
    for index, raw_sample in enumerate(raw_samples):
        sample = _object(raw_sample, location=f"samples[{index}]")
        _string(sample.get("sample_id"), location=f"samples[{index}].sample_id")
        samples.append(sample)
    ordered_samples = sorted(samples, key=lambda sample: str(sample["sample_id"]))
    identity = {
        field: ordered_samples if field == "samples" else payload[field]
        for field in MANIFEST_IDENTITY_FIELDS
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _object(value: Any, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"Expected an object at {location}.")
    return value


def _string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"Expected a non-empty string at {location}.")
    return value


def _positive_int(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ManifestError(f"Expected a positive integer at {location}.")
    return value


def _nonnegative_int(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"Expected a non-negative integer at {location}.")
    return value


def validate_relative_path(value: Any, *, location: str) -> str:
    """Reject absolute, drive-qualified, backslash, and traversal paths."""

    path = _string(value, location=location)
    if "\\" in path:
        raise ManifestError(f"Backslashes are not allowed at {location}: {path!r}")
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ManifestError(f"Absolute paths are not allowed at {location}: {path!r}")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise ManifestError(f"Unsafe relative path at {location}: {path!r}")
    return path


def resolve_under_root(root: Path, relative_path: str, *, must_exist: bool = True) -> Path:
    """Resolve a manifest path and prove it remains under the external data root."""

    try:
        resolved_root = root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"Dataset root does not exist: {root}") from exc
    try:
        candidate = (resolved_root / relative_path).resolve(strict=must_exist)
        candidate.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ManifestError(
            f"Manifest path is missing or escapes the dataset root: {relative_path!r}"
        ) from exc
    return candidate


def _hash(value: Any, *, location: str) -> str:
    digest = _string(value, location=location).lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise ManifestError(f"Expected a lowercase SHA-256 at {location}.")
    return digest


def _parse_sample(raw: Any, *, index: int, dataset_id: str) -> ManifestSample:
    sample = _object(raw, location=f"samples[{index}]")
    if set(sample) != MANIFEST_SAMPLE_FIELDS:
        raise ManifestError(
            f"Invalid sample keys at samples[{index}]: "
            f"missing={sorted(MANIFEST_SAMPLE_FIELDS - set(sample))}, "
            f"extra={sorted(set(sample) - MANIFEST_SAMPLE_FIELDS)}."
        )
    source_dataset = _string(
        sample.get("source_dataset"), location=f"samples[{index}].source_dataset"
    )
    if source_dataset != dataset_id:
        raise ManifestError(
            f"samples[{index}] declares source_dataset={source_dataset!r}, "
            f"but the manifest declares {dataset_id!r}."
        )
    raw_counts = _object(sample.get("class_counts"), location=f"samples[{index}].class_counts")
    counts: dict[int, int] = {}
    for key, value in raw_counts.items():
        if not isinstance(key, str) or not key.isdigit():
            raise ManifestError(f"Invalid class ID {key!r} in samples[{index}].class_counts.")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ManifestError(f"Invalid count for class {key} in samples[{index}].")
        counts[int(key)] = value
    expected_support = tuple(sorted(TARGET_DATASET_SUPPORT[dataset_id]))
    if not set(counts) <= set(expected_support):
        raise ManifestError(f"Unsupported class_counts IDs in samples[{index}].")
    target_split = _string(
        sample.get("target_split"), location=f"samples[{index}].target_split"
    )
    if target_split not in TARGET_SPLITS:
        raise ManifestError(f"Invalid target_split in samples[{index}].")
    source_split = _string(
        sample.get("source_split"), location=f"samples[{index}].source_split"
    )
    if source_split != SOURCE_SPLIT_FOR_TARGET[target_split]:
        raise ManifestError(f"Source/target split identity drift in samples[{index}].")
    width = _positive_int(sample.get("width"), location=f"samples[{index}].width")
    height = _positive_int(sample.get("height"), location=f"samples[{index}].height")
    image_hash = _hash(sample.get("image_hash"), location=f"samples[{index}].image_hash")
    target_image_hash = _hash(
        sample.get("target_image_hash"), location=f"samples[{index}].target_image_hash"
    )
    if target_image_hash != image_hash:
        raise ManifestError(f"target_image_hash must equal image_hash in samples[{index}].")
    preparation_version = _string(
        sample.get("preparation_version"), location=f"samples[{index}].preparation_version"
    )
    taxonomy_version = _string(
        sample.get("taxonomy_version"), location=f"samples[{index}].taxonomy_version"
    )
    if preparation_version != PREPARATION_VERSION or taxonomy_version != "segmentation-taxonomy-v2":
        raise ManifestError(f"Preparation/taxonomy drift in samples[{index}].")
    ignored_count = _nonnegative_int(
        sample.get("ignored_count"), location=f"samples[{index}].ignored_count"
    )
    invalid_count = _nonnegative_int(
        sample.get("invalid_count"), location=f"samples[{index}].invalid_count"
    )
    if ignored_count != 0 or invalid_count != 0:
        raise ManifestError(
            f"Final included sample has ignored/invalid pixels at samples[{index}]."
        )
    if sum(counts.values()) + ignored_count != width * height:
        raise ManifestError(f"Pixel class counts do not match dimensions at samples[{index}].")
    objects = sample.get("objects")
    if not isinstance(objects, list) or objects:
        raise ManifestError(f"objects must be an empty array at samples[{index}].")
    raw_valid_classes = sample.get("valid_supervision_classes")
    if not isinstance(raw_valid_classes, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw_valid_classes
    ):
        raise ManifestError(f"Invalid valid_supervision_classes at samples[{index}].")
    valid_classes = tuple(raw_valid_classes)
    if valid_classes != expected_support:
        raise ManifestError(
            f"valid_supervision_classes drifted for {dataset_id!r} at samples[{index}]."
        )
    source_schema = _string(
        sample.get("source_schema"), location=f"samples[{index}].source_schema"
    )
    mapping_version = _string(
        sample.get("target_mapping_version"),
        location=f"samples[{index}].target_mapping_version",
    )
    mapping_sha256 = _hash(
        sample.get("target_mapping_sha256"),
        location=f"samples[{index}].target_mapping_sha256",
    )
    if (
        source_schema != SOURCE_SCHEMAS[dataset_id]
        or mapping_version != MAPPING_VERSIONS[dataset_id]
        or mapping_sha256 != MAPPING_ASSETS[dataset_id][1]
    ):
        raise ManifestError(f"Source/mapping identity drift in samples[{index}].")
    ignore_index = _nonnegative_int(
        sample.get("ignore_index"), location=f"samples[{index}].ignore_index"
    )
    ignore_semantics = _string(
        sample.get("ignore_semantics"), location=f"samples[{index}].ignore_semantics"
    )
    if ignore_index != 255 or ignore_semantics != IGNORE_SEMANTICS:
        raise ManifestError(f"Ignore semantics drift in samples[{index}].")
    if sample.get("exclusion_status") != "INCLUDED" or sample.get("exclusion_reason") != "":
        raise ManifestError(f"Final manifests accept only included samples at samples[{index}].")
    sample_id = _string(sample.get("sample_id"), location=f"samples[{index}].sample_id")
    if not sample_id.startswith(f"{dataset_id}-{source_split}-"):
        raise ManifestError(f"Sample/source identity drift in samples[{index}].")
    image_path = validate_relative_path(
        sample.get("image_path"), location=f"samples[{index}].image_path"
    )
    source_annotation_path = validate_relative_path(
        sample.get("source_annotation_path"),
        location=f"samples[{index}].source_annotation_path",
    )
    target_annotation_path = validate_relative_path(
        sample.get("target_annotation_path"),
        location=f"samples[{index}].target_annotation_path",
    )
    if dataset_id == "floodnet":
        source_root = f"raw/floodnet/FloodNet-Supervised_v1.0/{source_split}"
        expected_image_prefix = f"{source_root}/{source_split}-org-img/"
        expected_source_mask_prefix = f"{source_root}/{source_split}-label-img/"
    else:
        source_root = "raw/rescuenet/official_2023_figshare_v1"
        expected_image_prefix = f"{source_root}/{source_split}-org-img/"
        expected_source_mask_prefix = f"{source_root}/{source_split}-label-img/"
    expected_target_mask_prefix = (
        f"processed/{PREPARATION_VERSION}/masks/{dataset_id}/{target_split}/"
    )
    if (
        not image_path.startswith(expected_image_prefix)
        or Path(image_path).suffix.lower() not in {".jpg", ".jpeg", ".png"}
        or not source_annotation_path.startswith(expected_source_mask_prefix)
        or Path(source_annotation_path).suffix.lower() != ".png"
        or not target_annotation_path.startswith(expected_target_mask_prefix)
        or Path(target_annotation_path).suffix.lower() != ".png"
        or Path(source_annotation_path).stem != f"{Path(image_path).stem}_lab"
        or Path(target_annotation_path).stem != sample_id
    ):
        raise ManifestError(f"Official source/processed path identity drift in samples[{index}].")
    return ManifestSample(
        sample_id=sample_id,
        source_dataset=source_dataset,
        source_split=source_split,
        target_split=target_split,
        image_path=image_path,
        source_annotation_path=source_annotation_path,
        target_annotation_path=target_annotation_path,
        width=width,
        height=height,
        image_hash=image_hash,
        annotation_hash=_hash(
            sample.get("annotation_hash"), location=f"samples[{index}].annotation_hash"
        ),
        target_image_hash=target_image_hash,
        target_annotation_hash=_hash(
            sample.get("target_annotation_hash"),
            location=f"samples[{index}].target_annotation_hash",
        ),
        class_counts=counts,
        ignored_count=ignored_count,
        invalid_count=invalid_count,
        preparation_version=preparation_version,
        taxonomy_version=taxonomy_version,
        source_schema=source_schema,
        target_mapping_version=mapping_version,
        target_mapping_sha256=mapping_sha256,
        valid_supervision_classes=valid_classes,
        ignore_index=ignore_index,
        ignore_semantics=ignore_semantics,
        exclusion_status="INCLUDED",
        exclusion_reason="",
    )


def load_frozen_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_fingerprint: str,
    expected_taxonomy: str,
    allowed_datasets: Iterable[str],
    selected_splits: Iterable[str] | None = None,
    require_full_integrity: bool = True,
) -> FrozenManifest:
    """Load a manifest only when its caller-supplied content hash matches."""

    raw_path = path.expanduser()
    if raw_path.is_symlink():
        raise ManifestError(f"Manifest must not be a symbolic link: {raw_path}")
    path = raw_path.resolve()
    expected = _hash(expected_sha256, location="expected_sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ManifestError(
            f"Manifest SHA-256 mismatch for {path}: expected {expected}, found {actual}."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Unable to parse manifest JSON: {path}") from exc
    payload = _object(raw, location="<root>")
    if set(payload) != MANIFEST_ROOT_FIELDS:
        raise ManifestError(
            "Manifest root fields do not match the frozen schema: "
            f"missing={sorted(MANIFEST_ROOT_FIELDS - set(payload))}, "
            f"extra={sorted(set(payload) - MANIFEST_ROOT_FIELDS)}."
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"Unsupported manifest schema in {path}.")
    if payload.get("task_type") != TASK_TYPE:
        raise ManifestError(f"Manifest is not a semantic-segmentation manifest: {path}")
    dataset_id = _string(payload.get("dataset_id"), location="dataset_id")
    allowed = frozenset(allowed_datasets)
    if dataset_id not in allowed:
        raise ManifestError(f"Unsupported segmentation dataset {dataset_id!r} in {path}.")
    source_version = _string(payload.get("source_version"), location="source_version")
    preparation_version = _string(
        payload.get("preparation_version"), location="preparation_version"
    )
    tool_version = _string(payload.get("tool_version"), location="tool_version")
    if (
        source_version != SOURCE_VERSIONS[dataset_id]
        or preparation_version != PREPARATION_VERSION
        or tool_version != TOOL_VERSION
    ):
        raise ManifestError(f"Final source/preparation/tool identity drifted in {path}.")
    created_at = _string(payload.get("created_at"), location="created_at")
    try:
        created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError(f"Invalid created_at timestamp in {path}.") from exc
    if created_time.tzinfo is None:
        raise ManifestError(f"created_at must include a timezone in {path}.")
    git_commit = _string(payload.get("git_commit"), location="git_commit")
    if git_commit != EXPECTED_GIT_COMMIT:
        raise ManifestError(f"Manifest Git baseline drifted in {path}.")
    taxonomy = _string(payload.get("taxonomy_version"), location="taxonomy_version")
    if taxonomy != expected_taxonomy:
        raise ManifestError(
            f"Taxonomy mismatch in {path}: expected {expected_taxonomy!r}, found {taxonomy!r}."
        )
    integrity_mode = _string(payload.get("integrity_mode"), location="integrity_mode")
    if integrity_mode != "full" or not require_full_integrity:
        raise ManifestError(f"Training requires a full-integrity manifest: {path}")
    fingerprint = _hash(payload.get("fingerprint"), location="fingerprint")
    expected_dataset_fingerprint = _hash(
        expected_fingerprint, location="expected_fingerprint"
    )
    recomputed_fingerprint = canonical_manifest_fingerprint(payload)
    if fingerprint != recomputed_fingerprint:
        raise ManifestError(
            f"Manifest fingerprint recomputation mismatch for {path}: "
            f"stored {fingerprint}, recomputed {recomputed_fingerprint}."
        )
    if recomputed_fingerprint != expected_dataset_fingerprint:
        raise ManifestError(
            f"Manifest fingerprint mismatch for {path}: expected "
            f"{expected_dataset_fingerprint}, found {recomputed_fingerprint}."
        )
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ManifestError("Expected an array at samples.")
    splits = frozenset(selected_splits) if selected_splits is not None else None
    parsed = tuple(
        _parse_sample(item, index=index, dataset_id=dataset_id)
        for index, item in enumerate(raw_samples)
    )
    if not parsed:
        raise ManifestError(f"Final manifest contains no samples: {path}")
    target_splits = {item.target_split for item in parsed}
    if len(target_splits) != 1:
        raise ManifestError(f"Final manifest must contain exactly one target split: {path}")
    target_split = next(iter(target_splits))
    manifest_id = _string(payload.get("manifest_id"), location="manifest_id")
    expected_manifest_id = f"{dataset_id}-{PREPARATION_VERSION}-{target_split}"
    if manifest_id != expected_manifest_id:
        raise ManifestError(
            f"Final manifest_id mismatch in {path}: expected {expected_manifest_id!r}."
        )
    samples = tuple(item for item in parsed if splits is None or item.target_split in splits)
    if not samples:
        requested = sorted(splits) if splits is not None else "all"
        raise ManifestError(f"Manifest {path} has no samples for requested splits {requested}.")
    ids = [item.sample_id for item in samples]
    if len(set(ids)) != len(ids):
        raise ManifestError(f"Manifest contains duplicate sample IDs: {path}")
    return FrozenManifest(
        path=path,
        sha256=actual,
        manifest_id=manifest_id,
        dataset_id=dataset_id,
        taxonomy_version=taxonomy,
        integrity_mode=integrity_mode,
        fingerprint=fingerprint,
        samples=samples,
    )


def load_manifest_collection(
    specs: Sequence[ManifestSpec],
    *,
    expected_taxonomy: str,
    allowed_datasets: Iterable[str],
    selected_splits: Iterable[str],
    require_full_integrity: bool = True,
) -> ManifestCollection:
    """Load, order, and deduplicate a set of content-addressed manifests."""

    if not specs:
        raise ManifestError("At least one frozen manifest is required.")
    manifests = tuple(
        load_frozen_manifest(
            path,
            expected_sha256=digest,
            expected_fingerprint=fingerprint,
            expected_taxonomy=expected_taxonomy,
            allowed_datasets=allowed_datasets,
            selected_splits=selected_splits,
            require_full_integrity=require_full_integrity,
        )
        for path, digest, fingerprint in specs
    )
    all_samples = sorted(
        (sample for manifest in manifests for sample in manifest.samples),
        key=lambda sample: (sample.source_dataset, sample.sample_id),
    )
    ids = [sample.sample_id for sample in all_samples]
    if len(ids) != len(set(ids)):
        raise ManifestError("Duplicate sample IDs were found across frozen manifests.")
    for label, identities in (
        ("image paths", [sample.image_path for sample in all_samples]),
        (
            "source annotation paths",
            [sample.source_annotation_path for sample in all_samples],
        ),
        (
            "target annotation paths",
            [sample.target_annotation_path for sample in all_samples],
        ),
        ("image SHA-256 values", [sample.target_image_hash for sample in all_samples]),
    ):
        if len(identities) != len(set(identities)):
            raise ManifestError(f"Duplicate {label} were found across frozen manifests.")
    return ManifestCollection(manifests=manifests, samples=tuple(all_samples))


def require_canonical_manifest_locks(collection: ManifestCollection) -> None:
    """Bind production operations to the exact frozen Stage-9 manifest files."""

    for manifest in collection.manifests:
        target_splits = {sample.target_split for sample in manifest.samples}
        if len(target_splits) != 1:
            raise ManifestError("Canonical manifest lock requires one target split per file.")
        key = (manifest.dataset_id, next(iter(target_splits)))
        try:
            expected_path, expected_sha256, expected_fingerprint = CANONICAL_MANIFEST_LOCKS[key]
        except KeyError as exc:
            raise ManifestError(f"No canonical Stage-9 manifest lock exists for {key!r}.") from exc
        if (
            manifest.path != expected_path.resolve()
            or manifest.sha256 != expected_sha256
            or manifest.fingerprint != expected_fingerprint
        ):
            raise ManifestError(
                "Manifest does not match the canonical Stage-9 "
                f"path/hash/fingerprint lock: {key!r}."
            )
