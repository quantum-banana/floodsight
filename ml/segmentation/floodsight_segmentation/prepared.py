"""Launch-verified prepared-target cache for the SegFormer data fast path.

The cache record is deliberately strict and content addressed.  Building a
cache copies only the already-audited Stage-9 target PNGs; raw images and raw
annotations are never written.  Loading a record revalidates its manifest,
ledger, Stage-9 report, and artifact-index bindings.  A launch snapshot then
hashes every current source image and cached target exactly once.

After a snapshot has been produced, :func:`read_verified_pair` performs only
an ``open``/``fstat`` identity guard and decode.  It does not hash, remap,
count pixels, inspect source annotations, or consume random-number state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any

from PIL import Image, UnidentifiedImageError

from .errors import ManifestError
from .manifest import (
    PREPARATION_VERSION,
    SHA256_PATTERN,
    ManifestCollection,
    ManifestSample,
    validate_relative_path,
)

PREPARED_CACHE_SCHEMA = "floodsight-segmentation-prepared-cache-v1"
PREPARED_CACHE_FINGERPRINT_ALGORITHM = (
    "sha256-canonical-prepared-cache-record-v1"
)
PREPARED_SNAPSHOT_FINGERPRINT_ALGORITHM = (
    "sha256-canonical-prepared-cache-content-snapshot-v1"
)
PREPARED_CACHE_RECORD_NAME = "prepared-cache-record.json"

_CACHE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "cache_id",
        "created_at",
        "fingerprint_algorithm",
        "record_fingerprint",
        "data_root",
        "cache_root",
        "preparation_version",
        "taxonomy_version",
        "manifest_set_fingerprint",
        "manifests",
        "prepared_cache_implementation_sha256",
        "user_instruction_sha256",
        "processed_mask_ledger",
        "stage09_report",
        "artifact_index",
        "samples",
    }
)
_PROVENANCE_FIELDS = frozenset({"path", "sha256"})
_MANIFEST_FIELDS = frozenset({"path", "sha256", "fingerprint"})
_SAMPLE_FIELDS = frozenset(
    {
        "sample_id",
        "source_dataset",
        "source_split",
        "target_split",
        "image_path",
        "image_sha256",
        "image_bytes",
        "source_annotation_path",
        "source_annotation_sha256",
        "stage09_target_path",
        "stage09_target_sha256",
        "cache_target_path",
        "cache_target_sha256",
        "cache_target_bytes",
        "width",
        "height",
        "target_mode",
        "target_dtype",
        "target_ids",
        "class_counts",
        "ignored_count",
        "invalid_count",
        "preparation_version",
        "taxonomy_version",
        "target_mapping_version",
        "target_mapping_sha256",
        "valid_supervision_classes",
        "ignore_index",
    }
)
_LEDGER_FIELDS = frozenset(
    {
        "bytes",
        "dtype",
        "height",
        "ignored_count",
        "invalid_count",
        "mode",
        "path",
        "sample_id",
        "sha256",
        "source_dataset",
        "target_ids",
        "target_split",
        "width",
    }
)


@dataclass(frozen=True, slots=True)
class ProvenanceFile:
    """One content-addressed audit artifact used by a prepared cache."""

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class PreparedRecordSample:
    """One strictly validated sample declaration before launch verification."""

    sample_id: str
    source_dataset: str
    source_split: str
    target_split: str
    image_path: Path
    image_relative_path: str
    image_sha256: str
    image_bytes: int
    source_annotation_path: Path
    source_annotation_relative_path: str
    source_annotation_sha256: str
    stage09_target_relative_path: str
    stage09_target_sha256: str
    cache_target_path: Path
    cache_target_relative_path: str
    cache_target_sha256: str
    cache_target_bytes: int
    width: int
    height: int
    target_mode: str
    target_dtype: str
    target_ids: tuple[int, ...]
    class_counts: tuple[tuple[int, int], ...]
    ignored_count: int
    invalid_count: int
    preparation_version: str
    taxonomy_version: str
    target_mapping_version: str
    target_mapping_sha256: str
    valid_supervision_classes: tuple[int, ...]
    ignore_index: int


@dataclass(frozen=True, slots=True)
class PreparedCacheRecord:
    """Validated on-disk cache record and its immutable audit bindings."""

    path: Path
    sha256: str
    record_fingerprint: str
    cache_id: str
    created_at: str
    data_root: Path
    cache_root: Path
    preparation_version: str
    taxonomy_version: str
    # Full cache record identity.  A smoke collection may be an exact subset.
    manifest_set_fingerprint: str
    selected_manifest_set_fingerprint: str
    prepared_cache_implementation_sha256: str
    user_instruction_sha256: str
    processed_mask_ledger: ProvenanceFile
    stage09_report: ProvenanceFile
    artifact_index: ProvenanceFile
    samples: tuple[PreparedRecordSample, ...]


@dataclass(frozen=True, slots=True)
class PreparedFileIdentity:
    """Content and filesystem identity captured from one stable open file."""

    path: Path
    sha256: str
    bytes: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int

    def matches_stat(self, value: os.stat_result) -> bool:
        """Return whether an open descriptor is still the launch-verified file."""

        return (
            stat.S_ISREG(value.st_mode)
            and value.st_dev == self.device
            and value.st_ino == self.inode
            and value.st_size == self.bytes
            and value.st_mtime_ns == self.mtime_ns
            and value.st_ctime_ns == self.ctime_ns
        )


@dataclass(frozen=True, slots=True)
class PreparedSample:
    """One sample authorized for the no-rehash epoch hot path."""

    sample_id: str
    source_dataset: str
    source_split: str
    target_split: str
    width: int
    height: int
    image: PreparedFileIdentity
    cache_target: PreparedFileIdentity
    source_annotation_sha256: str
    stage09_target_sha256: str
    target_mode: str
    target_dtype: str
    target_ids: tuple[int, ...]
    class_counts: tuple[tuple[int, int], ...]
    ignored_count: int
    invalid_count: int
    preparation_version: str
    taxonomy_version: str
    target_mapping_version: str
    target_mapping_sha256: str
    valid_supervision_classes: tuple[int, ...]
    ignore_index: int


@dataclass(frozen=True, slots=True)
class PreparedSnapshot:
    """Launch gate result required to activate the prepared-target fast path."""

    record_path: Path
    record_sha256: str
    record_fingerprint: str
    snapshot_fingerprint: str
    cache_id: str
    record_manifest_set_fingerprint: str
    # Exact collection verified for this launch (full production or a subset).
    manifest_set_fingerprint: str
    prepared_cache_implementation_sha256: str
    user_instruction_sha256: str
    samples: tuple[PreparedSample, ...]
    _sample_index: Mapping[str, PreparedSample] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        index = {sample.sample_id: sample for sample in self.samples}
        if len(index) != len(self.samples):
            raise ManifestError("Prepared snapshot contains duplicate sample IDs.")
        object.__setattr__(self, "_sample_index", MappingProxyType(index))

    def sample_for(self, sample_id: str) -> PreparedSample:
        """Return an exact snapshot sample or fail closed."""

        try:
            return self._sample_index[sample_id]
        except KeyError as exc:
            raise ManifestError(
                f"Sample {sample_id!r} is absent from the launch-verified prepared snapshot."
            ) from exc


@dataclass(frozen=True, slots=True)
class PreparedCacheBuild:
    """Identity of a newly completed, never-overwritten prepared cache."""

    cache_root: Path
    record_path: Path
    record_sha256: str
    record_fingerprint: str
    sample_count: int


@dataclass(frozen=True, slots=True)
class _LedgerRow:
    bytes: int
    dtype: str
    height: int
    ignored_count: int
    invalid_count: int
    mode: str
    path: str
    sample_id: str
    sha256: str
    source_dataset: str
    target_ids: tuple[int, ...]
    target_split: str
    width: int


def _mapping(value: Any, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"Expected an object at {location}.")
    return value


def _exact_fields(
    payload: Mapping[str, Any], expected: frozenset[str], *, location: str
) -> None:
    if set(payload) != expected:
        raise ManifestError(
            f"Invalid prepared-cache fields at {location}: "
            f"missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}."
        )


def _string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"Expected a non-empty string at {location}.")
    return value


def _sha256(value: Any, *, location: str) -> str:
    digest = _string(value, location=location)
    if not SHA256_PATTERN.fullmatch(digest):
        raise ManifestError(f"Expected a lowercase SHA-256 at {location}.")
    return digest


def _positive_int(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ManifestError(f"Expected a positive integer at {location}.")
    return value


def _nonnegative_int(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"Expected a non-negative integer at {location}.")
    return value


def _integer_list(value: Any, *, location: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise ManifestError(f"Expected a non-negative integer array at {location}.")
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        raise ManifestError(f"Expected sorted unique IDs at {location}.")
    return result


def _class_counts(value: Any, *, location: str) -> tuple[tuple[int, int], ...]:
    raw = _mapping(value, location=location)
    parsed: list[tuple[int, int]] = []
    for key, count in raw.items():
        if not isinstance(key, str) or not key.isdigit():
            raise ManifestError(f"Invalid class-count ID {key!r} at {location}.")
        parsed.append((int(key), _positive_int(count, location=f"{location}.{key}")))
    ordered = tuple(sorted(parsed))
    if len({key for key, _ in ordered}) != len(ordered):
        raise ManifestError(f"Duplicate class-count IDs at {location}.")
    return ordered


def _timestamp(value: Any, *, location: str) -> str:
    text = _string(value, location=location)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError(f"Expected an ISO-8601 timestamp at {location}.") from exc
    if parsed.tzinfo is None:
        raise ManifestError(f"Timestamp must include a timezone at {location}.")
    return text


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_prepared_record_fingerprint(payload: Mapping[str, Any]) -> str:
    """Fingerprint every record field except the fingerprint itself."""

    identity = dict(payload)
    identity.pop("record_fingerprint", None)
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _no_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ManifestError(f"Duplicate JSON object key {key!r}.")
        payload[key] = value
    return payload


def _parse_json_bytes(payload: bytes, *, location: str) -> Mapping[str, Any]:
    try:
        decoded = payload.decode("utf-8")
        parsed = json.loads(decoded, object_pairs_hook=_no_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Unable to parse JSON at {location}.") from exc
    return _mapping(parsed, location=location)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolved_root(path: Path, *, label: str) -> Path:
    raw = path.expanduser()
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"{label} does not exist: {raw}") from exc
    if not resolved.is_dir():
        raise ManifestError(f"{label} must be a directory: {resolved}")
    return resolved


def _resolve_under_root(root: Path, relative: str, *, label: str) -> Path:
    validated = validate_relative_path(relative, location=label)
    raw = root / validated
    if raw.is_symlink():
        raise ManifestError(f"{label} must not be a symbolic link: {validated!r}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"{label} is missing: {validated!r}") from exc
    if not _is_relative_to(resolved, root):
        raise ManifestError(f"{label} escapes its declared root: {validated!r}")
    if not resolved.is_file():
        raise ManifestError(f"{label} must resolve to a regular file: {validated!r}")
    return resolved


def _absolute_regular_path(path_value: Any, *, root: Path, location: str) -> Path:
    text = _string(path_value, location=location)
    raw = Path(text).expanduser()
    if not raw.is_absolute():
        raise ManifestError(f"Expected an absolute path at {location}.")
    if raw.is_symlink():
        raise ManifestError(f"Symbolic links are forbidden at {location}: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"Missing provenance file at {location}: {raw}") from exc
    if not _is_relative_to(resolved, root) or not resolved.is_file():
        raise ManifestError(f"Unsafe provenance path at {location}: {resolved}")
    return resolved


def _open_readonly(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManifestError(f"Unable to safely open {label}: {path}") from exc
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode):
            raise ManifestError(f"{label} is not a regular file: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, identity


def _stat_tuple(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_open_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
    label: str,
    chunk_size: int = 1024 * 1024,
) -> PreparedFileIdentity:
    descriptor, before = _open_readonly(path, label=label)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ManifestError(f"Unable to hash {label}: {path}") from exc
    if _stat_tuple(before) != _stat_tuple(after):
        raise ManifestError(f"{label} changed while it was being launch-verified: {path}")
    if expected_bytes is not None and before.st_size != expected_bytes:
        raise ManifestError(
            f"{label} size mismatch for {path}: expected {expected_bytes}, "
            f"found {before.st_size}."
        )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ManifestError(
            f"{label} SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"found {actual_sha256}."
        )
    return PreparedFileIdentity(
        path=path,
        sha256=actual_sha256,
        bytes=before.st_size,
        device=before.st_dev,
        inode=before.st_ino,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
    )


def _hash_regular_file(path: Path, *, expected_sha256: str, label: str) -> None:
    _hash_open_file(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=None,
        label=label,
    )


def _provenance_file(
    raw: Any, *, data_root: Path, location: str
) -> ProvenanceFile:
    payload = _mapping(raw, location=location)
    _exact_fields(payload, _PROVENANCE_FIELDS, location=location)
    path = _absolute_regular_path(
        payload.get("path"), root=data_root, location=f"{location}.path"
    )
    digest = _sha256(payload.get("sha256"), location=f"{location}.sha256")
    _hash_regular_file(path, expected_sha256=digest, label=location)
    return ProvenanceFile(path=path, sha256=digest)


def _parse_ledger(ref: ProvenanceFile) -> Mapping[str, _LedgerRow]:
    rows: dict[str, _LedgerRow] = {}
    try:
        lines = ref.path.read_bytes().splitlines()
    except OSError as exc:
        raise ManifestError(f"Unable to read processed-mask ledger: {ref.path}") from exc
    if not lines:
        raise ManifestError("Processed-mask ledger is empty.")
    for index, line in enumerate(lines, start=1):
        if not line:
            raise ManifestError(f"Blank line in processed-mask ledger at line {index}.")
        payload = _parse_json_bytes(line, location=f"processed_mask_ledger[{index}]")
        _exact_fields(payload, _LEDGER_FIELDS, location=f"processed_mask_ledger[{index}]")
        sample_id = _string(payload.get("sample_id"), location=f"ledger[{index}].sample_id")
        if sample_id in rows:
            raise ManifestError(f"Duplicate sample {sample_id!r} in processed-mask ledger.")
        row = _LedgerRow(
            bytes=_positive_int(payload.get("bytes"), location=f"ledger[{index}].bytes"),
            dtype=_string(payload.get("dtype"), location=f"ledger[{index}].dtype"),
            height=_positive_int(
                payload.get("height"), location=f"ledger[{index}].height"
            ),
            ignored_count=_nonnegative_int(
                payload.get("ignored_count"), location=f"ledger[{index}].ignored_count"
            ),
            invalid_count=_nonnegative_int(
                payload.get("invalid_count"), location=f"ledger[{index}].invalid_count"
            ),
            mode=_string(payload.get("mode"), location=f"ledger[{index}].mode"),
            path=validate_relative_path(
                payload.get("path"), location=f"ledger[{index}].path"
            ),
            sample_id=sample_id,
            sha256=_sha256(
                payload.get("sha256"), location=f"ledger[{index}].sha256"
            ),
            source_dataset=_string(
                payload.get("source_dataset"),
                location=f"ledger[{index}].source_dataset",
            ),
            target_ids=_integer_list(
                payload.get("target_ids"), location=f"ledger[{index}].target_ids"
            ),
            target_split=_string(
                payload.get("target_split"), location=f"ledger[{index}].target_split"
            ),
            width=_positive_int(
                payload.get("width"), location=f"ledger[{index}].width"
            ),
        )
        if row.dtype != "uint8" or row.mode != "L":
            raise ManifestError(
                f"Prepared target {sample_id!r} must be an L-mode uint8 mask."
            )
        rows[sample_id] = row
    return MappingProxyType(rows)


def _artifact_index_rows(
    ref: ProvenanceFile, *, data_root: Path
) -> Mapping[str, tuple[int, str]]:
    try:
        raw = ref.path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"Unable to read artifact index: {ref.path}") from exc
    payload = _parse_json_bytes(raw, location="artifact_index")
    artifacts = payload.get("artifacts")
    count = payload.get("artifact_count")
    if not isinstance(artifacts, list) or count != len(artifacts):
        raise ManifestError("Artifact index count/list contract is invalid.")
    selected: dict[str, tuple[int, str]] = {}
    for index, raw_item in enumerate(artifacts):
        item = _mapping(raw_item, location=f"artifact_index.artifacts[{index}]")
        if item.get("category") != "processed_mask":
            continue
        raw_path = Path(_string(item.get("path"), location=f"artifacts[{index}].path"))
        if not raw_path.is_absolute():
            raise ManifestError(f"Processed-mask artifact path is not absolute: {raw_path}")
        try:
            relative = raw_path.relative_to(data_root).as_posix()
        except ValueError as exc:
            raise ManifestError(
                f"Processed-mask artifact escapes the dataset root: {raw_path}"
            ) from exc
        if relative in selected:
            raise ManifestError(f"Duplicate processed-mask artifact path: {relative}")
        selected[relative] = (
            _positive_int(item.get("bytes"), location=f"artifacts[{index}].bytes"),
            _sha256(item.get("sha256"), location=f"artifacts[{index}].sha256"),
        )
    if not selected:
        raise ManifestError("Artifact index contains no processed masks.")
    return MappingProxyType(selected)


def _validate_stage09_report(
    ref: ProvenanceFile,
    *,
    data_root: Path,
    collection: ManifestCollection,
    ledger_count: int,
) -> None:
    try:
        payload = _parse_json_bytes(ref.path.read_bytes(), location="stage09_report")
    except OSError as exc:
        raise ManifestError(f"Unable to read Stage-9 report: {ref.path}") from exc
    if payload.get("preparation_version") != PREPARATION_VERSION:
        raise ManifestError("Stage-9 report preparation version drifted.")
    if payload.get("processed_mask_count") != ledger_count:
        raise ManifestError("Stage-9 report/processed-mask-ledger count mismatch.")
    _sha256(
        payload.get("processed_mask_tree_fingerprint"),
        location="stage09_report.processed_mask_tree_fingerprint",
    )
    expected_processed_root = (
        data_root / "processed" / PREPARATION_VERSION
    ).resolve(strict=True)
    report_processed_root = _absolute_regular_or_directory(
        payload.get("processed_root"), root=data_root, location="stage09_report.processed_root"
    )
    if report_processed_root != expected_processed_root or not report_processed_root.is_dir():
        raise ManifestError("Stage-9 report processed-root identity drifted.")
    checks = _mapping(payload.get("checks"), location="stage09_report.checks")
    required_checks = (
        "every_generated_mask_hash_id_and_dimension_validated",
        "every_manifest_exact_external_schema_validated",
        "every_manifest_fingerprint_recomputed_with_locked_helper",
        "every_manifest_loaded_with_full_integrity_contract",
        "every_source_hash_validated",
        "every_source_mask_id_and_dimension_validated",
    )
    if any(checks.get(key) is not True for key in required_checks):
        raise ManifestError("Stage-9 report does not preserve all required PASS checks.")
    raw_manifests = payload.get("manifests")
    if not isinstance(raw_manifests, list):
        raise ManifestError("Stage-9 report manifests must be an array.")
    report_manifests: dict[str, Mapping[str, Any]] = {}
    for index, raw_manifest in enumerate(raw_manifests):
        item = _mapping(raw_manifest, location=f"stage09_report.manifests[{index}]")
        path = _string(item.get("path"), location=f"stage09_report.manifests[{index}].path")
        if path in report_manifests:
            raise ManifestError(f"Duplicate manifest path in Stage-9 report: {path}")
        report_manifests[path] = item
    for manifest in collection.manifests:
        try:
            item = report_manifests[str(manifest.path)]
        except KeyError as exc:
            raise ManifestError(
                f"Selected manifest is absent from the Stage-9 report: {manifest.path}"
            ) from exc
        if (
            item.get("sha256") != manifest.sha256
            or item.get("fingerprint") != manifest.fingerprint
            or item.get("dataset_id") != manifest.dataset_id
            or item.get("sample_count") != len(manifest.samples)
        ):
            raise ManifestError(
                f"Selected manifest identity drifted in Stage-9 report: {manifest.path}"
            )


def _absolute_regular_or_directory(path_value: Any, *, root: Path, location: str) -> Path:
    text = _string(path_value, location=location)
    raw = Path(text).expanduser()
    if not raw.is_absolute() or raw.is_symlink():
        raise ManifestError(f"Unsafe absolute path at {location}: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"Missing path at {location}: {raw}") from exc
    if not _is_relative_to(resolved, root):
        raise ManifestError(f"Path escapes dataset root at {location}: {resolved}")
    return resolved


def _manifest_records(collection: ManifestCollection) -> list[dict[str, str]]:
    return [
        {
            "path": str(manifest.path),
            "sha256": manifest.sha256,
            "fingerprint": manifest.fingerprint,
        }
        for manifest in sorted(collection.manifests, key=lambda item: str(item.path))
    ]


def _validate_manifest_bindings(
    raw: Any, *, collection: ManifestCollection, data_root: Path
) -> str:
    """Validate the full record set and require the selected set as an exact subset."""

    if not isinstance(raw, list):
        raise ManifestError("Prepared-cache manifests must be an array.")
    parsed: list[dict[str, str]] = []
    for index, value in enumerate(raw):
        item = _mapping(value, location=f"manifests[{index}]")
        _exact_fields(item, _MANIFEST_FIELDS, location=f"manifests[{index}]")
        path = _absolute_regular_path(
            item.get("path"), root=data_root, location=f"manifests[{index}].path"
        )
        digest = _sha256(item.get("sha256"), location=f"manifests[{index}].sha256")
        fingerprint = _sha256(
            item.get("fingerprint"), location=f"manifests[{index}].fingerprint"
        )
        _hash_regular_file(path, expected_sha256=digest, label=f"manifest[{index}]")
        parsed.append(
            {"path": str(path), "sha256": digest, "fingerprint": fingerprint}
        )
    if not parsed:
        raise ManifestError("Prepared-cache manifest bindings are empty.")
    if parsed != sorted(parsed, key=lambda item: item["path"]):
        raise ManifestError("Prepared-cache manifest bindings are not canonically ordered.")
    parsed_by_path = {item["path"]: item for item in parsed}
    if len(parsed_by_path) != len(parsed):
        raise ManifestError("Prepared-cache manifest bindings contain duplicate paths.")
    for selected in _manifest_records(collection):
        if parsed_by_path.get(selected["path"]) != selected:
            raise ManifestError(
                "Selected manifest is missing or mismatched in prepared-cache bindings: "
                f"{selected['path']}"
            )
    fingerprint_records = [
        {
            "path": item["path"],
            "manifest_sha256": item["sha256"],
            "dataset_fingerprint": item["fingerprint"],
        }
        for item in parsed
    ]
    canonical = json.dumps(
        fingerprint_records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require_collection(collection: ManifestCollection) -> Mapping[str, ManifestSample]:
    if not collection.samples or not collection.manifests:
        raise ManifestError("Prepared cache requires a non-empty manifest collection.")
    index = {sample.sample_id: sample for sample in collection.samples}
    if len(index) != len(collection.samples):
        raise ManifestError("Manifest collection contains duplicate sample IDs.")
    return MappingProxyType(index)


def _validate_ledger_row(sample: ManifestSample, row: _LedgerRow) -> None:
    expected_target_ids = tuple(sorted(sample.class_counts))
    expected = (
        (row.path, sample.target_annotation_path, "path"),
        (row.sha256, sample.target_annotation_hash, "SHA-256"),
        (row.source_dataset, sample.source_dataset, "source dataset"),
        (row.target_split, sample.target_split, "target split"),
        (row.width, sample.width, "width"),
        (row.height, sample.height, "height"),
        (row.target_ids, expected_target_ids, "target IDs"),
        (row.ignored_count, sample.ignored_count, "ignored count"),
        (row.invalid_count, sample.invalid_count, "invalid count"),
    )
    for actual, wanted, label in expected:
        if actual != wanted:
            raise ManifestError(
                f"Processed-mask ledger {label} mismatch for {sample.sample_id!r}: "
                f"expected {wanted!r}, found {actual!r}."
            )


def _expected_cache_target(sample: ManifestSample) -> str:
    return f"masks/{sample.source_dataset}/{sample.target_split}/{sample.sample_id}.png"


def _sample_payload(
    sample: ManifestSample,
    *,
    image_bytes: int,
    cache_target_bytes: int,
    cache_target_path: str,
    ledger: _LedgerRow,
) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "source_dataset": sample.source_dataset,
        "source_split": sample.source_split,
        "target_split": sample.target_split,
        "image_path": sample.image_path,
        "image_sha256": sample.target_image_hash,
        "image_bytes": image_bytes,
        "source_annotation_path": sample.source_annotation_path,
        "source_annotation_sha256": sample.annotation_hash,
        "stage09_target_path": sample.target_annotation_path,
        "stage09_target_sha256": sample.target_annotation_hash,
        "cache_target_path": cache_target_path,
        "cache_target_sha256": sample.target_annotation_hash,
        "cache_target_bytes": cache_target_bytes,
        "width": sample.width,
        "height": sample.height,
        "target_mode": ledger.mode,
        "target_dtype": ledger.dtype,
        "target_ids": list(ledger.target_ids),
        "class_counts": {str(key): value for key, value in sorted(sample.class_counts.items())},
        "ignored_count": sample.ignored_count,
        "invalid_count": sample.invalid_count,
        "preparation_version": sample.preparation_version,
        "taxonomy_version": sample.taxonomy_version,
        "target_mapping_version": sample.target_mapping_version,
        "target_mapping_sha256": sample.target_mapping_sha256,
        "valid_supervision_classes": list(sample.valid_supervision_classes),
        "ignore_index": sample.ignore_index,
    }


def _copy_exclusive_and_hash(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> tuple[int, str]:
    source_fd, source_before = _open_readonly(source, label="Stage-9 target mask")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        destination_fd = os.open(destination, flags, 0o444)
    except OSError as exc:
        os.close(source_fd)
        raise ManifestError(
            f"Refusing to overwrite or create prepared target: {destination}"
        ) from exc
    digest = hashlib.sha256()
    copied = 0
    try:
        with (
            os.fdopen(source_fd, "rb", closefd=True) as source_stream,
            os.fdopen(destination_fd, "wb", closefd=True) as destination_stream,
        ):
            while chunk := source_stream.read(1024 * 1024):
                destination_stream.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
            source_after = os.fstat(source_stream.fileno())
    except OSError as exc:
        raise ManifestError(
            f"Unable to copy Stage-9 target into prepared cache: {source}"
        ) from exc
    if _stat_tuple(source_before) != _stat_tuple(source_after):
        raise ManifestError(f"Stage-9 target changed while being copied: {source}")
    actual_sha256 = digest.hexdigest()
    if copied != expected_bytes or actual_sha256 != expected_sha256:
        raise ManifestError(
            f"Stage-9 target identity mismatch while copying {source}: "
            f"bytes={copied}, sha256={actual_sha256}."
        )
    return copied, actual_sha256


def _atomic_write_new(path: Path, payload: bytes) -> str:
    """Publish bytes atomically without ever replacing an existing record."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except OSError as exc:
            raise ManifestError(f"Refusing to overwrite prepared-cache record: {path}") from exc
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise ManifestError(f"Unable to sync prepared-cache directory: {path.parent}") from exc
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def build_prepared_cache_record(
    *,
    collection: ManifestCollection,
    data_root: Path,
    cache_root: Path,
    cache_id: str,
    created_at: str,
    processed_mask_ledger_path: Path,
    processed_mask_ledger_sha256: str,
    stage09_report_path: Path,
    stage09_report_sha256: str,
    artifact_index_path: Path,
    artifact_index_sha256: str,
    prepared_cache_implementation_sha256: str,
    user_instruction_sha256: str,
) -> PreparedCacheBuild:
    """Copy frozen target PNGs into a new version and write its record last.

    ``cache_root`` must not already exist.  Every target is created with
    ``O_EXCL`` and the record is published with an atomic hard link, so this
    function never overwrites an existing cache file.  An interrupted build
    intentionally leaves no final record; such a directory is not loadable.
    """

    samples_by_id = _require_collection(collection)
    resolved_data_root = _resolved_root(data_root, label="Dataset root")
    cache_id = _string(cache_id, location="cache_id")
    if not _CACHE_ID_PATTERN.fullmatch(cache_id):
        raise ManifestError("cache_id must be a lowercase, path-safe version identifier.")
    created_at = _timestamp(created_at, location="created_at")
    implementation_sha = _sha256(
        prepared_cache_implementation_sha256,
        location="prepared_cache_implementation_sha256",
    )
    instruction_sha = _sha256(user_instruction_sha256, location="user_instruction_sha256")
    taxonomy_versions = {sample.taxonomy_version for sample in collection.samples}
    preparation_versions = {sample.preparation_version for sample in collection.samples}
    if len(taxonomy_versions) != 1 or preparation_versions != {PREPARATION_VERSION}:
        raise ManifestError("Manifest collection preparation/taxonomy identity is not uniform.")
    taxonomy_version = next(iter(taxonomy_versions))

    parent = cache_root.expanduser().parent.resolve(strict=True)
    final_cache_root = parent / cache_root.name
    if cache_root.name != cache_id:
        raise ManifestError("cache_id must equal the final cache-root directory name.")
    if final_cache_root.exists() or final_cache_root.is_symlink():
        raise ManifestError(
            "Prepared cache root already exists; refusing overwrite: "
            f"{final_cache_root}"
        )
    raw_root = (resolved_data_root / "raw").resolve(strict=True)
    if _is_relative_to(final_cache_root, raw_root):
        raise ManifestError("Prepared cache must never be created under the immutable raw tree.")

    provenance_specs = (
        (
            processed_mask_ledger_path,
            processed_mask_ledger_sha256,
            "processed_mask_ledger",
        ),
        (stage09_report_path, stage09_report_sha256, "stage09_report"),
        (artifact_index_path, artifact_index_sha256, "artifact_index"),
    )
    provenance: dict[str, ProvenanceFile] = {}
    for raw_path, raw_sha, label in provenance_specs:
        path = _absolute_regular_path(
            str(raw_path), root=resolved_data_root, location=label
        )
        digest = _sha256(raw_sha, location=f"{label}.sha256")
        _hash_regular_file(path, expected_sha256=digest, label=label)
        provenance[label] = ProvenanceFile(path=path, sha256=digest)
    ledger = _parse_ledger(provenance["processed_mask_ledger"])
    _validate_stage09_report(
        provenance["stage09_report"],
        data_root=resolved_data_root,
        collection=collection,
        ledger_count=len(ledger),
    )
    artifact_rows = _artifact_index_rows(
        provenance["artifact_index"], data_root=resolved_data_root
    )
    for manifest in collection.manifests:
        _hash_regular_file(
            manifest.path,
            expected_sha256=manifest.sha256,
            label=f"manifest {manifest.path}",
        )
    for sample in samples_by_id.values():
        try:
            row = ledger[sample.sample_id]
        except KeyError as exc:
            raise ManifestError(
                f"Sample {sample.sample_id!r} is absent from processed-mask ledger."
            ) from exc
        _validate_ledger_row(sample, row)
        try:
            artifact_bytes, artifact_sha = artifact_rows[sample.target_annotation_path]
        except KeyError as exc:
            raise ManifestError(
                f"Stage-9 target is absent from artifact index: {sample.target_annotation_path}"
            ) from exc
        if (artifact_bytes, artifact_sha) != (row.bytes, row.sha256):
            raise ManifestError(
                f"Ledger/artifact-index mismatch for {sample.sample_id!r}."
            )
        _resolve_under_root(
            resolved_data_root, sample.image_path, label=f"image {sample.sample_id!r}"
        )
        _resolve_under_root(
            resolved_data_root,
            sample.source_annotation_path,
            label=f"source annotation {sample.sample_id!r}",
        )
        _resolve_under_root(
            resolved_data_root,
            sample.target_annotation_path,
            label=f"Stage-9 target {sample.sample_id!r}",
        )

    try:
        final_cache_root.mkdir(mode=0o755)
    except OSError as exc:
        raise ManifestError(
            f"Unable to exclusively create prepared cache root: {final_cache_root}"
        ) from exc
    masks_root = final_cache_root / "masks"
    masks_root.mkdir(mode=0o755)
    for dataset in sorted({sample.source_dataset for sample in collection.samples}):
        dataset_root = masks_root / dataset
        dataset_root.mkdir(mode=0o755)
        for split in sorted(
            {
                sample.target_split
                for sample in collection.samples
                if sample.source_dataset == dataset
            }
        ):
            (dataset_root / split).mkdir(mode=0o755)

    sample_payloads: list[dict[str, Any]] = []
    for sample in sorted(
        collection.samples, key=lambda item: (item.source_dataset, item.sample_id)
    ):
        row = ledger[sample.sample_id]
        image_path = _resolve_under_root(
            resolved_data_root, sample.image_path, label=f"image {sample.sample_id!r}"
        )
        image_fd, image_stat = _open_readonly(image_path, label="source image")
        os.close(image_fd)
        relative_cache_target = _expected_cache_target(sample)
        cache_target = final_cache_root / relative_cache_target
        source_target = _resolve_under_root(
            resolved_data_root,
            sample.target_annotation_path,
            label=f"Stage-9 target {sample.sample_id!r}",
        )
        copied, copied_sha = _copy_exclusive_and_hash(
            source_target,
            cache_target,
            expected_sha256=row.sha256,
            expected_bytes=row.bytes,
        )
        if copied_sha != sample.target_annotation_hash:
            raise ManifestError(
                f"Copied target hash does not match manifest for {sample.sample_id!r}."
            )
        sample_payloads.append(
            _sample_payload(
                sample,
                image_bytes=image_stat.st_size,
                cache_target_bytes=copied,
                cache_target_path=relative_cache_target,
                ledger=row,
            )
        )

    payload: dict[str, Any] = {
        "schema_version": PREPARED_CACHE_SCHEMA,
        "cache_id": cache_id,
        "created_at": created_at,
        "fingerprint_algorithm": PREPARED_CACHE_FINGERPRINT_ALGORITHM,
        "data_root": str(resolved_data_root),
        "cache_root": str(final_cache_root),
        "preparation_version": PREPARATION_VERSION,
        "taxonomy_version": taxonomy_version,
        "manifest_set_fingerprint": collection.set_fingerprint,
        "manifests": _manifest_records(collection),
        "prepared_cache_implementation_sha256": implementation_sha,
        "user_instruction_sha256": instruction_sha,
        "processed_mask_ledger": {
            "path": str(provenance["processed_mask_ledger"].path),
            "sha256": provenance["processed_mask_ledger"].sha256,
        },
        "stage09_report": {
            "path": str(provenance["stage09_report"].path),
            "sha256": provenance["stage09_report"].sha256,
        },
        "artifact_index": {
            "path": str(provenance["artifact_index"].path),
            "sha256": provenance["artifact_index"].sha256,
        },
        "samples": sample_payloads,
    }
    payload["record_fingerprint"] = canonical_prepared_record_fingerprint(payload)
    record_path = final_cache_root / PREPARED_CACHE_RECORD_NAME
    record_sha256 = _atomic_write_new(record_path, _canonical_json_bytes(payload))
    return PreparedCacheBuild(
        cache_root=final_cache_root,
        record_path=record_path,
        record_sha256=record_sha256,
        record_fingerprint=str(payload["record_fingerprint"]),
        sample_count=len(sample_payloads),
    )


def _parse_record_sample(
    raw: Any,
    *,
    index: int,
    manifest_sample: ManifestSample,
    data_root: Path,
    cache_root: Path,
    ledger: _LedgerRow,
    artifact_rows: Mapping[str, tuple[int, str]],
) -> PreparedRecordSample:
    location = f"samples[{index}]"
    payload = _mapping(raw, location=location)
    _exact_fields(payload, _SAMPLE_FIELDS, location=location)
    sample_id = _string(payload.get("sample_id"), location=f"{location}.sample_id")
    if sample_id != manifest_sample.sample_id:
        raise ManifestError(f"Prepared-cache sample order/identity drift at {location}.")
    image_relative = validate_relative_path(
        payload.get("image_path"), location=f"{location}.image_path"
    )
    source_annotation_relative = validate_relative_path(
        payload.get("source_annotation_path"),
        location=f"{location}.source_annotation_path",
    )
    stage09_target_relative = validate_relative_path(
        payload.get("stage09_target_path"),
        location=f"{location}.stage09_target_path",
    )
    cache_target_relative = validate_relative_path(
        payload.get("cache_target_path"), location=f"{location}.cache_target_path"
    )
    image_sha = _sha256(payload.get("image_sha256"), location=f"{location}.image_sha256")
    source_annotation_sha = _sha256(
        payload.get("source_annotation_sha256"),
        location=f"{location}.source_annotation_sha256",
    )
    stage09_target_sha = _sha256(
        payload.get("stage09_target_sha256"),
        location=f"{location}.stage09_target_sha256",
    )
    cache_target_sha = _sha256(
        payload.get("cache_target_sha256"),
        location=f"{location}.cache_target_sha256",
    )
    image_bytes = _positive_int(
        payload.get("image_bytes"), location=f"{location}.image_bytes"
    )
    cache_target_bytes = _positive_int(
        payload.get("cache_target_bytes"), location=f"{location}.cache_target_bytes"
    )
    width = _positive_int(payload.get("width"), location=f"{location}.width")
    height = _positive_int(payload.get("height"), location=f"{location}.height")
    target_mode = _string(payload.get("target_mode"), location=f"{location}.target_mode")
    target_dtype = _string(
        payload.get("target_dtype"), location=f"{location}.target_dtype"
    )
    target_ids = _integer_list(
        payload.get("target_ids"), location=f"{location}.target_ids"
    )
    counts = _class_counts(payload.get("class_counts"), location=f"{location}.class_counts")
    valid_classes = _integer_list(
        payload.get("valid_supervision_classes"),
        location=f"{location}.valid_supervision_classes",
    )
    ignored_count = _nonnegative_int(
        payload.get("ignored_count"), location=f"{location}.ignored_count"
    )
    invalid_count = _nonnegative_int(
        payload.get("invalid_count"), location=f"{location}.invalid_count"
    )
    preparation_version = _string(
        payload.get("preparation_version"), location=f"{location}.preparation_version"
    )
    taxonomy_version = _string(
        payload.get("taxonomy_version"), location=f"{location}.taxonomy_version"
    )
    target_mapping_version = _string(
        payload.get("target_mapping_version"),
        location=f"{location}.target_mapping_version",
    )
    target_mapping_sha256 = _sha256(
        payload.get("target_mapping_sha256"),
        location=f"{location}.target_mapping_sha256",
    )
    ignore_index = _nonnegative_int(
        payload.get("ignore_index"), location=f"{location}.ignore_index"
    )
    values = (
        (payload.get("source_dataset"), manifest_sample.source_dataset, "source_dataset"),
        (payload.get("source_split"), manifest_sample.source_split, "source_split"),
        (payload.get("target_split"), manifest_sample.target_split, "target_split"),
        (image_relative, manifest_sample.image_path, "image_path"),
        (image_sha, manifest_sample.target_image_hash, "image_sha256"),
        (
            source_annotation_relative,
            manifest_sample.source_annotation_path,
            "source_annotation_path",
        ),
        (
            source_annotation_sha,
            manifest_sample.annotation_hash,
            "source_annotation_sha256",
        ),
        (
            stage09_target_relative,
            manifest_sample.target_annotation_path,
            "stage09_target_path",
        ),
        (
            stage09_target_sha,
            manifest_sample.target_annotation_hash,
            "stage09_target_sha256",
        ),
        (cache_target_relative, _expected_cache_target(manifest_sample), "cache_target_path"),
        (cache_target_sha, manifest_sample.target_annotation_hash, "cache_target_sha256"),
        (cache_target_bytes, ledger.bytes, "cache_target_bytes"),
        (width, manifest_sample.width, "width"),
        (height, manifest_sample.height, "height"),
        (target_mode, ledger.mode, "target_mode"),
        (target_dtype, ledger.dtype, "target_dtype"),
        (target_ids, ledger.target_ids, "target_ids"),
        (counts, tuple(sorted(manifest_sample.class_counts.items())), "class_counts"),
        (
            ignored_count,
            manifest_sample.ignored_count,
            "ignored_count",
        ),
        (
            invalid_count,
            manifest_sample.invalid_count,
            "invalid_count",
        ),
        (
            preparation_version,
            manifest_sample.preparation_version,
            "preparation_version",
        ),
        (
            taxonomy_version,
            manifest_sample.taxonomy_version,
            "taxonomy_version",
        ),
        (
            target_mapping_version,
            manifest_sample.target_mapping_version,
            "target_mapping_version",
        ),
        (
            target_mapping_sha256,
            manifest_sample.target_mapping_sha256,
            "target_mapping_sha256",
        ),
        (
            valid_classes,
            manifest_sample.valid_supervision_classes,
            "valid_supervision_classes",
        ),
        (ignore_index, manifest_sample.ignore_index, "ignore_index"),
    )
    for actual, expected, label in values:
        if actual != expected:
            raise ManifestError(
                f"Prepared-cache {label} mismatch for {sample_id!r}: "
                f"expected {expected!r}, found {actual!r}."
            )
    if image_bytes < 1:
        raise ManifestError(f"Prepared-cache image size is invalid for {sample_id!r}.")
    _validate_ledger_row(manifest_sample, ledger)
    try:
        artifact_identity = artifact_rows[stage09_target_relative]
    except KeyError as exc:
        raise ManifestError(
            f"Stage-9 target is absent from artifact index: {stage09_target_relative}"
        ) from exc
    if artifact_identity != (cache_target_bytes, cache_target_sha):
        raise ManifestError(f"Artifact-index mismatch for prepared sample {sample_id!r}.")
    return PreparedRecordSample(
        sample_id=sample_id,
        source_dataset=manifest_sample.source_dataset,
        source_split=manifest_sample.source_split,
        target_split=manifest_sample.target_split,
        image_path=_resolve_under_root(
            data_root, image_relative, label=f"image {sample_id!r}"
        ),
        image_relative_path=image_relative,
        image_sha256=image_sha,
        image_bytes=image_bytes,
        source_annotation_path=_resolve_under_root(
            data_root,
            source_annotation_relative,
            label=f"source annotation {sample_id!r}",
        ),
        source_annotation_relative_path=source_annotation_relative,
        source_annotation_sha256=source_annotation_sha,
        stage09_target_relative_path=stage09_target_relative,
        stage09_target_sha256=stage09_target_sha,
        cache_target_path=_resolve_under_root(
            cache_root, cache_target_relative, label=f"cache target {sample_id!r}"
        ),
        cache_target_relative_path=cache_target_relative,
        cache_target_sha256=cache_target_sha,
        cache_target_bytes=cache_target_bytes,
        width=width,
        height=height,
        target_mode=target_mode,
        target_dtype=target_dtype,
        target_ids=target_ids,
        class_counts=counts,
        ignored_count=manifest_sample.ignored_count,
        invalid_count=manifest_sample.invalid_count,
        preparation_version=manifest_sample.preparation_version,
        taxonomy_version=manifest_sample.taxonomy_version,
        target_mapping_version=manifest_sample.target_mapping_version,
        target_mapping_sha256=manifest_sample.target_mapping_sha256,
        valid_supervision_classes=valid_classes,
        ignore_index=manifest_sample.ignore_index,
    )


def load_prepared_cache_record(
    path: Path,
    *,
    expected_sha256: str,
    collection: ManifestCollection,
    data_root: Path,
) -> PreparedCacheRecord:
    """Load and fully cross-check one caller-hash-locked cache record."""

    _require_collection(collection)
    expected_record_sha = _sha256(expected_sha256, location="expected_sha256")
    raw_path = path.expanduser()
    if raw_path.is_symlink():
        raise ManifestError(f"Prepared-cache record must not be a symbolic link: {raw_path}")
    try:
        resolved_path = raw_path.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"Prepared-cache record is missing: {raw_path}") from exc
    record_identity = _hash_open_file(
        resolved_path,
        expected_sha256=expected_record_sha,
        expected_bytes=None,
        label="prepared-cache record",
    )
    try:
        raw_bytes = resolved_path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"Unable to read prepared-cache record: {resolved_path}") from exc
    if len(raw_bytes) != record_identity.bytes:
        raise ManifestError("Prepared-cache record changed after its hash validation.")
    if hashlib.sha256(raw_bytes).hexdigest() != expected_record_sha:
        raise ManifestError("Prepared-cache record changed after its hash validation.")
    payload = _parse_json_bytes(raw_bytes, location="prepared_cache_record")
    _exact_fields(payload, _RECORD_FIELDS, location="prepared_cache_record")
    if raw_bytes != _canonical_json_bytes(payload):
        raise ManifestError("Prepared-cache record is not canonical JSON.")
    if payload.get("schema_version") != PREPARED_CACHE_SCHEMA:
        raise ManifestError("Unsupported prepared-cache record schema.")
    if payload.get("fingerprint_algorithm") != PREPARED_CACHE_FINGERPRINT_ALGORITHM:
        raise ManifestError("Prepared-cache record fingerprint algorithm drifted.")
    cache_id = _string(payload.get("cache_id"), location="cache_id")
    if not _CACHE_ID_PATTERN.fullmatch(cache_id):
        raise ManifestError("Prepared-cache cache_id is not path safe.")
    created_at = _timestamp(payload.get("created_at"), location="created_at")
    record_fingerprint = _sha256(
        payload.get("record_fingerprint"), location="record_fingerprint"
    )
    recomputed = canonical_prepared_record_fingerprint(payload)
    if record_fingerprint != recomputed:
        raise ManifestError(
            "Prepared-cache record fingerprint mismatch: "
            f"stored {record_fingerprint}, recomputed {recomputed}."
        )
    resolved_data_root = _resolved_root(data_root, label="Dataset root")
    declared_data_root = _absolute_regular_or_directory(
        payload.get("data_root"), root=resolved_data_root, location="data_root"
    )
    if declared_data_root != resolved_data_root:
        raise ManifestError("Prepared-cache record binds a different dataset root.")
    cache_root_text = _string(payload.get("cache_root"), location="cache_root")
    declared_cache_root_raw = Path(cache_root_text).expanduser()
    if not declared_cache_root_raw.is_absolute() or declared_cache_root_raw.is_symlink():
        raise ManifestError("Prepared-cache root must be an absolute non-symlink directory.")
    try:
        declared_cache_root = declared_cache_root_raw.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"Prepared-cache root is missing: {declared_cache_root_raw}") from exc
    if not declared_cache_root.is_dir() or declared_cache_root.name != cache_id:
        raise ManifestError("Prepared-cache root/cache_id identity drifted.")
    if resolved_path != declared_cache_root / PREPARED_CACHE_RECORD_NAME:
        raise ManifestError("Prepared-cache record is not at its canonical cache-root path.")
    raw_root = (resolved_data_root / "raw").resolve(strict=True)
    if _is_relative_to(declared_cache_root, raw_root):
        raise ManifestError("Prepared cache is illegally located under the raw dataset tree.")
    preparation_version = _string(
        payload.get("preparation_version"), location="preparation_version"
    )
    taxonomy_version = _string(
        payload.get("taxonomy_version"), location="taxonomy_version"
    )
    if preparation_version != PREPARATION_VERSION:
        raise ManifestError("Prepared-cache preparation version drifted.")
    if {sample.taxonomy_version for sample in collection.samples} != {taxonomy_version}:
        raise ManifestError("Prepared-cache taxonomy version differs from the manifests.")
    manifest_set_fingerprint = _sha256(
        payload.get("manifest_set_fingerprint"), location="manifest_set_fingerprint"
    )
    recomputed_record_manifest_set = _validate_manifest_bindings(
        payload.get("manifests"), collection=collection, data_root=resolved_data_root
    )
    if manifest_set_fingerprint != recomputed_record_manifest_set:
        raise ManifestError("Prepared-cache full manifest-set fingerprint mismatch.")
    implementation_sha = _sha256(
        payload.get("prepared_cache_implementation_sha256"),
        location="prepared_cache_implementation_sha256",
    )
    instruction_sha = _sha256(
        payload.get("user_instruction_sha256"), location="user_instruction_sha256"
    )
    ledger_ref = _provenance_file(
        payload.get("processed_mask_ledger"),
        data_root=resolved_data_root,
        location="processed_mask_ledger",
    )
    report_ref = _provenance_file(
        payload.get("stage09_report"),
        data_root=resolved_data_root,
        location="stage09_report",
    )
    artifact_ref = _provenance_file(
        payload.get("artifact_index"),
        data_root=resolved_data_root,
        location="artifact_index",
    )
    ledger = _parse_ledger(ledger_ref)
    _validate_stage09_report(
        report_ref,
        data_root=resolved_data_root,
        collection=collection,
        ledger_count=len(ledger),
    )
    artifact_rows = _artifact_index_rows(artifact_ref, data_root=resolved_data_root)
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ManifestError("Prepared-cache samples must be an array.")
    ordered_manifest_samples = sorted(
        collection.samples, key=lambda item: (item.source_dataset, item.sample_id)
    )
    indexed_raw_samples: dict[str, tuple[int, Any]] = {}
    ordering: list[tuple[str, str]] = []
    for index, raw_sample in enumerate(raw_samples):
        raw_sample_payload = _mapping(raw_sample, location=f"samples[{index}]")
        _exact_fields(raw_sample_payload, _SAMPLE_FIELDS, location=f"samples[{index}]")
        raw_sample_id = _string(
            raw_sample_payload.get("sample_id"), location=f"samples[{index}].sample_id"
        )
        raw_dataset = _string(
            raw_sample_payload.get("source_dataset"),
            location=f"samples[{index}].source_dataset",
        )
        if raw_sample_id in indexed_raw_samples:
            raise ManifestError(
                f"Duplicate sample {raw_sample_id!r} in prepared-cache record."
            )
        indexed_raw_samples[raw_sample_id] = (index, raw_sample)
        ordering.append((raw_dataset, raw_sample_id))
    if ordering != sorted(ordering):
        raise ManifestError("Prepared-cache samples are not canonically ordered.")
    parsed_samples: list[PreparedRecordSample] = []
    for manifest_sample in ordered_manifest_samples:
        try:
            ledger_row = ledger[manifest_sample.sample_id]
        except KeyError as exc:
            raise ManifestError(
                f"Sample {manifest_sample.sample_id!r} is absent from the mask ledger."
            ) from exc
        try:
            index, raw_sample = indexed_raw_samples[manifest_sample.sample_id]
        except KeyError as exc:
            raise ManifestError(
                f"Selected sample {manifest_sample.sample_id!r} is absent from "
                "the prepared-cache record."
            ) from exc
        parsed_samples.append(
            _parse_record_sample(
                raw_sample,
                index=index,
                manifest_sample=manifest_sample,
                data_root=resolved_data_root,
                cache_root=declared_cache_root,
                ledger=ledger_row,
                artifact_rows=artifact_rows,
            )
        )
    return PreparedCacheRecord(
        path=resolved_path,
        sha256=expected_record_sha,
        record_fingerprint=record_fingerprint,
        cache_id=cache_id,
        created_at=created_at,
        data_root=resolved_data_root,
        cache_root=declared_cache_root,
        preparation_version=preparation_version,
        taxonomy_version=taxonomy_version,
        manifest_set_fingerprint=manifest_set_fingerprint,
        selected_manifest_set_fingerprint=collection.set_fingerprint,
        prepared_cache_implementation_sha256=implementation_sha,
        user_instruction_sha256=instruction_sha,
        processed_mask_ledger=ledger_ref,
        stage09_report=report_ref,
        artifact_index=artifact_ref,
        samples=tuple(parsed_samples),
    )


def _verify_record_sample(sample: PreparedRecordSample) -> PreparedSample:
    image = _hash_open_file(
        sample.image_path,
        expected_sha256=sample.image_sha256,
        expected_bytes=sample.image_bytes,
        label=f"source image {sample.sample_id!r}",
    )
    # The source mask remains an audit-only input.  Hash it once at launch so
    # pre-launch drift is rejected, then deliberately discard its file identity;
    # the epoch hot path reads only the image and prepared target.
    _hash_open_file(
        sample.source_annotation_path,
        expected_sha256=sample.source_annotation_sha256,
        expected_bytes=None,
        label=f"source annotation {sample.sample_id!r}",
    )
    cache_target = _hash_open_file(
        sample.cache_target_path,
        expected_sha256=sample.cache_target_sha256,
        expected_bytes=sample.cache_target_bytes,
        label=f"prepared target {sample.sample_id!r}",
    )
    return PreparedSample(
        sample_id=sample.sample_id,
        source_dataset=sample.source_dataset,
        source_split=sample.source_split,
        target_split=sample.target_split,
        width=sample.width,
        height=sample.height,
        image=image,
        cache_target=cache_target,
        source_annotation_sha256=sample.source_annotation_sha256,
        stage09_target_sha256=sample.stage09_target_sha256,
        target_mode=sample.target_mode,
        target_dtype=sample.target_dtype,
        target_ids=sample.target_ids,
        class_counts=sample.class_counts,
        ignored_count=sample.ignored_count,
        invalid_count=sample.invalid_count,
        preparation_version=sample.preparation_version,
        taxonomy_version=sample.taxonomy_version,
        target_mapping_version=sample.target_mapping_version,
        target_mapping_sha256=sample.target_mapping_sha256,
        valid_supervision_classes=sample.valid_supervision_classes,
        ignore_index=sample.ignore_index,
    )


def canonical_snapshot_fingerprint(
    *,
    record: PreparedCacheRecord,
    samples: Sequence[PreparedSample],
) -> str:
    """Return a migration-stable fingerprint based on content, not inode/path/time."""

    payload = {
        "algorithm": PREPARED_SNAPSHOT_FINGERPRINT_ALGORITHM,
        "schema_version": PREPARED_CACHE_SCHEMA,
        "cache_id": record.cache_id,
        "preparation_version": record.preparation_version,
        "taxonomy_version": record.taxonomy_version,
        "record_manifest_set_fingerprint": record.manifest_set_fingerprint,
        "selected_manifest_set_fingerprint": (
            record.selected_manifest_set_fingerprint
        ),
        "prepared_cache_implementation_sha256": (
            record.prepared_cache_implementation_sha256
        ),
        "user_instruction_sha256": record.user_instruction_sha256,
        "processed_mask_ledger_sha256": record.processed_mask_ledger.sha256,
        "stage09_report_sha256": record.stage09_report.sha256,
        "artifact_index_sha256": record.artifact_index.sha256,
        "samples": [
            {
                "sample_id": sample.sample_id,
                "source_dataset": sample.source_dataset,
                "source_split": sample.source_split,
                "target_split": sample.target_split,
                "width": sample.width,
                "height": sample.height,
                "image_sha256": sample.image.sha256,
                "image_bytes": sample.image.bytes,
                "source_annotation_sha256": sample.source_annotation_sha256,
                "stage09_target_sha256": sample.stage09_target_sha256,
                "cache_target_sha256": sample.cache_target.sha256,
                "cache_target_bytes": sample.cache_target.bytes,
                "target_mode": sample.target_mode,
                "target_dtype": sample.target_dtype,
                "target_ids": list(sample.target_ids),
                "class_counts": {str(key): value for key, value in sample.class_counts},
                "ignored_count": sample.ignored_count,
                "invalid_count": sample.invalid_count,
                "target_mapping_version": sample.target_mapping_version,
                "target_mapping_sha256": sample.target_mapping_sha256,
                "valid_supervision_classes": list(sample.valid_supervision_classes),
                "ignore_index": sample.ignore_index,
            }
            for sample in sorted(samples, key=lambda item: (item.source_dataset, item.sample_id))
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_prepared_cache_at_launch(
    record: PreparedCacheRecord,
    *,
    collection: ManifestCollection,
    data_root: Path,
    max_workers: int = 4,
) -> PreparedSnapshot:
    """Hash every current image/cache target once and freeze file identities."""

    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise ManifestError("max_workers must be an integer.")
    if max_workers < 1 or max_workers > 32:
        raise ManifestError("max_workers must be in the bounded range 1..32.")
    if _resolved_root(data_root, label="Dataset root") != record.data_root:
        raise ManifestError("Launch verification received a different dataset root.")
    if collection.set_fingerprint != record.selected_manifest_set_fingerprint:
        raise ManifestError("Launch verification received a different manifest collection.")
    expected_ids = tuple(
        sample.sample_id
        for sample in sorted(
            collection.samples, key=lambda item: (item.source_dataset, item.sample_id)
        )
    )
    if tuple(sample.sample_id for sample in record.samples) != expected_ids:
        raise ManifestError("Prepared record no longer matches the manifest sample order.")
    _hash_regular_file(
        record.path, expected_sha256=record.sha256, label="prepared-cache record"
    )
    for ref, label in (
        (record.processed_mask_ledger, "processed-mask ledger"),
        (record.stage09_report, "Stage-9 report"),
        (record.artifact_index, "artifact index"),
    ):
        _hash_regular_file(ref.path, expected_sha256=ref.sha256, label=label)
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="floodsight-prepared-verify"
    ) as executor:
        verified = tuple(executor.map(_verify_record_sample, record.samples))
    snapshot_fingerprint = canonical_snapshot_fingerprint(record=record, samples=verified)
    return PreparedSnapshot(
        record_path=record.path,
        record_sha256=record.sha256,
        record_fingerprint=record.record_fingerprint,
        snapshot_fingerprint=snapshot_fingerprint,
        cache_id=record.cache_id,
        record_manifest_set_fingerprint=record.manifest_set_fingerprint,
        manifest_set_fingerprint=record.selected_manifest_set_fingerprint,
        prepared_cache_implementation_sha256=(
            record.prepared_cache_implementation_sha256
        ),
        user_instruction_sha256=record.user_instruction_sha256,
        samples=verified,
    )


def _read_snapshot_bytes(identity: PreparedFileIdentity, *, label: str) -> bytes:
    descriptor, before = _open_readonly(identity.path, label=label)
    if not identity.matches_stat(before):
        os.close(descriptor)
        raise ManifestError(f"{label} changed after launch verification: {identity.path}")
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ManifestError(f"Unable to read launch-verified {label}: {identity.path}") from exc
    if not identity.matches_stat(after) or len(payload) != identity.bytes:
        raise ManifestError(f"{label} changed while it was being read: {identity.path}")
    return payload


def read_verified_pair(sample: PreparedSample) -> tuple[Image.Image, Image.Image]:
    """Decode a launch-verified image/target pair without hashing or RNG use."""

    image_bytes = _read_snapshot_bytes(sample.image, label=f"image {sample.sample_id!r}")
    target_bytes = _read_snapshot_bytes(
        sample.cache_target, label=f"prepared target {sample.sample_id!r}"
    )
    try:
        with Image.open(BytesIO(image_bytes)) as opened_image:
            image = opened_image.convert("RGB")
        with Image.open(BytesIO(target_bytes)) as opened_target:
            if opened_target.mode != sample.target_mode:
                raise ManifestError(
                    f"Prepared target mode changed for {sample.sample_id!r}: "
                    f"expected {sample.target_mode!r}, found {opened_target.mode!r}."
                )
            target = opened_target.copy()
    except ManifestError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise ManifestError(
            f"Unable to decode launch-verified sample {sample.sample_id!r}."
        ) from exc
    expected_size = (sample.width, sample.height)
    if image.size != expected_size or target.size != expected_size:
        raise ManifestError(
            f"Prepared sample dimension mismatch for {sample.sample_id!r}: "
            f"expected {expected_size}, image={image.size}, target={target.size}."
        )
    return image, target


__all__ = [
    "PREPARED_CACHE_FINGERPRINT_ALGORITHM",
    "PREPARED_CACHE_RECORD_NAME",
    "PREPARED_CACHE_SCHEMA",
    "PREPARED_SNAPSHOT_FINGERPRINT_ALGORITHM",
    "PreparedCacheBuild",
    "PreparedCacheRecord",
    "PreparedFileIdentity",
    "PreparedRecordSample",
    "PreparedSample",
    "PreparedSnapshot",
    "ProvenanceFile",
    "build_prepared_cache_record",
    "canonical_prepared_record_fingerprint",
    "canonical_snapshot_fingerprint",
    "load_prepared_cache_record",
    "read_verified_pair",
    "verify_prepared_cache_at_launch",
]
