"""Fail-closed validation for the one-time prepared-data fast-path transition.

The normal checkpoint contract deliberately requires byte-identical executable
source and authorization provenance.  This module defines the sole migration
envelope allowed to cross that boundary: a fully hash-bound transition from a
committed production checkpoint to the launch-verified prepared-data fast
path.  It does not load checkpoint tensors; :mod:`checkpoint` remains the only
module that restores training state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .errors import CheckpointError

TRANSITION_SCHEMA = "floodsight-segformer-source-transition-v1"
ALLOWED_TRANSITION = "LAUNCH_VERIFIED_PREPARED_DATA_FAST_PATH"

_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "allowed_change",
        "predecessor_checkpoint",
        "frozen_run",
        "training_state",
        "predecessor",
        "successor",
    }
)
_CHECKPOINT_KEYS = frozenset({"path", "sha256"})
_FROZEN_RUN_KEYS = frozenset(
    {
        "config_sha256",
        "manifest_sha256",
        "manifest_fingerprint",
        "taxonomy_sha256",
        "run_directory",
    }
)
_TRAINING_STATE_KEYS = frozenset({"epoch", "global_step", "best_metric"})
_PROVENANCE_KEYS = frozenset({"input_provenance", "authorization_provenance"})
_AUTHORIZATION_KEYS = frozenset(
    {
        "approval_record_sha256",
        "human_review_sha256",
        "real_smoke_report_sha256",
    }
)
_REQUIRED_FAST_PATH_INPUT_KEYS = frozenset(
    {
        "prepared_fast_path_record_path",
        "prepared_fast_path_record_sha256",
        "prepared_fast_path_snapshot_fingerprint",
    }
)


@dataclass(frozen=True, slots=True)
class TransitionTrainingState:
    """Exact committed state authorized to cross the source boundary."""

    epoch: int
    global_step: int
    best_metric: float


@dataclass(frozen=True, slots=True)
class CheckpointTransition:
    """Immutable, normalized view of a validated transition record."""

    record_path: Path
    record_sha256: str
    allowed_change: str
    predecessor_checkpoint_path: Path
    predecessor_checkpoint_sha256: str
    config_sha256: str
    manifest_sha256: Mapping[str, str]
    manifest_fingerprint: Mapping[str, str]
    taxonomy_sha256: Mapping[str, str]
    run_directory: Path
    training_state: TransitionTrainingState
    predecessor_input_provenance: Mapping[str, str]
    predecessor_authorization_provenance: Mapping[str, str]
    successor_input_provenance: Mapping[str, str]
    successor_authorization_provenance: Mapping[str, str]


def _fail(message: str) -> CheckpointError:
    return CheckpointError(f"Checkpoint source transition rejected: {message}")


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], location: str) -> None:
    actual = set(value)
    if actual != set(expected):
        raise _fail(
            f"invalid keys at {location}: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}."
        )


def _object(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"expected an object at {location}.")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"expected a non-empty string at {location}.")
    return value


def _sha256(value: Any, location: str) -> str:
    digest = _string(value, location)
    if not _SHA256_PATTERN.fullmatch(digest):
        raise _fail(f"expected a lowercase SHA-256 at {location}.")
    return digest


def _absolute_path(value: Any, location: str) -> Path:
    raw = Path(_string(value, location)).expanduser()
    if not raw.is_absolute():
        raise _fail(f"expected an absolute path at {location}.")
    if raw.is_symlink():
        raise _fail(f"symbolic links are forbidden at {location}.")
    return raw.resolve()


def _string_mapping(value: Any, location: str) -> Mapping[str, str]:
    raw = _object(value, location)
    if not raw:
        raise _fail(f"expected a non-empty object at {location}.")
    normalized: dict[str, str] = {}
    for key, item in raw.items():
        normalized[_string(key, f"{location} key")] = _string(
            item, f"{location}.{key}"
        )
    return MappingProxyType(dict(sorted(normalized.items())))


def _sha256_mapping(value: Any, location: str) -> Mapping[str, str]:
    raw = _object(value, location)
    if not raw:
        raise _fail(f"expected a non-empty object at {location}.")
    normalized: dict[str, str] = {}
    for key, item in raw.items():
        normalized[_string(key, f"{location} key")] = _sha256(
            item, f"{location}.{key}"
        )
    return MappingProxyType(dict(sorted(normalized.items())))


def _authorization(value: Any, location: str) -> Mapping[str, str]:
    raw = _object(value, location)
    _exact_keys(raw, _AUTHORIZATION_KEYS, location)
    return _sha256_mapping(raw, location)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_unique_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _fail(f"unable to parse transition record {path}.") from exc
    return _object(raw, "record")


def sha256_regular_file(path: Path, *, label: str) -> str:
    """Hash one regular non-symlink file without following its final component."""

    raw = path.expanduser()
    if raw.is_symlink():
        raise _fail(f"{label} must not be a symbolic link: {raw}")
    try:
        resolved = raw.resolve(strict=True)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise _fail(f"unable to open {label}: {raw}") from exc
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _fail(f"{label} must be a regular file: {resolved}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    except OSError as exc:
        raise _fail(f"unable to hash {label}: {resolved}") from exc
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _validate_provenance_change(
    predecessor: Mapping[str, str], successor: Mapping[str, str]
) -> None:
    predecessor_keys = set(predecessor)
    successor_keys = set(successor)
    if "training_source_sha256" not in predecessor_keys:
        raise _fail("predecessor input provenance has no training_source_sha256.")
    if "training_source_sha256" not in successor_keys:
        raise _fail("successor input provenance has no training_source_sha256.")
    old_source = _sha256(
        predecessor["training_source_sha256"],
        "predecessor.input_provenance.training_source_sha256",
    )
    new_source = _sha256(
        successor["training_source_sha256"],
        "successor.input_provenance.training_source_sha256",
    )
    if old_source == new_source:
        raise _fail("the executable-source SHA-256 did not change.")
    removed = predecessor_keys - successor_keys
    if removed:
        raise _fail(f"successor input provenance removed keys: {sorted(removed)}.")
    for key in predecessor_keys - {"training_source_sha256"}:
        if predecessor[key] != successor[key]:
            raise _fail(f"immutable input provenance changed at {key!r}.")
    added = successor_keys - predecessor_keys
    missing_fast_path = _REQUIRED_FAST_PATH_INPUT_KEYS - added
    if missing_fast_path:
        raise _fail(
            "successor input provenance lacks required fast-path bindings: "
            f"{sorted(missing_fast_path)}."
        )
    forbidden = {
        key
        for key in added
        if not key.startswith("prepared_fast_path_") and key != "torch_cpu_threads"
    }
    if forbidden:
        raise _fail(
            "successor input provenance added non-fast-path keys: "
            f"{sorted(forbidden)}."
        )
    _absolute_path(
        successor["prepared_fast_path_record_path"],
        "successor.input_provenance.prepared_fast_path_record_path",
    )
    _sha256(
        successor["prepared_fast_path_record_sha256"],
        "successor.input_provenance.prepared_fast_path_record_sha256",
    )
    _sha256(
        successor["prepared_fast_path_snapshot_fingerprint"],
        "successor.input_provenance.prepared_fast_path_snapshot_fingerprint",
    )


def _validate_authorization_change(
    predecessor: Mapping[str, str], successor: Mapping[str, str]
) -> None:
    for required_change in ("approval_record_sha256", "real_smoke_report_sha256"):
        if predecessor[required_change] == successor[required_change]:
            raise _fail(f"successor authorization did not refresh {required_change}.")


def _normalize_expected_mapping(
    value: Mapping[str, str], location: str, *, sha256_values: bool = False
) -> dict[str, str]:
    normalizer = _sha256_mapping if sha256_values else _string_mapping
    return dict(normalizer(value, location))


def validate_checkpoint_transition(
    record_path: Path,
    *,
    expected_record_sha256: str,
    expected_predecessor_checkpoint: Path,
    expected_config_sha256: str,
    expected_manifest_sha256: Mapping[str, str],
    expected_manifest_fingerprint: Mapping[str, str],
    expected_taxonomy_sha256: Mapping[str, str],
    expected_successor_input_provenance: Mapping[str, str],
    expected_successor_authorization_provenance: Mapping[str, str],
    expected_run_directory: Path,
) -> CheckpointTransition:
    """Validate and normalize an exact, externally SHA-bound transition record."""

    expected_record_digest = _sha256(expected_record_sha256, "expected_record_sha256")
    raw_record_path = record_path.expanduser()
    if raw_record_path.is_symlink():
        raise _fail("transition record must not be a symbolic link.")
    try:
        normalized_record_path = raw_record_path.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"transition record is missing: {raw_record_path}") from exc
    actual_record_digest = sha256_regular_file(
        normalized_record_path, label="transition record"
    )
    if actual_record_digest != expected_record_digest:
        raise _fail("transition-record SHA-256 does not match the expected digest.")

    raw = _read_unique_json(normalized_record_path)
    _exact_keys(raw, _ROOT_KEYS, "record")
    if raw["schema_version"] != TRANSITION_SCHEMA:
        raise _fail("unsupported transition-record schema/version.")
    if raw["allowed_change"] != ALLOWED_TRANSITION:
        raise _fail(
            f"allowed_change must be exactly {ALLOWED_TRANSITION!r}."
        )

    checkpoint = _object(raw["predecessor_checkpoint"], "predecessor_checkpoint")
    _exact_keys(checkpoint, _CHECKPOINT_KEYS, "predecessor_checkpoint")
    checkpoint_path = _absolute_path(checkpoint["path"], "predecessor_checkpoint.path")
    checkpoint_sha256 = _sha256(
        checkpoint["sha256"], "predecessor_checkpoint.sha256"
    )
    expected_checkpoint_path = expected_predecessor_checkpoint.expanduser()
    if expected_checkpoint_path.is_symlink():
        raise _fail("expected predecessor checkpoint must not be a symbolic link.")
    expected_checkpoint_path = expected_checkpoint_path.resolve()
    if checkpoint_path != expected_checkpoint_path:
        raise _fail("record names a different predecessor checkpoint path.")

    frozen = _object(raw["frozen_run"], "frozen_run")
    _exact_keys(frozen, _FROZEN_RUN_KEYS, "frozen_run")
    config_sha256 = _sha256(frozen["config_sha256"], "frozen_run.config_sha256")
    manifest_sha256 = _sha256_mapping(
        frozen["manifest_sha256"], "frozen_run.manifest_sha256"
    )
    manifest_fingerprint = _sha256_mapping(
        frozen["manifest_fingerprint"], "frozen_run.manifest_fingerprint"
    )
    taxonomy_sha256 = _sha256_mapping(
        frozen["taxonomy_sha256"], "frozen_run.taxonomy_sha256"
    )
    run_directory = _absolute_path(frozen["run_directory"], "frozen_run.run_directory")
    normalized_expected_run = expected_run_directory.expanduser().resolve()
    if run_directory != normalized_expected_run:
        raise _fail("frozen run directory changed across the source transition.")
    if checkpoint_path.parent != run_directory or checkpoint_path.name != "last.pt":
        raise _fail("predecessor must be the frozen run directory's exact last.pt.")

    if config_sha256 != _sha256(expected_config_sha256, "expected_config_sha256"):
        raise _fail("frozen configuration SHA-256 does not match the current run.")
    expected_manifest_hashes = _normalize_expected_mapping(
        expected_manifest_sha256,
        "expected_manifest_sha256",
        sha256_values=True,
    )
    if dict(manifest_sha256) != expected_manifest_hashes:
        raise _fail("frozen manifest SHA-256 set changed across the source transition.")
    expected_fingerprints = _normalize_expected_mapping(
        expected_manifest_fingerprint,
        "expected_manifest_fingerprint",
        sha256_values=True,
    )
    if dict(manifest_fingerprint) != expected_fingerprints:
        raise _fail("frozen manifest fingerprints changed across the source transition.")
    expected_taxonomy = _normalize_expected_mapping(
        expected_taxonomy_sha256,
        "expected_taxonomy_sha256",
        sha256_values=True,
    )
    if dict(taxonomy_sha256) != expected_taxonomy:
        raise _fail("frozen taxonomy/mapping hashes changed across the source transition.")

    raw_state = _object(raw["training_state"], "training_state")
    _exact_keys(raw_state, _TRAINING_STATE_KEYS, "training_state")
    epoch = raw_state["epoch"]
    global_step = raw_state["global_step"]
    best_metric = raw_state["best_metric"]
    if type(epoch) is not int or epoch < 1:
        raise _fail("transition training_state.epoch must be a positive integer.")
    if type(global_step) is not int or global_step < 1:
        raise _fail("transition training_state.global_step must be a positive integer.")
    if type(best_metric) not in {int, float}:
        raise _fail("transition training_state.best_metric must be numeric.")
    best_metric = float(best_metric)
    if not math.isfinite(best_metric):
        raise _fail("transition training_state.best_metric must be finite.")
    training_state = TransitionTrainingState(
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
    )

    predecessor = _object(raw["predecessor"], "predecessor")
    successor = _object(raw["successor"], "successor")
    _exact_keys(predecessor, _PROVENANCE_KEYS, "predecessor")
    _exact_keys(successor, _PROVENANCE_KEYS, "successor")
    predecessor_input = _string_mapping(
        predecessor["input_provenance"], "predecessor.input_provenance"
    )
    successor_input = _string_mapping(
        successor["input_provenance"], "successor.input_provenance"
    )
    predecessor_authorization = _authorization(
        predecessor["authorization_provenance"],
        "predecessor.authorization_provenance",
    )
    successor_authorization = _authorization(
        successor["authorization_provenance"],
        "successor.authorization_provenance",
    )
    _validate_provenance_change(predecessor_input, successor_input)
    _validate_authorization_change(predecessor_authorization, successor_authorization)

    expected_successor_input = _normalize_expected_mapping(
        expected_successor_input_provenance,
        "expected_successor_input_provenance",
    )
    if dict(successor_input) != expected_successor_input:
        raise _fail("successor input provenance does not match the current launch.")
    expected_successor_authorization = dict(
        _authorization(
            expected_successor_authorization_provenance,
            "expected_successor_authorization_provenance",
        )
    )
    if dict(successor_authorization) != expected_successor_authorization:
        raise _fail("successor authorization provenance does not match the current launch.")

    actual_checkpoint_sha256 = sha256_regular_file(
        checkpoint_path, label="predecessor checkpoint"
    )
    if actual_checkpoint_sha256 != checkpoint_sha256:
        raise _fail("predecessor checkpoint SHA-256 does not match the record.")

    return CheckpointTransition(
        record_path=normalized_record_path,
        record_sha256=actual_record_digest,
        allowed_change=ALLOWED_TRANSITION,
        predecessor_checkpoint_path=checkpoint_path,
        predecessor_checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
        manifest_fingerprint=manifest_fingerprint,
        taxonomy_sha256=taxonomy_sha256,
        run_directory=run_directory,
        training_state=training_state,
        predecessor_input_provenance=predecessor_input,
        predecessor_authorization_provenance=predecessor_authorization,
        successor_input_provenance=successor_input,
        successor_authorization_provenance=successor_authorization,
    )
