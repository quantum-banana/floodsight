"""Atomic, provenance-bound checkpoint save/load and exact RNG resume."""

from __future__ import annotations

import math
import os
import pickle
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .errors import CheckpointError
from .reproducibility import capture_rng_state, restore_rng_state
from .transition import sha256_regular_file, validate_checkpoint_transition

CHECKPOINT_FORMAT = "floodsight-segformer-checkpoint-v3"


@dataclass(frozen=True, slots=True)
class TrainingState:
    epoch: int
    global_step: int
    best_metric: float | None


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any | None,
    training_state: TrainingState,
    config_sha256: str,
    manifest_sha256: Mapping[str, str],
    manifest_fingerprint: Mapping[str, str],
    taxonomy_sha256: Mapping[str, str],
    input_provenance: Mapping[str, str],
    authorization_provenance: Mapping[str, str] | None = None,
    run_directory: Path,
    data_generator: torch.Generator,
    provenance: str,
) -> None:
    """Atomically save all state needed for an exact continuation."""

    if provenance not in {"REAL_ML_OUTPUT", "DEMO_SIMULATED"}:
        raise CheckpointError(f"Unsupported checkpoint provenance {provenance!r}.")
    path = path.expanduser().resolve()
    normalized_run_directory = run_directory.expanduser().resolve()
    if path.parent != normalized_run_directory:
        raise CheckpointError(
            f"Checkpoint must be written directly inside its frozen run directory: {path}"
        )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "provenance": provenance,
        "training_state": asdict(training_state),
        "config_sha256": config_sha256,
        "manifest_sha256": dict(sorted(manifest_sha256.items())),
        "manifest_fingerprint": dict(sorted(manifest_fingerprint.items())),
        "taxonomy_sha256": dict(sorted(taxonomy_sha256.items())),
        "input_provenance": dict(sorted(input_provenance.items())),
        "authorization_provenance": dict(sorted((authorization_provenance or {}).items())),
        "run_directory": str(normalized_run_directory),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng_state(data_generator=data_generator),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
        torch.save(payload, temporary_name)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: Any | None,
    expected_config_sha256: str,
    expected_manifest_sha256: Mapping[str, str],
    expected_manifest_fingerprint: Mapping[str, str],
    expected_taxonomy_sha256: Mapping[str, str],
    expected_input_provenance: Mapping[str, str],
    expected_authorization_provenance: Mapping[str, str] | None = None,
    expected_run_directory: Path,
    data_generator: torch.Generator | None,
    map_location: str | torch.device,
    expected_provenance: str | None = None,
    allow_manifest_subset: bool = False,
) -> TrainingState:
    """Strictly reload a trusted FloodSight checkpoint and restore RNG state."""

    raw_path = path.expanduser()
    if raw_path.is_symlink():
        raise CheckpointError("Checkpoint must not be a symbolic link.")
    path = raw_path.resolve()
    normalized_run_directory = expected_run_directory.expanduser().resolve()
    if path.parent != normalized_run_directory:
        raise CheckpointError("Checkpoint is outside the expected frozen run directory.")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"Unable to load checkpoint: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"Unsupported checkpoint format: {path}")
    if payload.get("config_sha256") != expected_config_sha256:
        raise CheckpointError("Checkpoint configuration SHA-256 does not match this run.")
    expected_manifests = dict(sorted(expected_manifest_sha256.items()))
    stored_manifests = payload.get("manifest_sha256")
    manifests_match = stored_manifests == expected_manifests
    if allow_manifest_subset and isinstance(stored_manifests, Mapping):
        manifests_match = all(
            stored_manifests.get(key) == value for key, value in expected_manifests.items()
        )
    if not manifests_match:
        raise CheckpointError("Checkpoint manifest SHA-256 set does not match this run.")
    expected_fingerprints = dict(sorted(expected_manifest_fingerprint.items()))
    stored_fingerprints = payload.get("manifest_fingerprint")
    fingerprints_match = stored_fingerprints == expected_fingerprints
    if allow_manifest_subset and isinstance(stored_fingerprints, Mapping):
        fingerprints_match = all(
            stored_fingerprints.get(key) == value for key, value in expected_fingerprints.items()
        )
    if not fingerprints_match:
        raise CheckpointError("Checkpoint manifest fingerprint set does not match this run.")
    expected_taxonomy = dict(sorted(expected_taxonomy_sha256.items()))
    if payload.get("taxonomy_sha256") != expected_taxonomy:
        raise CheckpointError("Checkpoint taxonomy/mapping SHA-256 set does not match this run.")
    if payload.get("run_directory") != str(normalized_run_directory):
        raise CheckpointError("Checkpoint approved run directory does not match this run.")
    if payload.get("input_provenance") != dict(sorted(expected_input_provenance.items())):
        raise CheckpointError("Checkpoint immutable input provenance does not match this run.")
    stored_authorization = payload.get("authorization_provenance")
    if not isinstance(stored_authorization, Mapping):
        raise CheckpointError("Checkpoint authorization provenance is missing or invalid.")
    if expected_authorization_provenance is not None and stored_authorization != dict(
        sorted(expected_authorization_provenance.items())
    ):
        raise CheckpointError("Checkpoint training authorization provenance does not match.")
    if expected_provenance is not None and payload.get("provenance") != expected_provenance:
        raise CheckpointError("Checkpoint provenance does not match this operation.")
    try:
        model.load_state_dict(payload["model"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        raw_state = payload["training_state"]
        state = TrainingState(
            epoch=int(raw_state["epoch"]),
            global_step=int(raw_state["global_step"]),
            best_metric=(
                float(raw_state["best_metric"]) if raw_state["best_metric"] is not None else None
            ),
        )
        if (
            state.epoch < 0
            or state.global_step < 0
            or (state.best_metric is not None and not math.isfinite(state.best_metric))
        ):
            raise ValueError("Checkpoint training state contains invalid numeric values.")
        restore_rng_state(payload["rng"], data_generator=data_generator)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError(f"Checkpoint state is incomplete or incompatible: {path}") from exc
    return state


def load_checkpoint_transition(
    path: Path,
    *,
    transition_record_path: Path,
    expected_transition_record_sha256: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any | None,
    expected_config_sha256: str,
    expected_manifest_sha256: Mapping[str, str],
    expected_manifest_fingerprint: Mapping[str, str],
    expected_taxonomy_sha256: Mapping[str, str],
    expected_input_provenance: Mapping[str, str],
    expected_authorization_provenance: Mapping[str, str],
    expected_run_directory: Path,
    data_generator: torch.Generator,
    map_location: str | torch.device,
) -> TrainingState:
    """Perform the sole approved one-time source transition from ``last.pt``.

    Ordinary resumes must continue to use :func:`load_checkpoint`, whose exact
    provenance behavior is intentionally unchanged.  This path first validates
    an externally SHA-bound transition record against both launch generations,
    then delegates all tensor/optimizer/scheduler/scaler/RNG restoration to the
    strict loader using the checkpoint's predecessor provenance.
    """

    transition = validate_checkpoint_transition(
        transition_record_path,
        expected_record_sha256=expected_transition_record_sha256,
        expected_predecessor_checkpoint=path,
        expected_config_sha256=expected_config_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_manifest_fingerprint=expected_manifest_fingerprint,
        expected_taxonomy_sha256=expected_taxonomy_sha256,
        expected_successor_input_provenance=expected_input_provenance,
        expected_successor_authorization_provenance=expected_authorization_provenance,
        expected_run_directory=expected_run_directory,
    )
    state = load_checkpoint(
        transition.predecessor_checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_config_sha256=transition.config_sha256,
        expected_manifest_sha256=transition.manifest_sha256,
        expected_manifest_fingerprint=transition.manifest_fingerprint,
        expected_taxonomy_sha256=transition.taxonomy_sha256,
        expected_input_provenance=transition.predecessor_input_provenance,
        expected_authorization_provenance=(
            transition.predecessor_authorization_provenance
        ),
        expected_run_directory=transition.run_directory,
        data_generator=data_generator,
        map_location=map_location,
        expected_provenance="REAL_ML_OUTPUT",
        allow_manifest_subset=False,
    )
    expected_state = TrainingState(
        epoch=transition.training_state.epoch,
        global_step=transition.training_state.global_step,
        best_metric=transition.training_state.best_metric,
    )
    if state != expected_state:
        raise CheckpointError(
            "Checkpoint source transition rejected: predecessor training state "
            "does not match the immutable transition record."
        )
    if (
        sha256_regular_file(
            transition.predecessor_checkpoint_path,
            label="predecessor checkpoint after load",
        )
        != transition.predecessor_checkpoint_sha256
    ):
        raise CheckpointError(
            "Checkpoint source transition rejected: predecessor checkpoint changed "
            "while it was being restored."
        )
    if (
        sha256_regular_file(
            transition.record_path,
            label="transition record after load",
        )
        != transition.record_sha256
    ):
        raise CheckpointError(
            "Checkpoint source transition rejected: transition record changed while "
            "the checkpoint was being restored."
        )
    return state
