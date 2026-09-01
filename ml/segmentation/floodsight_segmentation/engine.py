"""Guarded single-device SegFormer training and validation engine."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, SequentialSampler

from .approval import validate_human_approval
from .artifact import ModelArtifactSpec, validate_model_artifact
from .checkpoint import (
    TrainingState,
    load_checkpoint,
    load_checkpoint_transition,
    save_checkpoint,
)
from .checkpoint_probe import run_fresh_process_production_checkpoint_probe
from .config import AUDITED_SOURCE_TO_TARGET_IDS, SegmentationConfig, load_config
from .dataset import SegmentationManifestDataset, build_dataset_balanced_sampler
from .guard import require_real_smoke_authorization, require_training_authorization
from .integrity import training_source_sha256
from .manifest import (
    ManifestCollection,
    ManifestSpec,
    load_manifest_collection,
    require_canonical_manifest_locks,
    sha256_file,
)
from .metrics import SegmentationMetrics
from .model import build_segformer, logits_at_label_resolution
from .optim import build_optimizer, build_scheduler
from .prepared import (
    PreparedCacheRecord,
    PreparedSnapshot,
    load_prepared_cache_record,
    verify_prepared_cache_at_launch,
)
from .reproducibility import make_generator, seed_everything, seed_worker
from .runtime import require_h100, validate_runtime_versions
from .supervision import PartialCrossEntropyLoss
from .threaded_loader import ThreadedSampleLoader
from .transforms import PairedSegmentationTransform
from .transition import CheckpointTransition, validate_checkpoint_transition

POOL_CLASS_ID = 15
_PROCESS_RUN_LOCKS: dict[Path, Any] = {}


def _runtime_provenance(versions: Mapping[str, str]) -> dict[str, str]:
    return {f"runtime_{name}": value for name, value in sorted(versions.items())}


def _runtime_fingerprint(versions: Mapping[str, str]) -> str:
    canonical = json.dumps(dict(sorted(versions.items())), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _device_provenance(device: torch.device, *, precision: str) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device": str(device),
            "device_type": device.type,
            "effective_precision": precision,
        }
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "device": str(torch.device("cuda", index)),
        "device_type": "cuda",
        "device_name": torch.cuda.get_device_name(index),
        "device_capability": list(torch.cuda.get_device_capability(index)),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "effective_precision": precision,
    }


def _manifest_set_fingerprint(
    hashes: Mapping[str, str],
    fingerprints: Mapping[str, str],
) -> str:
    if set(hashes) != set(fingerprints):
        raise RuntimeError("Manifest hash/fingerprint path sets do not match.")
    records = [
        {
            "path": path,
            "manifest_sha256": hashes[path],
            "dataset_fingerprint": fingerprints[path],
        }
        for path in sorted(hashes)
    ]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError(f"{precision} is allowed only on CUDA in the frozen training path.")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _build_transform(config: SegmentationConfig, *, training: bool) -> PairedSegmentationTransform:
    transforms = config.transforms
    return PairedSegmentationTransform(
        height=transforms.height,
        width=transforms.width,
        mean=transforms.image_mean,
        std=transforms.image_std,
        training=training,
        scale=transforms.train_scale if training else (1.0, 1.0),
        ratio=transforms.train_ratio if training else (1.0, 1.0),
        horizontal_flip_probability=(
            transforms.horizontal_flip_probability if training else 0.0
        ),
    )


def _load_collection(
    specs: Sequence[ManifestSpec],
    config: SegmentationConfig,
    *,
    splits: Sequence[str],
) -> ManifestCollection:
    collection = load_manifest_collection(
        specs,
        expected_taxonomy=config.data.taxonomy_version,
        allowed_datasets=config.data.allowed_datasets,
        selected_splits=splits,
        require_full_integrity=config.data.require_full_integrity,
    )
    observed = sorted(manifest.dataset_id for manifest in collection.manifests)
    expected = sorted(config.data.allowed_datasets)
    if observed != expected:
        raise RuntimeError(
            "A frozen manifest is required exactly once for every segmentation dataset: "
            f"expected={expected}, observed={observed}."
        )
    require_canonical_manifest_locks(collection)
    return collection


def _build_dataset(
    collection: ManifestCollection,
    config: SegmentationConfig,
    *,
    data_root: Path,
    training: bool,
    prepared_snapshot: PreparedSnapshot | None = None,
) -> SegmentationManifestDataset:
    return SegmentationManifestDataset(
        collection,
        data_root=data_root,
        supported_class_ids=config.data.supported_class_ids,
        source_to_target_ids=AUDITED_SOURCE_TO_TARGET_IDS,
        num_labels=config.model.num_labels,
        ignore_index=config.data.ignore_index,
        transform=_build_transform(config, training=training),
        verify_sample_hashes=config.data.verify_sample_hashes,
        prepared_snapshot=prepared_snapshot,
    )


def _combined_collection(
    *collections: ManifestCollection,
) -> ManifestCollection:
    manifests = tuple(
        manifest for collection in collections for manifest in collection.manifests
    )
    samples = tuple(sample for collection in collections for sample in collection.samples)
    if len({str(manifest.path) for manifest in manifests}) != len(manifests):
        raise RuntimeError("Prepared fast path received duplicate manifest paths.")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise RuntimeError("Prepared fast path received duplicate sample IDs.")
    return ManifestCollection(manifests=manifests, samples=samples)


def _prepare_fast_path(
    *,
    collection: ManifestCollection,
    data_root: Path,
    prepared_fast_path_record: Path | None,
    prepared_fast_path_record_sha256: str | None,
    allow_prepared_fast_path: bool,
    loader_threads: int | None,
    loader_prefetch_samples: int | None,
    torch_cpu_threads: int | None,
) -> tuple[PreparedCacheRecord | None, PreparedSnapshot | None]:
    values = (
        prepared_fast_path_record,
        prepared_fast_path_record_sha256,
        loader_threads,
        loader_prefetch_samples,
        torch_cpu_threads,
    )
    if all(value is None for value in values) and allow_prepared_fast_path is False:
        return None, None
    if allow_prepared_fast_path is not True or any(value is None for value in values):
        raise RuntimeError(
            "Prepared fast path requires an explicit unlock and every record/runtime "
            "argument."
        )
    for label, value in (
        ("loader_threads", loader_threads),
        ("loader_prefetch_samples", loader_prefetch_samples),
        ("torch_cpu_threads", torch_cpu_threads),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"Prepared fast path {label} must be a positive integer.")
    assert prepared_fast_path_record is not None
    assert prepared_fast_path_record_sha256 is not None
    assert loader_threads is not None
    record = load_prepared_cache_record(
        prepared_fast_path_record,
        expected_sha256=prepared_fast_path_record_sha256,
        collection=collection,
        data_root=data_root,
    )
    current_source_sha256 = training_source_sha256()
    if record.prepared_cache_implementation_sha256 != current_source_sha256:
        raise RuntimeError(
            "Prepared cache was not built by the current executable training source."
        )
    snapshot = verify_prepared_cache_at_launch(
        record,
        collection=collection,
        data_root=data_root,
        max_workers=loader_threads,
    )
    return record, snapshot


def _prepared_input_provenance(
    *,
    record: PreparedCacheRecord,
    snapshot: PreparedSnapshot,
    loader_threads: int,
    loader_prefetch_samples: int,
    torch_cpu_threads: int,
) -> dict[str, str]:
    return {
        "prepared_fast_path_cache_id": snapshot.cache_id,
        "prepared_fast_path_record_path": str(record.path),
        "prepared_fast_path_record_sha256": record.sha256,
        "prepared_fast_path_record_fingerprint": record.record_fingerprint,
        "prepared_fast_path_snapshot_fingerprint": snapshot.snapshot_fingerprint,
        "prepared_fast_path_manifest_set_fingerprint": snapshot.manifest_set_fingerprint,
        "prepared_fast_path_implementation_sha256": (
            record.prepared_cache_implementation_sha256
        ),
        "prepared_fast_path_user_instruction_sha256": record.user_instruction_sha256,
        "prepared_fast_path_verification_policy": (
            "FULL_CONTENT_AT_LAUNCH_FSTAT_PER_FETCH"
        ),
        "prepared_fast_path_loader_threads": str(loader_threads),
        "prepared_fast_path_loader_prefetch_samples": str(loader_prefetch_samples),
        "torch_cpu_threads": str(torch_cpu_threads),
    }


def select_bounded_smoke_collection(
    collection: ManifestCollection,
    *,
    dataset_ids: Sequence[str],
    supported_class_ids: Mapping[str, frozenset[int]],
) -> ManifestCollection:
    """Select at most two samples per dataset covering Pool and source-specific labels."""

    selected = []
    for dataset_id in dataset_ids:
        candidates = tuple(
            sample for sample in collection.samples if sample.source_dataset == dataset_id
        )
        if not candidates:
            raise RuntimeError(f"Real smoke manifest set has no {dataset_id!r} training sample.")
        other_support = set().union(
            *(
                supported_class_ids[other]
                for other in dataset_ids
                if other != dataset_id
            )
        )
        source_specific = supported_class_ids[dataset_id] - other_support - {0, POOL_CLASS_ID}
        joint_sample = next(
            (
                sample
                for sample in candidates
                if sample.class_counts.get(POOL_CLASS_ID, 0) > 0
                and any(
                    sample.class_counts.get(class_id, 0) > 0
                    for class_id in source_specific
                )
            ),
            None,
        )
        pool_sample = next(
            (sample for sample in candidates if sample.class_counts.get(POOL_CLASS_ID, 0) > 0),
            None,
        )
        source_specific_sample = next(
            (
                sample
                for sample in candidates
                if any(sample.class_counts.get(class_id, 0) > 0 for class_id in source_specific)
            ),
            None,
        )
        if pool_sample is None or source_specific_sample is None:
            raise RuntimeError(
                f"Real smoke requires Pool and source-specific class coverage for {dataset_id!r}."
            )
        candidates_to_add = (
            (joint_sample,)
            if joint_sample is not None
            else (pool_sample, source_specific_sample)
        )
        for sample in candidates_to_add:
            if sample is None:
                raise AssertionError("Smoke coverage candidates unexpectedly disappeared.")
            if sample not in selected:
                selected.append(sample)
    if any(sum(sample.class_counts.values()) <= 0 for sample in selected):
        raise RuntimeError("Real smoke selected a sample with no supervised pixels.")
    return ManifestCollection(manifests=collection.manifests, samples=tuple(selected))


def validate_bounded_smoke_coverage(
    batch: Mapping[str, Any],
    *,
    dataset_ids: Sequence[str],
    supported_class_ids: Mapping[str, frozenset[int]],
    ignore_index: int,
) -> dict[str, list[int]]:
    """Prove the decoded, unaugmented target masks exercise the required semantics."""

    labels = batch["labels"]
    sources = tuple(str(item) for item in batch["source_dataset"])
    if labels.ndim != 3 or labels.shape[0] != len(sources):
        raise RuntimeError("Real smoke coverage batch has invalid labels/source metadata.")
    observed: dict[str, set[int]] = {dataset_id: set() for dataset_id in dataset_ids}
    for row, source in enumerate(sources):
        if source not in observed:
            raise RuntimeError(f"Unexpected real-smoke source dataset {source!r}.")
        observed[source].update(int(item) for item in torch.unique(labels[row]).tolist())
        observed[source].discard(ignore_index)
    for dataset_id in dataset_ids:
        other_support = set().union(
            *(
                supported_class_ids[other]
                for other in dataset_ids
                if other != dataset_id
            )
        )
        source_specific = supported_class_ids[dataset_id] - other_support - {0, POOL_CLASS_ID}
        if not observed[dataset_id]:
            raise RuntimeError(f"Real smoke has no supervised pixels for {dataset_id!r}.")
        if POOL_CLASS_ID not in observed[dataset_id]:
            raise RuntimeError(f"Real smoke did not decode Pool for {dataset_id!r}.")
        if not (observed[dataset_id] & source_specific):
            raise RuntimeError(
                f"Real smoke did not decode a source-specific class for {dataset_id!r}."
            )
    return {key: sorted(values) for key, values in observed.items()}


def require_disjoint_training_validation(
    train_collection: ManifestCollection,
    validation_collection: ManifestCollection,
) -> None:
    """Reject exact split leakage even when duplicate files use different IDs."""

    train_ids = {item.sample_id for item in train_collection.samples}
    validation_ids = {item.sample_id for item in validation_collection.samples}
    overlapping_ids = train_ids & validation_ids
    if overlapping_ids:
        raise RuntimeError(
            f"Train/validation sample IDs overlap: {sorted(overlapping_ids)[:10]}"
        )
    train_image_hashes = {
        item.target_image_hash or item.image_hash for item in train_collection.samples
    }
    validation_image_hashes = {
        item.target_image_hash or item.image_hash for item in validation_collection.samples
    }
    overlapping_hashes = train_image_hashes & validation_image_hashes
    if overlapping_hashes:
        raise RuntimeError(
            "Train/validation image SHA-256 values overlap: "
            f"{sorted(overlapping_hashes)[:10]}"
        )


def require_training_class_coverage(
    train_collection: ManifestCollection,
    *,
    num_labels: int,
) -> dict[int, int]:
    """Reject a one-shot run when any unified target class is absent from training."""

    totals = {class_id: 0 for class_id in range(num_labels)}
    for sample in train_collection.samples:
        for class_id, count in sample.class_counts.items():
            if class_id not in totals or count <= 0:
                raise RuntimeError("Training manifest contains invalid class-count evidence.")
            totals[class_id] += count
    missing = [class_id for class_id, count in totals.items() if count == 0]
    if missing:
        raise RuntimeError(f"Unified classes absent from frozen training manifests: {missing}")
    return totals

def _move_batch(batch: Mapping[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        batch["pixel_values"].to(device, non_blocking=True),
        batch["labels"].to(device, non_blocking=True),
        batch["class_availability"].to(device, non_blocking=True),
    )


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_function: PartialCrossEntropyLoss,
    device: torch.device,
    precision: str,
    num_labels: int,
    ignore_index: int,
) -> dict[str, Any]:
    """Run deterministic validation and return JSON-serializable metrics."""

    model.eval()
    metrics = SegmentationMetrics(num_labels=num_labels, ignore_index=ignore_index)
    loss_total = 0.0
    batch_count = 0
    with torch.inference_mode():
        for batch in loader:
            pixel_values, labels, availability = _move_batch(batch, device)
            with _autocast(device, precision):
                outputs = model(pixel_values=pixel_values)
                logits = logits_at_label_resolution(outputs.logits, labels)
                loss = loss_function(logits, labels, availability)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite validation loss encountered.")
            loss_total += float(loss)
            batch_count += 1
            metrics.update(logits.detach(), labels, availability)
    if batch_count == 0:
        raise RuntimeError("Validation loader is empty.")
    result = metrics.compute()
    result["loss"] = loss_total / batch_count
    return result


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    loss_function: PartialCrossEntropyLoss,
    device: torch.device,
    precision: str,
    accumulation_steps: int,
    gradient_clip_norm: float,
    global_step: int,
    num_labels: int,
    ignore_index: int,
) -> tuple[dict[str, Any], int]:
    model.train()
    metrics = SegmentationMetrics(num_labels=num_labels, ignore_index=ignore_index)
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    batch_count = 0
    for batch_index, batch in enumerate(loader):
        pixel_values, labels, availability = _move_batch(batch, device)
        with _autocast(device, precision):
            outputs = model(pixel_values=pixel_values)
            logits = logits_at_label_resolution(outputs.logits, labels)
            loss = loss_function(logits, labels, availability)
            scaled_loss = loss / accumulation_steps
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite training loss encountered.")
        scaler.scale(scaled_loss).backward()
        last_batch = batch_index + 1 == len(loader)
        if (batch_index + 1) % accumulation_steps == 0 or last_batch:
            scaler.unscale_(optimizer)
            clip_grad_norm_(
                model.parameters(),
                gradient_clip_norm,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        loss_total += float(loss.detach())
        batch_count += 1
        metrics.update(logits.detach(), labels, availability)
    if batch_count == 0:
        raise RuntimeError("Training loader is empty.")
    result = metrics.compute()
    result["loss"] = loss_total / batch_count
    return result, global_step


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
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


def prepare_run_directory(
    path: Path,
    *,
    resuming: bool,
    required_root: Path | None = None,
) -> Path:
    """Reserve a new run exclusively, or require the existing resume directory."""

    output = path.expanduser().resolve()
    if required_root is not None:
        root = required_root.expanduser().resolve()
        if output.parent != root or output == root:
            raise RuntimeError(
                f"Run directory must be one direct child of the frozen run root {root}: {output}"
            )
    if resuming:
        if not output.is_dir():
            raise RuntimeError(f"Resume run directory is missing: {output}")
        return output
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"Refusing to reuse an existing run directory: {output}") from exc
    return output


def acquire_process_run_lock(run_directory: Path) -> Path:
    """Hold a non-blocking advisory lock for this Python process's lifetime."""

    run_directory = run_directory.expanduser().resolve()
    if run_directory in _PROCESS_RUN_LOCKS:
        return run_directory / ".floodsight-segmentation-run.lock"
    lock_path = run_directory / ".floodsight-segmentation-run.lock"
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        handle = os.fdopen(descriptor, "r+", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Unable to open exclusive run lock: {lock_path}") from exc
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        handle.close()
        raise RuntimeError(
            f"Another process already owns the run directory: {run_directory}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    os.fsync(handle.fileno())
    _PROCESS_RUN_LOCKS[run_directory] = handle
    return lock_path


def require_approved_last_resume(
    output_dir: Path,
    resume_checkpoint: Path | None,
    config: SegmentationConfig,
) -> Path | None:
    """Allow continuation only from the approved run's exact ``last.pt``."""

    run_directory = output_dir.expanduser().resolve()
    if resume_checkpoint is None:
        return None
    expected = run_directory / config.output.last_checkpoint_filename
    raw_candidate = resume_checkpoint.expanduser()
    absolute_candidate = Path(os.path.abspath(raw_candidate))
    if (
        raw_candidate.is_symlink()
        or absolute_candidate != expected
        or absolute_candidate.name != "last.pt"
        or not absolute_candidate.is_file()
    ):
        raise RuntimeError(
            f"Resume requires the approved run checkpoint {expected}; found {absolute_candidate}."
        )
    candidate = absolute_candidate.resolve(strict=True)
    if candidate != expected:
        raise RuntimeError(
            f"Resume requires the approved run checkpoint {expected}; found {candidate}."
        )
    return candidate


def _transition_history_lineage(
    transition: CheckpointTransition,
) -> list[dict[str, Any]]:
    shared = {"transition_record_sha256": transition.record_sha256}
    return [
        {
            "start_epoch": 1,
            "end_epoch": transition.training_state.epoch,
            "input_provenance": dict(
                sorted(transition.predecessor_input_provenance.items())
            ),
            "authorization_provenance": dict(
                sorted(transition.predecessor_authorization_provenance.items())
            ),
            **shared,
        },
        {
            "start_epoch": transition.training_state.epoch + 1,
            "end_epoch": None,
            "input_provenance": dict(
                sorted(transition.successor_input_provenance.items())
            ),
            "authorization_provenance": dict(
                sorted(transition.successor_authorization_provenance.items())
            ),
            **shared,
        },
    ]


def _existing_history_lineage(path: Path) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load resume history lineage: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Invalid resume history lineage envelope: {path}")
    if payload.get("schema_version") == "training-history-v1":
        return None
    lineage = payload.get("execution_lineage")
    if not isinstance(lineage, list):
        raise RuntimeError(f"Invalid version-2 resume history lineage: {path}")
    return [dict(item) for item in lineage if isinstance(item, Mapping)]


def _load_resume_history(
    path: Path,
    *,
    state: TrainingState,
    config_sha256: str,
    manifest_sha256: Mapping[str, str],
    manifest_fingerprint: Mapping[str, str],
    taxonomy_sha256: Mapping[str, str],
    input_provenance: Mapping[str, str],
    authorization_provenance: Mapping[str, str],
    updates_per_epoch: int,
    configured_epochs: int,
    scheduler_last_epoch: int,
    monitor_metric: str,
    maximize_metric: bool,
    predecessor_input_provenance: Mapping[str, str] | None = None,
    predecessor_authorization_provenance: Mapping[str, str] | None = None,
    transition_record_sha256: str | None = None,
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load resume history: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") not in {
        "training-history-v1",
        "training-history-v2",
    }:
        raise RuntimeError(f"Unsupported resume history: {path}")
    schema_version = payload["schema_version"]
    if payload.get("config_sha256") != config_sha256:
        raise RuntimeError("Resume history configuration hash does not match.")
    if payload.get("manifest_sha256") != dict(sorted(manifest_sha256.items())):
        raise RuntimeError("Resume history manifest hashes do not match.")
    if payload.get("manifest_fingerprint") != dict(sorted(manifest_fingerprint.items())):
        raise RuntimeError("Resume history manifest fingerprints do not match.")
    if payload.get("manifest_set_fingerprint") != _manifest_set_fingerprint(
        manifest_sha256, manifest_fingerprint
    ):
        raise RuntimeError("Resume history manifest-set fingerprint does not match.")
    if payload.get("taxonomy_sha256") != dict(sorted(taxonomy_sha256.items())):
        raise RuntimeError("Resume history taxonomy/mapping hashes do not match.")
    expected_history_input = (
        predecessor_input_provenance
        if schema_version == "training-history-v1"
        and predecessor_input_provenance is not None
        else input_provenance
    )
    expected_history_authorization = (
        predecessor_authorization_provenance
        if schema_version == "training-history-v1"
        and predecessor_authorization_provenance is not None
        else authorization_provenance
    )
    if payload.get("input_provenance") != dict(
        sorted(expected_history_input.items())
    ):
        raise RuntimeError("Resume history model/approval provenance does not match.")
    if payload.get("authorization_provenance") != dict(
        sorted(expected_history_authorization.items())
    ):
        raise RuntimeError("Resume history training authorization provenance does not match.")
    if schema_version == "training-history-v2":
        lineage = payload.get("execution_lineage")
        if not isinstance(lineage, list) or len(lineage) != 2:
            raise RuntimeError("Version-2 resume history execution lineage is invalid.")
        predecessor_segment, successor_segment = lineage
        required_lineage_keys = {
            "start_epoch",
            "end_epoch",
            "input_provenance",
            "authorization_provenance",
            "transition_record_sha256",
        }
        if (
            not isinstance(predecessor_segment, Mapping)
            or not isinstance(successor_segment, Mapping)
            or set(predecessor_segment) != required_lineage_keys
            or set(successor_segment) != required_lineage_keys
            or predecessor_segment.get("start_epoch") != 1
            or not isinstance(predecessor_segment.get("end_epoch"), int)
            or successor_segment.get("start_epoch")
            != predecessor_segment["end_epoch"] + 1
            or successor_segment.get("end_epoch") is not None
            or successor_segment.get("input_provenance")
            != dict(sorted(input_provenance.items()))
            or successor_segment.get("authorization_provenance")
            != dict(sorted(authorization_provenance.items()))
            or predecessor_segment.get("transition_record_sha256")
            != successor_segment.get("transition_record_sha256")
        ):
            raise RuntimeError("Version-2 resume history execution lineage drifted.")
        if (
            predecessor_input_provenance is not None
            and predecessor_segment.get("input_provenance")
            != dict(sorted(predecessor_input_provenance.items()))
        ) or (
            predecessor_authorization_provenance is not None
            and predecessor_segment.get("authorization_provenance")
            != dict(sorted(predecessor_authorization_provenance.items()))
        ):
            raise RuntimeError("Resume history predecessor lineage does not match.")
        lineage_transition = predecessor_segment["transition_record_sha256"]
        if (
            not isinstance(lineage_transition, str)
            or len(lineage_transition) != 64
            or any(character not in "0123456789abcdef" for character in lineage_transition)
            or (
                transition_record_sha256 is not None
                and lineage_transition != transition_record_sha256
            )
        ):
            raise RuntimeError("Resume history source-transition identity does not match.")
    if state.epoch <= 0 or state.epoch > configured_epochs:
        raise RuntimeError(
            "Resume checkpoint epoch must be positive and no greater than the configured "
            "epoch count."
        )
    expected_global_step = state.epoch * updates_per_epoch
    if state.global_step != expected_global_step:
        raise RuntimeError(
            "Resume checkpoint global_step is inconsistent with its completed epochs."
        )
    if scheduler_last_epoch != state.global_step:
        raise RuntimeError("Resume scheduler position is inconsistent with checkpoint global_step.")
    epochs = payload.get("epochs")
    if not isinstance(epochs, list) or len(epochs) < state.epoch:
        history_epoch = len(epochs) if isinstance(epochs, list) else "invalid"
        raise RuntimeError(
            f"Resume history/checkpoint epoch mismatch: history={history_epoch}, "
            f"checkpoint={state.epoch}."
        )
    retained = epochs[: state.epoch]
    monitored_values: list[float] = []
    for expected_epoch, record in enumerate(retained, start=1):
        if (
            not isinstance(record, dict)
            or record.get("epoch") != expected_epoch
            or record.get("global_step") != expected_epoch * updates_per_epoch
        ):
            raise RuntimeError("Resume history has inconsistent epoch/global-step records.")
        validation = record.get("validation")
        if validation is not None:
            if not isinstance(validation, Mapping):
                raise RuntimeError("Resume history contains invalid validation metrics.")
            monitored = validation.get(monitor_metric)
            if (
                isinstance(monitored, bool)
                or not isinstance(monitored, (int, float))
                or not math.isfinite(float(monitored))
            ):
                raise RuntimeError(
                    "Resume history is missing a finite monitored validation metric."
                )
            monitored_values.append(float(monitored))
    expected_best = (
        (max(monitored_values) if maximize_metric else min(monitored_values))
        if monitored_values
        else None
    )
    if state.best_metric != expected_best:
        raise RuntimeError(
            "Resume checkpoint best metric is inconsistent with retained validation history."
        )
    # History is persisted before the atomic epoch checkpoint. A process
    # failure in that narrow window may leave history ahead of the selected
    # checkpoint; truncating to its epoch makes continuation crash-safe.
    return retained


def run_training(
    *,
    config_path: Path,
    data_root: Path,
    train_manifest_specs: Sequence[ManifestSpec],
    validation_manifest_specs: Sequence[ManifestSpec],
    output_dir: Path,
    device_name: str,
    resume_checkpoint: Path | None,
    model_artifact_spec: ModelArtifactSpec,
    approval_record: Path,
    approval_record_sha256: str,
    allow_training: bool,
    prepared_fast_path_record: Path | None = None,
    prepared_fast_path_record_sha256: str | None = None,
    allow_prepared_fast_path: bool = False,
    loader_threads: int | None = None,
    loader_prefetch_samples: int | None = None,
    torch_cpu_threads: int | None = None,
    source_transition_record: Path | None = None,
    source_transition_record_sha256: str | None = None,
) -> dict[str, Any]:
    """Run real training only after all explicit immutable-input gates pass."""

    require_training_authorization(allow_training)
    runtime_versions = validate_runtime_versions()
    config = load_config(config_path)
    resolved_output_dir = output_dir.expanduser().resolve()
    if resolved_output_dir.parent != config.output.run_root:
        raise RuntimeError(
            "Training output must be one direct child of the frozen segmentation run root."
        )
    all_manifest_specs = [*train_manifest_specs, *validation_manifest_specs]
    approval = validate_human_approval(
        approval_record,
        expected_record_sha256=approval_record_sha256,
        operation="TRAIN",
        config_sha256=config.sha256,
        manifest_specs=all_manifest_specs,
        taxonomy_sha256=config.taxonomy_assets.hashes,
        run_directory=resolved_output_dir,
        model_id=config.model.pretrained_model_name_or_path,
        model_revision=config.model.revision,
        model_artifact=model_artifact_spec,
        real_smoke_root=config.output.real_smoke_root,
    )
    model_artifact = validate_model_artifact(model_artifact_spec, config.model)
    resume_checkpoint = require_approved_last_resume(
        resolved_output_dir,
        resume_checkpoint,
        config,
    )
    output_dir = prepare_run_directory(
        resolved_output_dir,
        resuming=resume_checkpoint is not None,
        required_root=config.output.run_root,
    )
    run_lock_path = acquire_process_run_lock(output_dir)
    train_collection = _load_collection(
        train_manifest_specs,
        config,
        splits=config.data.train_splits,
    )
    validation_collection = _load_collection(
        validation_manifest_specs,
        config,
        splits=config.data.validation_splits,
    )
    require_disjoint_training_validation(train_collection, validation_collection)
    training_class_counts = require_training_class_coverage(
        train_collection,
        num_labels=config.model.num_labels,
    )
    combined_collection = _combined_collection(train_collection, validation_collection)
    prepared_record, prepared_snapshot = _prepare_fast_path(
        collection=combined_collection,
        data_root=data_root,
        prepared_fast_path_record=prepared_fast_path_record,
        prepared_fast_path_record_sha256=prepared_fast_path_record_sha256,
        allow_prepared_fast_path=allow_prepared_fast_path,
        loader_threads=loader_threads,
        loader_prefetch_samples=loader_prefetch_samples,
        torch_cpu_threads=torch_cpu_threads,
    )
    transition_supplied = (
        source_transition_record is not None
        or source_transition_record_sha256 is not None
    )
    if transition_supplied and (
        source_transition_record is None
        or source_transition_record_sha256 is None
        or resume_checkpoint is None
        or prepared_snapshot is None
    ):
        raise RuntimeError(
            "Source transition requires its record/hash, an exact last.pt resume, and "
            "the launch-verified prepared fast path."
        )
    if prepared_snapshot is not None:
        assert torch_cpu_threads is not None
        torch.set_num_threads(torch_cpu_threads)
    seed_everything(
        config.reproducibility.seed,
        deterministic_algorithms=config.reproducibility.deterministic_algorithms,
        cudnn_benchmark=config.reproducibility.cudnn_benchmark,
    )
    data_generator = make_generator(config.reproducibility.seed)
    train_dataset = _build_dataset(
        train_collection,
        config,
        data_root=data_root,
        training=True,
        prepared_snapshot=prepared_snapshot,
    )
    validation_dataset = _build_dataset(
        validation_collection,
        config,
        data_root=data_root,
        training=False,
        prepared_snapshot=prepared_snapshot,
    )
    sampler = build_dataset_balanced_sampler(
        train_dataset,
        target_mix=config.sampler.dataset_mix,
        generator=data_generator,
        replacement=config.sampler.replacement,
        num_samples_policy=config.sampler.num_samples_policy,
    )
    if prepared_snapshot is None:
        loader_options = {
            "batch_size": config.training.batch_size,
            "num_workers": config.training.num_workers,
            "pin_memory": True,
            "worker_init_fn": seed_worker,
            "generator": data_generator,
            # Recreating workers each epoch makes epoch-boundary RNG resume exact.
            "persistent_workers": False,
        }
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            drop_last=False,
            **loader_options,
        )
        validation_loader = DataLoader(
            validation_dataset,
            shuffle=False,
            drop_last=False,
            **loader_options,
        )
    else:
        assert loader_threads is not None
        assert loader_prefetch_samples is not None
        train_loader = ThreadedSampleLoader(
            train_dataset,
            sampler=sampler,
            batch_size=config.training.batch_size,
            num_threads=loader_threads,
            prefetch_samples=loader_prefetch_samples,
            drop_last=False,
            pin_memory=True,
        )
        validation_loader = ThreadedSampleLoader(
            validation_dataset,
            sampler=SequentialSampler(validation_dataset),
            batch_size=config.training.batch_size,
            num_threads=loader_threads,
            prefetch_samples=loader_prefetch_samples,
            drop_last=False,
            pin_memory=True,
        )
    device = _resolve_device(device_name)
    require_h100(device)
    device_provenance = _device_provenance(device, precision=config.training.precision)
    model = build_segformer(config.model, model_artifact).to(device)
    optimizer = build_optimizer(model, config.optimizer)
    updates_per_epoch = math.ceil(
        len(train_loader) / config.training.gradient_accumulation_steps
    )
    scheduler = build_scheduler(
        optimizer,
        config.scheduler,
        total_steps=updates_per_epoch * config.training.epochs,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and config.training.precision == "fp16",
    )
    loss_function = PartialCrossEntropyLoss(
        class_weights=config.loss.class_weights,
        ignore_index=config.data.ignore_index,
    )
    manifest_hashes = {**train_collection.hashes, **validation_collection.hashes}
    manifest_fingerprints = {
        **train_collection.fingerprints,
        **validation_collection.fingerprints,
    }
    manifest_set_fingerprint = _manifest_set_fingerprint(
        manifest_hashes,
        manifest_fingerprints,
    )
    taxonomy_hashes = config.taxonomy_assets.hashes
    input_provenance = {
        "model_revision": model_artifact.source_revision,
        "model_safetensors_path": str(model_artifact.safetensors_path),
        "model_safetensors_sha256": model_artifact.safetensors_sha256,
        "model_provenance_path": str(model_artifact.provenance_path),
        "model_provenance_sha256": model_artifact.provenance_sha256,
        "training_source_sha256": approval.training_code_sha256,
        "runtime_fingerprint": _runtime_fingerprint(runtime_versions),
        "device_name": str(device_provenance["device_name"]),
        "device_capability": ".".join(
            str(item) for item in device_provenance["device_capability"]
        ),
        "effective_precision": config.training.precision,
        **_runtime_provenance(runtime_versions),
    }
    if prepared_record is not None and prepared_snapshot is not None:
        assert loader_threads is not None
        assert loader_prefetch_samples is not None
        assert torch_cpu_threads is not None
        input_provenance.update(
            _prepared_input_provenance(
                record=prepared_record,
                snapshot=prepared_snapshot,
                loader_threads=loader_threads,
                loader_prefetch_samples=loader_prefetch_samples,
                torch_cpu_threads=torch_cpu_threads,
            )
        )
    authorization_provenance = {
        "approval_record_sha256": approval.record_sha256,
        "human_review_sha256": approval.human_review_sha256,
        "real_smoke_report_sha256": approval.real_smoke_sha256,
    }
    state = TrainingState(epoch=0, global_step=0, best_metric=None)
    validated_transition: CheckpointTransition | None = None
    if resume_checkpoint is not None:
        if transition_supplied:
            assert source_transition_record is not None
            assert source_transition_record_sha256 is not None
            validated_transition = validate_checkpoint_transition(
                source_transition_record,
                expected_record_sha256=source_transition_record_sha256,
                expected_predecessor_checkpoint=resume_checkpoint,
                expected_config_sha256=config.sha256,
                expected_manifest_sha256=manifest_hashes,
                expected_manifest_fingerprint=manifest_fingerprints,
                expected_taxonomy_sha256=taxonomy_hashes,
                expected_successor_input_provenance=input_provenance,
                expected_successor_authorization_provenance=(
                    authorization_provenance
                ),
                expected_run_directory=resolved_output_dir,
            )
            state = load_checkpoint_transition(
                resume_checkpoint,
                transition_record_path=source_transition_record,
                expected_transition_record_sha256=(
                    source_transition_record_sha256
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                expected_config_sha256=config.sha256,
                expected_manifest_sha256=manifest_hashes,
                expected_manifest_fingerprint=manifest_fingerprints,
                expected_taxonomy_sha256=taxonomy_hashes,
                expected_input_provenance=input_provenance,
                expected_authorization_provenance=authorization_provenance,
                expected_run_directory=resolved_output_dir,
                data_generator=data_generator,
                map_location=device,
            )
        else:
            state = load_checkpoint(
                resume_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                expected_config_sha256=config.sha256,
                expected_manifest_sha256=manifest_hashes,
                expected_manifest_fingerprint=manifest_fingerprints,
                expected_taxonomy_sha256=taxonomy_hashes,
                expected_input_provenance=input_provenance,
                expected_authorization_provenance=authorization_provenance,
                expected_run_directory=resolved_output_dir,
                data_generator=data_generator,
                map_location=device,
                expected_provenance="REAL_ML_OUTPUT",
            )
    history_path = output_dir / config.output.history_filename
    history: list[dict[str, Any]] = (
        _load_resume_history(
            history_path,
            state=state,
            config_sha256=config.sha256,
            manifest_sha256=manifest_hashes,
            manifest_fingerprint=manifest_fingerprints,
            taxonomy_sha256=taxonomy_hashes,
            input_provenance=input_provenance,
            authorization_provenance=authorization_provenance,
            updates_per_epoch=updates_per_epoch,
            configured_epochs=config.training.epochs,
            scheduler_last_epoch=scheduler.last_epoch,
            monitor_metric=config.training.monitor_metric,
            maximize_metric=config.training.maximize_metric,
            predecessor_input_provenance=(
                validated_transition.predecessor_input_provenance
                if validated_transition is not None
                else None
            ),
            predecessor_authorization_provenance=(
                validated_transition.predecessor_authorization_provenance
                if validated_transition is not None
                else None
            ),
            transition_record_sha256=(
                validated_transition.record_sha256
                if validated_transition is not None
                else None
            ),
        )
        if resume_checkpoint is not None
        else []
    )
    history_lineage = (
        _transition_history_lineage(validated_transition)
        if validated_transition is not None
        else (
            _existing_history_lineage(history_path)
            if resume_checkpoint is not None
            else None
        )
    )
    best_metric = state.best_metric
    global_step = state.global_step
    for epoch in range(state.epoch, config.training.epochs):
        training_metrics, global_step = _train_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loss_function=loss_function,
            device=device,
            precision=config.training.precision,
            accumulation_steps=config.training.gradient_accumulation_steps,
            gradient_clip_norm=config.training.gradient_clip_norm,
            global_step=global_step,
            num_labels=config.model.num_labels,
            ignore_index=config.data.ignore_index,
        )
        validation_metrics = None
        if (epoch + 1) % config.training.validate_every_epochs == 0:
            validation_metrics = validate_epoch(
                model,
                validation_loader,
                loss_function=loss_function,
                device=device,
                precision=config.training.precision,
                num_labels=config.model.num_labels,
                ignore_index=config.data.ignore_index,
            )
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "training": training_metrics,
            "validation": validation_metrics,
            "provenance": "REAL_ML_OUTPUT",
        }
        history.append(record)
        _write_json(
            history_path,
            {
                "schema_version": (
                    "training-history-v2"
                    if history_lineage is not None
                    else "training-history-v1"
                ),
                "provenance": "REAL_ML_OUTPUT",
                "config_sha256": config.sha256,
                "manifest_sha256": dict(sorted(manifest_hashes.items())),
                "manifest_fingerprint": dict(sorted(manifest_fingerprints.items())),
                "manifest_set_fingerprint": manifest_set_fingerprint,
                "taxonomy_sha256": dict(sorted(taxonomy_hashes.items())),
                "input_provenance": dict(sorted(input_provenance.items())),
                "authorization_provenance": dict(
                    sorted(authorization_provenance.items())
                ),
                **(
                    {"execution_lineage": history_lineage}
                    if history_lineage is not None
                    else {}
                ),
                "epochs": history,
            },
        )
        monitored = validation_metrics and validation_metrics.get(config.training.monitor_metric)
        improved = monitored is not None and (
            best_metric is None
            or (
                monitored > best_metric
                if config.training.maximize_metric
                else monitored < best_metric
            )
        )
        if improved:
            best_metric = float(monitored)
        checkpoint_state = TrainingState(
            epoch=epoch + 1,
            global_step=global_step,
            best_metric=best_metric,
        )
        # Write an improved best checkpoint before last.pt. The durable last.pt
        # is the epoch commit marker; a crash before it causes deterministic
        # replay from the preceding epoch instead of losing the best artifact.
        if improved:
            save_checkpoint(
                output_dir / config.output.best_checkpoint_filename,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                training_state=checkpoint_state,
                config_sha256=config.sha256,
                manifest_sha256=manifest_hashes,
                manifest_fingerprint=manifest_fingerprints,
                taxonomy_sha256=taxonomy_hashes,
                input_provenance=input_provenance,
                authorization_provenance=authorization_provenance,
                run_directory=output_dir,
                data_generator=data_generator,
                provenance="REAL_ML_OUTPUT",
            )
        if (epoch + 1) % config.training.checkpoint_every_epochs == 0:
            save_checkpoint(
                output_dir / config.output.last_checkpoint_filename,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                training_state=checkpoint_state,
                config_sha256=config.sha256,
                manifest_sha256=manifest_hashes,
                manifest_fingerprint=manifest_fingerprints,
                taxonomy_sha256=taxonomy_hashes,
                input_provenance=input_provenance,
                authorization_provenance=authorization_provenance,
                run_directory=output_dir,
                data_generator=data_generator,
                provenance="REAL_ML_OUTPUT",
            )
    report = {
        "status": "COMPLETE",
        "provenance": "REAL_ML_OUTPUT",
        "config_sha256": config.sha256,
        "manifest_sha256": manifest_hashes,
        "manifest_fingerprint": manifest_fingerprints,
        "manifest_set_fingerprint": manifest_set_fingerprint,
        "manifest_fingerprint_algorithm": config.data.manifest_fingerprint_algorithm,
        "taxonomy_sha256": taxonomy_hashes,
        "training_class_counts": training_class_counts,
        "loss_policy": {
            "name": config.loss.name,
            "class_weight_policy": config.loss.class_weight_policy,
            "normalization": config.loss.normalization,
            "class_weights": list(config.loss.class_weights),
        },
        "sampler_policy": {
            "name": config.sampler.name,
            "replacement": config.sampler.replacement,
            "num_samples_policy": config.sampler.num_samples_policy,
            "dataset_mix": dict(config.sampler.dataset_mix),
        },
        "model_safetensors_sha256": model_artifact.safetensors_sha256,
        "model_safetensors_path": str(model_artifact.safetensors_path),
        "model_provenance_path": str(model_artifact.provenance_path),
        "model_provenance_sha256": model_artifact.provenance_sha256,
        "model_revision": model_artifact.source_revision,
        "approval_id": approval.approval_id,
        "approval_record_path": str(approval.record_path),
        "approval_record_sha256": approval.record_sha256,
        "human_review_path": str(approval.human_review_path),
        "human_review_sha256": approval.human_review_sha256,
        "real_smoke_report_path": str(approval.real_smoke_path),
        "real_smoke_report_sha256": approval.real_smoke_sha256,
        "training_source_sha256": approval.training_code_sha256,
        "prepared_fast_path": (
            {
                "enabled": True,
                "record_path": str(prepared_record.path),
                "record_sha256": prepared_record.sha256,
                "record_fingerprint": prepared_record.record_fingerprint,
                "snapshot_fingerprint": prepared_snapshot.snapshot_fingerprint,
                "verification_policy": "FULL_CONTENT_AT_LAUNCH_FSTAT_PER_FETCH",
                "per_sample_hashing": False,
                "per_sample_source_mask_access": False,
                "per_sample_static_remap_or_pixel_counts": False,
                "loader_threads": loader_threads,
                "loader_prefetch_samples": loader_prefetch_samples,
                "torch_cpu_threads": torch_cpu_threads,
            }
            if prepared_record is not None and prepared_snapshot is not None
            else {"enabled": False}
        ),
        "source_transition": (
            {
                "record_path": str(validated_transition.record_path),
                "record_sha256": validated_transition.record_sha256,
                "allowed_change": validated_transition.allowed_change,
                "predecessor_checkpoint_sha256": (
                    validated_transition.predecessor_checkpoint_sha256
                ),
                "predecessor_epoch": validated_transition.training_state.epoch,
                "predecessor_global_step": (
                    validated_transition.training_state.global_step
                ),
            }
            if validated_transition is not None
            else None
        ),
        "runtime_versions": runtime_versions,
        "runtime_fingerprint": _runtime_fingerprint(runtime_versions),
        "device": device_provenance,
        "config_path": str(config.path),
        "epochs_completed": config.training.epochs,
        "global_step": global_step,
        "best_metric": best_metric,
        "output_dir": str(output_dir),
        "run_root": str(config.output.run_root),
        "resume_policy": config.output.resume_policy,
        "resumed_from": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "run_lock": str(run_lock_path),
    }
    _write_json(output_dir / config.output.report_filename, report)
    return report


def run_validation(
    *,
    config_path: Path,
    data_root: Path,
    manifest_specs: Sequence[ManifestSpec],
    checkpoint_path: Path,
    device_name: str,
    model_artifact_spec: ModelArtifactSpec,
    approval_record: Path,
    approval_record_sha256: str,
    allow_training: bool,
) -> dict[str, Any]:
    """Validate a real checkpoint behind the same explicit authorization gate."""

    require_training_authorization(allow_training)
    runtime_versions = validate_runtime_versions()
    config = load_config(config_path)
    resolved_checkpoint = checkpoint_path.expanduser().resolve()
    run_directory = resolved_checkpoint.parent
    if run_directory.parent != config.output.run_root:
        raise RuntimeError("Validation checkpoint is outside the frozen segmentation run root.")
    approval = validate_human_approval(
        approval_record,
        expected_record_sha256=approval_record_sha256,
        operation="VALIDATE",
        config_sha256=config.sha256,
        manifest_specs=manifest_specs,
        taxonomy_sha256=config.taxonomy_assets.hashes,
        run_directory=run_directory,
        model_id=config.model.pretrained_model_name_or_path,
        model_revision=config.model.revision,
        model_artifact=model_artifact_spec,
        real_smoke_root=config.output.real_smoke_root,
    )
    model_artifact = validate_model_artifact(model_artifact_spec, config.model)
    collection = _load_collection(
        manifest_specs,
        config,
        splits=config.data.validation_splits,
    )
    seed_everything(
        config.reproducibility.seed,
        deterministic_algorithms=config.reproducibility.deterministic_algorithms,
        cudnn_benchmark=config.reproducibility.cudnn_benchmark,
    )
    generator = make_generator(config.reproducibility.seed)
    dataset = _build_dataset(collection, config, data_root=data_root, training=False)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )
    device = _resolve_device(device_name)
    require_h100(device)
    device_provenance = _device_provenance(device, precision=config.training.precision)
    model = build_segformer(config.model, model_artifact).to(device)
    input_provenance = {
        "model_revision": model_artifact.source_revision,
        "model_safetensors_path": str(model_artifact.safetensors_path),
        "model_safetensors_sha256": model_artifact.safetensors_sha256,
        "model_provenance_path": str(model_artifact.provenance_path),
        "model_provenance_sha256": model_artifact.provenance_sha256,
        "training_source_sha256": approval.training_code_sha256,
        "runtime_fingerprint": _runtime_fingerprint(runtime_versions),
        "device_name": str(device_provenance["device_name"]),
        "device_capability": ".".join(
            str(item) for item in device_provenance["device_capability"]
        ),
        "effective_precision": config.training.precision,
        **_runtime_provenance(runtime_versions),
    }
    load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        expected_config_sha256=config.sha256,
        expected_manifest_sha256=collection.hashes,
        expected_manifest_fingerprint=collection.fingerprints,
        expected_taxonomy_sha256=config.taxonomy_assets.hashes,
        expected_input_provenance=input_provenance,
        expected_run_directory=run_directory,
        data_generator=generator,
        map_location=device,
        expected_provenance="REAL_ML_OUTPUT",
        allow_manifest_subset=True,
    )
    metrics = validate_epoch(
        model,
        loader,
        loss_function=PartialCrossEntropyLoss(
            class_weights=config.loss.class_weights,
            ignore_index=config.data.ignore_index,
        ),
        device=device,
        precision=config.training.precision,
        num_labels=config.model.num_labels,
        ignore_index=config.data.ignore_index,
    )
    return {
        "status": "COMPLETE",
        "provenance": "REAL_ML_OUTPUT",
        "config_sha256": config.sha256,
        "manifest_sha256": collection.hashes,
        "manifest_fingerprint": collection.fingerprints,
        "manifest_set_fingerprint": collection.set_fingerprint,
        "manifest_fingerprint_algorithm": config.data.manifest_fingerprint_algorithm,
        "taxonomy_sha256": config.taxonomy_assets.hashes,
        "loss_policy": {
            "name": config.loss.name,
            "class_weight_policy": config.loss.class_weight_policy,
            "normalization": config.loss.normalization,
            "class_weights": list(config.loss.class_weights),
        },
        "model_safetensors_sha256": model_artifact.safetensors_sha256,
        "model_safetensors_path": str(model_artifact.safetensors_path),
        "model_provenance_path": str(model_artifact.provenance_path),
        "model_provenance_sha256": model_artifact.provenance_sha256,
        "model_revision": model_artifact.source_revision,
        "approval_id": approval.approval_id,
        "approval_record_path": str(approval.record_path),
        "approval_record_sha256": approval.record_sha256,
        "human_review_path": str(approval.human_review_path),
        "human_review_sha256": approval.human_review_sha256,
        "real_smoke_report_path": str(approval.real_smoke_path),
        "real_smoke_report_sha256": approval.real_smoke_sha256,
        "training_source_sha256": approval.training_code_sha256,
        "runtime_versions": runtime_versions,
        "runtime_fingerprint": _runtime_fingerprint(runtime_versions),
        "device": device_provenance,
        "config_path": str(config.path),
        "checkpoint": str(resolved_checkpoint),
        "run_root": str(config.output.run_root),
        "run_directory": str(run_directory),
        "metrics": metrics,
    }


def run_real_manifest_smoke(
    *,
    config_path: Path,
    data_root: Path,
    manifest_specs: Sequence[ManifestSpec],
    output_dir: Path,
    device_name: str,
    model_artifact_spec: ModelArtifactSpec,
    allow_real_smoke: bool,
    prepared_fast_path_record: Path | None = None,
    prepared_fast_path_record_sha256: str | None = None,
    allow_prepared_fast_path: bool = False,
    loader_threads: int | None = None,
    loader_prefetch_samples: int | None = None,
    torch_cpu_threads: int | None = None,
) -> dict[str, Any]:
    """Run one real-data step on a tiny Pool/source-specific sample set."""

    require_real_smoke_authorization(allow_real_smoke)
    runtime_versions = validate_runtime_versions()
    config = load_config(config_path)
    model_artifact = validate_model_artifact(model_artifact_spec, config.model)
    collection = _load_collection(
        manifest_specs,
        config,
        splits=config.data.train_splits,
    )
    bounded_collection = select_bounded_smoke_collection(
        collection,
        dataset_ids=config.data.allowed_datasets,
        supported_class_ids=config.data.supported_class_ids,
    )
    selected = bounded_collection.samples
    prepared_record, prepared_snapshot = _prepare_fast_path(
        collection=bounded_collection,
        data_root=data_root,
        prepared_fast_path_record=prepared_fast_path_record,
        prepared_fast_path_record_sha256=prepared_fast_path_record_sha256,
        allow_prepared_fast_path=allow_prepared_fast_path,
        loader_threads=loader_threads,
        loader_prefetch_samples=loader_prefetch_samples,
        torch_cpu_threads=torch_cpu_threads,
    )
    if prepared_snapshot is not None:
        assert torch_cpu_threads is not None
        torch.set_num_threads(torch_cpu_threads)
    output_dir = prepare_run_directory(
        output_dir,
        resuming=False,
        required_root=config.output.real_smoke_root,
    )
    run_lock_path = acquire_process_run_lock(output_dir)
    seed_everything(
        config.reproducibility.seed,
        deterministic_algorithms=config.reproducibility.deterministic_algorithms,
        cudnn_benchmark=config.reproducibility.cudnn_benchmark,
    )
    generator = make_generator(config.reproducibility.seed)
    training_dataset = _build_dataset(
        bounded_collection,
        config,
        data_root=data_root,
        training=True,
        prepared_snapshot=prepared_snapshot,
    )
    validation_dataset = _build_dataset(
        bounded_collection,
        config,
        data_root=data_root,
        training=False,
        prepared_snapshot=prepared_snapshot,
    )
    if prepared_snapshot is None:
        training_loader = DataLoader(
            training_dataset,
            batch_size=len(training_dataset),
            shuffle=False,
            num_workers=0,
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=len(validation_dataset),
            shuffle=False,
            num_workers=0,
            generator=generator,
        )
    else:
        assert loader_threads is not None
        assert loader_prefetch_samples is not None
        training_loader = ThreadedSampleLoader(
            training_dataset,
            sampler=SequentialSampler(training_dataset),
            batch_size=len(training_dataset),
            num_threads=loader_threads,
            prefetch_samples=loader_prefetch_samples,
            pin_memory=True,
        )
        validation_loader = ThreadedSampleLoader(
            validation_dataset,
            sampler=SequentialSampler(validation_dataset),
            batch_size=len(validation_dataset),
            num_threads=loader_threads,
            prefetch_samples=loader_prefetch_samples,
            pin_memory=True,
        )
    if len(training_loader) != 1 or len(validation_loader) != 1:
        raise RuntimeError("Real smoke loader must contain exactly one bounded batch.")
    device = _resolve_device(device_name)
    require_h100(device)
    precision = config.training.precision
    if precision != "bf16":
        raise RuntimeError("Production real smoke requires the frozen bf16 precision.")
    device_provenance = _device_provenance(device, precision=precision)
    model = build_segformer(config.model, model_artifact).to(device)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler, total_steps=1)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False,
    )
    loss_function = PartialCrossEntropyLoss(
        class_weights=config.loss.class_weights,
        ignore_index=config.data.ignore_index,
    )
    coverage_batch = next(iter(validation_loader))
    validation_coverage = validate_bounded_smoke_coverage(
        coverage_batch,
        dataset_ids=config.data.allowed_datasets,
        supported_class_ids=config.data.supported_class_ids,
        ignore_index=config.data.ignore_index,
    )
    batch = next(iter(training_loader))
    training_coverage = validate_bounded_smoke_coverage(
        batch,
        dataset_ids=config.data.allowed_datasets,
        supported_class_ids=config.data.supported_class_ids,
        ignore_index=config.data.ignore_index,
    )
    pixel_values, labels, availability = _move_batch(batch, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, precision):
        outputs = model(pixel_values=pixel_values)
        logits = logits_at_label_resolution(outputs.logits, labels)
        loss = loss_function(logits, labels, availability)
    if not bool(torch.isfinite(loss)) or float(loss.detach()) <= 0:
        raise RuntimeError("Real-smoke loss must be positive and finite.")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = clip_grad_norm_(
        model.parameters(),
        config.training.gradient_clip_norm,
        error_if_nonfinite=True,
    )
    if not math.isfinite(float(gradient_norm)):
        raise RuntimeError("Non-finite real-smoke gradients encountered.")
    if float(gradient_norm) <= 0:
        raise RuntimeError("Real-smoke gradient norm must be positive.")
    changed_parameter_name = None
    parameter_before_step = None
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)):
            changed_parameter_name = name
            parameter_before_step = parameter.detach().clone()
            break
    if changed_parameter_name is None or parameter_before_step is None:
        raise RuntimeError("Real smoke produced no nonzero trainable parameter gradient.")
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    changed_parameter = dict(model.named_parameters())[changed_parameter_name]
    parameter_changed = not bool(torch.equal(changed_parameter.detach(), parameter_before_step))
    if not parameter_changed:
        raise RuntimeError("Real-smoke optimizer step did not change a parameter with gradient.")
    optimizer_steps = 1
    validation_metrics = validate_epoch(
        model,
        validation_loader,
        loss_function=loss_function,
        device=device,
        precision=precision,
        num_labels=config.model.num_labels,
        ignore_index=config.data.ignore_index,
    )
    checkpoint_path = output_dir / "real-manifest-smoke.pt"
    input_provenance = {
        "model_revision": model_artifact.source_revision,
        "model_safetensors_path": str(model_artifact.safetensors_path),
        "model_safetensors_sha256": model_artifact.safetensors_sha256,
        "model_provenance_path": str(model_artifact.provenance_path),
        "model_provenance_sha256": model_artifact.provenance_sha256,
        "human_review_status": model_artifact.human_review_status,
        "training_source_sha256": training_source_sha256(),
        "runtime_fingerprint": _runtime_fingerprint(runtime_versions),
        "device_name": str(device_provenance["device_name"]),
        "device_capability": ".".join(
            str(item) for item in device_provenance["device_capability"]
        ),
        "effective_precision": precision,
        **_runtime_provenance(runtime_versions),
    }
    if prepared_record is not None and prepared_snapshot is not None:
        assert loader_threads is not None
        assert loader_prefetch_samples is not None
        assert torch_cpu_threads is not None
        input_provenance.update(
            _prepared_input_provenance(
                record=prepared_record,
                snapshot=prepared_snapshot,
                loader_threads=loader_threads,
                loader_prefetch_samples=loader_prefetch_samples,
                torch_cpu_threads=torch_cpu_threads,
            )
        )
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        training_state=TrainingState(epoch=0, global_step=1, best_metric=None),
        config_sha256=config.sha256,
        manifest_sha256=collection.hashes,
        manifest_fingerprint=collection.fingerprints,
        taxonomy_sha256=config.taxonomy_assets.hashes,
        input_provenance=input_provenance,
        authorization_provenance={},
        run_directory=output_dir,
        data_generator=generator,
        provenance="REAL_ML_OUTPUT",
    )
    fresh_process_probe = run_fresh_process_production_checkpoint_probe(
        output_dir / "fresh-process-production-checkpoint-probe",
        checkpoint_path=checkpoint_path,
        config_path=config.path,
        model_artifact_spec=model_artifact_spec,
        training_state=TrainingState(epoch=0, global_step=1, best_metric=None),
        manifest_sha256=collection.hashes,
        manifest_fingerprint=collection.fingerprints,
        taxonomy_sha256=config.taxonomy_assets.hashes,
        input_provenance=input_provenance,
        run_directory=output_dir,
        data_generator=generator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )
    result = {
        "status": "PASS",
        "artifact_type": "BOUNDED_REAL_MANIFEST_SMOKE",
        "provenance": "REAL_ML_OUTPUT",
        "full_training": False,
        "training_authorized": False,
        "human_review_status": model_artifact.human_review_status,
        "epochs": 0,
        "unique_samples": len(selected),
        "samples_per_dataset": {
            dataset_id: sum(sample.source_dataset == dataset_id for sample in selected)
            for dataset_id in config.data.allowed_datasets
        },
        "sample_ids": [sample.sample_id for sample in selected],
        "decoded_class_coverage": validation_coverage,
        "post_training_transform_class_coverage": training_coverage,
        "optimizer_steps": optimizer_steps,
        "training_transform": "PASS",
        "validation_transform": "PASS",
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "parameter_changed": parameter_changed,
        "changed_parameter_name": changed_parameter_name,
        "validation": validation_metrics,
        "validation_scope": "SMOKE_ONLY_SAME_BOUNDED_SAMPLES",
        "checkpoint_reload": "FRESH_PYTHON_PROCESS_PASS",
        "fresh_process_checkpoint_proof": fresh_process_probe,
        "config_sha256": config.sha256,
        "manifest_sha256": collection.hashes,
        "manifest_fingerprint": collection.fingerprints,
        "manifest_set_fingerprint": collection.set_fingerprint,
        "manifest_fingerprint_algorithm": config.data.manifest_fingerprint_algorithm,
        "taxonomy_sha256": config.taxonomy_assets.hashes,
        "loss_policy": {
            "name": config.loss.name,
            "class_weight_policy": config.loss.class_weight_policy,
            "normalization": config.loss.normalization,
            "class_weights": list(config.loss.class_weights),
        },
        "sampler_policy": {
            "name": config.sampler.name,
            "replacement": config.sampler.replacement,
            "num_samples_policy": config.sampler.num_samples_policy,
            "dataset_mix": dict(config.sampler.dataset_mix),
        },
        "frozen_training_run_root": str(config.output.run_root),
        "frozen_real_smoke_root": str(config.output.real_smoke_root),
        "run_lock": str(run_lock_path),
        "config_path": str(config.path),
        "model_revision": model_artifact.source_revision,
        "model_safetensors_path": str(model_artifact.safetensors_path),
        "model_safetensors_sha256": model_artifact.safetensors_sha256,
        "model_provenance_path": str(model_artifact.provenance_path),
        "model_provenance_sha256": model_artifact.provenance_sha256,
        "training_source_sha256": input_provenance["training_source_sha256"],
        "prepared_fast_path": (
            {
                "enabled": True,
                "record_path": str(prepared_record.path),
                "record_sha256": prepared_record.sha256,
                "record_fingerprint": prepared_record.record_fingerprint,
                "snapshot_fingerprint": prepared_snapshot.snapshot_fingerprint,
                "verification_policy": "FULL_CONTENT_AT_LAUNCH_FSTAT_PER_FETCH",
                "per_sample_hashing": False,
                "per_sample_source_mask_access": False,
                "per_sample_static_remap_or_pixel_counts": False,
                "loader_threads": loader_threads,
                "loader_prefetch_samples": loader_prefetch_samples,
                "torch_cpu_threads": torch_cpu_threads,
            }
            if prepared_record is not None and prepared_snapshot is not None
            else {"enabled": False}
        ),
        "runtime_versions": runtime_versions,
        "runtime_fingerprint": _runtime_fingerprint(runtime_versions),
        "device": device_provenance,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    _write_json(output_dir / "real-manifest-smoke.json", result)
    return result
