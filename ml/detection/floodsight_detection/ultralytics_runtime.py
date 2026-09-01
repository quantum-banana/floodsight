"""Lazy Ultralytics boundary for authorized training and synthetic smoke."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from floodsight_detection.approval import TrainingApproval, load_training_approval
from floodsight_detection.checkpointing import (
    TrustedCheckpoint,
    capture_rng_state,
    publish_checkpoint_generation,
    restore_live_checkpoint,
    restore_rng_state,
)
from floodsight_detection.config import (
    PINNED_MODEL,
    ULTRALYTICS_TRAIN_VAL_KEYS,
    TrainingConfig,
    validate_frozen_model_artifacts,
    validate_ultralytics_default_contract,
    version_matches,
)
from floodsight_detection.contract import (
    DatasetContract,
    freeze_dataset_contract,
    validate_dataset_contract,
)
from floodsight_detection.determinism import configure_determinism, require_prestarted_hash_seed
from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file, stable_sha256, training_source_sha256
from floodsight_detection.runs import (
    RunReservation,
    exclusive_run_lock,
    reserve_new_run,
    validate_resume,
)
from floodsight_detection.runtime import (
    bound_training_device,
    bound_training_output_root,
    validate_full_training_runtime,
)
from floodsight_detection.weights import (
    WeightArtifact,
    load_weight_audit,
    validate_training_license_disposition,
)

REAL_SMOKE_MAX_TRAIN_BATCHES = 8
REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS = 8
REAL_SMOKE_BOUNDED_OVERRIDES: dict[str, Any] = {}
REAL_SMOKE_AUGMENTATION_KEYS = frozenset(
    {
        "imgsz",
        "rect",
        "multi_scale",
        "close_mosaic",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "copy_paste_mode",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "fliplr",
        "flipud",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "bgr",
    }
)
_REAL_SMOKE_REQUIRED_TRANSFORMS = frozenset(
    {
        "Mosaic",
        "CopyPaste",
        "RandomPerspective",
        "MixUp",
        "CutMix",
        "Albumentations",
        "RandomHSV",
        "RandomFlip",
        "Format",
    }
)

_REAL_SMOKE_BATCH_POLICY = "min(configured_batch_size,dataset_size)"


def bounded_loader_batch_size(configured_batch_size: Any, dataset_size: Any) -> int:
    """Return Ultralytics' audited small-dataset batch normalization."""

    if (
        isinstance(configured_batch_size, bool)
        or not isinstance(configured_batch_size, int)
        or configured_batch_size < 1
        or isinstance(dataset_size, bool)
        or not isinstance(dataset_size, int)
        or not 1 <= dataset_size <= 8
    ):
        raise DetectionInfrastructureError(
            "Real-smoke loader batch inputs are outside the bounded contract.",
            code="real_smoke_augmentation_drift",
        )
    return min(configured_batch_size, dataset_size)


def optimizer_training_state(trainer: Any) -> dict[str, Any]:
    """Return a JSON-safe optimizer/scaler progress summary for smoke evidence."""

    optimizer = trainer.optimizer
    optimizer_state_dict = optimizer.state_dict()
    param_groups = []
    for group in optimizer_state_dict["param_groups"]:
        sanitized_group: dict[str, Any] = {}
        for key, value in group.items():
            if key == "params":
                sanitized_group["parameter_count"] = len(value)
            elif isinstance(value, tuple):
                sanitized_group[key] = list(value)
            else:
                sanitized_group[key] = value
        param_groups.append(sanitized_group)
    parameter_steps: list[float] = []
    for state in optimizer.state.values():
        step = state.get("step") if isinstance(state, dict) else None
        if hasattr(step, "numel") and callable(step.numel):
            step = step.detach().cpu().item() if step.numel() == 1 else None
        if isinstance(step, (int, float)) and not isinstance(step, bool):
            parameter_steps.append(float(step))
    scaler_state = trainer.scaler.state_dict()
    scale = scaler_state.get("scale") if isinstance(scaler_state, dict) else None
    return {
        "optimizer_name": type(optimizer).__name__,
        "optimizer_state_parameter_count": len(optimizer.state),
        "optimizer_state_entries": sum(
            len(state) for state in optimizer.state.values() if isinstance(state, dict)
        ),
        "optimizer_param_group_count": len(optimizer.param_groups),
        "optimizer_param_groups_sha256": stable_sha256(param_groups),
        "optimizer_parameter_count": sum(
            len(group.get("params", ())) for group in optimizer.param_groups
        ),
        "optimizer_max_parameter_step": max(parameter_steps, default=0.0),
        "amp_scale": (
            float(scale)
            if isinstance(scale, (int, float)) and not isinstance(scale, bool)
            else None
        ),
        "amp_scaler_state_sha256": stable_sha256(scaler_state),
    }


def optimizer_step_evidence(
    trainer: Any,
    *,
    phase: str,
    before: dict[str, Any],
) -> dict[str, Any]:
    """Classify one AMP optimizer call as an update or an overflow backoff."""

    after = optimizer_training_state(trainer)
    before_step = before["optimizer_max_parameter_step"]
    after_step = after["optimizer_max_parameter_step"]
    before_scale = before["amp_scale"]
    after_scale = after["amp_scale"]
    successful_update = after_step > before_step
    amp_overflow_skip = (
        not successful_update
        and isinstance(before_scale, float)
        and isinstance(after_scale, float)
        and after_scale < before_scale
    )
    if not successful_update and not amp_overflow_skip:
        raise DetectionInfrastructureError(
            "Optimizer call neither advanced AdamW state nor proved an AMP overflow backoff.",
            code="smoke_optimizer_evidence_invalid",
        )
    return {
        "phase": phase,
        "successful_update": successful_update,
        "amp_overflow_skip": amp_overflow_skip,
        "before": before,
        "after": after,
    }


def optimizer_runtime_context(trainer: Any, *, phase_batch_index: int) -> dict[str, Any]:
    """Record scheduler/warmup state at the exact optimizer-call boundary."""

    learning_rates = [float(group["lr"]) for group in trainer.optimizer.param_groups]
    if (
        isinstance(phase_batch_index, bool)
        or not isinstance(phase_batch_index, int)
        or phase_batch_index < 1
        or not learning_rates
        or any(not math.isfinite(value) or value < 0 for value in learning_rates)
    ):
        raise DetectionInfrastructureError(
            "Optimizer runtime context is invalid.",
            code="smoke_optimizer_evidence_invalid",
        )
    batches_per_epoch = len(trainer.train_loader)
    if batches_per_epoch < 1:
        raise DetectionInfrastructureError(
            "Smoke trainer has no training batches.",
            code="smoke_bound_exceeded",
        )
    return {
        "phase_batch_index": phase_batch_index,
        "epoch": int(trainer.epoch),
        "batches_per_epoch": batches_per_epoch,
        "global_iteration": int(trainer.epoch) * batches_per_epoch
        + ((phase_batch_index - 1) % batches_per_epoch),
        "configured_batch_size": int(trainer.batch_size),
        "loader_batch_size": int(trainer.train_loader.batch_size),
        "accumulate": int(trainer.accumulate),
        "scheduler_last_epoch": int(trainer.scheduler.last_epoch),
        "learning_rates": learning_rates,
    }


def _fresh_process_environment(
    seed: int,
    authorization_token: str,
    cuda_visible_devices_at_parent_start: str | None,
) -> dict[str, str]:
    """Restore process-start controls that Ultralytics removes at teardown."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or not isinstance(authorization_token, str)
        or not authorization_token
        or (
            cuda_visible_devices_at_parent_start is not None
            and (
                not isinstance(cuda_visible_devices_at_parent_start, str)
                or not cuda_visible_devices_at_parent_start
            )
        )
    ):
        raise DetectionInfrastructureError(
            "Fresh-process smoke environment inputs are invalid.",
            code="real_smoke_worker_invalid",
        )
    environment = dict(os.environ)
    environment["FLOODSIGHT_REAL_SMOKE_WORKER_TOKEN"] = authorization_token
    # Ultralytics calls unset_deterministic() at trainer teardown, which removes
    # these variables from the parent environment. They must be set in the
    # child's environment before its interpreter starts.
    environment["PYTHONHASHSEED"] = str(seed)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if cuda_visible_devices_at_parent_start is None:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices_at_parent_start
    return environment


def _require_ultralytics(required_version: str) -> tuple[Any, str]:
    """Import Ultralytics only after the caller has passed an execution gate."""

    try:
        installed = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError as exc:
        raise DetectionInfrastructureError(
            "Ultralytics is absent from the active pinned ML environment.",
            code="ml_dependency_missing",
        ) from exc
    if not version_matches(installed, required_version):
        raise DetectionInfrastructureError(
            f"Ultralytics {installed} does not exactly match {required_version}.",
            code="ml_dependency_version_mismatch",
        )
    try:
        module = importlib.import_module("ultralytics")
        yolo_class = module.YOLO
    except (ImportError, AttributeError) as exc:
        raise DetectionInfrastructureError(
            "The pinned Ultralytics YOLO API is unavailable.",
            code="ml_dependency_api_mismatch",
        ) from exc
    return yolo_class, installed


def _require_package_version(distribution: str, required_version: str) -> str:
    try:
        installed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise DetectionInfrastructureError(
            f"{distribution} is absent from the active pinned ML environment.",
            code="ml_dependency_missing",
        ) from exc
    if not version_matches(installed, required_version):
        raise DetectionInfrastructureError(
            f"{distribution} {installed} does not exactly match {required_version}.",
            code="ml_dependency_version_mismatch",
        )
    return installed


def _append_event(run: RunReservation, event: str, **details: Any) -> None:
    payload = {"event": event, **details}
    path = run.run_directory / "floodsight-run-events.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _checkpoint_trainer_arguments(
    config: TrainingConfig,
    run_directory: Path,
    data_yaml: Path,
    selected_device: str,
) -> dict[str, Any]:
    if not data_yaml.is_absolute() or data_yaml.is_symlink() or not data_yaml.is_file():
        raise DetectionInfrastructureError(
            "The frozen dataset YAML is missing, non-absolute, or a symlink.",
            code="frozen_dataset_contract_invalid",
        )
    train = dict(config.train)
    train["device"] = selected_device
    return {
        "schema_version": "floodsight-production-trainer-arguments-v1",
        "task": "detect",
        "mode": "train",
        "data": str(data_yaml),
        "data_sha256": sha256_file(data_yaml),
        "project": str(run_directory.parent),
        "name": run_directory.name,
        "save_dir": str(run_directory),
        "exist_ok": True,
        "train": {key: _jsonable_argument(train[key]) for key in sorted(train)},
    }


def _audit_production_trainer(
    trainer: Any,
    *,
    expected: dict[str, Any],
    model_source: Path,
    resume: bool,
) -> None:
    observed_train = _observed_trainer_arguments(trainer)
    drift: dict[str, Any] = {}
    if observed_train != expected["train"]:
        drift["train"] = {
            key: {"expected": expected["train"].get(key), "actual": observed_train.get(key)}
            for key in sorted(set(expected["train"]) | set(observed_train))
            if expected["train"].get(key) != observed_train.get(key)
        }
    controls = {
        "task": getattr(trainer.args, "task", None),
        "mode": getattr(trainer.args, "mode", None),
        "data": str(getattr(trainer.args, "data", "")),
        "project": str(Path(getattr(trainer.args, "project", "")).expanduser().resolve()),
        "name": getattr(trainer.args, "name", None),
        "save_dir": str(Path(trainer.save_dir).expanduser().resolve()),
        "exist_ok": getattr(trainer.args, "exist_ok", None),
    }
    for key in ("task", "mode", "data", "project", "name", "save_dir", "exist_ok"):
        if controls[key] != expected[key]:
            drift[key] = {"expected": expected[key], "actual": controls[key]}
    observed_model = Path(str(getattr(trainer.args, "model", ""))).expanduser()
    observed_resume = getattr(trainer.args, "resume", None)
    if (
        observed_model.is_symlink()
        or not observed_model.is_absolute()
        or observed_model.resolve(strict=True) != model_source
    ):
        drift["model"] = {"expected": str(model_source), "actual": str(observed_model)}
    if resume:
        resume_path = Path(str(observed_resume)).expanduser()
        if (
            resume_path.is_symlink()
            or not resume_path.is_absolute()
            or resume_path.resolve(strict=True) != model_source
        ):
            drift["resume"] = {"expected": str(model_source), "actual": observed_resume}
    elif observed_resume is not False:
        drift["resume"] = {"expected": False, "actual": observed_resume}
    if drift:
        raise DetectionInfrastructureError(
            "Ultralytics production arguments or output containment drifted.",
            code="training_argument_drift",
            details=[drift],
        )


def execute_training(
    config: TrainingConfig,
    contract: DatasetContract,
    *,
    output_root: str | Path,
    run_name: str,
    allow_training: bool,
    weights: WeightArtifact | None = None,
    approval: TrainingApproval | None = None,
    real_smoke: Any | None = None,
    resume_checkpoint: str | Path | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Run real YOLO fine-tuning only behind an unambiguous authorization flag."""

    if not allow_training:
        raise DetectionInfrastructureError(
            "Real detection training is disabled; pass --allow-training explicitly.",
            code="training_not_authorized",
        )
    if weights is None or approval is None or real_smoke is None:
        raise DetectionInfrastructureError(
            "Training requires audited local weights, a current real-smoke report, "
            "and bound human approval.",
            code="training_not_authorized",
        )
    selected_device = bound_training_device(str(config.train["device"]), device_override)
    validate_frozen_model_artifacts(config)
    require_prestarted_hash_seed(int(config.train["seed"]))
    canonical_output_root = bound_training_output_root(config.output.run_root, output_root)
    canonical_contract = validate_dataset_contract(
        config.dataset.manifest_path,
        contract.data_root,
        expected_manifest_path=config.dataset.manifest_path,
        expected_manifest_sha256=config.dataset.manifest_sha256,
        expected_dataset_fingerprint=config.dataset.dataset_fingerprint,
        expected_source_version=config.dataset.source_version,
        verify_image_hashes=config.dataset.verify_image_hashes,
        require_full_integrity=config.dataset.require_full_integrity,
        required_splits=config.dataset.required_splits,
        require_all_train_classes=config.dataset.require_all_train_classes,
        reject_duplicate_images=config.dataset.reject_duplicate_images,
    )
    if canonical_contract.summary() != contract.summary():
        raise DetectionInfrastructureError(
            "The supplied dataset contract differs from a fresh canonical validation.",
            code="manifest_identity_mismatch",
        )
    contract = canonical_contract
    # The CLI validates these first; the execution boundary independently
    # reloads both records so direct Python callers cannot bypass the gate with
    # a fabricated dataclass instance.
    weights = load_weight_audit(
        weights.audit_path,
        expected_filename=config.model,
        expected_weight_path=config.model_path,
        expected_weight_sha256=config.model_sha256,
        expected_audit_path=config.weight_audit_path,
        expected_audit_sha256=config.weight_audit_sha256,
        require_license_approval=False,
    )
    from floodsight_detection.real_smoke import load_real_smoke_attestation

    real_smoke = load_real_smoke_attestation(
        real_smoke.path,
        config=config,
        contract=contract,
        weights=weights,
    )
    approval = load_training_approval(
        approval.path,
        config_sha256=config.sha256,
        manifest_sha256=contract.manifest_sha256,
        dataset_fingerprint=contract.dataset_fingerprint,
        weights_sha256=weights.sha256,
        weights_path=weights.path,
        weight_audit_path=weights.audit_path,
        weight_audit_sha256=weights.audit_sha256,
        run_name=run_name,
        output_root=canonical_output_root,
        device=selected_device,
        manifest_id=config.dataset.manifest_id,
        dataset_id=config.dataset.dataset_id,
        preparation_version=config.dataset.preparation_version,
        taxonomy_version=config.dataset.taxonomy_version,
        taxonomy_sha256=config.dataset.taxonomy_sha256,
        mapping_version=config.dataset.mapping_version,
        mapping_sha256=config.dataset.mapping_sha256,
        real_smoke_report_path=real_smoke.path,
        real_smoke_report_sha256=real_smoke.sha256,
    )
    validate_training_license_disposition(
        weights,
        review_disposition=approval.review_disposition,
    )
    configure_determinism(int(config.train["seed"]), include_ml_libraries=False)
    runtime = validate_full_training_runtime(selected_device)
    yolo_class, installed_version = _require_ultralytics(config.ultralytics_version)
    ultralytics_cfg = importlib.import_module("ultralytics.cfg")
    validate_ultralytics_default_contract(ultralytics_cfg.DEFAULT_CFG_DICT)
    torch_version = _require_package_version("torch", config.torch_version)
    torchvision_version = _require_package_version("torchvision", config.torchvision_version)
    configure_determinism(int(config.train["seed"]), include_ml_libraries=True)
    if resume_checkpoint is None:
        run = reserve_new_run(
            canonical_output_root,
            run_name,
            config_sha256=config.sha256,
            training_code_sha256=approval.training_code_sha256,
            manifest_id=config.dataset.manifest_id,
            dataset_id=config.dataset.dataset_id,
            preparation_version=config.dataset.preparation_version,
            manifest_sha256=contract.manifest_sha256,
            dataset_fingerprint=contract.dataset_fingerprint,
            taxonomy_version=config.dataset.taxonomy_version,
            taxonomy_sha256=config.dataset.taxonomy_sha256,
            mapping_version=config.dataset.mapping_version,
            mapping_sha256=config.dataset.mapping_sha256,
            weights_path=str(weights.path),
            weights_sha256=weights.sha256,
            weight_audit_path=str(weights.audit_path),
            weight_audit_sha256=weights.audit_sha256,
            approval_sha256=approval.sha256,
            approval_id=approval.approval_id,
            real_smoke_report_path=str(real_smoke.path),
            real_smoke_report_sha256=real_smoke.sha256,
            device=selected_device,
        )
    else:
        resume_data_yaml = canonical_output_root / run_name / "dataset-contract/dataset.yaml"
        checkpoint_arguments = _checkpoint_trainer_arguments(
            config,
            canonical_output_root / run_name,
            resume_data_yaml,
            selected_device,
        )
        run = validate_resume(
            resume_checkpoint,
            canonical_output_root,
            run_name=run_name,
            config_sha256=config.sha256,
            training_code_sha256=approval.training_code_sha256,
            manifest_id=config.dataset.manifest_id,
            dataset_id=config.dataset.dataset_id,
            preparation_version=config.dataset.preparation_version,
            manifest_sha256=contract.manifest_sha256,
            dataset_fingerprint=contract.dataset_fingerprint,
            taxonomy_version=config.dataset.taxonomy_version,
            taxonomy_sha256=config.dataset.taxonomy_sha256,
            mapping_version=config.dataset.mapping_version,
            mapping_sha256=config.dataset.mapping_sha256,
            weights_path=str(weights.path),
            weights_sha256=weights.sha256,
            weight_audit_path=str(weights.audit_path),
            weight_audit_sha256=weights.audit_sha256,
            approval_sha256=approval.sha256,
            approval_id=approval.approval_id,
            real_smoke_report_path=str(real_smoke.path),
            real_smoke_report_sha256=real_smoke.sha256,
            device=selected_device,
            checkpoint_trainer_arguments=checkpoint_arguments,
            last_checkpoint_filename=config.output.last_checkpoint_filename,
        )
    with exclusive_run_lock(run):
        latest_trusted: TrustedCheckpoint | None = run.trusted_checkpoint
        if not run.resumed:
            data_yaml = freeze_dataset_contract(contract, run.run_directory / "dataset-contract")
            checkpoint_arguments = _checkpoint_trainer_arguments(
                config, run.run_directory, data_yaml, selected_device
            )
            # Absolute, hash-audited local path: Ultralytics has no download fallback.
            model = yolo_class(str(weights.path), task="detect")
            kwargs = dict(config.train)
            kwargs["device"] = selected_device
            kwargs["resume"] = False
            kwargs.update(
                {
                    "data": str(data_yaml),
                    "project": str(run.run_directory.parent),
                    "name": run.run_directory.name,
                    # Safe because reserve_new_run atomically owns this exact directory.
                    "exist_ok": True,
                }
            )
            _append_event(
                run,
                "TRAINING_STARTED",
                ultralytics_version=installed_version,
                torch_version=torch_version,
                torchvision_version=torchvision_version,
                model=config.model,
                pretrained_weights=str(weights.path),
                pretrained_weights_sha256=weights.sha256,
                approval_id=approval.approval_id,
                real_smoke_report=str(real_smoke.path),
                real_smoke_report_sha256=real_smoke.sha256,
                runtime=runtime,
            )

            def on_train_start(trainer: Any) -> None:
                _audit_production_trainer(
                    trainer,
                    expected=checkpoint_arguments,
                    model_source=weights.path,
                    resume=False,
                )

            def on_model_save(trainer: Any) -> None:
                nonlocal latest_trusted
                _audit_production_trainer(
                    trainer,
                    expected=checkpoint_arguments,
                    model_source=weights.path,
                    resume=False,
                )
                latest_trusted = publish_checkpoint_generation(
                    run_directory=run.run_directory,
                    run_metadata_path=run.metadata_path,
                    live_checkpoint=Path(trainer.last),
                    epoch=int(trainer.epoch),
                    trainer_arguments=checkpoint_arguments,
                    data_yaml=data_yaml,
                    trainer=trainer,
                )
                _append_event(
                    run,
                    "TRUSTED_CHECKPOINT_PUBLISHED",
                    epoch=latest_trusted.epoch,
                    checkpoint=str(latest_trusted.path),
                    checkpoint_sha256=latest_trusted.sha256,
                    metadata_sha256=latest_trusted.metadata_sha256,
                )

            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_model_save", on_model_save)
            try:
                results = model.train(**kwargs)
            except Exception as exc:
                _append_event(run, "TRAINING_FAILED", exception_type=type(exc).__name__)
                raise
        else:
            if run.trusted_checkpoint is None:
                raise DetectionInfrastructureError(
                    "Resume did not resolve a trusted checkpoint generation.",
                    code="checkpoint_integrity_failed",
                )
            data_yaml = run.run_directory / "dataset-contract/dataset.yaml"
            checkpoint_arguments = _checkpoint_trainer_arguments(
                config, run.run_directory, data_yaml, selected_device
            )
            model = yolo_class(str(run.checkpoint), task="detect")
            kwargs = {"resume": True, "device": selected_device}
            _append_event(run, "RESUME_STARTED", checkpoint=str(run.checkpoint))

            def on_resume_start(trainer: Any) -> None:
                _audit_production_trainer(
                    trainer,
                    expected=checkpoint_arguments,
                    model_source=run.trusted_checkpoint.path,
                    resume=True,
                )
                restore_rng_state(run.trusted_checkpoint.rng_state, trainer)
                _append_event(
                    run,
                    "RESUME_RNG_RESTORED",
                    checkpoint_sha256=run.trusted_checkpoint.sha256,
                    rng_state_sha256=run.trusted_checkpoint.rng_state["state_sha256"],
                )

            def on_resumed_model_save(trainer: Any) -> None:
                nonlocal latest_trusted
                _audit_production_trainer(
                    trainer,
                    expected=checkpoint_arguments,
                    model_source=run.trusted_checkpoint.path,
                    resume=True,
                )
                latest_trusted = publish_checkpoint_generation(
                    run_directory=run.run_directory,
                    run_metadata_path=run.metadata_path,
                    live_checkpoint=Path(trainer.last),
                    epoch=int(trainer.epoch),
                    trainer_arguments=checkpoint_arguments,
                    data_yaml=data_yaml,
                    trainer=trainer,
                )
                _append_event(
                    run,
                    "TRUSTED_CHECKPOINT_PUBLISHED",
                    epoch=latest_trusted.epoch,
                    checkpoint=str(latest_trusted.path),
                    checkpoint_sha256=latest_trusted.sha256,
                    metadata_sha256=latest_trusted.metadata_sha256,
                )

            model.add_callback("on_train_start", on_resume_start)
            model.add_callback("on_model_save", on_resumed_model_save)
            try:
                results = model.train(**kwargs)
            except Exception as exc:
                _append_event(run, "RESUME_FAILED", exception_type=type(exc).__name__)
                raise
        last_checkpoint = (
            run.run_directory / "weights" / config.output.last_checkpoint_filename
        )
        best_checkpoint = (
            run.run_directory / "weights" / config.output.best_checkpoint_filename
        )
        if not last_checkpoint.is_file():
            _append_event(run, "CHECKPOINT_MISSING", expected=str(last_checkpoint))
            raise DetectionInfrastructureError(
                "Ultralytics returned without the required weights/last.pt checkpoint.",
                code="checkpoint_missing",
            )
        if latest_trusted is None:
            _append_event(run, "TRUSTED_CHECKPOINT_MISSING")
            raise DetectionInfrastructureError(
                "Ultralytics returned without a trusted resumable checkpoint generation.",
                code="checkpoint_missing",
            )
        restore_live_checkpoint(latest_trusted, last_checkpoint)
        _append_event(
            run,
            "TRAINING_RETURNED",
            last_checkpoint=str(last_checkpoint),
            best_checkpoint=str(best_checkpoint) if best_checkpoint.is_file() else None,
            trusted_checkpoint=str(latest_trusted.path),
            trusted_checkpoint_sha256=latest_trusted.sha256,
            trusted_checkpoint_metadata=str(latest_trusted.metadata_path),
            trusted_checkpoint_metadata_sha256=latest_trusted.metadata_sha256,
        )
        return {
            "status": "TRAINING_RETURNED",
            "resumed": run.resumed,
            "run_directory": str(run.run_directory),
            "last_checkpoint": str(last_checkpoint),
            "best_checkpoint": str(best_checkpoint) if best_checkpoint.is_file() else None,
            "trusted_checkpoint": str(latest_trusted.path),
            "trusted_checkpoint_sha256": latest_trusted.sha256,
            "trusted_checkpoint_metadata": str(latest_trusted.metadata_path),
            "trusted_checkpoint_metadata_sha256": latest_trusted.metadata_sha256,
            "ultralytics_version": installed_version,
            "torch_version": torch_version,
            "torchvision_version": torchvision_version,
            "pretrained_weights_sha256": weights.sha256,
            "approval_id": approval.approval_id,
            "real_smoke_report": str(real_smoke.path),
            "real_smoke_report_sha256": real_smoke.sha256,
            "runtime": runtime,
            "results_type": type(results).__name__,
        }


def _jsonable_argument(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable_argument(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_argument(item) for item in value]
    return value


def real_smoke_effective_arguments(train_settings: dict[str, Any]) -> dict[str, Any]:
    """Preserve every frozen train argument; callbacks impose the smoke bounds."""

    if set(train_settings) != set(ULTRALYTICS_TRAIN_VAL_KEYS):
        raise DetectionInfrastructureError(
            "Real smoke did not receive the exhaustive frozen Ultralytics settings.",
            code="real_smoke_config_drift",
        )
    configured_epochs = train_settings.get("epochs")
    if isinstance(configured_epochs, bool) or not isinstance(configured_epochs, int):
        raise DetectionInfrastructureError(
            "The frozen real-smoke epoch count is invalid.",
            code="real_smoke_config_drift",
        )
    return dict(train_settings)


def _observed_trainer_arguments(trainer: Any) -> dict[str, Any]:
    try:
        return {
            key: _jsonable_argument(getattr(trainer.args, key))
            for key in sorted(ULTRALYTICS_TRAIN_VAL_KEYS)
        }
    except AttributeError as exc:
        raise DetectionInfrastructureError(
            "Ultralytics trainer omitted a frozen effective argument.",
            code="real_smoke_config_drift",
        ) from exc


def _transform_class_names(transforms: Any) -> list[str]:
    names: list[str] = []

    def visit(transform: Any) -> None:
        names.append(type(transform).__name__)
        nested = getattr(transform, "transforms", None)
        if isinstance(nested, (list, tuple)):
            for child in nested:
                visit(child)

    visit(transforms)
    return names


def _transform_configuration(transforms: Any) -> list[dict[str, Any]]:
    """Capture the executable augmentation settings, not only class names."""

    fields = {
        "Mosaic": ("p", "imgsz", "n"),
        "CopyPaste": ("p", "mode"),
        "RandomPerspective": (
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
        ),
        "MixUp": ("p",),
        "CutMix": ("p",),
        "Albumentations": ("p",),
        "RandomHSV": ("hgain", "sgain", "vgain"),
        "RandomFlip": ("direction", "p"),
        "Format": ("bgr",),
    }
    records: list[dict[str, Any]] = []

    def visit(transform: Any) -> None:
        class_name = type(transform).__name__
        if class_name in fields:
            record = {"class_name": class_name}
            for field in fields[class_name]:
                if not hasattr(transform, field):
                    raise DetectionInfrastructureError(
                        f"Ultralytics {class_name} omitted audited field {field}.",
                        code="real_smoke_augmentation_drift",
                    )
                record[field] = _jsonable_argument(getattr(transform, field))
            records.append(record)
        nested = getattr(transform, "transforms", None)
        if isinstance(nested, (list, tuple)):
            for child in nested:
                visit(child)

    visit(transforms)
    return records


def expected_transform_configuration(
    effective_args: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the pinned executable transform contract for detection training."""

    return [
        {
            "class_name": "Mosaic",
            "p": effective_args["mosaic"],
            "imgsz": effective_args["imgsz"],
            "n": 4,
        },
        {
            "class_name": "CopyPaste",
            "p": effective_args["copy_paste"],
            "mode": effective_args["copy_paste_mode"],
        },
        {
            "class_name": "RandomPerspective",
            "degrees": effective_args["degrees"],
            "translate": effective_args["translate"],
            "scale": effective_args["scale"],
            "shear": effective_args["shear"],
            "perspective": effective_args["perspective"],
        },
        {"class_name": "MixUp", "p": effective_args["mixup"]},
        {"class_name": "CutMix", "p": effective_args["cutmix"]},
        {"class_name": "Albumentations", "p": 1.0},
        {
            "class_name": "RandomHSV",
            "hgain": effective_args["hsv_h"],
            "sgain": effective_args["hsv_s"],
            "vgain": effective_args["hsv_v"],
        },
        {
            "class_name": "RandomFlip",
            "direction": "vertical",
            "p": effective_args["flipud"],
        },
        {
            "class_name": "RandomFlip",
            "direction": "horizontal",
            "p": effective_args["fliplr"],
        },
        {"class_name": "Format", "bgr": effective_args["bgr"]},
    ]


def audited_transform_configuration(
    transforms: Any, effective_args: dict[str, Any]
) -> list[dict[str, Any]]:
    """Require the exact frozen detection augmentation probabilities and gains."""

    expected = expected_transform_configuration(effective_args)
    observed = _transform_configuration(transforms)
    if observed != expected:
        raise DetectionInfrastructureError(
            "Ultralytics real-smoke transform configuration drifted.",
            code="real_smoke_augmentation_drift",
            details=[{"expected": expected, "actual": observed}],
        )
    return observed


class UltralyticsSmokeBackend:
    """Exercise pinned Ultralytics in generated or config-bound real mode."""

    version_specification = "8.3.222"
    torch_version_specification = "2.13.0+cu130"
    torchvision_version_specification = "0.28.0+cu130"

    def __init__(self, model_source: str = "yolo11n.yaml") -> None:
        self.model_source = model_source

    def run(
        self,
        *,
        data_yaml: Path,
        output_root: Path,
        seed: int,
        device: str,
        train_settings: dict[str, Any] | None = None,
        config_sha256: str | None = None,
    ) -> dict[str, Any]:
        cuda_visible_devices_at_process_start = os.environ.get("CUDA_VISIBLE_DEVICES")
        real_mode = train_settings is not None or config_sha256 is not None
        if real_mode != (train_settings is not None and config_sha256 is not None):
            raise DetectionInfrastructureError(
                "Real smoke requires both frozen settings and their config hash.",
                code="real_smoke_config_drift",
            )

        runtime: dict[str, Any] | None = None
        hash_seed_runtime: dict[str, Any] | None = None
        effective_args: dict[str, Any] | None = None
        expected_args_sha256: str | None = None
        expected_augmentation_args: dict[str, Any] | None = None
        model_source = self.model_source
        selected_device = device
        if real_mode:
            if train_settings is None or config_sha256 is None:
                raise DetectionInfrastructureError(
                    "Real smoke settings are incomplete.", code="real_smoke_config_drift"
                )
            if len(config_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in config_sha256
            ):
                raise DetectionInfrastructureError(
                    "Real smoke received an invalid config SHA-256.",
                    code="real_smoke_config_drift",
                )
            model_path = Path(model_source).expanduser()
            if (
                model_path.name != PINNED_MODEL
                or model_path.is_symlink()
                or not model_path.is_absolute()
                or not model_path.is_file()
            ):
                raise DetectionInfrastructureError(
                    "Real smoke requires the audited absolute local yolo11l.pt artifact.",
                    code="real_smoke_model_drift",
                )
            model_path = model_path.resolve(strict=True)
            model_source = str(model_path)
            effective_args = real_smoke_effective_arguments(train_settings)
            selected_device = bound_training_device(str(effective_args["device"]), device)
            hash_seed_runtime = require_prestarted_hash_seed(seed)
            runtime = validate_full_training_runtime(selected_device)
            if effective_args["seed"] != seed:
                raise DetectionInfrastructureError(
                    "Real-smoke seed differs from the frozen configuration.",
                    code="real_smoke_config_drift",
                )
            expected_args_sha256 = stable_sha256(effective_args)
            expected_augmentation_args = {
                key: effective_args[key] for key in sorted(REAL_SMOKE_AUGMENTATION_KEYS)
            }

        configure_determinism(seed, include_ml_libraries=False)
        yolo_class, installed_version = _require_ultralytics(self.version_specification)
        ultralytics_cfg = importlib.import_module("ultralytics.cfg")
        validate_ultralytics_default_contract(ultralytics_cfg.DEFAULT_CFG_DICT)
        torch_version = _require_package_version("torch", self.torch_version_specification)
        torchvision_version = _require_package_version(
            "torchvision", self.torchvision_version_specification
        )
        import torch  # type: ignore[import-not-found]

        configure_determinism(seed, include_ml_libraries=True)
        try:
            dataset_payload = json.loads(data_yaml.read_text(encoding="utf-8"))
            train_list = Path(dataset_payload["train"])
            train_sample_count = len(
                [line for line in train_list.read_text(encoding="utf-8").splitlines() if line]
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise DetectionInfrastructureError(
                "Bounded smoke could not read its frozen training split.",
                code="smoke_contract_invalid",
            ) from exc
        if not 1 <= train_sample_count <= 8:
            raise DetectionInfrastructureError(
                "Bounded smoke training split must contain between one and eight samples.",
                code="smoke_bound_exceeded",
            )
        if (
            real_mode
            and effective_args is not None
            and train_sample_count > effective_args["batch"]
        ):
            raise DetectionInfrastructureError(
                "Frozen real-smoke batch would require more than one batch per epoch.",
                code="smoke_bound_exceeded",
            )

        flags = {
            "loader": False,
            "model_forward": False,
            "loss": False,
            "backward": False,
            "optimizer_step": False,
            "validation": False,
            "checkpoint": False,
            "resume": False,
        }
        if real_mode:
            flags["effective_args"] = False
            flags["augmentation"] = False
            flags["distinct_model_instance_resume"] = False
            flags["rng_restored"] = False
        observed_train_batches = 0
        observed_backward_calls = 0
        backward_gradient_finiteness: list[bool] = []
        observed_optimizer_step_calls = 0
        successful_optimizer_updates = 0
        amp_overflow_skips = 0
        observed_validation_runs = 0
        optimizer_evidence: list[dict[str, Any]] = []
        observed_effective_args_sha256: list[str] = []
        augmentation_evidence: list[dict[str, Any]] = []
        resume_state: dict[str, Any] = {}
        resume_rng_state: dict[str, Any] | None = None
        resume_checkpoint_state: dict[str, Any] | None = None
        initial_name = (
            "ultralytics-real-smoke-initial" if real_mode else "ultralytics-initial"
        )
        initial_dir = output_root / initial_name
        resumable_checkpoint = output_root / "pre-strip-resumable.pt"
        if initial_dir.exists() or resumable_checkpoint.exists():
            raise DetectionInfrastructureError(
                f"Ultralytics smoke output already exists beneath: {output_root}",
                code="smoke_collision",
            )

        model = yolo_class(model_source, task="detect")
        initial_model_instance_id = id(model)
        initial_pid = os.getpid()
        torch_device = torch.device(
            f"cuda:{selected_device}" if selected_device.isdigit() else selected_device
        )
        model.model.to(torch_device)
        forward_size = int(effective_args["imgsz"]) if effective_args is not None else 64
        with torch.no_grad():
            prediction = model.model(
                torch.zeros((1, 3, forward_size, forward_size), device=torch_device)
            )
        if prediction is None:
            raise DetectionInfrastructureError(
                "Ultralytics smoke model forward returned no output.",
                code="smoke_forward_failed",
            )
        flags["model_forward"] = True
        del prediction

        hook_handles: list[Any] = []

        def backward_hook(gradient: Any) -> Any:
            nonlocal observed_backward_calls
            if gradient is not None:
                flags["backward"] = True
                observed_backward_calls += 1
                backward_gradient_finiteness.append(
                    bool(torch.isfinite(gradient).all())
                )
            return gradient

        def install_backward_hook(trainer: Any) -> None:
            # Ultralytics rebuilds the architecture when it overrides nc, so
            # attach to the trainer's final model rather than the template.
            first_trainable = next(
                parameter for parameter in trainer.model.parameters() if parameter.requires_grad
            )
            hook_handles.append(first_trainable.register_hook(backward_hook))

        def install_optimizer_counter(trainer: Any, *, phase: str) -> None:
            nonlocal amp_overflow_skips
            nonlocal observed_optimizer_step_calls
            nonlocal successful_optimizer_updates
            original_optimizer_step = trainer.optimizer_step

            def applied_optimizer_step(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
                nonlocal successful_optimizer_updates
                successful_optimizer_updates += 1

            if real_mode:
                hook_handles.append(
                    trainer.optimizer.register_step_post_hook(applied_optimizer_step)
                )

            def counted_optimizer_step() -> None:
                before = optimizer_training_state(trainer) if real_mode else None
                applied_before = successful_optimizer_updates
                nonlocal amp_overflow_skips
                nonlocal observed_optimizer_step_calls
                if real_mode and (
                    observed_optimizer_step_calls >= REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS
                ):
                    raise DetectionInfrastructureError(
                        "Real smoke refused an optimizer call beyond its hard ceiling.",
                        code="smoke_bound_exceeded",
                    )
                context = (
                    optimizer_runtime_context(
                        trainer, phase_batch_index=observed_train_batches
                    )
                    if real_mode
                    else None
                )
                original_optimizer_step()
                observed_optimizer_step_calls += 1
                if before is not None:
                    evidence = optimizer_step_evidence(trainer, phase=phase, before=before)
                    evidence["phase_call_index"] = observed_optimizer_step_calls
                    evidence["runtime_context"] = context
                    evidence["underlying_optimizer_step"] = (
                        successful_optimizer_updates == applied_before + 1
                    )
                    if evidence["successful_update"] != evidence["underlying_optimizer_step"]:
                        raise DetectionInfrastructureError(
                            "Optimizer state progress differs from the underlying step hook.",
                            code="smoke_optimizer_evidence_invalid",
                        )
                    optimizer_evidence.append(evidence)
                    amp_overflow_skips += int(evidence["amp_overflow_skip"])

            trainer.optimizer_step = counted_optimizer_step

        def audit_real_train_start(trainer: Any, *, phase: str) -> None:
            if not real_mode:
                return
            if (
                effective_args is None
                or expected_args_sha256 is None
                or expected_augmentation_args is None
            ):
                raise DetectionInfrastructureError(
                    "Real-smoke evidence state is incomplete.",
                    code="real_smoke_config_drift",
                )
            observed_args = _observed_trainer_arguments(trainer)
            if observed_args != effective_args:
                drift = {
                    key: {"expected": effective_args[key], "actual": observed_args[key]}
                    for key in sorted(effective_args)
                    if observed_args[key] != effective_args[key]
                }
                raise DetectionInfrastructureError(
                    "Ultralytics normalized or replaced a frozen real-smoke argument.",
                    code="real_smoke_config_drift",
                    details=[drift],
                )
            observed_hash = stable_sha256(observed_args)
            if observed_hash != expected_args_sha256:
                raise DetectionInfrastructureError(
                    "Ultralytics effective argument hash drifted.",
                    code="real_smoke_config_drift",
                )
            observed_effective_args_sha256.append(observed_hash)
            dataset = trainer.train_loader.dataset
            loader_batch_size = trainer.train_loader.batch_size
            loader_num_workers = trainer.train_loader.num_workers
            dataset_size = len(dataset)
            expected_loader_batch_size = bounded_loader_batch_size(
                effective_args["batch"], dataset_size
            )
            transform_names = _transform_class_names(dataset.transforms)
            transform_configuration = audited_transform_configuration(
                dataset.transforms, effective_args
            )
            missing_transforms = sorted(
                _REAL_SMOKE_REQUIRED_TRANSFORMS - set(transform_names)
            )
            observed_augmentation_args = {
                key: observed_args[key] for key in sorted(REAL_SMOKE_AUGMENTATION_KEYS)
            }
            if (
                dataset.augment is not True
                or loader_batch_size != expected_loader_batch_size
                or loader_num_workers != effective_args["workers"]
                or missing_transforms
                or observed_augmentation_args != expected_augmentation_args
            ):
                raise DetectionInfrastructureError(
                    "Ultralytics real-smoke augmentation pipeline drifted.",
                    code="real_smoke_augmentation_drift",
                    details=[
                        {
                            "dataset_augment": dataset.augment,
                            "loader_batch_size": loader_batch_size,
                            "loader_num_workers": loader_num_workers,
                            "missing_transforms": missing_transforms,
                        }
                    ],
                )
            augmentation_evidence.append(
                {
                    "phase": phase,
                    "dataset_augment": True,
                    "configured_batch_size": effective_args["batch"],
                    "loader_batch_size": loader_batch_size,
                    "loader_batch_policy": _REAL_SMOKE_BATCH_POLICY,
                    "loader_num_workers": loader_num_workers,
                    "dataset_size": dataset_size,
                    "arguments": observed_augmentation_args,
                    "arguments_sha256": stable_sha256(observed_augmentation_args),
                    "transform_class_names": transform_names,
                    "transform_classes_sha256": stable_sha256(transform_names),
                    "transform_configuration": transform_configuration,
                    "transform_configuration_sha256": stable_sha256(
                        transform_configuration
                    ),
                }
            )
            flags["effective_args"] = True
            flags["augmentation"] = True

        def on_initial_train_start(trainer: Any) -> None:
            audit_real_train_start(trainer, phase="initial")
            install_backward_hook(trainer)
            install_optimizer_counter(trainer, phase="initial")

        def on_train_batch_start(_trainer: Any) -> None:
            nonlocal observed_train_batches
            if real_mode and observed_train_batches >= 1:
                raise DetectionInfrastructureError(
                    "Initial real smoke refused a second training batch.",
                    code="smoke_bound_exceeded",
                )
            flags["loader"] = True
            observed_train_batches += 1

        def on_train_batch_end(trainer: Any) -> None:
            flags["loss"] = bool(torch.isfinite(trainer.loss).all())

        def on_val_end(_validator: Any) -> None:
            nonlocal observed_validation_runs
            flags["validation"] = True
            observed_validation_runs += 1

        def on_model_save(trainer: Any) -> None:
            nonlocal resume_checkpoint_state, resume_rng_state
            flags["checkpoint"] = True
            # Ultralytics strips optimizer state during normal trainer shutdown.
            # Preserve the freshly written, pre-strip checkpoint so the smoke
            # exercises a genuine optimizer/scheduler resume on epoch two.
            shutil.copy2(Path(trainer.last), resumable_checkpoint)
            with resumable_checkpoint.open("rb") as stream:
                os.fsync(stream.fileno())
            resume_rng_state = capture_rng_state(trainer)
            resume_checkpoint_state = optimizer_training_state(trainer)

        def stop_after_first_epoch(trainer: Any) -> None:
            if int(trainer.epoch) == 0:
                trainer.stop = True

        for event, callback in (
            ("on_train_batch_start", on_train_batch_start),
            ("on_train_batch_end", on_train_batch_end),
            ("on_train_start", on_initial_train_start),
            ("on_val_end", on_val_end),
            ("on_model_save", on_model_save),
            ("on_train_epoch_end", stop_after_first_epoch),
        ):
            model.add_callback(event, callback)

        if effective_args is None:
            train_kwargs: dict[str, Any] = {
                "epochs": 2,
                "imgsz": 64,
                # One initial batch and one resumed batch are the hard envelope.
                "batch": train_sample_count,
                "workers": 0,
                "device": selected_device,
                "optimizer": "SGD",
                "lr0": 0.001,
                "amp": False,
                "cache": False,
                "pretrained": False,
                "deterministic": True,
                "seed": seed,
                "val": True,
                "plots": False,
                "save": True,
                "verbose": False,
            }
        else:
            train_kwargs = dict(effective_args)
        train_kwargs.update(
            {
                "data": str(data_yaml),
                "project": str(output_root),
                "name": initial_name,
                "exist_ok": False,
                "resume": False,
            }
        )
        model.train(**train_kwargs)
        for hook_handle in hook_handles:
            hook_handle.remove()
        hook_handles.clear()

        if real_mode:
            initial_phase_valid = (
                observed_train_batches == 1
                and observed_backward_calls == 1
                and len(backward_gradient_finiteness) == 1
                and type(backward_gradient_finiteness[0]) is bool
                and observed_optimizer_step_calls == 1
                and len(optimizer_evidence) == 1
                and amp_overflow_skips + successful_optimizer_updates == 1
            )
            if not initial_phase_valid:
                raise DetectionInfrastructureError(
                    "Initial real-smoke phase exceeded its one-batch envelope.",
                    code="smoke_bound_exceeded",
                    details=[
                        {
                            "observed_train_batches": observed_train_batches,
                            "observed_backward_calls": observed_backward_calls,
                            "backward_gradient_finiteness": (
                                backward_gradient_finiteness
                            ),
                            "observed_optimizer_step_calls": observed_optimizer_step_calls,
                            "successful_optimizer_updates": successful_optimizer_updates,
                            "amp_overflow_skips": amp_overflow_skips,
                        }
                    ],
                )
            if successful_optimizer_updates == 1:
                raise DetectionInfrastructureError(
                    "Initial smoke already applied the sole authorized optimizer update; "
                    "refusing a resumed training batch.",
                    code="smoke_first_update_before_resume",
                    details=[{"optimizer_step_evidence": optimizer_evidence}],
                )

        last = initial_dir / "weights" / "last.pt"
        if not last.is_file():
            raise DetectionInfrastructureError(
                "Ultralytics smoke did not produce weights/last.pt.",
                code="smoke_checkpoint_failed",
            )
        flags["checkpoint"] = True
        if resumable_checkpoint.is_symlink() or not resumable_checkpoint.is_file():
            raise DetectionInfrastructureError(
                "Ultralytics smoke did not preserve a safe resumable checkpoint.",
                code="smoke_checkpoint_failed",
            )
        checkpoint_sha256 = sha256_file(resumable_checkpoint)
        parent_cuda_release: dict[str, Any] | None = None
        if real_mode:
            # The resume is deliberately a fresh process. Release the parent
            # trainer/model and its cached CUDA blocks before spawning it so a
            # bounded smoke cannot transiently require two training footprints.
            torch.cuda.synchronize(torch_device)
            allocated_before = int(torch.cuda.memory_allocated(torch_device))
            reserved_before = int(torch.cuda.memory_reserved(torch_device))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch_device)
            allocated_after = int(torch.cuda.memory_allocated(torch_device))
            reserved_after = int(torch.cuda.memory_reserved(torch_device))
            if allocated_before <= 0 or allocated_after >= allocated_before:
                raise DetectionInfrastructureError(
                    "Parent CUDA training state was not released before fresh-process resume.",
                    code="smoke_resume_failed",
                )
            parent_cuda_release = {
                "status": "PASS",
                "released_before_fresh_process": True,
                "device": selected_device,
                "allocated_before_bytes": allocated_before,
                "allocated_after_bytes": allocated_after,
                "reserved_before_bytes": reserved_before,
                "reserved_after_bytes": reserved_after,
            }
        resumed_model_instance_id: int
        resume_pid: int
        fresh_process_resume = False
        resume_process_evidence: str | None = None
        resume_process_evidence_sha256: str | None = None
        if real_mode:
            if (
                train_settings is None
                or config_sha256 is None
                or resume_rng_state is None
                or resume_checkpoint_state is None
                or expected_args_sha256 is None
            ):
                raise DetectionInfrastructureError(
                    "Real smoke did not capture a resumable RNG/checkpoint state.",
                    code="smoke_checkpoint_failed",
                )
            token = secrets.token_urlsafe(32)
            result_path = output_root / "fresh-process-resume-result.json"
            request_path = output_root / "fresh-process-resume-request.json"
            request_body = {
                "schema_version": "floodsight-real-smoke-resume-request-v3",
                "authorization_token_sha256": stable_sha256(token),
                "checkpoint_path": str(resumable_checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "data_yaml_path": str(data_yaml.resolve(strict=True)),
                "data_yaml_sha256": sha256_file(data_yaml),
                "output_root": str(output_root.resolve(strict=True)),
                "result_path": str(result_path),
                "seed": seed,
                "device": selected_device,
                "train_settings": train_settings,
                "config_sha256": config_sha256,
                "training_source_sha256": training_source_sha256(),
                "rng_state": resume_rng_state,
                "checkpoint_training_state": resume_checkpoint_state,
                "maximum_resume_train_batches": REAL_SMOKE_MAX_TRAIN_BATCHES - 1,
                "maximum_resume_optimizer_step_calls": (
                    REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS - 1
                ),
                "cuda_visible_devices_at_parent_start": (
                    cuda_visible_devices_at_process_start
                ),
            }
            request = {**request_body, "request_sha256": stable_sha256(request_body)}
            with request_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(request, indent=2, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            worker_environment = _fresh_process_environment(
                seed,
                token,
                cuda_visible_devices_at_process_start,
            )
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "floodsight_detection.real_smoke_worker",
                        "--request",
                        str(request_path),
                    ],
                    env=worker_environment,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=7200,
                )
            except subprocess.TimeoutExpired as exc:
                raise DetectionInfrastructureError(
                    "Fresh-process real-smoke resume exceeded its bounded timeout.",
                    code="smoke_resume_failed",
                ) from exc
            if completed.returncode != 0 or result_path.is_symlink() or not result_path.is_file():
                raise DetectionInfrastructureError(
                    "Fresh-process real-smoke resume failed.",
                    code="smoke_resume_failed",
                    details=[
                        {
                            "returncode": completed.returncode,
                            "stdout_tail": completed.stdout[-2000:],
                            "stderr_tail": completed.stderr[-2000:],
                        }
                    ],
                )
            try:
                worker_result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DetectionInfrastructureError(
                    "Fresh-process real-smoke result is unreadable.",
                    code="smoke_resume_failed",
                ) from exc
            worker_unsigned = {
                key: value for key, value in worker_result.items() if key != "result_sha256"
            }
            worker_step_calls = worker_result.get("observed_optimizer_step_calls")
            worker_train_batches = worker_result.get("observed_train_batches")
            worker_validation_runs = worker_result.get("observed_validation_runs")
            worker_optimizer_evidence = worker_result.get("optimizer_step_evidence")
            worker_backward_finiteness = worker_result.get(
                "backward_gradient_finiteness"
            )
            worker_required_true = (
                "loader",
                "loss",
                "backward",
                "optimizer_step",
                "validation",
                "checkpoint",
                "resume",
                "effective_args",
                "augmentation",
                "rng_restored",
            )
            if (
                worker_result.get("schema_version")
                != "floodsight-real-smoke-resume-result-v3"
                or worker_result.get("status") != "PASS"
                or worker_result.get("result_sha256") != stable_sha256(worker_unsigned)
                or any(worker_result.get(key) is not True for key in worker_required_true)
                or worker_result.get("checkpoint_sha256") != checkpoint_sha256
                or worker_result.get("effective_args_sha256") != expected_args_sha256
                or worker_result.get("training_source_sha256") != training_source_sha256()
                or type(worker_step_calls) is not int
                or not 1 <= worker_step_calls <= REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS - 1
                or worker_result.get("maximum_optimizer_step_calls")
                != REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS - 1
                or type(worker_train_batches) is not int
                or not 1 <= worker_train_batches <= REAL_SMOKE_MAX_TRAIN_BATCHES - 1
                or worker_result.get("maximum_train_batches")
                != REAL_SMOKE_MAX_TRAIN_BATCHES - 1
                or worker_step_calls > worker_train_batches
                or worker_result.get("observed_backward_calls") != worker_train_batches
                or not isinstance(worker_backward_finiteness, list)
                or len(worker_backward_finiteness) != worker_train_batches
                or any(type(value) is not bool for value in worker_backward_finiteness)
                or worker_backward_finiteness[-1] is not True
                or type(worker_validation_runs) is not int
                or not 1 <= worker_validation_runs <= worker_train_batches + 1
                or worker_result.get("successful_optimizer_updates") != 1
                or worker_result.get("amp_overflow_skips") != worker_step_calls - 1
                or not isinstance(worker_optimizer_evidence, list)
                or len(worker_optimizer_evidence) != worker_step_calls
                or any(
                    not isinstance(evidence, dict)
                    or evidence.get("phase") != "resume"
                    or evidence.get("phase_call_index") != index
                    or evidence.get("successful_update") is not False
                    or evidence.get("amp_overflow_skip") is not True
                    or evidence.get("underlying_optimizer_step") is not False
                    for index, evidence in enumerate(
                        worker_optimizer_evidence[:-1], start=1
                    )
                )
                or worker_optimizer_evidence[-1].get("successful_update") is not True
                or worker_optimizer_evidence[-1].get("amp_overflow_skip") is not False
                or worker_optimizer_evidence[-1].get("underlying_optimizer_step") is not True
                or not isinstance(worker_result.get("resume_state"), dict)
                or worker_result["resume_state"].get("checkpoint_training_state_match")
                is not True
                or worker_result["resume_state"].get("checkpoint_training_state")
                != resume_checkpoint_state
                or worker_result["resume_state"].get("configured_effective_epochs")
                != effective_args["epochs"]
                or worker_result["resume_state"].get(
                    "optimizer_state_entries_after_batch", 0
                )
                < 1
                or worker_result["resume_state"].get(
                    "optimizer_max_parameter_step_after_batch", 0
                )
                <= 0
                or not isinstance(worker_result.get("hash_seed_runtime"), dict)
                or worker_result["hash_seed_runtime"].get("python_hash_seed")
                != str(seed)
                or worker_result["hash_seed_runtime"].get(
                    "python_hash_seed_prestarted"
                )
                is not True
                or worker_result.get("cuda_visible_devices_at_process_start")
                != cuda_visible_devices_at_process_start
            ):
                raise DetectionInfrastructureError(
                    "Fresh-process real-smoke evidence failed validation.",
                    code="real_smoke_evidence_invalid",
                )
            resume_pid = worker_result["pid"]
            resumed_model_instance_id = worker_result["model_instance_id"]
            if (
                type(resume_pid) is not int
                or resume_pid == initial_pid
                or worker_result.get("parent_pid") != initial_pid
                or type(resumed_model_instance_id) is not int
                or resumed_model_instance_id <= 0
            ):
                raise DetectionInfrastructureError(
                    "Real-smoke resume did not execute in a fresh process.",
                    code="smoke_resume_failed",
                )
            fresh_process_resume = True
            resume_process_evidence = str(result_path.resolve(strict=True))
            resume_process_evidence_sha256 = sha256_file(result_path)
            flags["resume"] = True
            flags["distinct_model_instance_resume"] = True
            flags["rng_restored"] = True
            observed_train_batches += worker_result["observed_train_batches"]
            observed_backward_calls += worker_result["observed_backward_calls"]
            backward_gradient_finiteness.extend(worker_backward_finiteness)
            observed_optimizer_step_calls += worker_step_calls
            observed_validation_runs += worker_validation_runs
            successful_optimizer_updates += 1
            amp_overflow_skips += worker_result["amp_overflow_skips"]
            optimizer_evidence.extend(worker_optimizer_evidence)
            observed_effective_args_sha256.append(worker_result["effective_args_sha256"])
            augmentation_evidence.append(worker_result["augmentation_evidence"])
            resume_state.update(worker_result["resume_state"])
        else:
            resumed = yolo_class(str(resumable_checkpoint), task="detect")
            resumed_model_instance_id = id(resumed)
            resume_pid = os.getpid()

            def on_resume_start(trainer: Any) -> None:
                flags["resume"] = True
                install_backward_hook(trainer)
                install_optimizer_counter(trainer, phase="resume")
                resume_state.update(
                    {
                        "start_epoch": int(trainer.start_epoch),
                        "configured_effective_epochs": int(trainer.epochs),
                        "optimizer_state_entries": sum(
                            len(state) for state in trainer.optimizer.state.values()
                        ),
                        "scheduler_last_epoch": int(trainer.scheduler.last_epoch),
                    }
                )

            resumed.add_callback("on_train_start", on_resume_start)
            resumed.add_callback("on_train_batch_start", on_train_batch_start)
            resumed.add_callback("on_train_batch_end", on_train_batch_end)
            resumed.add_callback("on_val_end", on_val_end)
            resumed.train(resume=True, device=selected_device)
            for hook_handle in hook_handles:
                hook_handle.remove()

        flags["optimizer_step"] = (
            (not real_mode and observed_optimizer_step_calls == 2)
            or (
                real_mode
                and 2
                <= observed_optimizer_step_calls
                <= REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS
                and successful_optimizer_updates == 1
            )
        )
        if (
            observed_backward_calls != observed_train_batches
            or len(backward_gradient_finiteness) != observed_train_batches
            or any(type(value) is not bool for value in backward_gradient_finiteness)
            or observed_optimizer_step_calls > observed_train_batches
            or (
                real_mode
                and not 2 <= observed_train_batches <= REAL_SMOKE_MAX_TRAIN_BATCHES
            )
            or (
                real_mode
                and not 2
                <= observed_validation_runs
                <= observed_train_batches + 2
            )
            or (real_mode and backward_gradient_finiteness[-1] is not True)
            or (
                not real_mode
                and observed_optimizer_step_calls != 2
            )
            or (
                real_mode
                and (
                    not 2
                    <= observed_optimizer_step_calls
                    <= REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS
                    or successful_optimizer_updates != 1
                    or amp_overflow_skips != observed_optimizer_step_calls - 1
                    or len(optimizer_evidence) != observed_optimizer_step_calls
                    or optimizer_evidence[0].get("phase") != "initial"
                    or optimizer_evidence[0].get("successful_update") is not False
                    or optimizer_evidence[0].get("amp_overflow_skip") is not True
                    or optimizer_evidence[0].get("underlying_optimizer_step") is not False
                    or any(
                        evidence.get("phase") != "resume"
                        or evidence.get("phase_call_index") != index
                        or evidence.get("successful_update") is not False
                        or evidence.get("amp_overflow_skip") is not True
                        or evidence.get("underlying_optimizer_step") is not False
                        for index, evidence in enumerate(
                            optimizer_evidence[1:-1], start=1
                        )
                    )
                    or optimizer_evidence[-1].get("phase") != "resume"
                    or optimizer_evidence[-1].get("phase_call_index")
                    != observed_optimizer_step_calls - 1
                    or optimizer_evidence[-1].get("successful_update") is not True
                    or optimizer_evidence[-1].get("amp_overflow_skip") is not False
                    or optimizer_evidence[-1].get("underlying_optimizer_step") is not True
                )
            )
        ):
            raise DetectionInfrastructureError(
                "Bounded smoke did not stop at its first applied optimizer update.",
                code="smoke_bound_exceeded",
                details=[
                    {
                        "observed_train_batches": observed_train_batches,
                        "observed_backward_calls": observed_backward_calls,
                        "backward_gradient_finiteness": backward_gradient_finiteness,
                        "observed_optimizer_step_calls": observed_optimizer_step_calls,
                        "observed_validation_runs": observed_validation_runs,
                        "successful_optimizer_updates": successful_optimizer_updates,
                        "amp_overflow_skips": amp_overflow_skips,
                        "maximum_train_batches": (
                            REAL_SMOKE_MAX_TRAIN_BATCHES if real_mode else 2
                        ),
                        "maximum_optimizer_step_calls": (
                            REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS if real_mode else 2
                        ),
                    }
                ],
            )
        if not flags["resume"]:
            raise DetectionInfrastructureError(
                "Ultralytics checkpoint resume callback was not reached.",
                code="smoke_resume_failed",
            )
        if sha256_file(resumable_checkpoint) != checkpoint_sha256:
            raise DetectionInfrastructureError(
                "The preserved resume checkpoint changed during resume.",
                code="smoke_checkpoint_failed",
            )
        missing = sorted(name for name, passed in flags.items() if not passed)
        if missing:
            raise DetectionInfrastructureError(
                f"Ultralytics smoke did not exercise: {', '.join(missing)}.",
                code=("real_smoke_incomplete" if real_mode else "synthetic_smoke_incomplete"),
            )

        result = {
            **flags,
            "mode": "config_bound_real" if real_mode else "generated_synthetic",
            "ultralytics_version": installed_version,
            "torch_version": torch_version,
            "torchvision_version": torchvision_version,
            "model": model_source,
            "initial_checkpoint": str(last),
            "resumed_checkpoint": str(resumable_checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "train_sample_count": train_sample_count,
            "observed_train_batches": observed_train_batches,
            "observed_backward_calls": observed_backward_calls,
            "backward_gradient_finiteness": backward_gradient_finiteness,
            "observed_optimizer_step_calls": observed_optimizer_step_calls,
            "observed_validation_runs": observed_validation_runs,
            "maximum_train_batches": (
                REAL_SMOKE_MAX_TRAIN_BATCHES if real_mode else 2
            ),
            "maximum_optimizer_step_calls": (
                REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS if real_mode else 2
            ),
            "initial_pid": initial_pid,
            "resume_pid": resume_pid,
            "resume_parent_pid": (
                worker_result["parent_pid"] if real_mode else initial_pid
            ),
            "initial_model_instance_id": initial_model_instance_id,
            "resume_model_instance_id": resumed_model_instance_id,
            "fresh_process_resume": fresh_process_resume,
            "resume_process_evidence": resume_process_evidence,
            "resume_process_evidence_sha256": resume_process_evidence_sha256,
            "resume_state": resume_state,
            "parent_cuda_release": parent_cuda_release,
        }
        if real_mode:
            if (
                train_settings is None
                or config_sha256 is None
                or expected_args_sha256 is None
                or expected_augmentation_args is None
            ):
                raise DetectionInfrastructureError(
                    "Real-smoke report state is incomplete.",
                    code="real_smoke_config_drift",
                )
            result.update(
                {
                    "config_sha256": config_sha256,
                    "model_source_sha256": sha256_file(Path(model_source)),
                    "runtime": runtime,
                    "hash_seed_runtime": hash_seed_runtime,
                    "bounded_override_allowlist": [],
                    "bounded_overrides": {},
                    "runtime_bounds": {
                        "configured_epochs_preserved": train_settings["epochs"],
                        "maximum_train_batches": REAL_SMOKE_MAX_TRAIN_BATCHES,
                        "maximum_optimizer_step_calls": (
                            REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS
                        ),
                        "stop_condition": (
                            "first_underlying_adamw_step_post_hook"
                        ),
                        "rationale": (
                            "four_calls_empirically_exhausted_at_amp_scale_4096;"
                            "bounded_backoff_window_expanded"
                        ),
                    },
                    "effective_args_sha256": expected_args_sha256,
                    "observed_effective_args_sha256": observed_effective_args_sha256,
                    "augmentation_args_sha256": stable_sha256(
                        expected_augmentation_args
                    ),
                    "augmentation_evidence": augmentation_evidence,
                    "successful_optimizer_updates": successful_optimizer_updates,
                    "amp_overflow_skips": amp_overflow_skips,
                    "optimizer_step_evidence": optimizer_evidence,
                    "minimum_successful_optimizer_updates": 1,
                    "training_source_sha256": training_source_sha256(),
                    "cuda_visible_devices_at_process_start": (
                        cuda_visible_devices_at_process_start
                    ),
                }
            )
        return result


# Compatibility for callers outside this tree; internal callers use the
# generalized name so a real-manifest run is never described as synthetic.
UltralyticsSyntheticBackend = UltralyticsSmokeBackend
