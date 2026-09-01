"""Inherently bounded fresh-process resume phase for the real YOLO smoke gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from floodsight_detection.checkpointing import capture_rng_state, restore_rng_state
from floodsight_detection.config import (
    ULTRALYTICS_TRAIN_VAL_KEYS,
    validate_ultralytics_default_contract,
)
from floodsight_detection.determinism import configure_determinism, require_prestarted_hash_seed
from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file, stable_sha256, training_source_sha256
from floodsight_detection.runtime import validate_full_training_runtime
from floodsight_detection.ultralytics_runtime import (
    REAL_SMOKE_AUGMENTATION_KEYS,
    REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS,
    REAL_SMOKE_MAX_TRAIN_BATCHES,
    _observed_trainer_arguments,
    _require_package_version,
    _require_ultralytics,
    audited_transform_configuration,
    bounded_loader_batch_size,
    optimizer_runtime_context,
    optimizer_step_evidence,
    optimizer_training_state,
    real_smoke_effective_arguments,
)

_REQUEST_FIELDS = {
    "schema_version",
    "request_sha256",
    "authorization_token_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "data_yaml_path",
    "data_yaml_sha256",
    "output_root",
    "result_path",
    "seed",
    "device",
    "train_settings",
    "config_sha256",
    "training_source_sha256",
    "rng_state",
    "checkpoint_training_state",
    "maximum_resume_train_batches",
    "maximum_resume_optimizer_step_calls",
    "cuda_visible_devices_at_parent_start",
}
_REQUIRED_TRANSFORMS = frozenset(
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


def _fail(message: str, code: str = "real_smoke_worker_invalid") -> None:
    raise DetectionInfrastructureError(message, code=code)


def _safe_file(path: Path, *, root: Path, expected_sha256: str) -> Path:
    if path.is_symlink() or not path.is_absolute() or not path.is_file():
        _fail(f"Fresh-process smoke input is unsafe: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DetectionInfrastructureError(
            "Fresh-process smoke input escapes its authorized output root.",
            code="real_smoke_worker_invalid",
        ) from exc
    if sha256_file(resolved) != expected_sha256:
        _fail(f"Fresh-process smoke input hash drifted: {resolved}")
    return resolved


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise DetectionInfrastructureError(
            "Fresh-process smoke result already exists.", code="smoke_collision"
        ) from exc


def run_worker(request_path: Path) -> dict[str, Any]:
    if request_path.is_symlink() or not request_path.is_absolute() or not request_path.is_file():
        _fail("Fresh-process smoke request is unsafe.")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionInfrastructureError(
            "Fresh-process smoke request is unreadable.", code="real_smoke_worker_invalid"
        ) from exc
    if not isinstance(request, dict) or set(request) != _REQUEST_FIELDS:
        _fail("Fresh-process smoke request fields drifted.")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    token = os.environ.pop("FLOODSIGHT_REAL_SMOKE_WORKER_TOKEN", None)
    if (
        request.get("schema_version") != "floodsight-real-smoke-resume-request-v3"
        or request.get("request_sha256") != stable_sha256(unsigned)
        or not isinstance(token, str)
        or stable_sha256(token) != request.get("authorization_token_sha256")
    ):
        _fail("Fresh-process smoke request authorization failed.")
    current_source = training_source_sha256()
    if request.get("training_source_sha256") != current_source:
        _fail("Fresh-process smoke source hash drifted.")
    expected_cuda_visible_devices = request.get("cuda_visible_devices_at_parent_start")
    if (
        expected_cuda_visible_devices is not None
        and (
            not isinstance(expected_cuda_visible_devices, str)
            or not expected_cuda_visible_devices
        )
    ) or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_cuda_visible_devices:
        _fail("Fresh-process CUDA visibility differs from the parent process start.")
    output_root = Path(request["output_root"])
    if output_root.is_symlink() or not output_root.is_absolute() or not output_root.is_dir():
        _fail("Fresh-process smoke output root is unsafe.")
    output_root = output_root.resolve(strict=True)
    request_resolved = request_path.resolve(strict=True)
    try:
        request_resolved.relative_to(output_root)
    except ValueError as exc:
        raise DetectionInfrastructureError(
            "Fresh-process smoke request is outside its output root.",
            code="real_smoke_worker_invalid",
        ) from exc
    checkpoint = _safe_file(
        Path(request["checkpoint_path"]),
        root=output_root,
        expected_sha256=request["checkpoint_sha256"],
    )
    # The dataset contract is a sibling beneath the one parent smoke output.
    smoke_root = output_root.parent
    data_yaml = _safe_file(
        Path(request["data_yaml_path"]),
        root=smoke_root,
        expected_sha256=request["data_yaml_sha256"],
    )
    result_path = Path(request["result_path"])
    if (
        result_path.is_symlink()
        or not result_path.is_absolute()
        or result_path.parent.resolve(strict=True) != output_root
        or result_path.exists()
    ):
        _fail("Fresh-process smoke result path is unsafe or occupied.")
    seed = request["seed"]
    device = request["device"]
    train_settings = request["train_settings"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not isinstance(device, str):
        _fail("Fresh-process smoke seed/device is invalid.")
    if not isinstance(train_settings, dict) or set(train_settings) != set(
        ULTRALYTICS_TRAIN_VAL_KEYS
    ):
        _fail("Fresh-process smoke training arguments are not exhaustive.")
    effective = real_smoke_effective_arguments(train_settings)
    if effective["seed"] != seed or effective["device"] != device:
        _fail("Fresh-process smoke seed/device differs from frozen arguments.")
    maximum_resume_train_batches = REAL_SMOKE_MAX_TRAIN_BATCHES - 1
    maximum_resume_step_calls = REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS - 1
    if (
        request.get("maximum_resume_train_batches") != maximum_resume_train_batches
        or request.get("maximum_resume_optimizer_step_calls")
        != maximum_resume_step_calls
        or effective["epochs"] != train_settings["epochs"]
    ):
        _fail("Fresh-process smoke execution bounds drifted.")
    hash_seed = require_prestarted_hash_seed(seed)
    runtime = validate_full_training_runtime(device)
    configure_determinism(seed, include_ml_libraries=False)
    yolo_class, ultralytics_version = _require_ultralytics("8.3.222")
    import importlib

    ultralytics_cfg = importlib.import_module("ultralytics.cfg")
    validate_ultralytics_default_contract(ultralytics_cfg.DEFAULT_CFG_DICT)
    torch_version = _require_package_version("torch", "2.13.0+cu130")
    torchvision_version = _require_package_version("torchvision", "0.28.0+cu130")
    import torch

    configure_determinism(seed, include_ml_libraries=True)
    model = yolo_class(str(checkpoint), task="detect")
    model_instance_id = id(model)
    flags = {
        "loader": False,
        "loss": False,
        "backward": False,
        "optimizer_step": False,
        "validation": False,
        "checkpoint": True,
        "resume": False,
        "effective_args": False,
        "augmentation": False,
        "rng_restored": False,
    }
    counts = {"batches": 0, "backward": 0, "optimizer": 0, "validation": 0}
    successful_optimizer_updates = 0
    amp_overflow_skips = 0
    resume_state: dict[str, Any] = {}
    augmentation_evidence: dict[str, Any] = {}
    optimizer_evidence: list[dict[str, Any]] = []
    backward_gradient_finiteness: list[bool] = []
    hook_handles: list[Any] = []
    expected_hash = stable_sha256(effective)
    expected_augmentation = {key: effective[key] for key in sorted(REAL_SMOKE_AUGMENTATION_KEYS)}

    def backward_hook(gradient: Any) -> Any:
        if gradient is not None:
            flags["backward"] = True
            counts["backward"] += 1
            backward_gradient_finiteness.append(
                bool(torch.isfinite(gradient).all())
            )
        return gradient

    def on_train_start(trainer: Any) -> None:
        nonlocal amp_overflow_skips, successful_optimizer_updates
        observed = _observed_trainer_arguments(trainer)
        if observed != effective or stable_sha256(observed) != expected_hash:
            _fail(
                "Fresh-process resumed trainer normalized frozen arguments.",
                "real_smoke_config_drift",
            )
        if Path(str(trainer.args.data)).resolve(strict=True) != data_yaml:
            _fail(
                "Fresh-process resumed trainer redirected its dataset.", "real_smoke_config_drift"
            )
        if Path(trainer.save_dir).resolve() != output_root / "ultralytics-real-smoke-initial":
            _fail("Fresh-process resumed trainer redirected output.", "real_smoke_config_drift")
        dataset = trainer.train_loader.dataset
        transform_names: list[str] = []

        def visit(transform: Any) -> None:
            transform_names.append(type(transform).__name__)
            nested = getattr(transform, "transforms", None)
            if isinstance(nested, (list, tuple)):
                for child in nested:
                    visit(child)

        visit(dataset.transforms)
        missing = sorted(_REQUIRED_TRANSFORMS - set(transform_names))
        transform_configuration = audited_transform_configuration(
            dataset.transforms, effective
        )
        dataset_size = len(dataset)
        expected_loader_batch_size = bounded_loader_batch_size(
            effective["batch"], dataset_size
        )
        if (
            dataset.augment is not True
            or trainer.train_loader.batch_size != expected_loader_batch_size
            or trainer.train_loader.num_workers != 0
            or missing
        ):
            _fail(
                "Fresh-process resume augmentation/loader contract drifted.",
                "real_smoke_augmentation_drift",
            )
        restore_rng_state(request["rng_state"], trainer)
        restored = capture_rng_state(trainer)
        if restored["state_sha256"] != request["rng_state"].get("state_sha256"):
            _fail(
                "Fresh-process RNG restoration did not reproduce the saved state.",
                "checkpoint_rng_invalid",
            )
        flags["rng_restored"] = True
        flags["resume"] = True
        flags["effective_args"] = True
        flags["augmentation"] = True
        checkpoint_training_state = request.get("checkpoint_training_state")
        restored_training_state = optimizer_training_state(trainer)
        if (
            not isinstance(checkpoint_training_state, dict)
            or restored_training_state != checkpoint_training_state
            or restored_training_state["optimizer_param_group_count"] < 1
            or restored_training_state["optimizer_parameter_count"] < 1
            or not isinstance(restored_training_state["amp_scale"], float)
            or restored_training_state["amp_scale"] <= 0
        ):
            _fail(
                "Fresh-process resume did not exactly restore checkpoint optimizer/scaler state.",
                "smoke_resume_failed",
            )
        resume_state.update(
            {
                "start_epoch": int(trainer.start_epoch),
                "configured_effective_epochs": int(trainer.epochs),
                "checkpoint_training_state_match": True,
                "checkpoint_training_state": restored_training_state,
                "optimizer_state_entries": restored_training_state[
                    "optimizer_state_entries"
                ],
                "scheduler_last_epoch": int(trainer.scheduler.last_epoch),
            }
        )
        if (
            resume_state["start_epoch"] != 1
            or resume_state["scheduler_last_epoch"]
            != resume_state["start_epoch"] - 1
        ):
            _fail(
                "Fresh-process resume did not reconstruct the epoch-one scheduler boundary.",
                "smoke_resume_failed",
            )
        augmentation_evidence.update(
            {
                "phase": "resume",
                "dataset_augment": True,
                "configured_batch_size": effective["batch"],
                "loader_batch_size": trainer.train_loader.batch_size,
                "loader_batch_policy": "min(configured_batch_size,dataset_size)",
                "loader_num_workers": trainer.train_loader.num_workers,
                "dataset_size": dataset_size,
                "arguments": expected_augmentation,
                "arguments_sha256": stable_sha256(expected_augmentation),
                "transform_class_names": transform_names,
                "transform_classes_sha256": stable_sha256(transform_names),
                "transform_configuration": transform_configuration,
                "transform_configuration_sha256": stable_sha256(
                    transform_configuration
                ),
            }
        )
        first_trainable = next(
            parameter for parameter in trainer.model.parameters() if parameter.requires_grad
        )
        hook_handles.append(first_trainable.register_hook(backward_hook))
        original_step = trainer.optimizer_step

        def applied_optimizer_step(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
            nonlocal successful_optimizer_updates
            successful_optimizer_updates += 1

        hook_handles.append(
            trainer.optimizer.register_step_post_hook(applied_optimizer_step)
        )

        def counted_step() -> None:
            nonlocal amp_overflow_skips
            if counts["optimizer"] >= maximum_resume_step_calls:
                _fail(
                    "Fresh-process smoke refused an optimizer call beyond its hard ceiling.",
                    "smoke_bound_exceeded",
                )
            before = optimizer_training_state(trainer)
            applied_before = successful_optimizer_updates
            context = optimizer_runtime_context(
                trainer, phase_batch_index=counts["batches"]
            )
            original_step()
            counts["optimizer"] += 1
            evidence = optimizer_step_evidence(trainer, phase="resume", before=before)
            evidence["phase_call_index"] = counts["optimizer"]
            evidence["runtime_context"] = context
            evidence["underlying_optimizer_step"] = (
                successful_optimizer_updates == applied_before + 1
            )
            if evidence["successful_update"] != evidence["underlying_optimizer_step"]:
                _fail(
                    "Fresh-process optimizer state differs from its underlying step hook.",
                    "smoke_optimizer_evidence_invalid",
                )
            amp_overflow_skips += int(evidence["amp_overflow_skip"])
            optimizer_evidence.append(evidence)
            if successful_optimizer_updates > 1:
                _fail(
                    "Fresh-process smoke exceeded one applied optimizer update.",
                    "smoke_bound_exceeded",
                )
            if successful_optimizer_updates == 1:
                trainer.stop = True

        trainer.optimizer_step = counted_step

    def on_batch_start(_trainer: Any) -> None:
        if counts["batches"] >= maximum_resume_train_batches:
            _fail(
                "Fresh-process resume refused a batch beyond its hard ceiling.",
                "smoke_bound_exceeded",
            )
        counts["batches"] += 1
        flags["loader"] = True

    def on_batch_end(trainer: Any) -> None:
        flags["loss"] = bool(torch.isfinite(trainer.loss).all())
        if (
            successful_optimizer_updates == 1
            or counts["batches"] == maximum_resume_train_batches
            or counts["optimizer"] == maximum_resume_step_calls
        ):
            trainer.stop = True

    def on_val_end(_validator: Any) -> None:
        flags["validation"] = True
        counts["validation"] += 1

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_start", on_batch_start)
    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_val_end", on_val_end)
    model.train(resume=True, device=device)
    for handle in hook_handles:
        handle.remove()
    flags["optimizer_step"] = successful_optimizer_updates == 1
    if optimizer_evidence:
        final_step_evidence = optimizer_evidence[-1]
        resume_state.update(
            {
                "optimizer_state_after_batch": final_step_evidence["after"],
                "optimizer_state_entries_after_batch": final_step_evidence["after"][
                    "optimizer_state_entries"
                ],
                "optimizer_max_parameter_step_after_batch": final_step_evidence[
                    "after"
                ]["optimizer_max_parameter_step"],
            }
        )
    if (
        not 1 <= counts["batches"] <= maximum_resume_train_batches
        or not 1 <= counts["optimizer"] <= maximum_resume_step_calls
        or counts["optimizer"] > counts["batches"]
        or counts["backward"] != counts["batches"]
        or len(backward_gradient_finiteness) != counts["batches"]
        or any(type(value) is not bool for value in backward_gradient_finiteness)
        or backward_gradient_finiteness[-1] is not True
        or not 1 <= counts["validation"] <= counts["batches"] + 1
        or successful_optimizer_updates != 1
        or amp_overflow_skips != counts["optimizer"] - 1
        or len(optimizer_evidence) != counts["optimizer"]
        or any(
            evidence.get("phase") != "resume"
            or evidence.get("phase_call_index") != index
            or evidence.get("successful_update") is not False
            or evidence.get("amp_overflow_skip") is not True
            or evidence.get("underlying_optimizer_step") is not False
            for index, evidence in enumerate(optimizer_evidence[:-1], start=1)
        )
        or optimizer_evidence[-1].get("successful_update") is not True
        or optimizer_evidence[-1].get("amp_overflow_skip") is not False
        or optimizer_evidence[-1].get("underlying_optimizer_step") is not True
        or resume_state.get("optimizer_state_entries_after_batch", 0) < 1
        or resume_state.get("optimizer_max_parameter_step_after_batch", 0) <= 0
        or not all(flags.values())
    ):
        raise DetectionInfrastructureError(
            "Fresh-process resume did not stop at its first applied optimizer update.",
            code="smoke_bound_exceeded",
            details=[
                {
                    "counts": counts,
                    "backward_gradient_finiteness": backward_gradient_finiteness,
                    "maximum_resume_train_batches": maximum_resume_train_batches,
                    "maximum_resume_optimizer_step_calls": maximum_resume_step_calls,
                    "successful_optimizer_updates": successful_optimizer_updates,
                    "amp_overflow_skips": amp_overflow_skips,
                    "optimizer_step_evidence": optimizer_evidence,
                }
            ],
        )
    result = {
        "schema_version": "floodsight-real-smoke-resume-result-v3",
        "status": "PASS",
        **flags,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "model_instance_id": model_instance_id,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "observed_train_batches": counts["batches"],
        "observed_backward_calls": counts["backward"],
        "backward_gradient_finiteness": backward_gradient_finiteness,
        "observed_optimizer_step_calls": counts["optimizer"],
        "observed_validation_runs": counts["validation"],
        "maximum_train_batches": maximum_resume_train_batches,
        "maximum_optimizer_step_calls": maximum_resume_step_calls,
        "successful_optimizer_updates": successful_optimizer_updates,
        "amp_overflow_skips": amp_overflow_skips,
        "optimizer_step_evidence": optimizer_evidence,
        "effective_args_sha256": expected_hash,
        "augmentation_evidence": augmentation_evidence,
        "resume_state": resume_state,
        "training_source_sha256": current_source,
        "runtime": runtime,
        "hash_seed_runtime": hash_seed,
        "cuda_visible_devices_at_process_start": expected_cuda_visible_devices,
        "ultralytics_version": ultralytics_version,
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
    }
    result["result_sha256"] = stable_sha256(result)
    _write_result(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_worker(args.request)
    except DetectionInfrastructureError as exc:
        print(json.dumps({"status": "FAIL", "error": exc.to_dict()}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
