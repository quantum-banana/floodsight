"""Bounded smoke over a tiny deterministic subset of a frozen real manifest."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from floodsight_detection.config import TrainingConfig, validate_frozen_model_artifacts
from floodsight_detection.contract import (
    DETECTION_CLASSES,
    DatasetContract,
    ValidatedSample,
    freeze_dataset_contract,
)
from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file, stable_sha256, training_source_sha256
from floodsight_detection.runtime import (
    bound_real_smoke_output_directory,
    bound_training_device,
)
from floodsight_detection.ultralytics_runtime import (
    REAL_SMOKE_AUGMENTATION_KEYS,
    REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS,
    REAL_SMOKE_MAX_TRAIN_BATCHES,
    expected_transform_configuration,
)
from floodsight_detection.weights import WeightArtifact


class RealSmokeBackend(Protocol):
    def run(
        self,
        *,
        data_yaml: Path,
        output_root: Path,
        seed: int,
        device: str,
        train_settings: dict[str, Any],
        config_sha256: str,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RealSmokeAttestation:
    path: Path
    sha256: str
    output_directory: Path
    training_source_sha256: str


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_REAL_SMOKE_CHECKS = (
    "loader",
    "model_forward",
    "loss",
    "backward",
    "optimizer_step",
    "validation",
    "checkpoint",
    "resume",
    "effective_args",
    "augmentation",
    "distinct_model_instance_resume",
    "rng_restored",
)


def _representative_subset(contract: DatasetContract) -> DatasetContract:
    train = [sample for sample in contract.samples if sample.split == "train"]
    validation = [sample for sample in contract.samples if sample.split == "val"]
    selected_train: list[ValidatedSample] = []
    covered: set[int] = set()
    for sample in train:
        new_classes = set(sample.class_counts) - covered
        if new_classes:
            selected_train.append(sample)
            covered.update(sample.class_counts)
        if covered == set(DETECTION_CLASSES):
            break
    if covered != set(DETECTION_CLASSES):
        raise DetectionInfrastructureError(
            "Real smoke cannot cover every frozen detector class.",
            code="real_smoke_subset_incomplete",
        )
    selected_val = [sample for sample in validation if sample.box_count > 0][:2]
    if not selected_val:
        raise DetectionInfrastructureError(
            "Real smoke requires at least one non-empty validation sample.",
            code="real_smoke_subset_incomplete",
        )
    selected = tuple(selected_train + selected_val)
    split_counts = Counter(sample.split for sample in selected)
    class_counts: Counter[int] = Counter()
    for sample in selected:
        class_counts.update(sample.class_counts)
    return DatasetContract(
        manifest_path=contract.manifest_path,
        manifest_sha256=contract.manifest_sha256,
        dataset_fingerprint=contract.dataset_fingerprint,
        data_root=contract.data_root,
        samples=selected,
        split_counts=dict(split_counts),
        class_counts=dict(class_counts),
        total_boxes=sum(class_counts.values()),
        image_hashes_verified=contract.image_hashes_verified,
    )


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _optimizer_state_summary_valid(state: Any) -> bool:
    return (
        isinstance(state, dict)
        and state.get("optimizer_name") == "AdamW"
        and type(state.get("optimizer_state_parameter_count")) is int
        and state["optimizer_state_parameter_count"] >= 0
        and type(state.get("optimizer_state_entries")) is int
        and state["optimizer_state_entries"] >= 0
        and type(state.get("optimizer_param_group_count")) is int
        and state["optimizer_param_group_count"] > 0
        and isinstance(state.get("optimizer_param_groups_sha256"), str)
        and _HEX64.fullmatch(state["optimizer_param_groups_sha256"]) is not None
        and type(state.get("optimizer_parameter_count")) is int
        and state["optimizer_parameter_count"] > 0
        and isinstance(state.get("optimizer_max_parameter_step"), (int, float))
        and not isinstance(state["optimizer_max_parameter_step"], bool)
        and state["optimizer_max_parameter_step"] >= 0
        and isinstance(state.get("amp_scale"), (int, float))
        and not isinstance(state["amp_scale"], bool)
        and state["amp_scale"] > 0
        and isinstance(state.get("amp_scaler_state_sha256"), str)
        and _HEX64.fullmatch(state["amp_scaler_state_sha256"]) is not None
    )


def _optimizer_state_continues(current: dict[str, Any], previous: dict[str, Any]) -> bool:
    """Allow scheduler/warmup hyperparameters to change while state remains continuous."""

    keys = (
        "optimizer_name",
        "optimizer_state_parameter_count",
        "optimizer_state_entries",
        "optimizer_param_group_count",
        "optimizer_parameter_count",
        "optimizer_max_parameter_step",
        "amp_scale",
        "amp_scaler_state_sha256",
    )
    return all(current.get(key) == previous.get(key) for key in keys)


def _optimizer_runtime_context_valid(
    context: Any,
    *,
    expected_phase_batch_index: int,
    expected_epoch: int,
    expected_configured_batch_size: int,
    expected_loader_batch_size: int,
    expected_param_groups: int,
) -> bool:
    if not isinstance(context, dict) or set(context) != {
        "phase_batch_index",
        "epoch",
        "batches_per_epoch",
        "global_iteration",
        "configured_batch_size",
        "loader_batch_size",
        "accumulate",
        "scheduler_last_epoch",
        "learning_rates",
    }:
        return False
    learning_rates = context.get("learning_rates")
    return (
        context.get("phase_batch_index") == expected_phase_batch_index
        and context.get("epoch") == expected_epoch
        and context.get("batches_per_epoch") == 1
        and context.get("global_iteration") == expected_epoch
        and context.get("configured_batch_size") == expected_configured_batch_size
        and context.get("loader_batch_size") == expected_loader_batch_size
        and type(context.get("accumulate")) is int
        and context["accumulate"] >= 1
        and context.get("scheduler_last_epoch") == expected_epoch
        and isinstance(learning_rates, list)
        and len(learning_rates) == expected_param_groups
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for value in learning_rates
        )
    )


def _optimizer_evidence_valid(
    backend: Any,
    *,
    configured_epochs: int,
    configured_batch_size: int,
    loader_batch_size: int,
) -> bool:
    if not isinstance(backend, dict):
        return False
    evidence = backend.get("optimizer_step_evidence")
    step_calls = backend.get("observed_optimizer_step_calls")
    if (
        type(step_calls) is not int
        or not 2 <= step_calls <= REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS
        or type(backend.get("observed_train_batches")) is not int
        or not 2
        <= backend["observed_train_batches"]
        <= REAL_SMOKE_MAX_TRAIN_BATCHES
        or step_calls > backend["observed_train_batches"]
        or backend.get("observed_backward_calls")
        != backend["observed_train_batches"]
        or not isinstance(backend.get("backward_gradient_finiteness"), list)
        or len(backend["backward_gradient_finiteness"])
        != backend["observed_train_batches"]
        or any(
            type(value) is not bool
            for value in backend["backward_gradient_finiteness"]
        )
        or backend["backward_gradient_finiteness"][-1] is not True
        or backend.get("maximum_train_batches") != REAL_SMOKE_MAX_TRAIN_BATCHES
        or backend.get("maximum_optimizer_step_calls")
        != REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS
        or backend.get("successful_optimizer_updates") != 1
        or backend.get("minimum_successful_optimizer_updates") != 1
        or backend.get("amp_overflow_skips") != step_calls - 1
        or not isinstance(evidence, list)
        or len(evidence) != step_calls
        or not all(isinstance(item, dict) for item in evidence)
    ):
        return False
    initial = evidence[0]
    initial_before = initial.get("before")
    initial_after = initial.get("after")
    if not all(_optimizer_state_summary_valid(state) for state in (initial_before, initial_after)):
        return False
    assert isinstance(initial_before, dict)
    assert isinstance(initial_after, dict)
    if (
        initial.get("phase") != "initial"
        or initial.get("phase_call_index") != 1
        or initial.get("successful_update") is not False
        or initial.get("amp_overflow_skip") is not True
        or initial.get("underlying_optimizer_step") is not False
        or initial_before["optimizer_state_entries"] != 0
        or initial_before["optimizer_max_parameter_step"] != 0
        or initial_after["optimizer_state_entries"] != 0
        or initial_after["optimizer_max_parameter_step"] != 0
        or initial_after["amp_scale"] != initial_before["amp_scale"] * 0.5
        or initial_after["optimizer_param_groups_sha256"]
        != initial_before["optimizer_param_groups_sha256"]
        or not _optimizer_runtime_context_valid(
            initial.get("runtime_context"),
            expected_phase_batch_index=1,
            expected_epoch=0,
            expected_configured_batch_size=configured_batch_size,
            expected_loader_batch_size=loader_batch_size,
            expected_param_groups=initial_before["optimizer_param_group_count"],
        )
    ):
        return False
    previous = initial_after
    previous_phase_batch_index = 0
    for index, resumed in enumerate(evidence[1:], start=1):
        resume_before = resumed.get("before")
        resume_after = resumed.get("after")
        if (
            not _optimizer_state_summary_valid(resume_before)
            or not _optimizer_state_summary_valid(resume_after)
            or not _optimizer_state_continues(resume_before, previous)
            or resumed.get("phase") != "resume"
            or resumed.get("phase_call_index") != index
            or not isinstance(resumed.get("runtime_context"), dict)
        ):
            return False
        assert isinstance(resume_before, dict)
        assert isinstance(resume_after, dict)
        is_final = index == step_calls - 1
        runtime_context = resumed["runtime_context"]
        phase_batch_index = runtime_context.get("phase_batch_index")
        if (
            type(phase_batch_index) is not int
            or phase_batch_index < index
            or phase_batch_index <= previous_phase_batch_index
            or not _optimizer_runtime_context_valid(
                runtime_context,
                expected_phase_batch_index=phase_batch_index,
                expected_epoch=phase_batch_index,
                expected_configured_batch_size=configured_batch_size,
                expected_loader_batch_size=loader_batch_size,
                expected_param_groups=resume_before["optimizer_param_group_count"],
            )
        ):
            return False
        previous_phase_batch_index = phase_batch_index
        if is_final:
            if (
                resumed.get("successful_update") is not True
                or resumed.get("amp_overflow_skip") is not False
                or resumed.get("underlying_optimizer_step") is not True
                or resume_after["optimizer_state_entries"] <= 0
                or resume_after["optimizer_state_parameter_count"] <= 0
                or resume_after["optimizer_max_parameter_step"]
                <= resume_before["optimizer_max_parameter_step"]
                or resume_after["amp_scale"] < resume_before["amp_scale"]
            ):
                return False
        elif (
            resumed.get("successful_update") is not False
            or resumed.get("amp_overflow_skip") is not True
            or resumed.get("underlying_optimizer_step") is not False
            or resume_after["optimizer_state_entries"] != 0
            or resume_after["optimizer_max_parameter_step"] != 0
            or resume_after["amp_scale"] != resume_before["amp_scale"] * 0.5
        ):
            return False
        if (
            resume_after["optimizer_param_groups_sha256"]
            != resume_before["optimizer_param_groups_sha256"]
        ):
            return False
        previous = resume_after
    final_state = previous
    resume_state = backend.get("resume_state")
    return (
        isinstance(resume_state, dict)
        and resume_state.get("start_epoch") == 1
        and resume_state.get("configured_effective_epochs")
        == configured_epochs
        and resume_state.get("scheduler_last_epoch") == 0
        and resume_state.get("checkpoint_training_state_match") is True
        and resume_state.get("checkpoint_training_state") == initial_after
        and resume_state.get("optimizer_state_entries") == 0
        and resume_state.get("optimizer_state_after_batch") == final_state
        and resume_state.get("optimizer_state_entries_after_batch")
        == final_state["optimizer_state_entries"]
        and resume_state.get("optimizer_max_parameter_step_after_batch")
        == final_state["optimizer_max_parameter_step"]
    )


def run_real_manifest_smoke(
    config: TrainingConfig,
    contract: DatasetContract,
    weights: WeightArtifact,
    output_directory: str | Path,
    *,
    allow_real_smoke: bool,
    device: str,
    backend: RealSmokeBackend | None = None,
) -> dict[str, Any]:
    """Run bounded calls until exactly one AMP-applied optimizer update."""

    if not allow_real_smoke:
        raise DetectionInfrastructureError(
            "Real-manifest smoke is disabled; pass --allow-real-smoke explicitly.",
            code="real_smoke_not_authorized",
        )
    validate_frozen_model_artifacts(config)
    if (
        weights.path != config.model_path
        or weights.sha256 != config.model_sha256
        or weights.audit_path != config.weight_audit_path
        or weights.audit_sha256 != config.weight_audit_sha256
    ):
        raise DetectionInfrastructureError(
            "Real smoke weight/audit identity differs from the frozen config.",
            code="real_smoke_model_drift",
        )
    selected_device = bound_training_device(str(config.train["device"]), device)
    output = bound_real_smoke_output_directory(
        config.output.real_smoke_root, output_directory
    )
    try:
        output.mkdir(parents=True, mode=0o750, exist_ok=False)
    except FileExistsError as exc:
        raise DetectionInfrastructureError(
            f"Refusing to overwrite real smoke output: {output}", code="smoke_collision"
        ) from exc
    subset = _representative_subset(contract)
    data_yaml = freeze_dataset_contract(subset, output / "dataset-contract")
    if backend is None:
        from floodsight_detection.ultralytics_runtime import UltralyticsSmokeBackend

        backend = UltralyticsSmokeBackend(model_source=str(weights.path))
    result = backend.run(
        data_yaml=data_yaml,
        output_root=output / "runtime",
        seed=int(config.train["seed"]),
        device=selected_device,
        train_settings=dict(config.train),
        config_sha256=config.sha256,
    )
    required = _REQUIRED_REAL_SMOKE_CHECKS
    failed = [name for name in required if result.get(name) is not True]
    if failed:
        raise DetectionInfrastructureError(
            f"Real-manifest smoke did not prove: {', '.join(failed)}.",
            code="real_smoke_incomplete",
        )
    observed_train_batches = result.get("observed_train_batches")
    observed_step_calls = result.get("observed_optimizer_step_calls")
    observed_validation_runs = result.get("observed_validation_runs")
    train_sample_count = result.get("train_sample_count")
    if (
        type(observed_step_calls) is not int
        or not 2 <= observed_step_calls <= REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS
        or type(observed_train_batches) is not int
        or not 2 <= observed_train_batches <= REAL_SMOKE_MAX_TRAIN_BATCHES
        or observed_step_calls > observed_train_batches
        or result.get("observed_backward_calls") != observed_train_batches
        or type(observed_validation_runs) is not int
        or not 2 <= observed_validation_runs <= observed_train_batches + 2
        or result.get("maximum_train_batches") != REAL_SMOKE_MAX_TRAIN_BATCHES
        or result.get("maximum_optimizer_step_calls")
        != REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS
        or result.get("successful_optimizer_updates") != 1
        or result.get("amp_overflow_skips") != observed_step_calls - 1
        or type(train_sample_count) is not int
        or not 1 <= train_sample_count <= 8
        or not _optimizer_evidence_valid(
            result,
            configured_epochs=int(config.train["epochs"]),
            configured_batch_size=int(config.train["batch"]),
            loader_batch_size=min(int(config.train["batch"]), train_sample_count),
        )
    ):
        raise DetectionInfrastructureError(
            "Real smoke did not stop at one update within its optimizer-call bound.",
            code="smoke_bound_exceeded",
        )
    expected_effective_args = dict(config.train)
    effective_args_sha256 = stable_sha256(expected_effective_args)
    expected_augmentation_args = {
        key: expected_effective_args[key]
        for key in sorted(REAL_SMOKE_AUGMENTATION_KEYS)
    }
    augmentation_args_sha256 = stable_sha256(expected_augmentation_args)
    expected_runtime_bounds = {
        "configured_epochs_preserved": config.train["epochs"],
        "maximum_train_batches": REAL_SMOKE_MAX_TRAIN_BATCHES,
        "maximum_optimizer_step_calls": REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS,
        "stop_condition": "first_underlying_adamw_step_post_hook",
        "rationale": (
            "four_calls_empirically_exhausted_at_amp_scale_4096;"
            "bounded_backoff_window_expanded"
        ),
    }
    effective_evidence_valid = (
        result.get("mode") == "config_bound_real"
        and result.get("config_sha256") == config.sha256
        and result.get("model") == str(weights.path)
        and result.get("model_source_sha256") == weights.sha256
        and result.get("bounded_override_allowlist") == []
        and result.get("bounded_overrides") == {}
        and result.get("runtime_bounds") == expected_runtime_bounds
        and result.get("effective_args_sha256") == effective_args_sha256
        and result.get("observed_effective_args_sha256")
        == [effective_args_sha256, effective_args_sha256]
        and result.get("augmentation_args_sha256") == augmentation_args_sha256
        and result.get("training_source_sha256") == training_source_sha256()
    )
    augmentation_evidence = result.get("augmentation_evidence")
    expected_transform_configuration_records = expected_transform_configuration(
        expected_effective_args
    )
    expected_transform_configuration_sha256 = stable_sha256(
        expected_transform_configuration_records
    )
    augmentation_evidence_valid = (
        isinstance(augmentation_evidence, list)
        and len(augmentation_evidence) == 2
        and all(isinstance(evidence, dict) for evidence in augmentation_evidence)
        and all(
            evidence.get("dataset_augment") is True
            and evidence.get("configured_batch_size") == config.train["batch"]
            and type(evidence.get("dataset_size")) is int
            and 1 <= evidence["dataset_size"] <= 8
            and evidence.get("loader_batch_size")
            == min(config.train["batch"], evidence["dataset_size"])
            and evidence.get("loader_batch_policy")
            == "min(configured_batch_size,dataset_size)"
            and evidence.get("loader_num_workers") == config.train["workers"]
            and evidence.get("arguments") == expected_augmentation_args
            and evidence.get("arguments_sha256") == augmentation_args_sha256
            and evidence.get("transform_configuration")
            == expected_transform_configuration_records
            and evidence.get("transform_configuration_sha256")
            == expected_transform_configuration_sha256
            for evidence in augmentation_evidence
        )
        and [evidence.get("phase") for evidence in augmentation_evidence]
        == ["initial", "resume"]
    )
    runtime = result.get("runtime")
    hash_seed_runtime = result.get("hash_seed_runtime")
    runtime_evidence_valid = (
        isinstance(runtime, dict)
        and runtime.get("offline_assets_verified") is True
        and runtime.get("environment_marker_sha256")
        == "11ec5e2bc107465862ab04f8a01d58719c5012356489168cf28387f6848f96bd"
        and runtime.get("resolved_lock_sha256")
        == "33e7ca74a272659827d10c3bc882de1aa6e39b871c36435eb52279bd88eb58e1"
        and runtime.get("installed_distributions_sha256")
        == "ee7f9ce2704ddaea38312d0e11dacb8d01270f8be04ac1aad7e31095878ce775"
        and isinstance(runtime.get("gpu_name"), str)
        and "H100" in runtime["gpu_name"]
        and runtime.get("device") == int(selected_device)
        and isinstance(hash_seed_runtime, dict)
        and hash_seed_runtime.get("python_hash_seed") == str(config.train["seed"])
        and hash_seed_runtime.get("python_hash_seed_prestarted") is True
    )
    resume_state = result.get("resume_state")
    parent_cuda_release = result.get("parent_cuda_release")
    resume_evidence_valid = (
        result.get("fresh_process_resume") is True
        and result.get("rng_restored") is True
        and isinstance(result.get("resume_process_evidence"), str)
        and isinstance(result.get("resume_process_evidence_sha256"), str)
        and type(result.get("initial_pid")) is int
        and type(result.get("resume_pid")) is int
        and result.get("resume_pid") != result.get("initial_pid")
        and result.get("resume_parent_pid") == result.get("initial_pid")
        and type(result.get("initial_model_instance_id")) is int
        and result["initial_model_instance_id"] > 0
        and type(result.get("resume_model_instance_id")) is int
        and result["resume_model_instance_id"] > 0
        and isinstance(resume_state, dict)
        and resume_state.get("start_epoch") == 1
        and resume_state.get("configured_effective_epochs")
        == config.train["epochs"]
        and resume_state.get("checkpoint_training_state_match") is True
        and resume_state.get("optimizer_state_entries") == 0
        and type(resume_state.get("optimizer_state_entries_after_batch")) is int
        and resume_state["optimizer_state_entries_after_batch"] > 0
        and type(resume_state.get("optimizer_max_parameter_step_after_batch"))
        in {int, float}
        and resume_state["optimizer_max_parameter_step_after_batch"] > 0
        and isinstance(parent_cuda_release, dict)
        and parent_cuda_release.get("status") == "PASS"
        and parent_cuda_release.get("released_before_fresh_process") is True
        and parent_cuda_release.get("device") == selected_device
        and type(parent_cuda_release.get("allocated_before_bytes")) is int
        and type(parent_cuda_release.get("allocated_after_bytes")) is int
        and parent_cuda_release["allocated_before_bytes"]
        > parent_cuda_release["allocated_after_bytes"]
        >= 0
        and type(parent_cuda_release.get("reserved_before_bytes")) is int
        and type(parent_cuda_release.get("reserved_after_bytes")) is int
        and parent_cuda_release["reserved_before_bytes"]
        >= parent_cuda_release["reserved_after_bytes"]
        >= 0
    )
    checkpoint_path_raw = result.get("resumed_checkpoint")
    checkpoint_evidence_valid = False
    if isinstance(checkpoint_path_raw, str):
        declared_checkpoint = Path(checkpoint_path_raw).expanduser()
        if not declared_checkpoint.is_symlink() and declared_checkpoint.is_file():
            checkpoint = declared_checkpoint.resolve(strict=True)
            if checkpoint == (output / "runtime" / "pre-strip-resumable.pt").resolve(
                strict=True
            ):
                checkpoint_evidence_valid = (
                    result.get("checkpoint_sha256") == sha256_file(checkpoint)
                )
    process_evidence_path_raw = result.get("resume_process_evidence")
    process_evidence_valid = False
    if isinstance(process_evidence_path_raw, str):
        declared_process_evidence = Path(process_evidence_path_raw).expanduser()
        if (
            not declared_process_evidence.is_symlink()
            and declared_process_evidence.is_file()
        ):
            process_evidence = declared_process_evidence.resolve(strict=True)
            if process_evidence == (
                output / "runtime" / "fresh-process-resume-result.json"
            ).resolve(strict=True):
                process_evidence_valid = result.get(
                    "resume_process_evidence_sha256"
                ) == sha256_file(process_evidence)
    if not all(
        (
            effective_evidence_valid,
            augmentation_evidence_valid,
            runtime_evidence_valid,
            resume_evidence_valid,
            checkpoint_evidence_valid,
            process_evidence_valid,
        )
    ):
        raise DetectionInfrastructureError(
            "Real-manifest smoke evidence drifted from the frozen execution contract.",
            code="real_smoke_evidence_invalid",
        )
    report = {
        "schema_version": "floodsight-detection-real-manifest-smoke-v5",
        "status": "PASS",
        "bounded_real_manifest_smoke": True,
        "full_training_started": False,
        "manifest_sha256": contract.manifest_sha256,
        "dataset_fingerprint": contract.dataset_fingerprint,
        "config_sha256": config.sha256,
        "training_source_sha256": training_source_sha256(),
        "weights_sha256": weights.sha256,
        "weights_path": str(weights.path),
        "weight_audit_path": str(weights.audit_path),
        "weight_audit_sha256": weights.audit_sha256,
        "weight_license_review_status": weights.license_review_status,
        "maximum_train_batches": REAL_SMOKE_MAX_TRAIN_BATCHES,
        "maximum_optimizer_step_calls": REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS,
        "observed_train_batches": observed_train_batches,
        "observed_optimizer_step_calls": observed_step_calls,
        "observed_validation_runs": observed_validation_runs,
        "backward_gradient_finiteness": result["backward_gradient_finiteness"],
        "successful_optimizer_updates": result["successful_optimizer_updates"],
        "amp_overflow_skips": result["amp_overflow_skips"],
        "output_directory": str(output),
        "frozen_real_smoke_root": str(config.output.real_smoke_root),
        "output_policy": config.output.new_run_policy,
        "effective_args_sha256": effective_args_sha256,
        "augmentation_args_sha256": augmentation_args_sha256,
        "fresh_process_resume": True,
        "rng_state_restored": True,
        "initial_pid": result["initial_pid"],
        "resume_pid": result["resume_pid"],
        "resume_parent_pid": result["resume_parent_pid"],
        "resume_process_evidence": result["resume_process_evidence"],
        "resume_process_evidence_sha256": result["resume_process_evidence_sha256"],
        "subset_sample_ids": [sample.sample_id for sample in subset.samples],
        "subset_split_counts": subset.split_counts,
        "checks": {name: True for name in required},
        "backend": result,
    }
    report_path = output / "real-manifest-smoke-report.json"
    _write_exclusive(report_path, report)
    return report | {
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
    }


def load_real_smoke_attestation(
    path: str | Path,
    *,
    config: TrainingConfig,
    contract: DatasetContract,
    weights: WeightArtifact,
) -> RealSmokeAttestation:
    """Validate one exact current-source real-smoke report before TRAIN can be approved."""

    declared = Path(path).expanduser()
    if declared.is_symlink() or not declared.is_absolute() or not declared.is_file():
        raise DetectionInfrastructureError(
            "Real-smoke report must be an absolute regular non-symlink file.",
            code="real_smoke_attestation_invalid",
        )
    report_path = declared.resolve(strict=True)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectionInfrastructureError(
            "Real-smoke report is unreadable.", code="real_smoke_attestation_invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise DetectionInfrastructureError(
            "Real-smoke report must contain an object.", code="real_smoke_attestation_invalid"
        )
    output_raw = payload.get("output_directory")
    if not isinstance(output_raw, str) or not Path(output_raw).is_absolute():
        raise DetectionInfrastructureError(
            "Real-smoke report output identity is invalid.",
            code="real_smoke_attestation_invalid",
        )
    output = Path(output_raw)
    if output.is_symlink() or output.resolve(strict=True) != report_path.parent:
        raise DetectionInfrastructureError(
            "Real-smoke report is not inside its declared immutable output.",
            code="real_smoke_attestation_invalid",
        )
    if output.parent.resolve() != config.output.real_smoke_root.resolve():
        raise DetectionInfrastructureError(
            "Real-smoke report is outside the frozen smoke root.",
            code="real_smoke_attestation_invalid",
        )
    expected = {
        "schema_version": "floodsight-detection-real-manifest-smoke-v5",
        "status": "PASS",
        "bounded_real_manifest_smoke": True,
        "full_training_started": False,
        "manifest_sha256": contract.manifest_sha256,
        "dataset_fingerprint": contract.dataset_fingerprint,
        "config_sha256": config.sha256,
        "training_source_sha256": training_source_sha256(),
        "weights_sha256": weights.sha256,
        "weights_path": str(weights.path),
        "weight_audit_path": str(weights.audit_path),
        "weight_audit_sha256": weights.audit_sha256,
        "weight_license_review_status": weights.license_review_status,
        "frozen_real_smoke_root": str(config.output.real_smoke_root),
        "output_policy": config.output.new_run_policy,
        "maximum_train_batches": REAL_SMOKE_MAX_TRAIN_BATCHES,
        "maximum_optimizer_step_calls": REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS,
        "successful_optimizer_updates": 1,
        "fresh_process_resume": True,
        "rng_state_restored": True,
        "initial_pid": payload.get("initial_pid"),
        "resume_pid": payload.get("resume_pid"),
        "resume_parent_pid": payload.get("initial_pid"),
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    checks = payload.get("checks")
    backend = payload.get("backend")
    runtime_root = output / "runtime"
    artifact_evidence_valid = True
    process_evidence_payload: dict[str, Any] | None = None
    for path_key, hash_key, expected_path in (
        (
            "resumed_checkpoint",
            "checkpoint_sha256",
            runtime_root / "pre-strip-resumable.pt",
        ),
        (
            "resume_process_evidence",
            "resume_process_evidence_sha256",
            runtime_root / "fresh-process-resume-result.json",
        ),
    ):
        path_raw = backend.get(path_key) if isinstance(backend, dict) else None
        expected_hash = backend.get(hash_key) if isinstance(backend, dict) else None
        if (
            not isinstance(path_raw, str)
            or not isinstance(expected_hash, str)
            or _HEX64.fullmatch(expected_hash) is None
        ):
            artifact_evidence_valid = False
            continue
        artifact = Path(path_raw).expanduser()
        if artifact.is_symlink() or not artifact.is_absolute() or not artifact.is_file():
            artifact_evidence_valid = False
            continue
        artifact = artifact.resolve(strict=True)
        if artifact != expected_path.resolve(strict=True):
            artifact_evidence_valid = False
            continue
        try:
            artifact.relative_to(runtime_root.resolve(strict=True))
        except ValueError:
            artifact_evidence_valid = False
            continue
        if sha256_file(artifact) != expected_hash:
            artifact_evidence_valid = False
            continue
        if path_key == "resume_process_evidence":
            try:
                parsed = json.loads(artifact.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                artifact_evidence_valid = False
            else:
                if isinstance(parsed, dict):
                    process_evidence_payload = parsed
                else:
                    artifact_evidence_valid = False
    process_evidence_valid = False
    if process_evidence_payload is not None and isinstance(backend, dict):
        worker_step_calls = process_evidence_payload.get("observed_optimizer_step_calls")
        worker_train_batches = process_evidence_payload.get("observed_train_batches")
        worker_validation_runs = process_evidence_payload.get("observed_validation_runs")
        backend_step_calls = backend.get("observed_optimizer_step_calls")
        backend_train_batches = backend.get("observed_train_batches")
        backend_validation_runs = backend.get("observed_validation_runs")
        worker_unsigned = {
            key: value
            for key, value in process_evidence_payload.items()
            if key != "result_sha256"
        }
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
        process_evidence_valid = (
            process_evidence_payload.get("schema_version")
            == "floodsight-real-smoke-resume-result-v3"
            and process_evidence_payload.get("status") == "PASS"
            and process_evidence_payload.get("result_sha256")
            == stable_sha256(worker_unsigned)
            and process_evidence_payload.get("pid") == backend.get("resume_pid")
            and process_evidence_payload.get("parent_pid") == backend.get("initial_pid")
            and process_evidence_payload.get("parent_pid")
            == backend.get("resume_parent_pid")
            and process_evidence_payload.get("model_instance_id")
            == backend.get("resume_model_instance_id")
            and all(
                process_evidence_payload.get(key) is True
                for key in worker_required_true
            )
            and process_evidence_payload.get("checkpoint_path")
            == backend.get("resumed_checkpoint")
            and process_evidence_payload.get("checkpoint_sha256")
            == backend.get("checkpoint_sha256")
            and process_evidence_payload.get("training_source_sha256")
            == training_source_sha256()
            and type(worker_step_calls) is int
            and type(backend_step_calls) is int
            and worker_step_calls == backend_step_calls - 1
            and type(worker_train_batches) is int
            and type(backend_train_batches) is int
            and worker_train_batches == backend_train_batches - 1
            and 1 <= worker_step_calls <= worker_train_batches
            and worker_train_batches <= REAL_SMOKE_MAX_TRAIN_BATCHES - 1
            and process_evidence_payload.get("maximum_optimizer_step_calls")
            == REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS - 1
            and process_evidence_payload.get("maximum_train_batches")
            == REAL_SMOKE_MAX_TRAIN_BATCHES - 1
            and process_evidence_payload.get("observed_backward_calls")
            == worker_train_batches
            and isinstance(
                process_evidence_payload.get("backward_gradient_finiteness"), list
            )
            and process_evidence_payload["backward_gradient_finiteness"]
            == backend.get("backward_gradient_finiteness", [])[1:]
            and type(worker_validation_runs) is int
            and 1 <= worker_validation_runs <= worker_train_batches + 1
            and type(backend_validation_runs) is int
            and 1 <= backend_validation_runs - worker_validation_runs <= 2
            and process_evidence_payload.get("successful_optimizer_updates") == 1
            and process_evidence_payload.get("amp_overflow_skips")
            == worker_step_calls - 1
            and isinstance(backend.get("optimizer_step_evidence"), list)
            and len(backend["optimizer_step_evidence"]) == backend_step_calls
            and process_evidence_payload.get("optimizer_step_evidence")
            == backend["optimizer_step_evidence"][1:]
            and process_evidence_payload.get("resume_state")
            == backend.get("resume_state")
            and process_evidence_payload.get("effective_args_sha256")
            == backend.get("effective_args_sha256")
            and isinstance(backend.get("augmentation_evidence"), list)
            and len(backend["augmentation_evidence"]) == 2
            and process_evidence_payload.get("augmentation_evidence")
            == backend["augmentation_evidence"][1]
            and process_evidence_payload.get("runtime") == backend.get("runtime")
            and process_evidence_payload.get("rng_restored") is True
            and isinstance(process_evidence_payload.get("hash_seed_runtime"), dict)
            and process_evidence_payload["hash_seed_runtime"].get("python_hash_seed")
            == str(config.train["seed"])
            and process_evidence_payload["hash_seed_runtime"].get(
                "python_hash_seed_prestarted"
            )
            is True
            and process_evidence_payload.get("cuda_visible_devices_at_process_start")
            == backend.get("cuda_visible_devices_at_process_start")
            and process_evidence_payload.get("ultralytics_version")
            == config.ultralytics_version
            and process_evidence_payload.get("torch_version")
            == config.torch_version
            and process_evidence_payload.get("torchvision_version")
            == config.torchvision_version
        )
    parent_cuda_release = backend.get("parent_cuda_release") if isinstance(backend, dict) else None
    parent_cuda_release_valid = (
        isinstance(parent_cuda_release, dict)
        and parent_cuda_release.get("status") == "PASS"
        and parent_cuda_release.get("released_before_fresh_process") is True
        and parent_cuda_release.get("device") == str(config.train["device"])
        and type(parent_cuda_release.get("allocated_before_bytes")) is int
        and type(parent_cuda_release.get("allocated_after_bytes")) is int
        and parent_cuda_release["allocated_before_bytes"]
        > parent_cuda_release["allocated_after_bytes"]
        >= 0
        and type(parent_cuda_release.get("reserved_before_bytes")) is int
        and type(parent_cuda_release.get("reserved_after_bytes")) is int
        and parent_cuda_release["reserved_before_bytes"]
        >= parent_cuda_release["reserved_after_bytes"]
        >= 0
    )
    expected_effective_args = dict(config.train)
    effective_args_sha256 = stable_sha256(expected_effective_args)
    expected_augmentation_args = {
        key: expected_effective_args[key]
        for key in sorted(REAL_SMOKE_AUGMENTATION_KEYS)
    }
    augmentation_args_sha256 = stable_sha256(expected_augmentation_args)
    expected_transform_records = expected_transform_configuration(
        expected_effective_args
    )
    transform_configuration_sha256 = stable_sha256(expected_transform_records)
    augmentation_evidence = (
        backend.get("augmentation_evidence") if isinstance(backend, dict) else None
    )
    augmentation_evidence_valid = (
        isinstance(augmentation_evidence, list)
        and len(augmentation_evidence) == 2
        and all(isinstance(item, dict) for item in augmentation_evidence)
        and [item.get("phase") for item in augmentation_evidence]
        == ["initial", "resume"]
        and all(
            item.get("dataset_augment") is True
            and item.get("arguments") == expected_augmentation_args
            and item.get("arguments_sha256") == augmentation_args_sha256
            and item.get("transform_configuration") == expected_transform_records
            and item.get("transform_configuration_sha256")
            == transform_configuration_sha256
            and item.get("configured_batch_size") == config.train["batch"]
            and item.get("loader_num_workers") == config.train["workers"]
            and type(item.get("dataset_size")) is int
            and 1 <= item["dataset_size"] <= 8
            and item.get("loader_batch_size")
            == min(config.train["batch"], item["dataset_size"])
            for item in augmentation_evidence
        )
    )
    runtime_bounds_valid = (
        isinstance(backend, dict)
        and backend.get("mode") == "config_bound_real"
        and backend.get("config_sha256") == config.sha256
        and backend.get("model") == str(weights.path)
        and backend.get("model_source_sha256") == weights.sha256
        and backend.get("bounded_override_allowlist") == []
        and backend.get("bounded_overrides") == {}
        and backend.get("runtime_bounds")
        == {
            "configured_epochs_preserved": config.train["epochs"],
            "maximum_train_batches": REAL_SMOKE_MAX_TRAIN_BATCHES,
            "maximum_optimizer_step_calls": REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS,
            "stop_condition": "first_underlying_adamw_step_post_hook",
            "rationale": (
                "four_calls_empirically_exhausted_at_amp_scale_4096;"
                "bounded_backoff_window_expanded"
            ),
        }
        and backend.get("effective_args_sha256") == effective_args_sha256
        and backend.get("observed_effective_args_sha256")
        == [effective_args_sha256, effective_args_sha256]
        and backend.get("augmentation_args_sha256") == augmentation_args_sha256
    )
    backend_runtime = backend.get("runtime") if isinstance(backend, dict) else None
    backend_hash_seed = (
        backend.get("hash_seed_runtime") if isinstance(backend, dict) else None
    )
    runtime_evidence_valid = (
        isinstance(backend_runtime, dict)
        and backend_runtime.get("offline_assets_verified") is True
        and backend_runtime.get("environment_marker_sha256")
        == "11ec5e2bc107465862ab04f8a01d58719c5012356489168cf28387f6848f96bd"
        and backend_runtime.get("resolved_lock_sha256")
        == "33e7ca74a272659827d10c3bc882de1aa6e39b871c36435eb52279bd88eb58e1"
        and backend_runtime.get("installed_distributions_sha256")
        == "ee7f9ce2704ddaea38312d0e11dacb8d01270f8be04ac1aad7e31095878ce775"
        and isinstance(backend_runtime.get("gpu_name"), str)
        and "H100" in backend_runtime["gpu_name"]
        and backend_runtime.get("device") == int(config.train["device"])
        and isinstance(backend_hash_seed, dict)
        and backend_hash_seed.get("python_hash_seed") == str(config.train["seed"])
        and backend_hash_seed.get("python_hash_seed_prestarted") is True
        and backend.get("ultralytics_version") == config.ultralytics_version
        and backend.get("torch_version") == config.torch_version
        and backend.get("torchvision_version") == config.torchvision_version
    )
    backend_train_sample_count = (
        backend.get("train_sample_count") if isinstance(backend, dict) else None
    )
    optimizer_evidence_valid = (
        isinstance(backend_train_sample_count, int)
        and not isinstance(backend_train_sample_count, bool)
        and 1 <= backend_train_sample_count <= 8
        and _optimizer_evidence_valid(
            backend,
            configured_epochs=int(config.train["epochs"]),
            configured_batch_size=int(config.train["batch"]),
            loader_batch_size=min(
                int(config.train["batch"]), backend_train_sample_count
            ),
        )
    )
    if (
        drift
        or type(payload.get("initial_pid")) is not int
        or payload["initial_pid"] <= 0
        or type(payload.get("resume_pid")) is not int
        or payload["resume_pid"] <= 0
        or payload.get("initial_pid") == payload.get("resume_pid")
        or payload.get("resume_parent_pid") != payload.get("initial_pid")
        or not isinstance(backend, dict)
        or payload.get("resume_process_evidence")
        != backend.get("resume_process_evidence")
        or payload.get("resume_process_evidence_sha256")
        != backend.get("resume_process_evidence_sha256")
        or not isinstance(checks, dict)
        or set(checks) != set(_REQUIRED_REAL_SMOKE_CHECKS)
        or any(checks.get(key) is not True for key in _REQUIRED_REAL_SMOKE_CHECKS)
        or type(backend.get("initial_model_instance_id")) is not int
        or backend["initial_model_instance_id"] <= 0
        or type(backend.get("resume_model_instance_id")) is not int
        or backend["resume_model_instance_id"] <= 0
        or backend.get("initial_pid") != payload.get("initial_pid")
        or backend.get("resume_pid") != payload.get("resume_pid")
        or backend.get("resume_parent_pid") != payload.get("resume_parent_pid")
        or payload.get("maximum_train_batches")
        != backend.get("maximum_train_batches")
        or payload.get("observed_train_batches")
        != backend.get("observed_train_batches")
        or payload.get("observed_optimizer_step_calls")
        != backend.get("observed_optimizer_step_calls")
        or payload.get("observed_validation_runs")
        != backend.get("observed_validation_runs")
        or payload.get("backward_gradient_finiteness")
        != backend.get("backward_gradient_finiteness")
        or payload.get("effective_args_sha256")
        != backend.get("effective_args_sha256")
        or payload.get("augmentation_args_sha256")
        != backend.get("augmentation_args_sha256")
        or payload.get("amp_overflow_skips") != backend.get("amp_overflow_skips")
        or backend.get("observed_backward_calls")
        != backend.get("observed_train_batches")
        or not optimizer_evidence_valid
        or not augmentation_evidence_valid
        or not runtime_bounds_valid
        or not runtime_evidence_valid
        or backend.get("fresh_process_resume") is not True
        or backend.get("rng_restored") is not True
        or backend.get("training_source_sha256") != training_source_sha256()
        or not artifact_evidence_valid
        or not process_evidence_valid
        or not parent_cuda_release_valid
    ):
        raise DetectionInfrastructureError(
            "Real-smoke report is not bound to the exact current frozen run.",
            code="real_smoke_attestation_mismatch",
            details=[drift] if drift else None,
        )
    report_sha256 = sha256_file(report_path)
    if _HEX64.fullmatch(report_sha256) is None:
        raise DetectionInfrastructureError(
            "Real-smoke report SHA-256 is invalid.", code="real_smoke_attestation_invalid"
        )
    return RealSmokeAttestation(
        path=report_path,
        sha256=report_sha256,
        output_directory=output,
        training_source_sha256=payload["training_source_sha256"],
    )
