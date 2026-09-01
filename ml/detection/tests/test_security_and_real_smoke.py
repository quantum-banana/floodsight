from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from floodsight_detection.approval import load_training_approval
from floodsight_detection.cli import run
from floodsight_detection.config import load_training_config
from floodsight_detection.contract import validate_dataset_contract
from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file, stable_sha256, training_source_sha256
from floodsight_detection.real_smoke import (
    load_real_smoke_attestation,
    run_real_manifest_smoke,
)
from floodsight_detection.ultralytics_runtime import (
    REAL_SMOKE_AUGMENTATION_KEYS,
    REAL_SMOKE_BOUNDED_OVERRIDES,
    REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS,
    REAL_SMOKE_MAX_TRAIN_BATCHES,
    _fresh_process_environment,
    audited_transform_configuration,
    bounded_loader_batch_size,
    expected_transform_configuration,
    real_smoke_effective_arguments,
)
from floodsight_detection.weights import (
    load_weight_audit,
    validate_training_license_disposition,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/yolo11l_h100_max_quality.yaml"


def _weight_audit(
    tmp_path: Path,
    *,
    license_review_status: str = "APPROVED_FOR_RESEARCH_DEMO",
) -> tuple[Path, str]:
    weight = tmp_path / "yolo11l.pt"
    weight.write_bytes(b"audited-local-test-weight")
    digest = sha256_file(weight)
    payload = {
        "schema_version": "floodsight-yolo-weight-audit-v1",
        "architecture": "yolo11l",
        "local_path": str(weight.resolve()),
        "sha256": digest,
        "source_url": "https://github.com/ultralytics/assets/releases/tag/v8.3.0",
        "source_release": "test-fixture-only",
        "license_review_status": license_review_status,
        "reviewed_by": "unit-test-reviewer",
        "reviewed_at": "2026-08-31T00:00:00Z",
    }
    audit = tmp_path / "weight-audit.json"
    audit.write_text(json.dumps(payload), encoding="utf-8")
    return audit, digest


def _load_fixture_weight(
    audit: Path, *, require_license_approval: bool = True
) -> Any:
    payload = json.loads(audit.read_text(encoding="utf-8"))
    return load_weight_audit(
        audit,
        expected_filename="yolo11l.pt",
        expected_weight_path=payload["local_path"],
        expected_weight_sha256=payload["sha256"],
        expected_audit_path=audit,
        expected_audit_sha256=sha256_file(audit),
        require_license_approval=require_license_approval,
    )


def _approval(
    tmp_path: Path,
    *,
    config_sha256: str,
    manifest_sha256: str,
    dataset_fingerprint: str,
    weights_sha256: str,
    run_name: str = "approved-run",
    device: str = "0",
    deferred_by_user: bool = False,
) -> Path:
    if deferred_by_user:
        instruction = tmp_path / "instruction.txt"
        instruction.write_text("Authorize exact production runs.\n", encoding="utf-8")
        review = tmp_path / "launch-override.json"
        review.write_text(
            json.dumps(
                {
                    "schema_version": "floodsight-user-training-launch-override-v1",
                    "instruction_accepted_utc": "2026-08-31T20:06:25Z",
                    "target_training_launch_utc": "2026-08-31T22:06:25Z",
                    "instruction_source_path": str(instruction.resolve()),
                    "instruction_source_sha256": sha256_file(instruction),
                    "authorized_by": "unit-test-human",
                    "decision": (
                        "AUTHORIZE_PRODUCTION_TRAINING_WHEN_EXISTING_TECHNICAL_"
                        "REQUIREMENTS_PASS_AND_A_SAFE_H100_IS_AVAILABLE"
                    ),
                    "authorized_models": ["SEGFORMER", "YOLO"],
                    "human_review_status": "DEFERRED_BY_USER",
                    "provenance_review_status": "DEFERRED_BY_USER",
                    "human_review_completed": False,
                    "provenance_review_completed": False,
                    "full_training_explicitly_authorized": True,
                    "additional_broad_audits_authorized": False,
                    "persistent_tmux_required": True,
                    "source_freeze_required_before_each_launch": True,
                    "notes": ["Fixture user deferral."],
                }
            ),
            encoding="utf-8",
        )
    else:
        review = tmp_path / "human-review.md"
        review.write_text("# Human review\n\nAPPROVED fixture.\n", encoding="utf-8")
    real_smoke = tmp_path / "real-manifest-smoke-report.json"
    real_smoke.write_text('{"status":"PASS"}\n', encoding="utf-8")
    payload = {
        "schema_version": (
            "floodsight-full-training-approval-v5"
            if deferred_by_user
            else "floodsight-full-training-approval-v4"
        ),
        "approval_id": "unit-test-approval",
        "decision": "APPROVE_FULL_TRAINING",
        "approved_by": "unit-test-human",
        "approved_at": (
            "2026-08-31T20:07:00Z"
            if deferred_by_user
            else "2026-08-31T00:00:00Z"
        ),
        "run_name": run_name,
        "output_root": str((tmp_path / "runs").resolve()),
        "device": device,
        "training_code_sha256": training_source_sha256(),
        "config_sha256": config_sha256,
        "manifest_id": "visdrone_det-detection_v2",
        "dataset_id": "visdrone_det",
        "preparation_version": "detection_v2",
        "manifest_sha256": manifest_sha256,
        "dataset_fingerprint": dataset_fingerprint,
        "taxonomy_version": "detection-taxonomy-v1",
        "taxonomy_sha256": (
            "4000f1bb75b2e4687d60027a72dd3f428c9bed8ba8de918c0892e3f115bdf535"
        ),
        "mapping_version": "visdrone-mapping-v1",
        "mapping_sha256": (
            "ad0d67626195744bfc908cfc64c36b15a046d53f35a466791101256aa8681ad8"
        ),
        "weights_path": str((tmp_path / "yolo11l.pt").resolve()),
        "weights_sha256": weights_sha256,
        "weight_audit_path": str((tmp_path / "weight-audit.json").resolve()),
        "weight_audit_sha256": "2" * 64,
        "real_smoke_report_path": str(real_smoke.resolve()),
        "real_smoke_report_sha256": sha256_file(real_smoke),
        "human_review_path": str(review.resolve()),
        "human_review_sha256": sha256_file(review),
        "acknowledgements": (
            [
                "FULL_TRAINING_EXPLICITLY_AUTHORIZED",
                "DATASET_AND_LABEL_REVIEW_DEFERRED_BY_USER",
                "LICENSE_REVIEW_DEFERRED_BY_USER",
                "HUMAN_DECISION_SUPPORT_ONLY",
                "REAL_SMOKE_TECHNICALLY_VERIFIED",
                "USER_OVERRIDE_HASH_BOUND",
            ]
            if deferred_by_user
            else [
                "FULL_TRAINING_EXPLICITLY_AUTHORIZED",
                "DATASET_AND_LABEL_REVIEW_COMPLETE",
                "LICENSE_REVIEW_COMPLETE",
                "HUMAN_DECISION_SUPPORT_ONLY",
                "REAL_SMOKE_REVIEW_COMPLETE",
            ]
        ),
    }
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _approval_expected(tmp_path: Path) -> dict[str, Any]:
    return {
        "config_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "dataset_fingerprint": "c" * 64,
        "weights_sha256": "d" * 64,
        "weights_path": tmp_path / "yolo11l.pt",
        "weight_audit_path": tmp_path / "weight-audit.json",
        "weight_audit_sha256": "2" * 64,
        "run_name": "approved-run",
        "output_root": tmp_path / "runs",
        "device": "0",
        "manifest_id": "visdrone_det-detection_v2",
        "dataset_id": "visdrone_det",
        "preparation_version": "detection_v2",
        "taxonomy_version": "detection-taxonomy-v1",
        "taxonomy_sha256": (
            "4000f1bb75b2e4687d60027a72dd3f428c9bed8ba8de918c0892e3f115bdf535"
        ),
        "mapping_version": "visdrone-mapping-v1",
        "mapping_sha256": (
            "ad0d67626195744bfc908cfc64c36b15a046d53f35a466791101256aa8681ad8"
        ),
        "real_smoke_report_path": tmp_path / "real-manifest-smoke-report.json",
        "real_smoke_report_sha256": sha256_file(
            tmp_path / "real-manifest-smoke-report.json"
        ),
    }


def test_weight_audit_requires_exact_local_file_and_hash(tmp_path: Path) -> None:
    audit, digest = _weight_audit(tmp_path)

    artifact = _load_fixture_weight(audit)

    assert artifact.path.is_absolute()
    assert artifact.sha256 == digest


def test_weight_audit_blocks_hash_drift(tmp_path: Path) -> None:
    audit, _digest = _weight_audit(tmp_path)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    Path(payload["local_path"]).write_bytes(b"changed")

    with pytest.raises(DetectionInfrastructureError) as error:
        _load_fixture_weight(audit)

    assert error.value.code == "weight_hash_mismatch"


def test_pending_weight_review_is_allowed_only_for_pre_review_smoke(tmp_path: Path) -> None:
    audit, digest = _weight_audit(
        tmp_path,
        license_review_status="PENDING_HUMAN_SIGNOFF",
    )

    smoke_artifact = _load_fixture_weight(audit, require_license_approval=False)

    assert smoke_artifact.sha256 == digest
    assert smoke_artifact.license_review_status == "PENDING_HUMAN_SIGNOFF"
    with pytest.raises(DetectionInfrastructureError) as error:
        _load_fixture_weight(audit)
    assert error.value.code == "weight_license_not_approved"


def test_human_approval_is_bound_to_every_frozen_identity(tmp_path: Path) -> None:
    path = _approval(
        tmp_path,
        config_sha256="a" * 64,
        manifest_sha256="b" * 64,
        dataset_fingerprint="c" * 64,
        weights_sha256="d" * 64,
    )

    approval = load_training_approval(
        path,
        **_approval_expected(tmp_path),
    )

    assert approval.approval_id == "unit-test-approval"

    with pytest.raises(DetectionInfrastructureError) as error:
        load_training_approval(
            path,
            **(_approval_expected(tmp_path) | {"config_sha256": "e" * 64}),
        )
    assert error.value.code == "training_approval_mismatch"


def test_hash_bound_user_override_allows_pending_license_without_false_completion(
    tmp_path: Path,
) -> None:
    audit, digest = _weight_audit(
        tmp_path,
        license_review_status="PENDING_HUMAN_SIGNOFF",
    )
    weights = _load_fixture_weight(audit, require_license_approval=False)
    path = _approval(
        tmp_path,
        config_sha256="a" * 64,
        manifest_sha256="b" * 64,
        dataset_fingerprint="c" * 64,
        weights_sha256=digest,
        deferred_by_user=True,
    )
    expected = _approval_expected(tmp_path) | {"weights_sha256": digest}

    approval = load_training_approval(path, **expected)
    assert approval.review_disposition == "DEFERRED_BY_USER"
    validate_training_license_disposition(
        weights,
        review_disposition=approval.review_disposition,
    )

    override = tmp_path / "launch-override.json"
    payload = json.loads(override.read_text(encoding="utf-8"))
    payload["provenance_review_completed"] = 0
    override.write_text(json.dumps(payload), encoding="utf-8")
    approval_payload = json.loads(path.read_text(encoding="utf-8"))
    approval_payload["human_review_sha256"] = sha256_file(override)
    path.write_text(json.dumps(approval_payload), encoding="utf-8")
    with pytest.raises(DetectionInfrastructureError, match="deferred-review"):
        load_training_approval(path, **expected)


def test_human_approval_rejects_post_approval_real_smoke_drift(tmp_path: Path) -> None:
    path = _approval(
        tmp_path,
        config_sha256="a" * 64,
        manifest_sha256="b" * 64,
        dataset_fingerprint="c" * 64,
        weights_sha256="d" * 64,
    )
    (tmp_path / "real-manifest-smoke-report.json").write_text(
        '{"status":"DRIFTED"}\n', encoding="utf-8"
    )

    with pytest.raises(DetectionInfrastructureError) as error:
        load_training_approval(path, **_approval_expected(tmp_path))

    assert error.value.code == "training_approval_mismatch"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "1"),
        ("output_root", "/different/output"),
        ("manifest_id", "different-manifest"),
        ("dataset_id", "different-dataset"),
        ("taxonomy_sha256", "0" * 64),
        ("mapping_sha256", "1" * 64),
        ("training_code_sha256", "2" * 64),
    ],
)
def test_human_approval_rejects_runtime_source_or_dataset_identity_drift(
    tmp_path: Path, field: str, value: str
) -> None:
    path = _approval(
        tmp_path,
        config_sha256="a" * 64,
        manifest_sha256="b" * 64,
        dataset_fingerprint="c" * 64,
        weights_sha256="d" * 64,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        load_training_approval(path, **_approval_expected(tmp_path))

    assert error.value.code == "training_approval_mismatch"


def test_allow_training_alone_is_insufficient(tmp_path: Path) -> None:
    with pytest.raises(DetectionInfrastructureError) as error:
        run(
            [
                "train",
                "--config",
                str(tmp_path / "absent"),
                "--manifest",
                str(tmp_path / "absent"),
                "--data-root",
                str(tmp_path / "absent"),
                "--output-root",
                str(tmp_path / "runs"),
                "--run-name",
                "forbidden",
                "--allow-training",
            ]
        )

    assert error.value.code == "training_not_authorized"
    assert not (tmp_path / "runs").exists()


def test_real_smoke_guard_fails_before_dataset_or_weight_access(tmp_path: Path) -> None:
    with pytest.raises(DetectionInfrastructureError) as error:
        run(
            [
                "real-smoke",
                "--config",
                str(tmp_path / "absent"),
                "--manifest",
                str(tmp_path / "absent"),
                "--data-root",
                str(tmp_path / "absent"),
                "--output",
                str(tmp_path / "smoke"),
                "--device",
                "cpu",
            ]
        )

    assert error.value.code == "real_smoke_not_authorized"
    assert not (tmp_path / "smoke").exists()


class PassingRealBackend:
    def __init__(
        self,
        model_path: Path,
        *,
        corrupt_effective_hash: bool = False,
        corrupt_transform_configuration: bool = False,
        corrupt_runtime_context: bool = False,
        corrupt_resume_parent_pid: bool = False,
        corrupt_backward_gradient_finiteness: bool = False,
        corrupt_worker_backward_gradient_finiteness: bool = False,
        resume_train_batches: int = 1,
    ) -> None:
        self.model_path = model_path
        self.corrupt_effective_hash = corrupt_effective_hash
        self.corrupt_transform_configuration = corrupt_transform_configuration
        self.corrupt_runtime_context = corrupt_runtime_context
        self.corrupt_resume_parent_pid = corrupt_resume_parent_pid
        self.corrupt_backward_gradient_finiteness = (
            corrupt_backward_gradient_finiteness
        )
        self.corrupt_worker_backward_gradient_finiteness = (
            corrupt_worker_backward_gradient_finiteness
        )
        self.resume_train_batches = resume_train_batches

    def run(
        self,
        *,
        data_yaml: Path,
        output_root: Path,
        seed: int,
        device: str,
        train_settings: dict[str, Any],
        config_sha256: str,
    ) -> dict[str, Any]:
        assert data_yaml.is_file()
        assert seed == 20260831
        assert device == "0"
        split = Path(json.loads(data_yaml.read_text(encoding="utf-8"))["train"])
        train_sample_count = len(split.read_text(encoding="utf-8").splitlines())
        assert 1 <= train_sample_count <= 8
        assert 1 <= self.resume_train_batches <= REAL_SMOKE_MAX_TRAIN_BATCHES - 1
        output_root.mkdir(parents=True)
        checkpoint = output_root / "pre-strip-resumable.pt"
        checkpoint.write_bytes(b"bounded-real-smoke-checkpoint")
        resume_evidence = output_root / "fresh-process-resume-result.json"
        effective = dict(train_settings)
        effective.update(REAL_SMOKE_BOUNDED_OVERRIDES)
        assert effective == train_settings
        effective_hash = stable_sha256(effective)
        if self.corrupt_effective_hash:
            effective_hash = "0" * 64
        augmentation_args = {
            key: effective[key] for key in sorted(REAL_SMOKE_AUGMENTATION_KEYS)
        }
        augmentation_hash = stable_sha256(augmentation_args)
        transform_configuration = expected_transform_configuration(effective)
        if self.corrupt_transform_configuration:
            transform_configuration[0]["p"] = 0.0
        transform_configuration_hash = stable_sha256(transform_configuration)
        pid = os.getpid()
        resume_parent_pid = pid + int(self.corrupt_resume_parent_pid)
        runtime = {
            "offline_assets_verified": True,
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "device": 0,
            "environment_marker_sha256": (
                "11ec5e2bc107465862ab04f8a01d58719c5012356489168cf28387f6848f96bd"
            ),
            "resolved_lock_sha256": (
                "33e7ca74a272659827d10c3bc882de1aa6e39b871c36435eb52279bd88eb58e1"
            ),
            "installed_distributions_sha256": (
                "ee7f9ce2704ddaea38312d0e11dacb8d01270f8be04ac1aad7e31095878ce775"
            ),
        }
        optimizer_group_hash = "4" * 64
        initial_before = {
            "optimizer_name": "AdamW",
            "optimizer_state_parameter_count": 0,
            "optimizer_state_entries": 0,
            "optimizer_param_group_count": 3,
            "optimizer_param_groups_sha256": optimizer_group_hash,
            "optimizer_parameter_count": 514,
            "optimizer_max_parameter_step": 0.0,
            "amp_scale": 65536.0,
            "amp_scaler_state_sha256": "5" * 64,
        }
        checkpoint_state = {
            **initial_before,
            "amp_scale": 32768.0,
            "amp_scaler_state_sha256": "6" * 64,
        }
        resumed_after = {
            **checkpoint_state,
            "optimizer_state_parameter_count": 514,
            "optimizer_state_entries": 1542,
            "optimizer_max_parameter_step": 1.0,
            "amp_scaler_state_sha256": "7" * 64,
        }
        initial_step_evidence = {
            "phase": "initial",
            "phase_call_index": 1,
            "successful_update": False,
            "amp_overflow_skip": True,
            "underlying_optimizer_step": False,
            "before": initial_before,
            "after": checkpoint_state,
            "runtime_context": {
                "phase_batch_index": 1,
                "epoch": 0,
                "batches_per_epoch": 1,
                "global_iteration": 0,
                "configured_batch_size": train_settings["batch"],
                "loader_batch_size": train_sample_count,
                "accumulate": 1,
                "scheduler_last_epoch": 0,
                "learning_rates": [0.1, 0.0, 0.0],
            },
        }
        resumed_scheduler_epoch = self.resume_train_batches
        if self.corrupt_runtime_context:
            resumed_scheduler_epoch -= 1
        resumed_step_evidence = {
            "phase": "resume",
            "phase_call_index": 1,
            "successful_update": True,
            "amp_overflow_skip": False,
            "underlying_optimizer_step": True,
            "before": checkpoint_state,
            "after": resumed_after,
            "runtime_context": {
                "phase_batch_index": self.resume_train_batches,
                "epoch": self.resume_train_batches,
                "batches_per_epoch": 1,
                "global_iteration": self.resume_train_batches,
                "configured_batch_size": train_settings["batch"],
                "loader_batch_size": train_sample_count,
                "accumulate": 1,
                "scheduler_last_epoch": resumed_scheduler_epoch,
                "learning_rates": [0.099, 0.00001, 0.00001],
            },
        }
        resume_state = {
            "start_epoch": 1,
            "configured_effective_epochs": train_settings["epochs"],
            "checkpoint_training_state_match": True,
            "checkpoint_training_state": checkpoint_state,
            "optimizer_state_entries": 0,
            "scheduler_last_epoch": 0,
            "optimizer_state_after_batch": resumed_after,
            "optimizer_state_entries_after_batch": 1542,
            "optimizer_max_parameter_step_after_batch": 1.0,
        }
        augmentation_evidence = [
            {
                "phase": phase,
                "dataset_augment": True,
                "configured_batch_size": train_settings["batch"],
                "loader_batch_size": train_sample_count,
                "loader_batch_policy": "min(configured_batch_size,dataset_size)",
                "loader_num_workers": train_settings["workers"],
                "dataset_size": train_sample_count,
                "arguments": augmentation_args,
                "arguments_sha256": augmentation_hash,
                "transform_configuration": transform_configuration,
                "transform_configuration_sha256": transform_configuration_hash,
            }
            for phase in ("initial", "resume")
        ]
        worker_validation_runs = self.resume_train_batches
        worker_backward_gradient_finiteness = [False] * (
            self.resume_train_batches - 1
        ) + [True]
        backward_gradient_finiteness = [
            False,
            *worker_backward_gradient_finiteness,
        ]
        if self.corrupt_backward_gradient_finiteness:
            backward_gradient_finiteness[-1] = False
        if self.corrupt_worker_backward_gradient_finiteness:
            assert self.resume_train_batches >= 2
            worker_backward_gradient_finiteness[0] = True
        resume_evidence_body = {
            "schema_version": "floodsight-real-smoke-resume-result-v3",
            "status": "PASS",
            "loader": True,
            "loss": True,
            "backward": True,
            "optimizer_step": True,
            "validation": True,
            "checkpoint": True,
            "resume": True,
            "effective_args": True,
            "augmentation": True,
            "rng_restored": True,
            "pid": pid + 1,
            "parent_pid": resume_parent_pid,
            "model_instance_id": 1,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "training_source_sha256": training_source_sha256(),
            "observed_train_batches": self.resume_train_batches,
            "observed_backward_calls": self.resume_train_batches,
            "backward_gradient_finiteness": worker_backward_gradient_finiteness,
            "observed_optimizer_step_calls": 1,
            "observed_validation_runs": worker_validation_runs,
            "maximum_train_batches": REAL_SMOKE_MAX_TRAIN_BATCHES - 1,
            "maximum_optimizer_step_calls": REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS - 1,
            "successful_optimizer_updates": 1,
            "amp_overflow_skips": 0,
            "optimizer_step_evidence": [resumed_step_evidence],
            "effective_args_sha256": effective_hash,
            "augmentation_evidence": augmentation_evidence[1],
            "resume_state": resume_state,
            "runtime": runtime,
            "hash_seed_runtime": {
                "python_hash_seed": str(seed),
                "python_hash_seed_prestarted": True,
            },
            "cuda_visible_devices_at_process_start": None,
            "ultralytics_version": "8.3.222",
            "torch_version": "2.13.0+cu130",
            "torchvision_version": "0.28.0+cu130",
        }
        resume_evidence_payload = {
            **resume_evidence_body,
            "result_sha256": stable_sha256(resume_evidence_body),
        }
        resume_evidence.write_text(
            json.dumps(resume_evidence_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "loader": True,
            "model_forward": True,
            "loss": True,
            "backward": True,
            "optimizer_step": True,
            "validation": True,
            "checkpoint": True,
            "resume": True,
            "effective_args": True,
            "augmentation": True,
            "distinct_model_instance_resume": True,
            "mode": "config_bound_real",
            "config_sha256": config_sha256,
            "model": str(self.model_path),
            "model_source_sha256": sha256_file(self.model_path),
            "ultralytics_version": "8.3.222",
            "torch_version": "2.13.0+cu130",
            "torchvision_version": "0.28.0+cu130",
            "runtime": runtime,
            "hash_seed_runtime": {
                "python_hash_seed": "20260831",
                "python_hash_seed_prestarted": True,
            },
            "bounded_override_allowlist": [],
            "bounded_overrides": {},
            "runtime_bounds": {
                "configured_epochs_preserved": train_settings["epochs"],
                "maximum_train_batches": REAL_SMOKE_MAX_TRAIN_BATCHES,
                "maximum_optimizer_step_calls": REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS,
                "stop_condition": "first_underlying_adamw_step_post_hook",
                "rationale": (
                    "four_calls_empirically_exhausted_at_amp_scale_4096;"
                    "bounded_backoff_window_expanded"
                ),
            },
            "effective_args_sha256": effective_hash,
            "observed_effective_args_sha256": [effective_hash, effective_hash],
            "augmentation_args_sha256": augmentation_hash,
            "augmentation_evidence": augmentation_evidence,
            "resumed_checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "fresh_process_resume": True,
            "rng_restored": True,
            "resume_process_evidence": str(resume_evidence),
            "resume_process_evidence_sha256": sha256_file(resume_evidence),
            "initial_pid": pid,
            "resume_pid": pid + 1,
            "resume_parent_pid": resume_parent_pid,
            "initial_model_instance_id": 1,
            "resume_model_instance_id": 1,
            "resume_state": resume_state,
            "parent_cuda_release": {
                "status": "PASS",
                "released_before_fresh_process": True,
                "device": device,
                "allocated_before_bytes": 1024,
                "allocated_after_bytes": 0,
                "reserved_before_bytes": 2048,
                "reserved_after_bytes": 0,
            },
            "train_sample_count": train_sample_count,
            "maximum_train_batches": REAL_SMOKE_MAX_TRAIN_BATCHES,
            "maximum_optimizer_step_calls": REAL_SMOKE_MAX_OPTIMIZER_STEP_CALLS,
            "minimum_successful_optimizer_updates": 1,
            "observed_train_batches": self.resume_train_batches + 1,
            "observed_backward_calls": self.resume_train_batches + 1,
            "backward_gradient_finiteness": backward_gradient_finiteness,
            "observed_optimizer_step_calls": 2,
            "observed_validation_runs": worker_validation_runs + 1,
            "successful_optimizer_updates": 1,
            "amp_overflow_skips": 1,
            "optimizer_step_evidence": [
                initial_step_evidence,
                resumed_step_evidence,
            ],
            "training_source_sha256": training_source_sha256(),
            "cuda_visible_devices_at_process_start": None,
        }


def test_real_smoke_preserves_every_frozen_training_argument() -> None:
    config = load_training_config(CONFIG)

    effective = real_smoke_effective_arguments(config.train)
    drift = {
        key: (config.train[key], effective[key])
        for key in config.train
        if config.train[key] != effective[key]
    }

    assert REAL_SMOKE_BOUNDED_OVERRIDES == {}
    assert set(effective) == set(config.train)
    assert effective == config.train
    assert effective["epochs"] == 200
    assert drift == {}


def test_real_smoke_audits_executable_transform_configuration() -> None:
    config = load_training_config(CONFIG)
    expected = expected_transform_configuration(config.train)
    transform_instances = []
    for record in expected:
        transform = type(record["class_name"], (), {})()
        for name, value in record.items():
            if name != "class_name":
                setattr(transform, name, value)
        transform_instances.append(transform)
    pipeline = SimpleNamespace(transforms=transform_instances)

    assert audited_transform_configuration(pipeline, config.train) == expected
    transform_instances[0].p = 0.0
    with pytest.raises(DetectionInfrastructureError) as error:
        audited_transform_configuration(pipeline, config.train)
    assert error.value.code == "real_smoke_augmentation_drift"


def test_initial_applied_update_guard_precedes_fresh_resume_spawn() -> None:
    source = (
        CONFIG.parents[1] / "floodsight_detection/ultralytics_runtime.py"
    ).read_text(encoding="utf-8")

    guard = source.index('code="smoke_first_update_before_resume"')
    child_spawn = source.index("completed = subprocess.run(", guard)

    assert guard < child_spawn


def test_real_smoke_loader_batch_is_explicitly_clamped_to_bounded_subset() -> None:
    assert bounded_loader_batch_size(12, 2) == 2
    assert bounded_loader_batch_size(12, 8) == 8
    with pytest.raises(DetectionInfrastructureError):
        bounded_loader_batch_size(12, 9)


def test_real_smoke_restores_process_start_controls_for_fresh_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    environment = _fresh_process_environment(20260831, "unit-test-token", None)

    assert environment["PYTHONHASHSEED"] == "20260831"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["FLOODSIGHT_REAL_SMOKE_WORKER_TOKEN"] == "unit-test-token"
    assert "CUDA_VISIBLE_DEVICES" not in environment

    visible_environment = _fresh_process_environment(
        20260831,
        "unit-test-token",
        "0,1",
    )
    assert visible_environment["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_real_manifest_smoke_uses_only_bounded_representative_subset(
    detection_manifest: tuple[Path, Path], tmp_path: Path
) -> None:
    root, manifest = detection_manifest
    config = load_training_config(CONFIG)
    config = replace(
        config,
        output=replace(config.output, real_smoke_root=tmp_path / "real-smoke-root"),
    )
    contract = validate_dataset_contract(manifest, root)
    weights = load_weight_audit(
        config.weight_audit_path,
        expected_filename=config.model,
        expected_weight_path=config.model_path,
        expected_weight_sha256=config.model_sha256,
        expected_audit_path=config.weight_audit_path,
        expected_audit_sha256=config.weight_audit_sha256,
        require_license_approval=False,
    )

    report = run_real_manifest_smoke(
        config,
        contract,
        weights,
        config.output.real_smoke_root / "gate-001",
        allow_real_smoke=True,
        device="0",
        backend=PassingRealBackend(weights.path),
    )

    assert report["status"] == "PASS"
    assert report["bounded_real_manifest_smoke"] is True
    assert report["full_training_started"] is False
    assert report["schema_version"] == "floodsight-detection-real-manifest-smoke-v5"
    assert report["weight_license_review_status"] == "PENDING_HUMAN_SIGNOFF"
    assert report["maximum_train_batches"] == 8
    assert report["maximum_optimizer_step_calls"] == 8
    assert report["observed_train_batches"] == 2
    assert report["observed_optimizer_step_calls"] == 2
    assert report["observed_validation_runs"] == 2
    assert report["backward_gradient_finiteness"] == [False, True]
    assert report["successful_optimizer_updates"] == 1
    assert report["amp_overflow_skips"] == 1
    assert report["effective_args_sha256"] == report["backend"]["effective_args_sha256"]
    assert report["fresh_process_resume"] is True
    assert report["rng_state_restored"] is True
    assert report["resume_parent_pid"] == report["initial_pid"]
    assert report["backend"]["observed_train_batches"] == 2
    assert report["backend"]["observed_optimizer_step_calls"] == 2
    assert report["backend"]["backward_gradient_finiteness"] == report[
        "backward_gradient_finiteness"
    ]
    assert report["backend"]["successful_optimizer_updates"] == 1
    assert report["backend"]["bounded_override_allowlist"] == []
    assert report["backend"]["bounded_overrides"] == {}
    assert report["backend"]["resume_state"]["configured_effective_epochs"] == 200
    assert (
        report["backend"]["initial_model_instance_id"]
        == report["backend"]["resume_model_instance_id"]
    )
    assert report["backend"]["augmentation_evidence"][0]["transform_configuration"] == (
        expected_transform_configuration(config.train)
    )
    process_evidence = json.loads(
        Path(report["resume_process_evidence"]).read_text(encoding="utf-8")
    )
    assert process_evidence["schema_version"] == "floodsight-real-smoke-resume-result-v3"
    assert process_evidence["parent_pid"] == report["initial_pid"]
    assert process_evidence["maximum_train_batches"] == 7
    assert process_evidence["maximum_optimizer_step_calls"] == 7
    assert process_evidence["backward_gradient_finiteness"] == [True]
    assert process_evidence["backward_gradient_finiteness"] == report["backend"][
        "backward_gradient_finiteness"
    ][1:]
    assert len(report["subset_sample_ids"]) <= 10
    assert report["subset_split_counts"]["val"] <= 2
    attestation = load_real_smoke_attestation(
        report["report_path"], config=config, contract=contract, weights=weights
    )
    assert attestation.sha256 == report["report_sha256"]
    Path(report["backend"]["resumed_checkpoint"]).write_bytes(b"post-report-drift")
    with pytest.raises(DetectionInfrastructureError) as error:
        load_real_smoke_attestation(
            report["report_path"], config=config, contract=contract, weights=weights
        )
    assert error.value.code == "real_smoke_attestation_mismatch"


def test_real_manifest_smoke_accepts_independent_batch_and_optimizer_call_counts(
    detection_manifest: tuple[Path, Path], tmp_path: Path
) -> None:
    root, manifest = detection_manifest
    config = load_training_config(CONFIG)
    config = replace(
        config,
        output=replace(config.output, real_smoke_root=tmp_path / "real-smoke-root"),
    )
    contract = validate_dataset_contract(manifest, root)
    weights = load_weight_audit(
        config.weight_audit_path,
        expected_filename=config.model,
        expected_weight_path=config.model_path,
        expected_weight_sha256=config.model_sha256,
        expected_audit_path=config.weight_audit_path,
        expected_audit_sha256=config.weight_audit_sha256,
        require_license_approval=False,
    )

    report = run_real_manifest_smoke(
        config,
        contract,
        weights,
        config.output.real_smoke_root / "gate-independent-counts",
        allow_real_smoke=True,
        device="0",
        backend=PassingRealBackend(weights.path, resume_train_batches=2),
    )

    assert report["observed_train_batches"] == 3
    assert report["observed_optimizer_step_calls"] == 2
    assert report["observed_validation_runs"] == 3
    assert report["backend"]["observed_backward_calls"] == 3
    assert report["backward_gradient_finiteness"] == [False, False, True]
    assert report["backend"]["backward_gradient_finiteness"] == report[
        "backward_gradient_finiteness"
    ]
    assert report["backend"]["optimizer_step_evidence"][-1]["runtime_context"] == {
        "phase_batch_index": 2,
        "epoch": 2,
        "batches_per_epoch": 1,
        "global_iteration": 2,
        "configured_batch_size": 12,
        "loader_batch_size": 8,
        "accumulate": 1,
        "scheduler_last_epoch": 2,
        "learning_rates": [0.099, 0.00001, 0.00001],
    }
    attestation = load_real_smoke_attestation(
        report["report_path"], config=config, contract=contract, weights=weights
    )
    assert attestation.sha256 == report["report_sha256"]
    process_evidence = json.loads(
        Path(report["resume_process_evidence"]).read_text(encoding="utf-8")
    )
    assert process_evidence["backward_gradient_finiteness"] == [False, True]
    assert process_evidence["backward_gradient_finiteness"] == report["backend"][
        "backward_gradient_finiteness"
    ][1:]


@pytest.mark.parametrize(
    ("backend_kwargs", "expected_code"),
    [
        ({"corrupt_effective_hash": True}, "real_smoke_evidence_invalid"),
        ({"corrupt_transform_configuration": True}, "real_smoke_evidence_invalid"),
        ({"corrupt_runtime_context": True}, "smoke_bound_exceeded"),
        ({"corrupt_resume_parent_pid": True}, "real_smoke_evidence_invalid"),
        ({"corrupt_backward_gradient_finiteness": True}, "smoke_bound_exceeded"),
    ],
)
def test_real_manifest_smoke_rejects_evidence_drift(
    detection_manifest: tuple[Path, Path],
    tmp_path: Path,
    backend_kwargs: dict[str, Any],
    expected_code: str,
) -> None:
    root, manifest = detection_manifest
    config = load_training_config(CONFIG)
    config = replace(
        config,
        output=replace(config.output, real_smoke_root=tmp_path / "real-smoke-root"),
    )
    contract = validate_dataset_contract(manifest, root)
    weights = load_weight_audit(
        config.weight_audit_path,
        expected_filename=config.model,
        expected_weight_path=config.model_path,
        expected_weight_sha256=config.model_sha256,
        expected_audit_path=config.weight_audit_path,
        expected_audit_sha256=config.weight_audit_sha256,
        require_license_approval=False,
    )

    with pytest.raises(DetectionInfrastructureError) as error:
        run_real_manifest_smoke(
            config,
            contract,
            weights,
            config.output.real_smoke_root / "gate-drift",
            allow_real_smoke=True,
            device="0",
            backend=PassingRealBackend(weights.path, **backend_kwargs),
        )

    assert error.value.code == expected_code


def test_real_smoke_attestation_rejects_worker_backward_trace_drift(
    detection_manifest: tuple[Path, Path], tmp_path: Path
) -> None:
    root, manifest = detection_manifest
    config = load_training_config(CONFIG)
    config = replace(
        config,
        output=replace(config.output, real_smoke_root=tmp_path / "real-smoke-root"),
    )
    contract = validate_dataset_contract(manifest, root)
    weights = load_weight_audit(
        config.weight_audit_path,
        expected_filename=config.model,
        expected_weight_path=config.model_path,
        expected_weight_sha256=config.model_sha256,
        expected_audit_path=config.weight_audit_path,
        expected_audit_sha256=config.weight_audit_sha256,
        require_license_approval=False,
    )

    report = run_real_manifest_smoke(
        config,
        contract,
        weights,
        config.output.real_smoke_root / "gate-worker-trace-drift",
        allow_real_smoke=True,
        device="0",
        backend=PassingRealBackend(
            weights.path,
            resume_train_batches=2,
            corrupt_worker_backward_gradient_finiteness=True,
        ),
    )

    assert report["backward_gradient_finiteness"] == [False, False, True]
    with pytest.raises(DetectionInfrastructureError) as error:
        load_real_smoke_attestation(
            report["report_path"], config=config, contract=contract, weights=weights
        )
    assert error.value.code == "real_smoke_attestation_mismatch"
