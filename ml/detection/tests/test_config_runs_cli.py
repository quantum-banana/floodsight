from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from floodsight_detection.checkpointing import publish_checkpoint_generation
from floodsight_detection.cli import run
from floodsight_detection.config import (
    DETECTION_REAL_SMOKE_ROOT,
    DETECTION_RUN_ROOT,
    EXPECTED_DATASET_FINGERPRINT,
    EXPECTED_MANIFEST_PATH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_VERSION,
    ULTRALYTICS_INAPPLICABLE_KEYS,
    ULTRALYTICS_RUNTIME_KEYS,
    ULTRALYTICS_TRAIN_VAL_KEYS,
    load_training_config,
    validate_ultralytics_default_contract,
    version_matches,
)
from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file, stable_sha256
from floodsight_detection.runs import exclusive_run_lock, reserve_new_run, validate_resume
from floodsight_detection.runtime import (
    bound_real_smoke_output_directory,
    bound_training_device,
    bound_training_output_root,
    validate_accepted_environment,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/yolo11l_h100_max_quality.yaml"
REPO = Path(__file__).resolve().parents[3]


def _run_identity() -> dict[str, str]:
    return {
        "config_sha256": "a" * 64,
        "training_code_sha256": "0" * 64,
        "manifest_id": "visdrone_det-detection_v2",
        "dataset_id": "visdrone_det",
        "preparation_version": "detection_v2",
        "manifest_sha256": "b" * 64,
        "dataset_fingerprint": "c" * 64,
        "taxonomy_version": "detection-taxonomy-v1",
        "taxonomy_sha256": "d" * 64,
        "mapping_version": "visdrone-mapping-v1",
        "mapping_sha256": "e" * 64,
        "weights_path": "/canonical/yolo11l.pt",
        "weights_sha256": "f" * 64,
        "weight_audit_path": "/canonical/yolo11l-weight-audit.json",
        "weight_audit_sha256": "2" * 64,
        "approval_sha256": "1" * 64,
        "approval_id": "approval-1",
        "real_smoke_report_path": "/canonical/real-manifest-smoke-report.json",
        "real_smoke_report_sha256": "3" * 64,
        "device": "0",
    }


def _checkpoint_arguments() -> dict[str, object]:
    return {"schema_version": "unit-test-trainer-arguments-v1", "data": "bound"}


def _publish_test_checkpoint(run_state: object) -> Path:
    import torch

    run_directory = run_state.run_directory
    data_yaml = run_directory / "dataset-contract/dataset.yaml"
    data_yaml.parent.mkdir(exist_ok=True)
    data_yaml.write_text("{}\n", encoding="utf-8")
    checkpoint = run_directory / "weights/last.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.write_bytes(b"content-bound-test-checkpoint")
    generator = torch.Generator().manual_seed(7)
    trainer = SimpleNamespace(
        train_loader=SimpleNamespace(
            generator=generator,
            sampler=SimpleNamespace(generator=generator),
        )
    )
    publish_checkpoint_generation(
        run_directory=run_directory,
        run_metadata_path=run_state.metadata_path,
        live_checkpoint=checkpoint,
        epoch=0,
        trainer_arguments=_checkpoint_arguments(),
        data_yaml=data_yaml,
        trainer=trainer,
    )
    return checkpoint


def test_primary_h100_configuration_is_strict_and_complete() -> None:
    config = load_training_config(CONFIG)

    assert config.model == "yolo11l.pt"
    assert config.ultralytics_version == "8.3.222"
    assert config.torch_version == "2.13.0+cu130"
    assert config.torchvision_version == "0.28.0+cu130"
    assert config.train["imgsz"] == 1536
    assert config.train["workers"] == 0
    assert config.train["deterministic"] is True
    assert config.dataset.verify_image_hashes is True
    assert config.dataset.manifest_path == EXPECTED_MANIFEST_PATH
    assert config.dataset.manifest_sha256 == EXPECTED_MANIFEST_SHA256
    assert config.dataset.dataset_fingerprint == EXPECTED_DATASET_FINGERPRINT
    assert config.dataset.source_version == EXPECTED_SOURCE_VERSION
    assert config.output.run_root == DETECTION_RUN_ROOT
    assert config.output.real_smoke_root == DETECTION_REAL_SMOKE_ROOT
    assert config.output.new_run_policy == "exclusive_direct_child"
    assert config.output.resume_policy == "approved_run_last_only"
    assert config.output.last_checkpoint_filename == "last.pt"
    assert config.output.best_checkpoint_filename == "best.pt"
    assert len(config.train) == 68
    assert config.dataset.taxonomy_sha256 == (
        "4000f1bb75b2e4687d60027a72dd3f428c9bed8ba8de918c0892e3f115bdf535"
    )


def test_loader_is_in_process_because_one_raw_batch_exceeds_host_shm() -> None:
    config = load_training_config(CONFIG)
    raw_batch_bytes = config.train["batch"] * 3 * config.train["imgsz"] ** 2

    assert raw_batch_bytes == 84_934_656
    assert raw_batch_bytes > 64 * 1024 * 1024
    assert config.train["workers"] == 0


def test_ultralytics_83222_default_keys_are_exhaustively_partitioned() -> None:
    from ultralytics.cfg import DEFAULT_CFG_DICT

    validate_ultralytics_default_contract(DEFAULT_CFG_DICT)
    sets = (
        ULTRALYTICS_TRAIN_VAL_KEYS,
        ULTRALYTICS_RUNTIME_KEYS,
        ULTRALYTICS_INAPPLICABLE_KEYS,
    )
    assert set(DEFAULT_CFG_DICT) == set().union(*sets)
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


@pytest.mark.parametrize("version", ["8.3.999", "8.4.0", "8.3.221"])
def test_dependency_versions_must_match_exactly(version: str) -> None:
    assert version_matches(version, "8.3.222") is False


def test_dependency_exact_version_match() -> None:
    assert version_matches("2.13.0+cu130", "2.13.0+cu130") is True


def test_shared_accepted_environment_is_bound_to_full_snapshot() -> None:
    evidence = validate_accepted_environment()

    assert evidence["environment_marker_sha256"] == (
        "11ec5e2bc107465862ab04f8a01d58719c5012356489168cf28387f6848f96bd"
    )
    assert evidence["resolved_lock_sha256"] == (
        "33e7ca74a272659827d10c3bc882de1aa6e39b871c36435eb52279bd88eb58e1"
    )
    assert evidence["installed_distributions_sha256"] == (
        "ee7f9ce2704ddaea38312d0e11dacb8d01270f8be04ac1aad7e31095878ce775"
    )
    assert evidence["installed_distribution_count"] == "103"


def test_full_training_device_override_must_match_frozen_config() -> None:
    assert bound_training_device("0", None) == "0"
    assert bound_training_device("0", "0") == "0"
    with pytest.raises(DetectionInfrastructureError) as error:
        bound_training_device("0", "1")
    assert error.value.code == "training_device_drift"


def test_output_roots_and_direct_child_policy_are_config_bound(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    smoke_root = tmp_path / "real-smoke"

    assert bound_training_output_root(training_root, training_root) == training_root.resolve()
    with pytest.raises(DetectionInfrastructureError) as training_error:
        bound_training_output_root(training_root, tmp_path / "other")
    assert training_error.value.code == "training_output_root_drift"

    smoke = smoke_root / "gate-001"
    assert bound_real_smoke_output_directory(smoke_root, smoke) == smoke.resolve()
    for invalid in (smoke_root, tmp_path / "outside", smoke / "nested"):
        with pytest.raises(DetectionInfrastructureError) as smoke_error:
            bound_real_smoke_output_directory(smoke_root, invalid)
        assert smoke_error.value.code == "real_smoke_output_policy_drift"


def test_unknown_configuration_key_is_blocking(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["train"]["mystery"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        load_training_config(path)

    assert error.value.code == "training_config_invalid"


def test_omitted_frozen_training_key_cannot_fall_back_to_library_default(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    del payload["train"]["mixup"]
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        load_training_config(path)

    assert error.value.code == "training_config_invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_root", "/different/run-root"),
        ("real_smoke_root", "/different/smoke-root"),
        ("new_run_policy", "reuse"),
        ("resume_policy", "any-checkpoint"),
        ("last_checkpoint_filename", "latest.pt"),
    ],
)
def test_frozen_output_policy_drift_is_blocking(
    tmp_path: Path, field: str, value: str
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["output"][field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        load_training_config(path)

    assert error.value.code == "training_config_invalid"


def test_invalid_detector_stride_or_switch_type_is_blocking(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["train"]["imgsz"] = 1537
    payload["train"]["amp"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        load_training_config(path)

    assert error.value.code == "training_config_invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [("single_cls", True), ("save_json", True), ("pretrained", False), ("fraction", 0.5)],
)
def test_semantically_unsafe_validly_typed_switch_drift_is_blocking(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["train"][field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        load_training_config(path)

    assert error.value.code == "training_config_invalid"


def test_worker_processes_are_blocked_for_the_64_mib_shm_host(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["train"]["workers"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        load_training_config(path)

    assert error.value.code == "training_config_invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fraction", 0.0),
        ("iou", 1.1),
        ("warmup_momentum", float("nan")),
        ("max_det", True),
        ("copy_paste_mode", "unknown"),
        ("freeze", 1),
    ],
)
def test_every_new_frozen_train_key_family_is_type_and_range_checked(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["train"][field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        load_training_config(path)

    assert error.value.code == "training_config_invalid"


def test_taxonomy_drift_in_configuration_is_blocking(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["classes"]["0"] = "pedestrian"
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        load_training_config(path)

    assert error.value.code == "training_config_invalid"


def test_taxonomy_or_mapping_fingerprint_drift_is_blocking(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["dataset_gate"]["mapping_sha256"] = "0" * 64
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        load_training_config(path)

    assert error.value.code == "training_config_invalid"


def test_real_train_cli_guard_fails_before_reading_any_paths(tmp_path: Path) -> None:
    with pytest.raises(DetectionInfrastructureError) as error:
        run(
            [
                "train",
                "--config",
                str(tmp_path / "absent-config"),
                "--manifest",
                str(tmp_path / "absent-manifest"),
                "--data-root",
                str(tmp_path / "absent-data"),
                "--output-root",
                str(tmp_path / "runs"),
                "--run-name",
                "forbidden",
            ]
        )

    assert error.value.code == "training_not_authorized"
    assert not (tmp_path / "runs").exists()


def test_locked_cli_rejects_unfrozen_train_and_real_smoke_outputs(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "20260831"
    launcher = REPO / "scripts/training/run-locked.sh"
    cases = (
        (
            [
                "train",
                "--config",
                str(CONFIG),
                "--manifest",
                str(tmp_path / "absent-manifest"),
                "--data-root",
                str(tmp_path / "absent-data"),
                "--output-root",
                str(tmp_path / "wrong-training-root"),
                "--run-name",
                "blocked-run",
                "--weights-audit",
                str(tmp_path / "absent-weight-audit"),
                    "--approval",
                    str(tmp_path / "absent-approval"),
                    "--real-smoke-report",
                    str(tmp_path / "absent-real-smoke"),
                "--allow-training",
            ],
            "training_output_root_drift",
            tmp_path / "wrong-training-root",
        ),
        (
            [
                "real-smoke",
                "--config",
                str(CONFIG),
                "--manifest",
                str(tmp_path / "absent-manifest"),
                "--data-root",
                str(tmp_path / "absent-data"),
                "--weights-audit",
                str(tmp_path / "absent-weight-audit"),
                "--output",
                str(tmp_path / "wrong-real-smoke-root"),
                "--device",
                "0",
                "--allow-real-smoke",
            ],
            "real_smoke_output_policy_drift",
            tmp_path / "wrong-real-smoke-root",
        ),
    )
    for arguments, expected_code, forbidden_output in cases:
        result = subprocess.run(
            ["bash", str(launcher), "detection", *arguments],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 2
        assert expected_code in result.stdout
        assert not forbidden_output.exists()


def test_synthetic_smoke_cli_guard_fails_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "smoke"

    with pytest.raises(DetectionInfrastructureError) as error:
        run(["smoke", "--output", str(output)])

    assert error.value.code == "synthetic_smoke_not_authorized"
    assert not output.exists()


def test_run_reservation_is_atomic_and_collision_safe(tmp_path: Path) -> None:
    first = reserve_new_run(
        tmp_path,
        "detector-run-001",
        **_run_identity(),
    )
    assert first.metadata_path.is_file()
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["output_root"] == str(tmp_path.resolve())
    assert metadata["training_code_sha256"] == "0" * 64
    assert metadata["device"] == "0"

    with pytest.raises(DetectionInfrastructureError) as error:
        reserve_new_run(
            tmp_path,
            "detector-run-001",
            **_run_identity(),
        )

    assert error.value.code == "run_collision"


def test_process_lifetime_run_lock_is_exclusive_and_recoverable(tmp_path: Path) -> None:
    run_state = reserve_new_run(tmp_path, "detector-run-001", **_run_identity())

    with exclusive_run_lock(run_state) as lock_path:
        assert lock_path.is_file()
        assert lock_path.read_text(encoding="utf-8").startswith("pid=")
        with (
            pytest.raises(DetectionInfrastructureError) as error,
            exclusive_run_lock(run_state),
        ):
            pass
        assert error.value.code == "run_already_active"

    with exclusive_run_lock(run_state):
        pass


def test_resume_requires_last_checkpoint_and_matching_frozen_identity(tmp_path: Path) -> None:
    run_state = reserve_new_run(
        tmp_path,
        "detector-run-001",
        **_run_identity(),
    )
    checkpoint = run_state.run_directory / "weights/last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(DetectionInfrastructureError) as plain_error:
        validate_resume(
            checkpoint,
            tmp_path,
            run_name="detector-run-001",
            checkpoint_trainer_arguments=_checkpoint_arguments(),
            **_run_identity(),
        )
    assert plain_error.value.code == "checkpoint_integrity_failed"

    checkpoint.unlink()
    checkpoint = _publish_test_checkpoint(run_state)

    resumed = validate_resume(
        checkpoint,
        tmp_path,
        run_name="detector-run-001",
        checkpoint_trainer_arguments=_checkpoint_arguments(),
        **_run_identity(),
    )

    assert resumed.resumed is True
    assert resumed.checkpoint is not None
    assert resumed.checkpoint.parent.name == ".floodsight-checkpoints"
    assert resumed.trusted_checkpoint is not None
    assert resumed.trusted_checkpoint.sha256 == sha256_file(checkpoint)

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_resume(
            checkpoint,
            tmp_path,
            run_name="detector-run-001",
            checkpoint_trainer_arguments=_checkpoint_arguments(),
            **(_run_identity() | {"config_sha256": "2" * 64}),
        )
    assert error.value.code == "resume_contract_mismatch"


def test_resume_rejects_checkpoint_outside_run_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside/weights/last.pt"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"checkpoint")
    root = tmp_path / "runs"
    root.mkdir()

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_resume(
            outside,
            root,
            run_name="detector-run-001",
            checkpoint_trainer_arguments=_checkpoint_arguments(),
            **_run_identity(),
        )

    assert error.value.code == "unsafe_resume_checkpoint"


def test_resume_rejects_live_checkpoint_tampering_after_trusted_publish(
    tmp_path: Path,
) -> None:
    run_state = reserve_new_run(tmp_path, "detector-run-001", **_run_identity())
    checkpoint = _publish_test_checkpoint(run_state)
    checkpoint.write_bytes(b"tampered-after-publish")

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_resume(
            checkpoint,
            tmp_path,
            run_name="detector-run-001",
            checkpoint_trainer_arguments=_checkpoint_arguments(),
            **_run_identity(),
        )

    assert error.value.code == "checkpoint_integrity_failed"


def test_resume_rejects_rehashed_dataset_contract_path_escape(tmp_path: Path) -> None:
    run_state = reserve_new_run(tmp_path, "detector-run-001", **_run_identity())
    checkpoint = _publish_test_checkpoint(run_state)
    escaped_yaml = run_state.run_directory / "outside-dataset.yaml"
    escaped_yaml.write_text("{}\n", encoding="utf-8")
    metadata_path = next(
        (run_state.run_directory / ".floodsight-checkpoints").glob("epoch-*.json")
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["data_yaml_relative_path"] = escaped_yaml.name
    metadata["data_yaml_sha256"] = sha256_file(escaped_yaml)
    metadata["metadata_sha256"] = stable_sha256(
        {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    )
    metadata_path.chmod(0o640)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    pointer_path = run_state.run_directory / ".floodsight-last-checkpoint.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["metadata_sha256"] = metadata["metadata_sha256"]
    pointer["pointer_sha256"] = stable_sha256(
        {key: value for key, value in pointer.items() if key != "pointer_sha256"}
    )
    pointer_path.chmod(0o640)
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_resume(
            checkpoint,
            tmp_path,
            run_name="detector-run-001",
            checkpoint_trainer_arguments=_checkpoint_arguments(),
            **_run_identity(),
        )

    assert error.value.code == "checkpoint_integrity_failed"


def test_resume_rejects_metadata_tampering_even_before_identity_comparison(tmp_path: Path) -> None:
    run_state = reserve_new_run(tmp_path, "detector-run-001", **_run_identity())
    checkpoint = run_state.run_directory / "weights/last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    metadata = json.loads(run_state.metadata_path.read_text(encoding="utf-8"))
    metadata["device"] = "1"
    run_state.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_resume(
            checkpoint,
            tmp_path,
            run_name="detector-run-001",
            checkpoint_trainer_arguments=_checkpoint_arguments(),
            **_run_identity(),
        )

    assert error.value.code == "resume_metadata_invalid"


def test_resume_rejects_rehashed_approval_identity_drift(tmp_path: Path) -> None:
    run_state = reserve_new_run(tmp_path, "detector-run-001", **_run_identity())
    checkpoint = run_state.run_directory / "weights/last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    metadata = json.loads(run_state.metadata_path.read_text(encoding="utf-8"))
    metadata["approval_id"] = "different-approval"
    unsigned = {key: value for key, value in metadata.items() if key != "reservation_sha256"}
    metadata["reservation_sha256"] = stable_sha256(unsigned)
    run_state.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_resume(
            checkpoint,
            tmp_path,
            run_name="detector-run-001",
            checkpoint_trainer_arguments=_checkpoint_arguments(),
            **_run_identity(),
        )

    assert error.value.code == "resume_contract_mismatch"


def test_hash_seed_must_be_active_at_interpreter_start_not_set_late() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO / "ml/detection")
    environment["PYTHONHASHSEED"] = "1"
    script = (
        "import os; "
        "os.environ['PYTHONHASHSEED']='20260831'; "
        "from floodsight_detection.determinism import require_prestarted_hash_seed; "
        "from floodsight_detection.errors import DetectionInfrastructureError; "
        "\ntry: require_prestarted_hash_seed(20260831)\n"
        "except DetectionInfrastructureError as exc: print(exc.code); raise SystemExit(2)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "python_hash_seed_not_prestarted" in result.stdout


def test_pinned_hash_seed_is_accepted_when_active_at_interpreter_start() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO / "ml/detection")
    environment["PYTHONHASHSEED"] = "20260831"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from floodsight_detection.determinism import require_prestarted_hash_seed; "
                "print(require_prestarted_hash_seed(20260831)['python_hash_seed_prestarted'])"
            ),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_locked_launcher_rejects_conflicting_hash_seed_before_python() -> None:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "1"
    result = subprocess.run(
        ["bash", str(REPO / "scripts/training/run-locked.sh"), "detection", "--help"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "requires PYTHONHASHSEED=20260831" in result.stderr


def test_offline_runtime_accepts_only_the_exact_ultralytics_newline_rewrite(
    tmp_path: Path,
) -> None:
    template = (REPO / "ml/training/ultralytics-settings-v1.json").read_bytes()
    assert template.endswith(b"\n")
    settings = tmp_path / "runtime/yolo-config/Ultralytics/settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_bytes(template[:-1])
    environment = dict(os.environ)
    environment["FLOODSIGHT_ML_RUNTIME_CACHE"] = str(tmp_path / "runtime")
    source_runtime = (
        'set -euo pipefail; repo_root="$1"; '
        'source "$repo_root/scripts/training/runtime-offline.sh"'
    )

    accepted = subprocess.run(
        ["bash", "-c", source_runtime, "runtime-test", str(REPO)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr

    corrupted = bytearray(template[:-1])
    corrupted[10] ^= 1
    settings.write_bytes(corrupted)
    rejected = subprocess.run(
        ["bash", "-c", source_runtime, "runtime-test", str(REPO)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "Refusing drifted Ultralytics runtime settings" in rejected.stderr


def test_locked_check_uses_frozen_audit_attestation_without_network() -> None:
    check_script = REPO / "scripts/training/check.sh"
    text = check_script.read_text(encoding="utf-8")

    assert 'accepted_marker_sha256="11ec5e2bc107465862ab04f8a01d58719' in text
    assert "Accepted-environment marker hash drifted" in text
    assert "-m pip_audit" not in text
    assert "HTTP_PROXY=" not in text
    assert "torch==2.13.0+cu130" in text
    assert "torchvision==0.28.0+cu130" in text
    subprocess.run(["bash", "-n", str(check_script)], check=True)
