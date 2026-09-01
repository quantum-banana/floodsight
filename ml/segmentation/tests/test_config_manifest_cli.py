from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from floodsight_segmentation.approval import (
    REQUIRED_REVIEW_ACKNOWLEDGEMENTS,
    _validate_human_review,
    validate_human_approval,
)
from floodsight_segmentation.artifact import ModelArtifactSpec, validate_model_artifact
from floodsight_segmentation.cli import DEFAULT_CONFIG, main
from floodsight_segmentation.config import (
    FINAL_CLASS_WEIGHTS,
    MANIFEST_FINGERPRINT_ALGORITHM,
    PRODUCTION_MODEL_ID,
    PRODUCTION_MODEL_REVISION,
    PRODUCTION_PROVENANCE_PATH,
    PRODUCTION_PROVENANCE_SHA256,
    PRODUCTION_SAFETENSORS_PATH,
    PRODUCTION_SAFETENSORS_SHA256,
    SEGMENTATION_REAL_SMOKE_ROOT,
    SEGMENTATION_RUN_ROOT,
    STAGE10_CLASS_WEIGHT_REPORT_PATH,
    STAGE10_CLASS_WEIGHT_REPORT_SHA256,
    UPSTREAM_PYTORCH_MODEL_SHA256,
    load_config,
)
from floodsight_segmentation.errors import (
    ConfigurationError,
    ManifestError,
    TrainingAuthorizationError,
)
from floodsight_segmentation.integrity import training_source_sha256
from floodsight_segmentation.manifest import (
    CANONICAL_MANIFEST_LOCKS,
    IGNORE_SEMANTICS,
    PREPARATION_VERSION,
    TOOL_VERSION,
    canonical_manifest_fingerprint,
    load_frozen_manifest,
    load_manifest_collection,
    require_canonical_manifest_locks,
    validate_relative_path,
)
from floodsight_segmentation.runtime import validate_runtime_versions

HASH_A = "a" * 64
HASH_B = "b" * 64


def _manifest(
    path: Path, *, sample_id: str = "floodnet-train-sample-1", split: str = "train"
) -> str:
    payload = {
        "schema_version": "dataset-manifest-v1",
        "manifest_id": f"floodnet-{PREPARATION_VERSION}-{split}",
        "dataset_id": "floodnet",
        "task_type": "SEMANTIC_SEGMENTATION",
        "source_version": "FloodNet-Supervised_v1.0",
        "preparation_version": PREPARATION_VERSION,
        "taxonomy_version": "segmentation-taxonomy-v2",
        "integrity_mode": "full",
        "created_at": "2026-01-01T00:00:00Z",
        "tool_version": TOOL_VERSION,
        "git_commit": "f272a31ae05e9cb8e532e939e3ca02365755e6a9",
        "fingerprint": HASH_A,
        "samples": [
            {
                "sample_id": sample_id,
                "source_dataset": "floodnet",
                "source_split": split,
                "target_split": split,
                "image_path": (
                    f"raw/floodnet/FloodNet-Supervised_v1.0/{split}/"
                    f"{split}-org-img/{sample_id}.jpg"
                ),
                "source_annotation_path": (
                    f"raw/floodnet/FloodNet-Supervised_v1.0/{split}/"
                    f"{split}-label-img/{sample_id}_lab.png"
                ),
                "target_annotation_path": (
                    f"processed/{PREPARATION_VERSION}/masks/floodnet/{split}/{sample_id}.png"
                ),
                "width": 8,
                "height": 6,
                "image_hash": HASH_A,
                "annotation_hash": HASH_B,
                "target_image_hash": HASH_A,
                "target_annotation_hash": HASH_B,
                "class_counts": {"0": 40, "3": 8},
                "ignored_count": 0,
                "invalid_count": 0,
                "preparation_version": PREPARATION_VERSION,
                "taxonomy_version": "segmentation-taxonomy-v2",
                "objects": [],
                "source_schema": "floodnet-supervised-v1.0-indexed-mask-ids-0-9",
                "target_mapping_version": "floodnet-mapping-v2",
                "target_mapping_sha256": (
                    "fdfbbba84c1cf8ea0176429b8d236693030abc16452f507c94922cc2f0769760"
                ),
                "valid_supervision_classes": [0, 1, 2, 3, 6, 7, 12, 13, 14, 15],
                "ignore_index": 255,
                "ignore_semantics": IGNORE_SEMANTICS,
                "exclusion_status": "INCLUDED",
                "exclusion_reason": "",
            }
        ],
    }
    payload["fingerprint"] = canonical_manifest_fingerprint(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_fingerprint(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["fingerprint"])


def test_frozen_config_pins_b2_taxonomy_and_partial_supervision() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.frozen
    assert config.model.pretrained_model_name_or_path == PRODUCTION_MODEL_ID
    assert config.model.revision == PRODUCTION_MODEL_REVISION
    assert config.model.upstream_pytorch_model_sha256 == UPSTREAM_PYTORCH_MODEL_SHA256
    assert config.model.safetensors_path == PRODUCTION_SAFETENSORS_PATH
    assert config.model.safetensors_sha256 == PRODUCTION_SAFETENSORS_SHA256
    assert config.model.provenance_path == PRODUCTION_PROVENANCE_PATH
    assert config.model.provenance_sha256 == PRODUCTION_PROVENANCE_SHA256
    assert config.model.local_files_only is True
    assert config.model.trust_remote_code is False
    assert config.model.num_labels == 16
    assert (config.transforms.height, config.transforms.width) == (1024, 1024)
    assert config.training.batch_size == 4
    assert config.training.gradient_accumulation_steps == 4
    assert config.data.ignore_index == 255
    assert config.data.manifest_fingerprint_algorithm == MANIFEST_FINGERPRINT_ALGORITHM
    assert config.data.supported_class_ids["floodnet"] == frozenset(
        {0, 1, 2, 3, 6, 7, 12, 13, 14, 15}
    )
    assert config.data.supported_class_ids["rescuenet"] == frozenset(
        {0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 15}
    )
    assert config.taxonomy_assets.taxonomy.sha256 == (
        "2e10de69f9920aa113ae65c77b275ca506ed27ea7369bd8c9067bd899eca9ccf"
    )
    assert set(config.taxonomy_assets.mappings) == {"floodnet", "rescuenet"}
    assert config.loss.class_weight_policy == "fixed_explicit"
    assert config.loss.class_weight_source_path == STAGE10_CLASS_WEIGHT_REPORT_PATH
    assert config.loss.class_weight_source_sha256 == STAGE10_CLASS_WEIGHT_REPORT_SHA256
    assert len(config.loss.class_weights) == 16
    assert config.loss.class_weights == FINAL_CLASS_WEIGHTS
    assert config.sampler.dataset_mix == {"floodnet": 0.5, "rescuenet": 0.5}
    assert config.output.run_root == SEGMENTATION_RUN_ROOT
    assert config.output.real_smoke_root == SEGMENTATION_REAL_SMOKE_ROOT
    assert config.output.resume_policy == "approved_run_last_only"
    assert config.data.require_full_integrity is True
    assert config.data.verify_sample_hashes is True
    assert config.training.precision == "bf16"
    assert config.training.num_workers == 0
    assert config.optimizer.epsilon == 1e-8
    assert config.optimizer.foreach is False
    assert config.optimizer.fused is False
    assert len(config.sha256) == 64


def test_locked_segmentation_runtime_versions_are_exact() -> None:
    versions = validate_runtime_versions()
    assert versions["torch"] == "2.13.0+cu130"
    assert versions["transformers"] == "5.15.1"
    assert versions["PyYAML"] == "6.0.3"
    assert versions["torch_cuda"] == "13.0"
    assert versions["environment_marker_sha256"] == (
        "11ec5e2bc107465862ab04f8a01d58719c5012356489168cf28387f6848f96bd"
    )
    assert versions["resolved_lock_sha256"] == (
        "33e7ca74a272659827d10c3bc882de1aa6e39b871c36435eb52279bd88eb58e1"
    )
    assert versions["installed_distributions_sha256"] == (
        "ee7f9ce2704ddaea38312d0e11dacb8d01270f8be04ac1aad7e31095878ce775"
    )
    assert versions["installed_distribution_count"] == "103"


def test_config_rejects_unfrozen_copy(tmp_path: Path) -> None:
    payload = DEFAULT_CONFIG.read_text(encoding="utf-8").replace("frozen: true", "frozen: false")
    path = tmp_path / "config.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="frozen"):
        load_config(path)


def test_config_rejects_impossible_random_crop_area(tmp_path: Path) -> None:
    payload = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
        "train_scale: [0.5, 1.0]", "train_scale: [0.5, 1.1]"
    )
    path = tmp_path / "config.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="source-area fraction"):
        load_config(path)


def test_config_rejects_non_50_50_dataset_mix_even_when_sum_is_one(tmp_path: Path) -> None:
    payload = DEFAULT_CONFIG.read_text(encoding="utf-8")
    payload = payload.replace("    floodnet: 0.5", "    floodnet: 0.4")
    payload = payload.replace("    rescuenet: 0.5", "    rescuenet: 0.6")
    path = tmp_path / "config.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="50/50"):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("    - background_other", "    - wrong_background", "taxonomy order"),
        ("  ignore_index: 255", "  ignore_index: 254", "frozen value 255"),
        (
            "    floodnet: [0, 1, 2, 3, 6, 7, 12, 13, 14, 15]",
            "    floodnet: [0, 1, 2, 3, 6, 7, 12, 13, 14]",
            "partial-supervision map",
        ),
        ("  learning_rate: 0.00006", "  learning_rate: .nan", "Expected a number"),
        (
            "  require_full_integrity: true",
            "  require_full_integrity: false",
            "full manifest integrity",
        ),
        (
            "  verify_sample_hashes: true",
            "  verify_sample_hashes: false",
            "per-sample hashes",
        ),
        ("  train_splits: [train]", "  train_splits: [val]", "split roles"),
        (
            "  validation_splits: [val]",
            "  validation_splits: [holdout]",
            "split roles",
        ),
        ("  precision: bf16", "  precision: fp32", "must be bf16"),
        ("    floodnet: 0.5", "    floodnet: 0.4", "sum to 1.0"),
    ],
)
def test_config_rejects_semantic_or_nonfinite_drift(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    payload = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(old, new)
    assert payload != DEFAULT_CONFIG.read_text(encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    "path",
    ["../secret", "/absolute/path", r"C:\\dataset\\image.jpg", r"nested\\image.jpg"],
)
def test_manifest_relative_paths_reject_escape_and_nonportable_forms(path: str) -> None:
    with pytest.raises(ManifestError):
        validate_relative_path(path, location="fixture")


def test_manifest_requires_exact_hash_full_integrity_and_split(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    digest = _manifest(path)
    loaded = load_frozen_manifest(
        path,
        expected_sha256=digest,
        expected_fingerprint=_manifest_fingerprint(path),
        expected_taxonomy="segmentation-taxonomy-v2",
        allowed_datasets=("floodnet", "rescuenet"),
        selected_splits=("train",),
    )
    assert loaded.sha256 == digest
    assert loaded.samples[0].sample_id == "floodnet-train-sample-1"
    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        load_frozen_manifest(
            path,
            expected_sha256="0" * 64,
            expected_fingerprint=_manifest_fingerprint(path),
            expected_taxonomy="segmentation-taxonomy-v2",
            allowed_datasets=("floodnet",),
            selected_splits=("train",),
        )
    with pytest.raises(ManifestError, match="no samples"):
        load_frozen_manifest(
            path,
            expected_sha256=digest,
            expected_fingerprint=_manifest_fingerprint(path),
            expected_taxonomy="segmentation-taxonomy-v2",
            allowed_datasets=("floodnet",),
            selected_splits=("validation",),
        )


def test_manifest_fingerprint_is_recomputed_from_canonical_persisted_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    _manifest(path)
    original = json.loads(path.read_text(encoding="utf-8"))
    declared = original["fingerprint"]
    assert declared == canonical_manifest_fingerprint(original)
    original["samples"][0]["width"] = 9
    path.write_text(json.dumps(original), encoding="utf-8")
    tampered_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ManifestError, match="fingerprint recomputation mismatch"):
        load_frozen_manifest(
            path,
            expected_sha256=tampered_sha256,
            expected_fingerprint=declared,
            expected_taxonomy="segmentation-taxonomy-v2",
            allowed_datasets=("floodnet",),
            selected_splits=("train",),
        )


def test_manifest_requires_the_exact_stage9_sample_contract(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _manifest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["samples"][0]["target_mapping_version"]
    payload["fingerprint"] = canonical_manifest_fingerprint(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="Invalid sample keys"):
        load_frozen_manifest(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_fingerprint=payload["fingerprint"],
            expected_taxonomy="segmentation-taxonomy-v2",
            allowed_datasets=("floodnet",),
            selected_splits=("train",),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_split", "val", "split identity drift"),
        (
            "image_path",
            "raw/rescuenet/legacy/train-org-img/floodnet-train-sample-1.jpg",
            "path identity drift",
        ),
        ("class_counts", {"0": 39, "3": 8}, "do not match dimensions"),
    ],
)
def test_manifest_rejects_split_path_and_pixel_count_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path = tmp_path / "manifest.json"
    _manifest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0][field] = value
    payload["fingerprint"] = canonical_manifest_fingerprint(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match=message):
        load_frozen_manifest(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_fingerprint=payload["fingerprint"],
            expected_taxonomy="segmentation-taxonomy-v2",
            allowed_datasets=("floodnet",),
            selected_splits=("train",),
        )


def test_collection_rejects_cross_manifest_duplicate_sample_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_hash = _manifest(first)
    second_hash = _manifest(second)
    with pytest.raises(ManifestError, match="across frozen manifests"):
        load_manifest_collection(
            [
                (first, first_hash, _manifest_fingerprint(first)),
                (second, second_hash, _manifest_fingerprint(second)),
            ],
            expected_taxonomy="segmentation-taxonomy-v2",
            allowed_datasets=("floodnet",),
            selected_splits=("train",),
        )


def test_collection_rejects_duplicate_image_hashes_under_different_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_hash = _manifest(first, sample_id="floodnet-train-first")
    second_hash = _manifest(second, sample_id="floodnet-train-second")
    with pytest.raises(ManifestError, match="image SHA-256 values"):
        load_manifest_collection(
            [
                (first, first_hash, _manifest_fingerprint(first)),
                (second, second_hash, _manifest_fingerprint(second)),
            ],
            expected_taxonomy="segmentation-taxonomy-v2",
            allowed_datasets=("floodnet",),
            selected_splits=("train",),
        )


def test_fixture_manifest_does_not_satisfy_canonical_stage9_lock(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    digest = _manifest(path)
    collection = load_manifest_collection(
        [(path, digest, _manifest_fingerprint(path))],
        expected_taxonomy="segmentation-taxonomy-v2",
        allowed_datasets=("floodnet",),
        selected_splits=("train",),
    )
    with pytest.raises(ManifestError, match="canonical Stage-9"):
        require_canonical_manifest_locks(collection)


def _write_minimal_safetensors(path: Path) -> str:
    descriptor = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    header = json.dumps(descriptor, separators=(",", ":")).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header + struct.pack("<f", 1.0))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_fixture(tmp_path: Path) -> tuple[ModelArtifactSpec, object]:
    config = load_config(DEFAULT_CONFIG)
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    weights = artifact_dir / "model.safetensors"
    weights_hash = _write_minimal_safetensors(weights)
    provenance = artifact_dir / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "floodsight-model-artifact-v1",
                "artifact_kind": "SEGFORMER_B2_SAFETENSORS",
                "model_id": PRODUCTION_MODEL_ID,
                "source_revision": PRODUCTION_MODEL_REVISION,
                "source_filename": "pytorch_model.bin",
                "source_sha256": UPSTREAM_PYTORCH_MODEL_SHA256,
                "conversion_tool": "synthetic-test-only",
                "converted_at": "2026-08-31T00:00:00Z",
                "safetensors_filename": "model.safetensors",
                "safetensors_sha256": weights_hash,
                "audit_status": "PASS",
                "human_review_status": "PENDING_HUMAN_SIGNOFF",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    spec = ModelArtifactSpec(
        safetensors_path=weights,
        safetensors_sha256=weights_hash,
        provenance_path=provenance,
        provenance_sha256=hashlib.sha256(provenance.read_bytes()).hexdigest(),
    )
    bound_model = replace(
        config.model,
        safetensors_path=spec.safetensors_path.resolve(),
        safetensors_sha256=spec.safetensors_sha256,
        provenance_path=spec.provenance_path.resolve(),
        provenance_sha256=spec.provenance_sha256,
    )
    return spec, replace(config, model=bound_model)


def test_local_safetensors_requires_exact_conversion_provenance(tmp_path: Path) -> None:
    spec, config = _artifact_fixture(tmp_path)
    artifact = validate_model_artifact(spec, config.model)
    assert artifact.source_revision == PRODUCTION_MODEL_REVISION
    assert artifact.safetensors_sha256 == spec.safetensors_sha256
    tampered = ModelArtifactSpec(
        safetensors_path=spec.safetensors_path,
        safetensors_sha256="0" * 64,
        provenance_path=spec.provenance_path,
        provenance_sha256=spec.provenance_sha256,
    )
    with pytest.raises(Exception, match="exact frozen model identity"):
        validate_model_artifact(tampered, config.model)


def _real_smoke_fixture(
    tmp_path: Path, *, config: object, spec: ModelArtifactSpec
) -> tuple[Path, Path]:
    smoke_root = tmp_path / "real-smoke-root"
    smoke_run = smoke_root / "gate-fixture"
    smoke_run.mkdir(parents=True)
    checkpoint = smoke_run / "real-manifest-smoke.pt"
    checkpoint.write_bytes(b"content-addressed-segformer-smoke-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest_hashes = {
        str(path.resolve()): digest
        for (_dataset_id, split), (path, digest, _fingerprint) in CANONICAL_MANIFEST_LOCKS.items()
        if split == "train"
    }
    manifest_fingerprints = {
        str(path.resolve()): fingerprint
        for (_dataset_id, split), (path, _digest, fingerprint) in CANONICAL_MANIFEST_LOCKS.items()
        if split == "train"
    }
    source_hash = training_source_sha256()
    report = smoke_run / "real-manifest-smoke.json"
    report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "artifact_type": "BOUNDED_REAL_MANIFEST_SMOKE",
                "provenance": "REAL_ML_OUTPUT",
                "full_training": False,
                "training_authorized": False,
                "epochs": 0,
                "optimizer_steps": 1,
                "training_transform": "PASS",
                "validation_transform": "PASS",
                "parameter_changed": True,
                "checkpoint_reload": "FRESH_PYTHON_PROCESS_PASS",
                "config_sha256": config.sha256,
                "training_source_sha256": source_hash,
                "manifest_sha256": manifest_hashes,
                "manifest_fingerprint": manifest_fingerprints,
                "taxonomy_sha256": config.taxonomy_assets.hashes,
                "model_safetensors_path": str(spec.safetensors_path.resolve()),
                "model_safetensors_sha256": spec.safetensors_sha256,
                "model_provenance_path": str(spec.provenance_path.resolve()),
                "model_provenance_sha256": spec.provenance_sha256,
                "sample_ids": ["floodnet-train-fixture", "rescuenet-train-fixture"],
                "samples_per_dataset": {"floodnet": 1, "rescuenet": 1},
                "loss": 1.0,
                "gradient_norm": 2.0,
                "validation": {"loss": 1.0, "mean_iou": 0.1},
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "fresh_process_checkpoint_proof": {
                    "status": "PASS",
                    "provenance": "REAL_ML_OUTPUT",
                    "fresh_python_process": True,
                    "creator_pid": 100,
                    "resumer_pid": 101,
                    "training_state": "PASS",
                    "model_state": "PASS",
                    "optimizer_state": "PASS",
                    "scheduler_state": "PASS",
                    "scaler_state": "PASS",
                    "python_numpy_torch_cpu_cuda_generator_rng": "PASS",
                    "child_executable_source_rehash": "PASS",
                    "training_source_sha256": source_hash,
                    "checkpoint": str(checkpoint.resolve()),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return smoke_root, report


def test_human_approval_is_bound_to_config_manifests_model_and_review(tmp_path: Path) -> None:
    spec, config = _artifact_fixture(tmp_path)
    smoke_root, smoke_report = _real_smoke_fixture(tmp_path, config=config, spec=spec)
    manifest = tmp_path / "manifest.json"
    manifest_hash = _manifest(manifest)
    manifest_fingerprint = _manifest_fingerprint(manifest)
    run_directory = tmp_path / "approved-run"
    review = tmp_path / "human-review.json"
    review.write_text(
        json.dumps(
            {
                "schema_version": "floodsight-segmentation-human-review-v1",
                "review_kind": "FLOODSIGHT_FINAL_PRETRAINING_REVIEW",
                "decision": "APPROVED",
                "reviewed_by": "Fixture Reviewer",
                "reviewed_at": "2026-08-30T23:59:00Z",
                "acknowledgements": {
                    key: True for key in sorted(REQUIRED_REVIEW_ACKNOWLEDGEMENTS)
                },
                "open_blockers": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": "floodsight-training-approval-v3",
                "approval_kind": "HUMAN",
                "approval_id": "approval-fixture",
                "decision": "APPROVED",
                "approved_by": "Fixture Reviewer",
                "approved_at": "2026-08-31T00:00:00Z",
                "operations": ["VALIDATE"],
                "training_code_sha256": training_source_sha256(),
                "config_sha256": config.sha256,
                "manifest_sha256": {str(manifest.resolve()): manifest_hash},
                "manifest_fingerprint": {
                    str(manifest.resolve()): manifest_fingerprint,
                },
                "taxonomy_sha256": config.taxonomy_assets.hashes,
                "run_directory": str(run_directory.resolve()),
                "model": {
                    "model_id": PRODUCTION_MODEL_ID,
                    "revision": PRODUCTION_MODEL_REVISION,
                    "safetensors_path": str(spec.safetensors_path.resolve()),
                    "safetensors_sha256": spec.safetensors_sha256,
                    "provenance_path": str(spec.provenance_path.resolve()),
                    "provenance_sha256": spec.provenance_sha256,
                },
                "real_smoke": {
                    "report_path": str(smoke_report.resolve()),
                    "report_sha256": hashlib.sha256(smoke_report.read_bytes()).hexdigest(),
                },
                "human_review": {
                    "report_path": str(review.resolve()),
                    "report_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    approval_hash = hashlib.sha256(approval.read_bytes()).hexdigest()
    accepted = validate_human_approval(
        approval,
        expected_record_sha256=approval_hash,
        operation="VALIDATE",
        config_sha256=config.sha256,
        manifest_specs=[(manifest, manifest_hash, manifest_fingerprint)],
        taxonomy_sha256=config.taxonomy_assets.hashes,
        run_directory=run_directory,
        model_id=PRODUCTION_MODEL_ID,
        model_revision=PRODUCTION_MODEL_REVISION,
        model_artifact=spec,
        real_smoke_root=smoke_root,
    )
    assert accepted.approval_id == "approval-fixture"
    assert accepted.real_smoke_path == smoke_report.resolve()
    approval_payload = json.loads(approval.read_text(encoding="utf-8"))
    instruction = tmp_path / "override-instruction.txt"
    instruction.write_text("Authorize exact production runs.\n", encoding="utf-8")
    override = tmp_path / "launch-override.json"
    override.write_text(
        json.dumps(
            {
                "schema_version": "floodsight-user-training-launch-override-v1",
                "instruction_accepted_utc": "2026-08-31T20:06:25Z",
                "target_training_launch_utc": "2026-08-31T22:06:25Z",
                "instruction_source_path": str(instruction.resolve()),
                "instruction_source_sha256": hashlib.sha256(
                    instruction.read_bytes()
                ).hexdigest(),
                "authorized_by": "Fixture Reviewer",
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
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    approval_payload["approved_at"] = "2026-08-31T20:07:00Z"
    approval_payload["human_review"] = {
        "report_path": str(override.resolve()),
        "report_sha256": hashlib.sha256(override.read_bytes()).hexdigest(),
    }
    approval.write_text(json.dumps(approval_payload, sort_keys=True), encoding="utf-8")
    deferred = validate_human_approval(
        approval,
        expected_record_sha256=hashlib.sha256(approval.read_bytes()).hexdigest(),
        operation="VALIDATE",
        config_sha256=config.sha256,
        manifest_specs=[(manifest, manifest_hash, manifest_fingerprint)],
        taxonomy_sha256=config.taxonomy_assets.hashes,
        run_directory=run_directory,
        model_id=PRODUCTION_MODEL_ID,
        model_revision=PRODUCTION_MODEL_REVISION,
        model_artifact=spec,
        real_smoke_root=smoke_root,
    )
    assert deferred.human_review_path == override.resolve()
    approval_payload["approved_at"] = "2026-08-31T00:00:00Z"
    approval_payload["human_review"] = {
        "report_path": str(review.resolve()),
        "report_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    }
    approval.write_text(json.dumps(approval_payload, sort_keys=True), encoding="utf-8")
    smoke_payload = json.loads(smoke_report.read_text(encoding="utf-8"))
    smoke_payload["optimizer_steps"] = 2
    smoke_report.write_text(json.dumps(smoke_payload, sort_keys=True), encoding="utf-8")
    approval_payload["real_smoke"]["report_sha256"] = hashlib.sha256(
        smoke_report.read_bytes()
    ).hexdigest()
    approval.write_text(json.dumps(approval_payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(TrainingAuthorizationError, match="exact PASS envelope"):
        validate_human_approval(
            approval,
            expected_record_sha256=hashlib.sha256(approval.read_bytes()).hexdigest(),
            operation="VALIDATE",
            config_sha256=config.sha256,
            manifest_specs=[(manifest, manifest_hash, manifest_fingerprint)],
            taxonomy_sha256=config.taxonomy_assets.hashes,
            run_directory=run_directory,
            model_id=PRODUCTION_MODEL_ID,
            model_revision=PRODUCTION_MODEL_REVISION,
            model_artifact=spec,
            real_smoke_root=smoke_root,
        )
    smoke_payload["optimizer_steps"] = 1
    smoke_report.write_text(json.dumps(smoke_payload, sort_keys=True), encoding="utf-8")
    approval_payload["real_smoke"]["report_sha256"] = hashlib.sha256(
        smoke_report.read_bytes()
    ).hexdigest()
    approval.write_text(json.dumps(approval_payload, sort_keys=True), encoding="utf-8")
    review_payload = json.loads(review.read_text(encoding="utf-8"))
    review_payload["acknowledgements"]["pool_separate_from_water"] = False
    review.write_text(json.dumps(review_payload), encoding="utf-8")
    approval_payload = json.loads(approval.read_text(encoding="utf-8"))
    approval_payload["human_review"]["report_sha256"] = hashlib.sha256(
        review.read_bytes()
    ).hexdigest()
    approval.write_text(json.dumps(approval_payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(TrainingAuthorizationError, match="acknowledgement"):
        validate_human_approval(
            approval,
            expected_record_sha256=hashlib.sha256(approval.read_bytes()).hexdigest(),
            operation="VALIDATE",
            config_sha256=config.sha256,
            manifest_specs=[(manifest, manifest_hash, manifest_fingerprint)],
            taxonomy_sha256=config.taxonomy_assets.hashes,
            run_directory=run_directory,
            model_id=PRODUCTION_MODEL_ID,
            model_revision=PRODUCTION_MODEL_REVISION,
            model_artifact=spec,
            real_smoke_root=smoke_root,
        )
    review_payload["acknowledgements"]["pool_separate_from_water"] = True
    review.write_text(json.dumps(review_payload), encoding="utf-8")
    approval_payload["human_review"]["report_sha256"] = hashlib.sha256(
        review.read_bytes()
    ).hexdigest()
    approval.write_text(json.dumps(approval_payload, sort_keys=True), encoding="utf-8")
    approval_hash = hashlib.sha256(approval.read_bytes()).hexdigest()
    with pytest.raises(TrainingAuthorizationError, match="does not authorize"):
        validate_human_approval(
            approval,
            expected_record_sha256=approval_hash,
            operation="TRAIN",
            config_sha256=config.sha256,
            manifest_specs=[(manifest, manifest_hash, manifest_fingerprint)],
            taxonomy_sha256=config.taxonomy_assets.hashes,
            run_directory=run_directory,
            model_id=PRODUCTION_MODEL_ID,
            model_revision=PRODUCTION_MODEL_REVISION,
            model_artifact=spec,
            real_smoke_root=smoke_root,
        )


def test_explicit_user_launch_override_truthfully_defers_human_review(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("Authorize exact production runs.\n", encoding="utf-8")
    override = tmp_path / "launch-override.json"
    payload = {
        "schema_version": "floodsight-user-training-launch-override-v1",
        "instruction_accepted_utc": "2026-08-31T20:06:25Z",
        "target_training_launch_utc": "2026-08-31T22:06:25Z",
        "instruction_source_path": str(instruction.resolve()),
        "instruction_source_sha256": hashlib.sha256(instruction.read_bytes()).hexdigest(),
        "authorized_by": "Fixture User",
        "decision": (
            "AUTHORIZE_PRODUCTION_TRAINING_WHEN_EXISTING_TECHNICAL_REQUIREMENTS_"
            "PASS_AND_A_SAFE_H100_IS_AVAILABLE"
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
    override.write_text(json.dumps(payload), encoding="utf-8")

    _validate_human_review(
        override,
        approved_by="Fixture User",
        approval_time=datetime(2026, 8, 31, 20, 7, tzinfo=UTC),
    )

    payload["human_review_completed"] = 0
    override.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingAuthorizationError, match="deferred review"):
        _validate_human_review(
            override,
            approved_by="Fixture User",
            approval_time=datetime(2026, 8, 31, 20, 7, tzinfo=UTC),
        )


def test_real_cli_refuses_before_reading_paths_or_importing_ml(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "train",
            "--data-root",
            "/does/not/exist",
            "--train-manifest",
            "/does/not/exist/train.json",
            "--train-manifest-sha256",
            HASH_A,
            "--train-manifest-fingerprint",
            HASH_A,
            "--validation-manifest",
            "/does/not/exist/val.json",
            "--validation-manifest-sha256",
            HASH_B,
            "--validation-manifest-fingerprint",
            HASH_B,
            "--output-dir",
            "/does/not/exist/output",
            "--model-safetensors",
            "/does/not/exist/model.safetensors",
            "--model-safetensors-sha256",
            HASH_A,
            "--model-provenance-record",
            "/does/not/exist/provenance.json",
            "--model-provenance-record-sha256",
            HASH_B,
            "--approval-record",
            "/does/not/exist/approval.json",
            "--approval-record-sha256",
            HASH_A,
        ]
    )
    assert code == 2
    assert "--allow-training" in capsys.readouterr().err


def test_lone_training_boolean_still_requires_human_approval(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "train",
            "--data-root",
            "/does/not/exist",
            "--train-manifest",
            "/missing/train.json",
            "--train-manifest-sha256",
            HASH_A,
            "--train-manifest-fingerprint",
            HASH_A,
            "--validation-manifest",
            "/missing/val.json",
            "--validation-manifest-sha256",
            HASH_B,
            "--validation-manifest-fingerprint",
            HASH_B,
            "--output-dir",
            "/missing/output",
            "--model-safetensors",
            "/missing/model.safetensors",
            "--model-safetensors-sha256",
            HASH_A,
            "--model-provenance-record",
            "/missing/provenance.json",
            "--model-provenance-record-sha256",
            HASH_B,
            "--approval-record",
            "/missing/approval.json",
            "--approval-record-sha256",
            HASH_A,
            "--allow-training",
        ]
    )
    assert code == 2
    assert "approval record is missing" in capsys.readouterr().err


def test_real_smoke_has_separate_explicit_guard(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "real-smoke",
            "--data-root",
            "/missing",
            "--manifest",
            "/missing/manifest.json",
            "--manifest-sha256",
            HASH_A,
            "--manifest-fingerprint",
            HASH_A,
            "--output-dir",
            "/missing/output",
            "--model-safetensors",
            "/missing/model.safetensors",
            "--model-safetensors-sha256",
            HASH_A,
            "--model-provenance-record",
            "/missing/provenance.json",
            "--model-provenance-record-sha256",
            HASH_B,
        ]
    )
    assert code == 2
    assert "--allow-real-smoke" in capsys.readouterr().err
