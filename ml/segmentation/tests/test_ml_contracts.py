# ruff: noqa: E402, I001

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from PIL import Image

from floodsight_segmentation.checkpoint import TrainingState, load_checkpoint, save_checkpoint
from floodsight_segmentation.checkpoint_probe import (
    run_fresh_process_checkpoint_probe,
    state_fingerprint,
)
from floodsight_segmentation.config import AUDITED_SOURCE_TO_TARGET_IDS
from floodsight_segmentation.dataset import SegmentationManifestDataset
from floodsight_segmentation.engine import (
    _load_resume_history,
    acquire_process_run_lock,
    prepare_run_directory,
    require_approved_last_resume,
    require_disjoint_training_validation,
    require_training_class_coverage,
    run_training,
    select_bounded_smoke_collection,
    validate_bounded_smoke_coverage,
)
from floodsight_segmentation.errors import ManifestError
from floodsight_segmentation.manifest import (
    FrozenManifest,
    ManifestCollection,
    ManifestSample,
    require_canonical_manifest_locks,
)
from floodsight_segmentation.metrics import SegmentationMetrics
from floodsight_segmentation.optim import build_optimizer
from floodsight_segmentation import model as model_module
from floodsight_segmentation.artifact import ModelArtifact
from floodsight_segmentation.cli import DEFAULT_CONFIG
from floodsight_segmentation.config import load_config
from floodsight_segmentation.reproducibility import (
    capture_rng_state,
    make_generator,
    restore_rng_state,
    seed_everything,
)
from floodsight_segmentation.smoke import run_synthetic_smoke
from floodsight_segmentation.supervision import PartialCrossEntropyLoss
from floodsight_segmentation.transforms import PairedSegmentationTransform


def test_partial_cross_entropy_masks_unsupported_logits_and_rejects_labels() -> None:
    logits = torch.tensor([[[[0.0]], [[0.0]], [[100.0]], [[-100.0]]]], requires_grad=True)
    labels = torch.tensor([[[1]]])
    availability = torch.tensor([[True, True, False, False]])
    loss = PartialCrossEntropyLoss(class_weights=(1.0,) * 4, ignore_index=255)(
        logits, labels, availability
    )
    assert float(loss.detach()) == pytest.approx(float(np.log(2.0)), rel=1e-6)
    loss.backward()
    assert torch.equal(logits.grad[:, 2:], torch.zeros_like(logits.grad[:, 2:]))
    with pytest.raises(ValueError, match="unsupported"):
        PartialCrossEntropyLoss(class_weights=(1.0,) * 4, ignore_index=255)(
            logits.detach(), torch.tensor([[[2]]]), availability
        )


def test_metrics_mask_predictions_but_preserve_unified_space() -> None:
    logits = torch.zeros((1, 4, 2, 2))
    logits[:, 2] = 100  # Unsupported and therefore ineligible for prediction.
    logits[:, 1] = 2
    labels = torch.ones((1, 2, 2), dtype=torch.long)
    metrics = SegmentationMetrics(num_labels=4, ignore_index=255)
    metrics.update(logits, labels, torch.tensor([[True, True, False, False]]))
    result = metrics.compute()
    assert result["confusion_matrix"][1][1] == 4
    assert result["mean_iou"] == pytest.approx(1.0)
    assert len(result["per_class_iou"]) == 4


def test_paired_transform_uses_nearest_mask_interpolation() -> None:
    image_array = np.zeros((7, 9, 3), dtype=np.uint8)
    image_array[:, 4:] = 255
    mask_array = np.zeros((7, 9), dtype=np.uint8)
    mask_array[:, 4:] = 11
    transform = PairedSegmentationTransform(
        height=31,
        width=29,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        training=False,
    )
    pixels, labels = transform(Image.fromarray(image_array), Image.fromarray(mask_array))
    assert pixels.shape == (3, 31, 29)
    assert labels.shape == (31, 29)
    assert set(labels.unique().tolist()) == {0, 11}
    assert float(pixels[:, :, 2].mean()) < float(pixels[:, :, -3].mean())


def test_random_crop_and_flip_keep_index_mask_discrete_and_aligned() -> None:
    torch.manual_seed(44)
    image_array = np.zeros((32, 48, 3), dtype=np.uint8)
    image_array[:, 24:] = 255
    mask_array = np.zeros((32, 48), dtype=np.uint8)
    mask_array[:, 24:] = 3
    transform = PairedSegmentationTransform(
        height=24,
        width=24,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        training=True,
        scale=(0.5, 1.0),
        ratio=(0.75, 1.25),
        horizontal_flip_probability=1.0,
    )
    pixels, labels = transform(Image.fromarray(image_array), Image.fromarray(mask_array))
    assert pixels.shape == (3, 24, 24)
    assert labels.shape == (24, 24)
    assert set(labels.unique().tolist()) <= {0, 3}
    bright = pixels.mean(dim=0) > 0.5
    assert float((bright == (labels == 3)).float().mean()) > 0.9


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_collection(root: Path, *, mask_value: int) -> ManifestCollection:
    image_path = root / "processed/images/example.png"
    source_mask_path = root / "raw/masks/example.png"
    mask_path = root / "processed/masks/example.png"
    image_path.parent.mkdir(parents=True)
    source_mask_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(np.full((8, 10, 3), 127, dtype=np.uint8)).save(image_path)
    Image.fromarray(np.full((8, 10), mask_value, dtype=np.uint8)).save(source_mask_path)
    Image.fromarray(np.full((8, 10), mask_value, dtype=np.uint8)).save(mask_path)
    sample = ManifestSample(
        sample_id="example",
        source_dataset="floodnet",
        source_split="train",
        target_split="train",
        image_path="processed/images/example.png",
        source_annotation_path="raw/masks/example.png",
        target_annotation_path="processed/masks/example.png",
        width=10,
        height=8,
        image_hash=_hash(image_path),
        annotation_hash=_hash(source_mask_path),
        target_image_hash=_hash(image_path),
        target_annotation_hash=_hash(mask_path),
        class_counts={mask_value: 80},
        ignored_count=0,
        invalid_count=0,
        preparation_version="segmentation_v2_20260831T131322Z_v1",
        taxonomy_version="segmentation-taxonomy-v2",
        source_schema="floodnet-supervised-v1.0-indexed-mask-ids-0-9",
        target_mapping_version="floodnet-mapping-v2",
        target_mapping_sha256=(
            "fdfbbba84c1cf8ea0176429b8d236693030abc16452f507c94922cc2f0769760"
        ),
        valid_supervision_classes=(0, 1, 2, 3, 6, 7, 12, 13, 14, 15),
        ignore_index=255,
        ignore_semantics=(
            "255_reserved_for_genuine_invalid_or_unlabelled_pixels;"
            "audited_remaps_emit_no_ignore_pixels;"
            "unsupported_classes_are_removed_from_dataset_aware_softmax_not_relabelled_as_background"
        ),
        exclusion_status="INCLUDED",
        exclusion_reason="",
    )
    manifest = FrozenManifest(
        path=root / "manifest.json",
        sha256="a" * 64,
        manifest_id="fixture",
        dataset_id="floodnet",
        taxonomy_version="segmentation-taxonomy-v2",
        integrity_mode="full",
        fingerprint="b" * 64,
        samples=(sample,),
    )
    return ManifestCollection(manifests=(manifest,), samples=(sample,))


def _dataset(root: Path, *, mask_value: int) -> SegmentationManifestDataset:
    return SegmentationManifestDataset(
        _synthetic_collection(root, mask_value=mask_value),
        data_root=root,
        supported_class_ids={"floodnet": frozenset({0, 1, 2, 3, 6, 7, 12, 13, 14, 15})},
        source_to_target_ids=AUDITED_SOURCE_TO_TARGET_IDS,
        num_labels=16,
        ignore_index=255,
        transform=PairedSegmentationTransform(
            height=8,
            width=10,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            training=False,
        ),
        verify_sample_hashes=True,
    )


def test_dataset_returns_partial_supervision_and_rejects_unsupported_id(tmp_path: Path) -> None:
    item = _dataset(tmp_path / "valid", mask_value=3)[0]
    assert item["pixel_values"].shape == (3, 8, 10)
    assert item["class_availability"].sum().item() == 10
    with pytest.raises(Exception, match="frozen mapping"):
        _dataset(tmp_path / "invalid", mask_value=4)[0]


def test_checkpoint_restores_model_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    seed_everything(9, deterministic_algorithms=True, cudnn_benchmark=False)
    generator = make_generator(10)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        training_state=TrainingState(epoch=2, global_step=7, best_metric=0.4),
        config_sha256="a" * 64,
        manifest_sha256={"manifest": "b" * 64},
        manifest_fingerprint={"manifest": "c" * 64},
        taxonomy_sha256={"taxonomy": "d" * 64},
        input_provenance={"mode": "DEMO_SIMULATED"},
        run_directory=tmp_path,
        data_generator=generator,
        provenance="DEMO_SIMULATED",
    )
    expected = torch.rand(3)
    with torch.no_grad():
        model.weight.zero_()
    _ = torch.rand(10)
    state = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_config_sha256="a" * 64,
        expected_manifest_sha256={"manifest": "b" * 64},
        expected_manifest_fingerprint={"manifest": "c" * 64},
        expected_taxonomy_sha256={"taxonomy": "d" * 64},
        expected_input_provenance={"mode": "DEMO_SIMULATED"},
        expected_run_directory=tmp_path,
        data_generator=generator,
        map_location="cpu",
        expected_provenance="DEMO_SIMULATED",
    )
    assert state == TrainingState(epoch=2, global_step=7, best_metric=0.4)
    assert torch.equal(torch.rand(3), expected)
    assert not torch.equal(model.weight, torch.zeros_like(model.weight))


def test_checkpoint_rng_continuation_across_fresh_python_processes(tmp_path: Path) -> None:
    report = run_fresh_process_checkpoint_probe(tmp_path / "fresh-process")
    assert report["status"] == "PASS"
    assert report["fresh_python_processes"] is True
    assert report["creator_pid"] != report["resumer_pid"]
    assert report["model_optimizer_scheduler_continuation"] == "PASS"


def test_checkpoint_validation_accepts_exact_manifest_subset_and_separate_approval(
    tmp_path: Path,
) -> None:
    seed_everything(31, deterministic_algorithms=True, cudnn_benchmark=False)
    generator = make_generator(32)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        training_state=TrainingState(epoch=1, global_step=1, best_metric=0.4),
        config_sha256="a" * 64,
        manifest_sha256={"train": "b" * 64, "val": "c" * 64},
        manifest_fingerprint={"train": "d" * 64, "val": "e" * 64},
        taxonomy_sha256={"taxonomy": "f" * 64},
        input_provenance={"model": "immutable"},
        authorization_provenance={"approval": "training-approval"},
        run_directory=tmp_path,
        data_generator=generator,
        provenance="REAL_ML_OUTPUT",
    )
    state = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        expected_config_sha256="a" * 64,
        expected_manifest_sha256={"val": "c" * 64},
        expected_manifest_fingerprint={"val": "e" * 64},
        expected_taxonomy_sha256={"taxonomy": "f" * 64},
        expected_input_provenance={"model": "immutable"},
        expected_authorization_provenance=None,
        expected_run_directory=tmp_path,
        data_generator=generator,
        map_location="cpu",
        expected_provenance="REAL_ML_OUTPUT",
        allow_manifest_subset=True,
    )
    assert state.epoch == 1


def test_checkpoint_state_fingerprint_is_content_and_order_stable() -> None:
    first = {"tensor": torch.tensor([1.0, 2.0]), "nested": {"value": 3}}
    second = {"nested": {"value": 3}, "tensor": torch.tensor([1.0, 2.0])}
    assert state_fingerprint(first) == state_fingerprint(second)
    second["tensor"][0] = 9
    assert state_fingerprint(first) != state_fingerprint(second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rng_restore_normalizes_cuda_mapped_checkpoint_tensors() -> None:
    """Direct-to-CUDA checkpoint loads must still restore CPU RNG APIs."""

    seed_everything(19, deterministic_algorithms=True, cudnn_benchmark=False)
    generator = make_generator(20)
    state = capture_rng_state(data_generator=generator)
    expected = torch.rand(3)
    _ = torch.rand(11)
    mapped = {
        **state,
        "torch_cpu": state["torch_cpu"].cuda(),
        "torch_cuda": [item.cuda() for item in state["torch_cuda"]],
        "data_generator": state["data_generator"].cuda(),
    }
    restore_rng_state(mapped, data_generator=generator)
    assert torch.equal(torch.rand(3), expected)


def test_full_tiny_segformer_smoke_is_offline_and_synthetic(tmp_path: Path) -> None:
    report = run_synthetic_smoke(tmp_path / "smoke")
    assert report["status"] == "PASS"
    assert report["provenance"] == "DEMO_SIMULATED"
    assert report["real_dataset_access"] is False
    assert report["real_training"] is False
    assert report["checkpoint_reload"] == "PASS"
    assert Path(report["report_path"]).is_file()


def test_real_smoke_selection_requires_pool_and_source_specific_coverage(tmp_path: Path) -> None:
    collection = _synthetic_collection(tmp_path, mask_value=3)
    floodnet_specific = collection.samples[0]
    floodnet_pool = replace(
        floodnet_specific,
        sample_id="floodnet-pool",
        class_counts={15: 80},
    )
    rescuenet_pool = replace(
        floodnet_specific,
        sample_id="rescuenet-pool",
        source_dataset="rescuenet",
        class_counts={15: 80},
    )
    rescuenet_specific = replace(
        rescuenet_pool,
        sample_id="rescuenet-specific",
        class_counts={5: 80},
    )
    expanded = ManifestCollection(
        manifests=collection.manifests,
        samples=(floodnet_pool, floodnet_specific, rescuenet_pool, rescuenet_specific),
    )
    bounded = select_bounded_smoke_collection(
        expanded,
        dataset_ids=("floodnet", "rescuenet"),
        supported_class_ids={
            "floodnet": frozenset({0, 1, 2, 3, 6, 7, 12, 13, 14, 15}),
            "rescuenet": frozenset({0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 15}),
        },
    )
    assert [sample.sample_id for sample in bounded.samples] == [
        "floodnet-pool",
        "example",
        "rescuenet-pool",
        "rescuenet-specific",
    ]


def test_real_smoke_selection_prefers_one_joint_coverage_sample_per_dataset(
    tmp_path: Path,
) -> None:
    collection = _synthetic_collection(tmp_path, mask_value=3)
    flood_joint = replace(
        collection.samples[0],
        sample_id="flood-joint",
        class_counts={3: 4, 15: 4},
    )
    rescue_joint = replace(
        collection.samples[0],
        sample_id="rescue-joint",
        source_dataset="rescuenet",
        class_counts={5: 4, 15: 4},
    )
    bounded = select_bounded_smoke_collection(
        ManifestCollection(
            manifests=collection.manifests,
            samples=(flood_joint, rescue_joint),
        ),
        dataset_ids=("floodnet", "rescuenet"),
        supported_class_ids={
            "floodnet": frozenset({0, 1, 2, 3, 6, 7, 12, 13, 14, 15}),
            "rescuenet": frozenset({0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 15}),
        },
    )
    assert [sample.sample_id for sample in bounded.samples] == [
        "flood-joint",
        "rescue-joint",
    ]


def test_real_smoke_decoded_coverage_rejects_missing_pool() -> None:
    batch = {
        "labels": torch.tensor([[[3]], [[5]]]),
        "source_dataset": ["floodnet", "rescuenet"],
    }
    with pytest.raises(RuntimeError, match="did not decode Pool"):
        validate_bounded_smoke_coverage(
            batch,
            dataset_ids=("floodnet", "rescuenet"),
            supported_class_ids={
                "floodnet": frozenset({0, 1, 2, 3, 6, 7, 12, 13, 14, 15}),
                "rescuenet": frozenset({0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 15}),
            },
            ignore_index=255,
        )


def test_training_validation_rejects_exact_image_hash_leakage(tmp_path: Path) -> None:
    training = _synthetic_collection(tmp_path / "training", mask_value=3)
    leaked = replace(
        training.samples[0],
        sample_id="different-id-same-image",
        source_split="val",
        target_split="val",
    )
    validation = ManifestCollection(
        manifests=training.manifests,
        samples=(leaked,),
    )
    with pytest.raises(RuntimeError, match="image SHA-256 values overlap"):
        require_disjoint_training_validation(training, validation)


def test_training_validation_accepts_distinct_image_hashes(tmp_path: Path) -> None:
    training = _synthetic_collection(tmp_path / "training", mask_value=3)
    distinct = replace(
        training.samples[0],
        sample_id="validation-example",
        source_split="val",
        target_split="val",
        image_hash="c" * 64,
        target_image_hash="c" * 64,
    )
    validation = ManifestCollection(
        manifests=training.manifests,
        samples=(distinct,),
    )
    require_disjoint_training_validation(training, validation)


def test_training_requires_all_unified_classes_in_frozen_manifests(tmp_path: Path) -> None:
    collection = _synthetic_collection(tmp_path, mask_value=3)
    with pytest.raises(RuntimeError, match="Unified classes absent"):
        require_training_class_coverage(collection, num_labels=16)


def test_production_operations_reject_noncanonical_manifest_identity(tmp_path: Path) -> None:
    collection = _synthetic_collection(tmp_path, mask_value=3)
    with pytest.raises(ManifestError, match="canonical Stage-9"):
        require_canonical_manifest_locks(collection)


def test_run_directory_reservation_refuses_even_an_empty_collision(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    run = root / "run"
    run.mkdir()
    with pytest.raises(RuntimeError, match="existing run directory"):
        prepare_run_directory(run, resuming=False, required_root=root)
    assert prepare_run_directory(run, resuming=True, required_root=root) == run.resolve()
    with pytest.raises(RuntimeError, match="direct child"):
        prepare_run_directory(run / "nested", resuming=False, required_root=root)


def test_process_lifetime_run_lock_rejects_a_second_python_process(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    lock_path = acquire_process_run_lock(run)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from floodsight_segmentation.engine import acquire_process_run_lock; "
                "acquire_process_run_lock(Path(__import__('sys').argv[1]))"
            ),
            str(run),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert lock_path.is_file()
    assert completed.returncode != 0
    assert "already owns" in completed.stderr


def test_resume_requires_the_approved_run_last_checkpoint(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    run = tmp_path / "approved-run"
    run.mkdir()
    last = run / "last.pt"
    last.touch()
    best = run / "best.pt"
    best.touch()
    assert require_approved_last_resume(run, last, config) == last.resolve()
    with pytest.raises(RuntimeError, match="approved run checkpoint"):
        require_approved_last_resume(run, best, config)


def test_resume_rejects_a_symlink_named_last_checkpoint(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    run = tmp_path / "approved-run"
    run.mkdir()
    target = run / "checkpoint-target.pt"
    target.touch()
    symlink = run / "last.pt"
    symlink.symlink_to(target)
    with pytest.raises(RuntimeError, match="approved run checkpoint"):
        require_approved_last_resume(run, symlink, config)


def test_resume_history_recovers_a_record_written_ahead_of_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    manifests = {"manifest": "b" * 64}
    fingerprints = {"manifest": "c" * 64}
    taxonomy = {"taxonomy": "d" * 64}
    provenance = {"mode": "REAL_ML_OUTPUT"}
    authorization = {"approval": "e" * 64}
    path.write_text(
        json.dumps(
            {
                "schema_version": "training-history-v1",
                "config_sha256": "a" * 64,
                "manifest_sha256": manifests,
                "manifest_fingerprint": fingerprints,
                "manifest_set_fingerprint": hashlib.sha256(
                    json.dumps(
                        [
                            {
                                "path": "manifest",
                                "manifest_sha256": "b" * 64,
                                "dataset_fingerprint": "c" * 64,
                            }
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "taxonomy_sha256": taxonomy,
                "input_provenance": provenance,
                "authorization_provenance": authorization,
                "epochs": [
                    {
                        "epoch": 1,
                        "global_step": 1,
                        "validation": {"mean_iou": 0.2},
                    },
                    {
                        "epoch": 2,
                        "global_step": 2,
                        "validation": {"mean_iou": 0.3},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    retained = _load_resume_history(
        path,
        state=TrainingState(epoch=1, global_step=1, best_metric=0.2),
        config_sha256="a" * 64,
        manifest_sha256=manifests,
        manifest_fingerprint=fingerprints,
        taxonomy_sha256=taxonomy,
        input_provenance=provenance,
        authorization_provenance=authorization,
        updates_per_epoch=1,
        configured_epochs=2,
        scheduler_last_epoch=1,
        monitor_metric="mean_iou",
        maximize_metric=True,
    )

    assert retained == [
        {"epoch": 1, "global_step": 1, "validation": {"mean_iou": 0.2}}
    ]
    completed = _load_resume_history(
        path,
        state=TrainingState(epoch=2, global_step=2, best_metric=0.3),
        config_sha256="a" * 64,
        manifest_sha256=manifests,
        manifest_fingerprint=fingerprints,
        taxonomy_sha256=taxonomy,
        input_provenance=provenance,
        authorization_provenance=authorization,
        updates_per_epoch=1,
        configured_epochs=2,
        scheduler_last_epoch=2,
        monitor_metric="mean_iou",
        maximize_metric=True,
    )
    assert len(completed) == 2
    with pytest.raises(RuntimeError, match="best metric"):
        _load_resume_history(
            path,
            state=TrainingState(epoch=2, global_step=2, best_metric=0.2),
            config_sha256="a" * 64,
            manifest_sha256=manifests,
            manifest_fingerprint=fingerprints,
            taxonomy_sha256=taxonomy,
            input_provenance=provenance,
            authorization_provenance=authorization,
            updates_per_epoch=1,
            configured_epochs=2,
            scheduler_last_epoch=2,
            monitor_metric="mean_iou",
            maximize_metric=True,
        )


def test_epoch_checkpoint_commit_writes_best_before_authoritative_last() -> None:
    source = inspect.getsource(run_training)
    epoch_commit = source[source.index("record = {") : source.index("report = {")]
    assert epoch_commit.index("best_checkpoint_filename") < epoch_commit.index(
        "last_checkpoint_filename"
    )


def test_production_builder_forces_local_safetensors_and_no_remote_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    class FakeConfig:
        def __init__(self, **kwargs: object) -> None:
            calls["config"] = kwargs

    class FakeModelType:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return object(), {
                "missing_keys": ["decode_head.classifier.weight"],
                "unexpected_keys": [],
                "mismatched_keys": ["decode_head.classifier.bias"],
                "error_msgs": [],
            }

    monkeypatch.setattr(model_module, "_transformers", lambda: (FakeConfig, FakeModelType))
    config = load_config(DEFAULT_CONFIG)
    artifact = ModelArtifact(
        safetensors_path=tmp_path / "model.safetensors",
        safetensors_sha256="a" * 64,
        provenance_path=tmp_path / "provenance.json",
        provenance_sha256="b" * 64,
        source_revision=config.model.revision,
        source_sha256=config.model.upstream_pytorch_model_sha256,
        human_review_status="PENDING_HUMAN_SIGNOFF",
    )
    model_module.build_segformer(config.model, artifact)
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["local_files_only"] is True
    assert kwargs["trust_remote_code"] is False
    assert kwargs["use_safetensors"] is True
    assert calls["path"] == str(tmp_path)
    explicit = calls["config"]
    assert isinstance(explicit, dict)
    assert explicit["hidden_dropout_prob"] == 0.0
    assert explicit["attention_probs_dropout_prob"] == 0.0
    assert explicit["classifier_dropout_prob"] == 0.1
    assert explicit["drop_path_rate"] == 0.1
    assert explicit["layer_norm_eps"] == 1e-6
    assert explicit["reshape_last_stage"] is True


def test_adamw_execution_defaults_are_explicit() -> None:
    config = load_config(DEFAULT_CONFIG)
    optimizer = build_optimizer(torch.nn.Linear(3, 2), config.optimizer)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["amsgrad"] is False
    assert optimizer.defaults["maximize"] is False
    assert optimizer.defaults["foreach"] is False
    assert optimizer.defaults["capturable"] is False
    assert optimizer.defaults["differentiable"] is False
    assert optimizer.defaults["fused"] is False
