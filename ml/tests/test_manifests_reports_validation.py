from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from floodsight_data.cli import main
from floodsight_data.common.atomic import atomic_write_text
from floodsight_data.common.discovery import SourcePair
from floodsight_data.common.materialize import MaterializationStrategy
from floodsight_data.common.segmentation_converter import convert_segmentation_dataset
from floodsight_data.config import PROJECT_ROOT
from floodsight_data.hashing import (
    IntegrityMode,
    file_integrity,
    sha256_file,
    stable_sample_id,
)
from floodsight_data.manifests import build_manifest, read_json, validate_schema
from floodsight_data.paths import DataPaths
from floodsight_data.reports import derive_readiness, read_review, record_review
from floodsight_data.validation import audit_duplicates, guard_repository_artifacts
from floodsight_data.visualization import generate_inspection


def _sample(sample_id: str, image_path: str) -> dict[str, object]:
    digest = "a" * 64
    return {
        "sample_id": sample_id,
        "source_dataset": "floodnet",
        "source_split": "train",
        "target_split": "train",
        "image_path": image_path,
        "source_annotation_path": "raw/floodnet/train/masks/a.png",
        "target_annotation_path": "processed/segmentation_v1/floodnet/train/masks/a.png",
        "width": 4,
        "height": 4,
        "image_hash": digest,
        "annotation_hash": digest,
        "class_counts": {"0": 16},
        "ignored_count": 0,
        "invalid_count": 0,
        "preparation_version": "segmentation_v1",
        "taxonomy_version": "segmentation-taxonomy-v1",
        "objects": [],
    }


def test_shared_dataset_examples_validate_against_schemas() -> None:
    for stem in ("dataset-manifest", "dataset-report", "dataset-lock"):
        payload = read_json(PROJECT_ROOT / "shared" / "examples" / f"{stem}.sample.json")
        validate_schema(payload, f"{stem}.schema.json")


def test_manifest_order_sample_ids_and_fingerprint_are_stable() -> None:
    samples = [
        _sample("sample-b", "raw/floodnet/train/images/b.jpg"),
        _sample("sample-a", "raw/floodnet/train/images/a.jpg"),
    ]
    options = {
        "dataset_id": "floodnet",
        "task_type": "SEMANTIC_SEGMENTATION",
        "source_version": "fixture",
        "preparation_version": "segmentation_v1",
        "taxonomy_version": "segmentation-taxonomy-v1",
        "integrity_mode": "fast",
        "source_records": [{"path": "a", "size": 1, "mtime_ns": 2}],
        "mapping_hashes": {"mapping": "b" * 64},
        "preparation": {"splits_preserved": True},
        "created_at": "2026-01-01T00:00:00Z",
    }

    first = build_manifest(samples=samples, **options)
    second = build_manifest(samples=reversed(samples), **options)

    assert [sample["sample_id"] for sample in first["samples"]] == ["sample-a", "sample-b"]
    assert first == second
    assert stable_sample_id("floodnet", "train", "images/a.jpg") == stable_sample_id(
        "floodnet", "train", "images/a.jpg"
    )


def test_fast_and_full_integrity_modes(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"floodsight")

    fast = file_integrity(path, relative_path="sample.bin", mode=IntegrityMode.FAST)
    fast_annotation = file_integrity(
        path, relative_path="sample.bin", mode=IntegrityMode.FAST, annotation=True
    )
    full = file_integrity(path, relative_path="sample.bin", mode=IntegrityMode.FULL)

    assert "sha256" not in fast
    assert fast_annotation["sha256"] == sha256_file(path)
    assert full["sha256"] == sha256_file(path)


def test_duplicate_and_cross_split_leakage_detection(tmp_path: Path) -> None:
    images = []
    annotations = []
    for name in ("train-a", "train-b", "val-a"):
        image = tmp_path / f"{name}.jpg"
        annotation = tmp_path / f"{name}.png"
        image.write_bytes(b"same-image")
        annotation.write_bytes(b"same-annotation")
        images.append(image)
        annotations.append(annotation)
    pairs = (
        SourcePair("train", images[0], annotations[0]),
        SourcePair("train", images[1], annotations[1]),
        SourcePair("val", images[2], annotations[2]),
    )

    audit = audit_duplicates(pairs, tmp_path)

    assert len(audit.duplicate_images) == 1
    assert len(audit.duplicate_annotations) == 1
    assert len(audit.cross_split_leakage) == 1


def test_conflicting_masks_for_identical_image_are_reported(tmp_path: Path) -> None:
    first_image = tmp_path / "first.jpg"
    second_image = tmp_path / "second.jpg"
    first_mask = tmp_path / "first.png"
    second_mask = tmp_path / "second.png"
    first_image.write_bytes(b"same")
    second_image.write_bytes(b"same")
    first_mask.write_bytes(b"mask-one")
    second_mask.write_bytes(b"mask-two")

    audit = audit_duplicates(
        (
            SourcePair("train", first_image, first_mask),
            SourcePair("val", second_image, second_mask),
        ),
        tmp_path,
    )

    assert len(audit.conflicting_annotations) == 1


def test_readiness_is_false_with_blocking_errors_and_review_gaps() -> None:
    report = {
        "acquisition_status": "IMPORTED",
        "blocking_errors": ["leakage"],
        "unknown_labels": [],
        "image_count": 3,
        "annotation_count": 3,
        "conversion_count": 3,
        "failed_conversion_count": 0,
        "split_counts": {"train": 1, "val": 1, "test": 1},
    }

    assert not derive_readiness(
        report,
        required_splits=("train", "val", "test"),
        license_reviewed=True,
        mapping_reviewed=True,
        visual_reviewed=True,
    )
    report["blocking_errors"] = []
    assert not derive_readiness(
        report,
        required_splits=("train", "val", "test"),
        license_reviewed=True,
        mapping_reviewed=False,
        visual_reviewed=True,
    )


def test_readiness_is_true_only_after_required_checks_and_reviews() -> None:
    report = {
        "acquisition_status": "VERIFIED",
        "blocking_errors": [],
        "unknown_labels": [],
        "image_count": 3,
        "annotation_count": 3,
        "conversion_count": 3,
        "failed_conversion_count": 0,
        "split_counts": {"train": 1, "val": 1, "test": 1},
    }

    assert derive_readiness(
        report,
        required_splits=("train", "val", "test"),
        license_reviewed=True,
        mapping_reviewed=True,
        visual_reviewed=True,
    )


def test_atomic_text_output_replaces_partial_content(tmp_path: Path) -> None:
    path = tmp_path / "atomic.txt"
    path.write_text("old")

    atomic_write_text(path, "new")

    assert path.read_text() == "new"
    assert not list(tmp_path.glob("*.tmp"))


def test_visual_inspection_is_deterministic_on_synthetic_fixture(
    data_paths: DataPaths, write_rgb_image: object, write_indexed_mask: object
) -> None:
    image = data_paths.raw / "floodnet" / "train" / "images" / "sample.jpg"
    mask = data_paths.raw / "floodnet" / "train" / "masks" / "sample.png"
    write_rgb_image(image, (8, 6), 80)
    values = np.array(
        [[0, 0, 1, 1, 3, 3, 5, 5]] * 6,
        dtype=np.uint8,
    )
    write_indexed_mask(mask, values)
    convert_segmentation_dataset(
        data_paths,
        "floodnet",
        integrity=IntegrityMode.FULL,
        materialization=MaterializationStrategy.COPY,
    )

    first = generate_inspection(data_paths, "floodnet", split="train", count=1, seed=9)
    sheet = data_paths.root / first["contact_sheet"]
    first_hash = sha256_file(sheet)
    second = generate_inspection(data_paths, "floodnet", split="train", count=1, seed=9)

    assert first == second
    assert sha256_file(sheet) == first_hash


def test_human_review_is_bound_to_current_dataset_fingerprint(
    data_paths: DataPaths, write_rgb_image: object, write_indexed_mask: object
) -> None:
    image = data_paths.raw / "floodnet" / "train" / "images" / "sample.jpg"
    mask = data_paths.raw / "floodnet" / "train" / "masks" / "sample.png"
    write_rgb_image(image, (8, 6), 80)
    write_indexed_mask(mask, np.zeros((6, 8), dtype=np.uint8))
    result = convert_segmentation_dataset(
        data_paths,
        "floodnet",
        integrity=IntegrityMode.FULL,
        materialization=MaterializationStrategy.COPY,
    )

    record_review(
        data_paths,
        "floodnet",
        reviewer="Fixture reviewer",
        license_reviewed=True,
        mapping_reviewed=True,
        visual_reviewed=True,
    )
    current = read_review(data_paths, "floodnet", result["fingerprint"])
    stale = read_review(data_paths, "floodnet", "f" * 64)

    assert current["fingerprint_matches"] is True
    assert current["mapping_reviewed"] is True
    assert stale["fingerprint_matches"] is False
    assert stale["visual_reviewed"] is False


def test_repository_artifact_safeguard_detects_archives(tmp_path: Path) -> None:
    (tmp_path / "source.zip").write_bytes(b"archive")

    assert guard_repository_artifacts(tmp_path) == ["source.zip"]
    assert guard_repository_artifacts(PROJECT_ROOT) == []


def test_cli_doctor_works_without_datasets(monkeypatch: object, capsys: object) -> None:
    monkeypatch.delenv("FLOODSIGHT_DATA_ROOT", raising=False)

    exit_code = main(["--json", "doctor"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["data_root_configured"] is False
    assert payload["datasets"]["floodnet"]["status"] == "DATA_ROOT_NOT_CONFIGURED"


def test_cli_missing_data_root_is_structured(monkeypatch: object, capsys: object) -> None:
    monkeypatch.delenv("FLOODSIGHT_DATA_ROOT", raising=False)

    exit_code = main(["--json", "validate", "--dataset", "floodnet"])
    payload = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert payload["error"]["code"] == "data_root_missing"
