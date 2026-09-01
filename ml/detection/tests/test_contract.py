from __future__ import annotations

import json
from pathlib import Path

import pytest
from floodsight_detection.contract import (
    DETECTION_CLASSES,
    freeze_dataset_contract,
    validate_dataset_contract,
)
from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file


def test_validates_all_manifest_samples_labels_hashes_and_classes(
    detection_manifest: tuple[Path, Path],
) -> None:
    root, manifest = detection_manifest

    contract = validate_dataset_contract(manifest, root)

    assert len(contract.samples) == 10
    assert contract.split_counts == {"train": 8, "val": 2}
    assert contract.class_counts == {0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
    assert contract.total_boxes == 10
    assert contract.image_hashes_verified is True


def test_manifest_path_hash_fingerprint_and_source_version_are_config_bindable(
    detection_manifest: tuple[Path, Path],
) -> None:
    root, manifest = detection_manifest
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    contract = validate_dataset_contract(
        manifest,
        root,
        expected_manifest_path=manifest,
        expected_manifest_sha256=sha256_file(manifest),
        expected_dataset_fingerprint=payload["fingerprint"],
        expected_source_version=payload["source_version"],
    )
    assert contract.manifest_sha256 == sha256_file(manifest)

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_dataset_contract(
            manifest,
            root,
            expected_manifest_path=manifest,
            expected_manifest_sha256="0" * 64,
            expected_dataset_fingerprint=payload["fingerprint"],
            expected_source_version=payload["source_version"],
        )
    assert error.value.code == "manifest_identity_mismatch"


def test_freeze_writes_absolute_split_lists_and_json_yaml_without_touching_data(
    detection_manifest: tuple[Path, Path], tmp_path: Path
) -> None:
    root, manifest = detection_manifest
    original_manifest = manifest.read_bytes()
    contract = validate_dataset_contract(manifest, root)

    data_yaml = freeze_dataset_contract(contract, tmp_path / "frozen")

    payload = json.loads(data_yaml.read_text(encoding="utf-8"))
    assert payload["names"] == {str(key): value for key, value in DETECTION_CLASSES.items()}
    assert Path(payload["train"]).is_absolute()
    assert Path(payload["val"]).is_absolute()
    train_images = Path(payload["train"]).read_text(encoding="utf-8").splitlines()
    assert len(train_images) == 8
    assert all(Path(path).is_absolute() for path in train_images)
    snapshot = json.loads((data_yaml.parent / "contract.json").read_text(encoding="utf-8"))
    assert snapshot["labels_exhaustively_validated"] is True
    assert len(snapshot["contract_sha256"]) == 64
    assert manifest.read_bytes() == original_manifest


def test_freeze_refuses_existing_output(
    detection_manifest: tuple[Path, Path], tmp_path: Path
) -> None:
    root, manifest = detection_manifest
    contract = validate_dataset_contract(manifest, root)
    output = tmp_path / "frozen"
    output.mkdir()

    with pytest.raises(DetectionInfrastructureError) as error:
        freeze_dataset_contract(contract, output)

    assert error.value.code == "contract_collision"


@pytest.mark.parametrize(
    ("label", "code"),
    [
        ("8 0.5 0.5 0.2 0.2\n", "unsupported_target_class"),
        ("0 0.5 0.5 0.0 0.2\n", "label_box_invalid"),
        ("0 0.95 0.5 0.2 0.2\n", "label_box_invalid"),
        ("0 nan 0.5 0.2 0.2\n", "label_box_invalid"),
        ("0 0.5 0.5 0.2\n", "label_row_invalid"),
        ("zero 0.5 0.5 0.2 0.2\n", "label_row_invalid"),
        ("0 0.5 0.5 0.2 0.2\n\n0 0.5 0.5 0.2 0.2\n", "label_row_invalid"),
        ("0 0.5 0.5 0.2 0.2\n0 0.500000 0.500000 0.200000 0.200000\n", "duplicate_label_row"),
    ],
)
def test_exhaustive_label_validation_blocks_malformed_rows(
    detection_manifest: tuple[Path, Path],
    mutate_manifest: object,
    label: str,
    code: str,
) -> None:
    root, manifest = detection_manifest
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    label_path = root / sample["target_annotation_path"]
    label_path.write_text(label, encoding="utf-8")
    sample["target_annotation_hash"] = sha256_file(label_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_dataset_contract(manifest, root)

    assert error.value.code == code


def test_label_hash_mismatch_is_blocking(detection_manifest: tuple[Path, Path]) -> None:
    root, manifest = detection_manifest
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    label = root / payload["samples"][0]["target_annotation_path"]
    label.write_text("0 0.4 0.4 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_dataset_contract(manifest, root)

    assert error.value.code == "label_hash_mismatch"


def test_image_hash_mismatch_is_blocking(detection_manifest: tuple[Path, Path]) -> None:
    root, manifest = detection_manifest
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    image = root / payload["samples"][0]["image_path"]
    image.write_bytes(b"drift")

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_dataset_contract(manifest, root)

    assert error.value.code == "image_hash_mismatch"


def test_manifest_label_count_mismatch_is_blocking(
    detection_manifest: tuple[Path, Path], mutate_manifest: object
) -> None:
    root, manifest = detection_manifest
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["samples"][0]["class_counts"] = {"0": 2}
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_dataset_contract(manifest, root)

    assert error.value.code == "class_count_mismatch"


def test_manifest_path_escape_is_blocking(
    detection_manifest: tuple[Path, Path], tmp_path: Path
) -> None:
    root, manifest = detection_manifest
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["samples"][0]["image_path"] = "../outside.jpg"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_dataset_contract(manifest, root)

    assert error.value.code == "unsafe_dataset_path"


def test_duplicate_image_content_is_blocking(detection_manifest: tuple[Path, Path]) -> None:
    root, manifest = detection_manifest
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    first, second = payload["samples"][:2]
    first_image = root / first["image_path"]
    second_image = root / second["image_path"]
    second_image.write_bytes(first_image.read_bytes())
    duplicate_hash = sha256_file(second_image)
    second["image_hash"] = duplicate_hash
    second["target_image_hash"] = duplicate_hash
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_dataset_contract(manifest, root)

    assert error.value.code == "duplicate_image_content"


def test_missing_train_class_is_blocking(detection_manifest: tuple[Path, Path]) -> None:
    root, manifest = detection_manifest
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample = next(item for item in payload["samples"] if item["class_counts"] == {"7": 1})
    sample["target_split"] = "val"
    sample["source_split"] = "val"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_dataset_contract(manifest, root)

    assert error.value.code == "train_class_coverage_incomplete"


def test_fast_integrity_manifest_is_rejected(detection_manifest: tuple[Path, Path]) -> None:
    root, manifest = detection_manifest
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["integrity_mode"] = "fast"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DetectionInfrastructureError) as error:
        validate_dataset_contract(manifest, root)

    assert error.value.code == "manifest_contract_mismatch"
