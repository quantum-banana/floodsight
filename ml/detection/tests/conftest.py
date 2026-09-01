from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from floodsight_detection.hashing import sha256_file, stable_sha256
from PIL import Image

_TARGET_TO_SOURCE = {0: 1, 1: 4, 2: 5, 3: 6, 4: 9, 5: 3, 6: 10, 7: 7}


@pytest.fixture
def detection_manifest(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "data"
    samples: list[dict[str, object]] = []
    for split, class_ids in (("train", tuple(range(8))), ("val", (0, 1))):
        for index, class_id in enumerate(class_ids):
            stem = f"{split}-{index:02d}"
            image = root / "processed/detection_v2/images" / split / f"{stem}.jpg"
            source_annotation = (
                root / "raw/visdrone_det/VisDrone2019-DET-test/annotations" / f"{stem}.txt"
            )
            label = root / "processed/detection_v2/labels" / split / f"{stem}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            source_annotation.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (100, 100),
                (20 + index * 17, 40 + (0 if split == "train" else 70), 90),
            ).save(image, format="JPEG", quality=95)
            source_annotation.write_text(
                f"37.5,37.5,25,25,1,{_TARGET_TO_SOURCE[class_id]},0,0\n",
                encoding="utf-8",
            )
            label.write_text(
                f"{class_id} 0.500000 0.500000 0.250000 0.250000\n",
                encoding="utf-8",
            )
            image_hash = sha256_file(image)
            source_annotation_hash = sha256_file(source_annotation)
            label_hash = sha256_file(label)
            samples.append(
                {
                    "sample_id": f"visdrone-{stem}",
                    "source_dataset": "visdrone_det",
                    "source_split": split,
                    "target_split": split,
                    "image_path": image.relative_to(root).as_posix(),
                    "source_annotation_path": source_annotation.relative_to(root).as_posix(),
                    "target_annotation_path": label.relative_to(root).as_posix(),
                    "width": 100,
                    "height": 100,
                    "image_hash": image_hash,
                    "annotation_hash": source_annotation_hash,
                    "target_image_hash": image_hash,
                    "target_annotation_hash": label_hash,
                    "class_counts": {str(class_id): 1},
                    "ignored_count": 0,
                    "invalid_count": 0,
                    "preparation_version": "detection_v2",
                    "taxonomy_version": "detection-taxonomy-v1",
                    "objects": [
                        {
                            "source_class_id": _TARGET_TO_SOURCE[class_id],
                            "target_class_id": class_id,
                            "truncation": 0,
                            "occlusion": 0,
                            "clamped": False,
                        }
                    ],
                }
            )
    payload = {
        "schema_version": "dataset-manifest-v1",
        "manifest_id": "visdrone_det-detection_v2",
        "dataset_id": "visdrone_det",
        "task_type": "AERIAL_DETECTION",
        "source_version": "VisDrone2019-DET-test",
        "preparation_version": "detection_v2",
        "taxonomy_version": "detection-taxonomy-v1",
        "integrity_mode": "full",
        "created_at": "2026-08-31T00:00:00Z",
        "tool_version": "test",
        "git_commit": "UNKNOWN",
        "fingerprint": stable_sha256([sample["sample_id"] for sample in samples]),
        "samples": samples,
    }
    manifest = root / "manifests/visdrone_det-detection_v2.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return root, manifest


@pytest.fixture
def mutate_manifest() -> Callable[[Path, Callable[[dict[str, object]], None]], None]:
    def mutate(path: Path, callback: Callable[[dict[str, object]], None]) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        callback(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    return mutate
