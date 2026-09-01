"""Explicit, isolated, synthetic-only detector smoke orchestration."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from floodsight_detection.contract import freeze_dataset_contract, validate_dataset_contract
from floodsight_detection.errors import DetectionInfrastructureError
from floodsight_detection.hashing import sha256_file, stable_sha256

_TARGET_TO_SOURCE = {0: 1, 1: 4, 2: 5, 3: 6, 4: 9, 5: 3, 6: 10, 7: 7}


class SyntheticBackend(Protocol):
    def run(
        self,
        *,
        data_yaml: Path,
        output_root: Path,
        seed: int,
        device: str,
    ) -> dict[str, Any]: ...


def _write_exclusive(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _create_synthetic_manifest(root: Path) -> Path:
    """Generate six tiny images; this function is called only after authorization."""

    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DetectionInfrastructureError(
            "Pillow is required by the explicit synthetic smoke path.",
            code="ml_dependency_missing",
        ) from exc
    samples: list[dict[str, Any]] = []
    layout = (("train", 4), ("val", 2))
    for split, count in layout:
        for index in range(count):
            stem = f"{split}-{index:02d}"
            image = root / "processed" / "detection_v2" / "images" / split / f"{stem}.jpg"
            source_annotation = (
                root
                / "raw"
                / "visdrone_det"
                / "SYNTHETIC_SMOKE_ONLY"
                / "annotations"
                / f"{stem}.txt"
            )
            label = root / "processed" / "detection_v2" / "labels" / split / f"{stem}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            source_annotation.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            # Different deterministic pixels prevent duplicate-hash false positives.
            Image.new(
                "RGB",
                (64, 64),
                (20 + index * 17, 40 + (0 if split == "train" else 70), 90),
            ).save(image, format="JPEG", quality=95)
            class_ids = (
                (
                    (index * 2) % 8,
                    (index * 2 + 1) % 8,
                )
                if split == "train"
                else (index,)
            )
            lines = [f"{class_id} 0.500000 0.500000 0.250000 0.250000" for class_id in class_ids]
            source_lines = [
                f"24,24,16,16,1,{_TARGET_TO_SOURCE[class_id]},0,0"
                for class_id in class_ids
            ]
            _write_exclusive(source_annotation, "\n".join(source_lines) + "\n")
            _write_exclusive(label, "\n".join(lines) + "\n")
            counts = Counter(class_ids)
            image_hash = sha256_file(image)
            source_annotation_hash = sha256_file(source_annotation)
            label_hash = sha256_file(label)
            samples.append(
                {
                    "sample_id": f"synthetic-{stem}",
                    "source_dataset": "visdrone_det",
                    "source_split": split,
                    "target_split": split,
                    "image_path": image.relative_to(root).as_posix(),
                    "source_annotation_path": source_annotation.relative_to(root).as_posix(),
                    "target_annotation_path": label.relative_to(root).as_posix(),
                    "width": 64,
                    "height": 64,
                    "image_hash": image_hash,
                    "annotation_hash": source_annotation_hash,
                    "target_image_hash": image_hash,
                    "target_annotation_hash": label_hash,
                    "class_counts": {str(key): value for key, value in counts.items()},
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
                        for class_id in class_ids
                    ],
                }
            )
    identity = {
        "samples": [sample["sample_id"] for sample in samples],
        "purpose": "floodsight-yolo-synthetic-smoke",
    }
    payload = {
        "schema_version": "dataset-manifest-v1",
        "manifest_id": "visdrone_det-detection_v2",
        "dataset_id": "visdrone_det",
        "task_type": "AERIAL_DETECTION",
        "source_version": "SYNTHETIC_SMOKE_ONLY",
        "preparation_version": "detection_v2",
        "taxonomy_version": "detection-taxonomy-v1",
        "integrity_mode": "full",
        "created_at": "1970-01-01T00:00:00Z",
        "tool_version": "synthetic-smoke-v1",
        "git_commit": "UNKNOWN",
        "fingerprint": stable_sha256(identity),
        "samples": samples,
    }
    manifest = root / "manifests" / "visdrone_det-detection_v2.json"
    _write_exclusive(manifest, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest


def run_synthetic_smoke(
    output_directory: str | Path,
    *,
    allow_synthetic_smoke: bool,
    device: str = "cpu",
    seed: int = 20260831,
    backend: SyntheticBackend | None = None,
) -> dict[str, Any]:
    """Run a bounded generated-data smoke; never accepts a real dataset path."""

    if not allow_synthetic_smoke:
        raise DetectionInfrastructureError(
            "Synthetic smoke is disabled; pass --allow-synthetic-smoke explicitly.",
            code="synthetic_smoke_not_authorized",
        )
    output = Path(output_directory).expanduser().resolve()
    try:
        output.mkdir(parents=True, mode=0o750, exist_ok=False)
    except FileExistsError as exc:
        raise DetectionInfrastructureError(
            f"Refusing to overwrite synthetic smoke output: {output}",
            code="smoke_collision",
        ) from exc
    synthetic_root = output / "synthetic-data"
    synthetic_root.mkdir()
    manifest = _create_synthetic_manifest(synthetic_root)
    contract = validate_dataset_contract(manifest, synthetic_root)
    data_yaml = freeze_dataset_contract(contract, output / "dataset-contract")
    if backend is None:
        # PyTorch and Ultralytics are imported only on this authorized path.
        from floodsight_detection.ultralytics_runtime import UltralyticsSmokeBackend

        backend = UltralyticsSmokeBackend()
    result = backend.run(
        data_yaml=data_yaml,
        output_root=output / "runtime",
        seed=seed,
        device=device,
    )
    required = (
        "loader",
        "model_forward",
        "loss",
        "backward",
        "validation",
        "checkpoint",
        "resume",
    )
    failed = [name for name in required if result.get(name) is not True]
    if failed:
        raise DetectionInfrastructureError(
            f"Synthetic backend did not prove: {', '.join(failed)}.",
            code="synthetic_smoke_incomplete",
        )
    report = {
        "schema_version": "floodsight-detection-synthetic-smoke-v1",
        "status": "PASS",
        "synthetic_only": True,
        "real_dataset_accessed": False,
        "real_training_started": False,
        "seed": seed,
        "device": device,
        "manifest_sha256": contract.manifest_sha256,
        "dataset_fingerprint": contract.dataset_fingerprint,
        "checks": {name: True for name in required},
        "backend": result,
    }
    report_path = output / "synthetic-smoke-report.json"
    _write_exclusive(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report | {"report_path": str(report_path)}
