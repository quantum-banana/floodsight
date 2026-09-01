from pathlib import Path

import numpy as np
import pytest

from app.inference.contracts import decode_mask, encode_mask
from app.inference.model_registry import (
    ModelLifecycleStatus,
    ModelRecord,
    ModelRegistry,
    ModelType,
    RegistryProvenance,
    ResolvedModel,
)
from app.inference.segformer import (
    SegFormerAdapter,
    SegmentationPrediction,
)
from app.inference.yolo import COCO_TO_FLOODSIGHT, RawDetection, YoloAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEGMENTATION_TAXONOMY = PROJECT_ROOT / "shared/taxonomy/segmentation-taxonomy-v2.yaml"
DETECTION_TAXONOMY = PROJECT_ROOT / "shared/taxonomy/detection-taxonomy-v1.yaml"


class StubSegmentationRuntime:
    device = "cpu"

    def load(self) -> None:
        return None

    def predict(self, normalized_chw: np.ndarray) -> SegmentationPrediction:
        assert normalized_chw.shape == (3, 8, 8)
        class_map = np.ones((8, 8), dtype=np.uint8)
        class_map[:, 4:] = 15
        return SegmentationPrediction(
            class_map=class_map,
            confidence_map=np.full((8, 8), 0.9, dtype=np.float32),
        )


class StubDetectionRuntime:
    device = "cpu"

    def load(self) -> None:
        return None

    def predict(self, frame_bgr: np.ndarray) -> list[RawDetection]:
        assert frame_bgr.shape == (20, 40, 3)
        return [
            RawDetection(0, "person", 0.93, (4, 2, 12, 10)),
            RawDetection(99, "dog", 0.99, (20, 2, 30, 12)),
        ]


def _record(
    checkpoint: Path,
    *,
    model_type: ModelType,
    taxonomy: Path,
    taxonomy_version: str,
    provenance: RegistryProvenance,
    label_space: str,
) -> ResolvedModel:
    record = ModelRecord(
        model_id="fixture-model",
        model_type=model_type,
        architecture="fixture",
        version="test-v1",
        checkpoint_path=str(checkpoint),
        checkpoint_format="FIXTURE",
        taxonomy_path=str(taxonomy),
        taxonomy_version=taxonomy_version,
        source_training_identity="synthetic contract fixture",
        status=ModelLifecycleStatus.INTEGRATION,
        provenance=provenance,
        enabled=True,
        label_space=label_space,
    )
    return ResolvedModel(record, checkpoint, taxonomy)


def test_segmentation_contract_resizes_and_preserves_pool(tmp_path: Path) -> None:
    checkpoint = tmp_path / "seg.pt"
    checkpoint.write_bytes(b"fixture")
    model = _record(
        checkpoint,
        model_type=ModelType.SEGMENTATION,
        taxonomy=SEGMENTATION_TAXONOMY,
        taxonomy_version="segmentation-taxonomy-v2",
        provenance=RegistryProvenance.REAL_MODEL,
        label_space="FLOODSIGHT_SEGMENTATION_V2",
    )
    adapter = SegFormerAdapter(
        model,
        inference_resolution=8,
        runtime=StubSegmentationRuntime(),
    )
    adapter.load()
    result = adapter.infer(np.zeros((20, 40, 3), dtype=np.uint8), frame_id=3, timestamp_ms=100)

    decoded = decode_mask(result.mask)
    assert decoded.shape == (20, 40)
    assert result.provenance_mode.value == "REAL_MODEL"
    statistics = {item.class_name: item for item in result.class_statistics}
    assert statistics["water"].coverage_percent == 50
    assert statistics["pool"].coverage_percent == 50


def test_detector_contract_keeps_source_label_and_marks_pretrained_fallback(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "det.pt"
    checkpoint.write_bytes(b"fixture")
    model = _record(
        checkpoint,
        model_type=ModelType.DETECTION,
        taxonomy=DETECTION_TAXONOMY,
        taxonomy_version="detection-taxonomy-v1",
        provenance=RegistryProvenance.PRETRAINED_FALLBACK,
        label_space="COCO",
    )
    adapter = YoloAdapter(model, runtime=StubDetectionRuntime())
    adapter.load()
    result = adapter.infer(np.zeros((20, 40, 3), dtype=np.uint8), frame_id=4, timestamp_ms=200)

    assert result.provenance_mode.value == "PRETRAINED_FALLBACK"
    assert len(result.detections) == 1
    assert result.detections[0].source_class == "person"
    assert result.detections[0].source_class_id == 0
    assert result.detections[0].application_class == "person"
    assert result.detections[0].bbox.x == 0.1
    assert result.model.checkpoint_sha256 is not None


def test_coco_fallback_mapping_is_explicit_and_does_not_invent_visdrone_labels() -> None:
    assert COCO_TO_FLOODSIGHT == {
        "person": "person",
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        "bicycle": "bicycle",
        "motorcycle": "motorcycle",
    }
    assert "van" not in COCO_TO_FLOODSIGHT
    assert "tricycle" not in COCO_TO_FLOODSIGHT


def test_mask_rle_round_trip_and_missing_checkpoint_behavior(tmp_path: Path) -> None:
    class_map = np.asarray([[1, 1, 15], [5, 5, 5]], dtype=np.uint8)
    assert np.array_equal(decode_mask(encode_mask(class_map)), class_map)

    missing = tmp_path / "missing.pt"
    model = _record(
        missing,
        model_type=ModelType.DETECTION,
        taxonomy=DETECTION_TAXONOMY,
        taxonomy_version="detection-taxonomy-v1",
        provenance=RegistryProvenance.PRETRAINED_FALLBACK,
        label_space="COCO",
    )
    with pytest.raises(FileNotFoundError):
        ModelRegistry.verify_checkpoint(model)


def test_registry_resolves_explicit_environment_values(tmp_path: Path) -> None:
    checkpoint = tmp_path / "segmentation.pt"
    checkpoint.write_bytes(b"fixture")
    registry = ModelRegistry(
        path=PROJECT_ROOT / "configs/models/registry.json",
        project_root=PROJECT_ROOT,
        environment={"FLOODSIGHT_SEGMENTATION_CHECKPOINT": str(checkpoint)},
    )

    model = registry.enabled(ModelType.SEGMENTATION)[0]

    assert model.checkpoint == checkpoint.resolve()
