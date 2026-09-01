from pathlib import Path

import numpy as np
import pytest

from app.inference.contracts import (
    DetectorInferenceMode,
    ModelIdentity,
    SegmentationClassStatistic,
    SegmentationProvenance,
    SegmentationResult,
    decode_mask,
    encode_mask,
)
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
from app.inference.yolo import (
    COCO_TO_FLOODSIGHT,
    RawDetection,
    YoloAdapter,
    overlapping_tile_windows,
)

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

    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        inference_resolution: int | None = None,
    ) -> list[RawDetection]:
        assert frame_bgr.shape == (20, 40, 3)
        assert inference_resolution == 768
        return [
            RawDetection(0, "person", 0.93, (4, 2, 12, 10)),
            RawDetection(99, "dog", 0.99, (20, 2, 30, 12)),
        ]


class StubAerialDetectionRuntime:
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int | None]] = []

    def load(self) -> None:
        return None

    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        inference_resolution: int | None = None,
    ) -> list[RawDetection]:
        call_index = len(self.calls)
        self.calls.append((frame_bgr.shape, inference_resolution))
        if call_index == 0:
            return [
                RawDetection(0, "person", 0.9, (5, 5, 45, 45)),
                RawDetection(2, "car", 0.7, (20, 20, 40, 40)),
            ]
        if call_index == 1:
            return [
                RawDetection(0, "person", 0.85, (10, 10, 30, 25)),
                RawDetection(7, "truck", 0.8, (20, 20, 40, 40)),
            ]
        if call_index == 2:
            return [RawDetection(2, "car", 0.75, (10, 10, 20, 20))]
        return []


class StubReinspectionRuntime:
    device = "cpu"

    def __init__(self, reinspection_class: str) -> None:
        self.calls = 0
        self.reinspection_class = reinspection_class

    def load(self) -> None:
        return None

    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        inference_resolution: int | None = None,
    ) -> list[RawDetection]:
        del frame_bgr, inference_resolution
        self.calls += 1
        if self.calls == 15:
            class_id = 2 if self.reinspection_class == "car" else 67
            return [
                RawDetection(class_id, self.reinspection_class, 0.8, (10, 10, 30, 30))
            ]
        return []


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


def test_aerial_mode_maps_original_coordinates_and_suppresses_vehicle_duplicates(
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
    runtime = StubAerialDetectionRuntime()
    adapter = YoloAdapter(
        model,
        inference_resolution=640,
        aerial_inference_resolution=1280,
        aerial_tile_overlap=0.2,
        aerial_fusion_iou_threshold=0.5,
        runtime=runtime,
    )
    adapter.load()

    result = adapter.infer(
        np.zeros((100, 100, 3), dtype=np.uint8),
        frame_id=8,
        timestamp_ms=300,
        detector_mode=DetectorInferenceMode.AERIAL,
    )

    windows = overlapping_tile_windows(100, 100, rows=2, columns=2, overlap=0.2)
    assert [(item.x1, item.y1, item.x2, item.y2) for item in windows] == [
        (0, 0, 56, 56),
        (44, 0, 100, 56),
        (0, 44, 56, 100),
        (44, 44, 100, 100),
    ]
    assert runtime.calls == [
        ((100, 100, 3), 640),
        ((56, 56, 3), 1280),
        ((56, 56, 3), 1280),
        ((56, 56, 3), 1280),
        ((56, 56, 3), 1280),
    ]
    assert [item.application_class for item in result.detections] == ["person", "truck", "car"]
    assert sum(item.application_class == "person" for item in result.detections) == 1
    assert result.detections[1].bbox.x == pytest.approx(0.2)
    assert result.detections[2].bbox.x == pytest.approx(0.54)
    assert result.detections[2].bbox.y == pytest.approx(0.1)
    assert result.provenance_mode.value == "PRETRAINED_FALLBACK"


def test_aerial_high_recall_adds_deterministic_three_by_three_pass(tmp_path: Path) -> None:
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
    runtime = StubAerialDetectionRuntime()
    adapter = YoloAdapter(
        model,
        inference_resolution=640,
        aerial_inference_resolution=1280,
        aerial_tile_overlap=0.2,
        aerial_high_recall_resolution=1280,
        aerial_high_recall_tile_overlap=0.25,
        runtime=runtime,
    )
    adapter.load()

    result = adapter.infer(
        np.zeros((100, 100, 3), dtype=np.uint8),
        frame_id=9,
        timestamp_ms=400,
        detector_mode=DetectorInferenceMode.AERIAL_HIGH_RECALL,
    )

    high_recall_windows = overlapping_tile_windows(
        100,
        100,
        rows=3,
        columns=3,
        overlap=0.25,
    )
    assert [(item.x1, item.y1, item.x2, item.y2) for item in high_recall_windows] == [
        (0, 0, 40, 40),
        (30, 0, 70, 40),
        (60, 0, 100, 40),
        (0, 30, 40, 70),
        (30, 30, 70, 70),
        (60, 30, 100, 70),
        (0, 60, 40, 100),
        (30, 60, 70, 100),
        (60, 60, 100, 100),
    ]
    assert len(runtime.calls) == 14
    assert runtime.calls[0][1] == 640
    assert [resolution for _, resolution in runtime.calls[1:]] == [1280] * 13
    assert all(item.source_confidence == item.confidence for item in result.detections)


@pytest.mark.parametrize(
    ("reinspection_class", "expected_count"),
    [("car", 1), ("cell phone", 0)],
)
def test_segformer_reinspection_requires_yolo_supported_class(
    tmp_path: Path,
    reinspection_class: str,
    expected_count: int,
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
    runtime = StubReinspectionRuntime(reinspection_class)
    adapter = YoloAdapter(
        model,
        inference_resolution=640,
        aerial_inference_resolution=1280,
        aerial_high_recall_resolution=1280,
        segformer_reinspection_min_pixels=4,
        runtime=runtime,
    )
    adapter.load()
    class_map = np.zeros((100, 100), dtype=np.uint8)
    class_map[40:50, 40:50] = 12
    segmentation = SegmentationResult(
        frame_id=1,
        timestamp_ms=1_000,
        source_width=100,
        source_height=100,
        model=ModelIdentity(
            model_id="segformer-fixture",
            architecture="SegFormer-B2",
            version="test",
        ),
        taxonomy_version="segmentation-taxonomy-v2",
        mask=encode_mask(class_map),
        class_statistics=[
            SegmentationClassStatistic(
                class_id=12,
                class_name="vehicle",
                pixel_count=100,
                coverage_percent=1,
                mean_confidence=0.9,
            )
        ],
        inference_latency_ms=1,
        device="cpu",
        provenance_mode=SegmentationProvenance.REAL_MODEL,
        source_frame_id=1,
    )

    result = adapter.infer(
        np.zeros((100, 100, 3), dtype=np.uint8),
        frame_id=1,
        timestamp_ms=1_000,
        detector_mode=DetectorInferenceMode.AERIAL_HIGH_RECALL,
        segmentation=segmentation,
    )

    assert runtime.calls == 15
    assert len(result.detections) == expected_count
    if result.detections:
        assert result.detections[0].source_class == "car"
        assert result.detections[0].application_class == "car"


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
