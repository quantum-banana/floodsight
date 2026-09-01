from app.inference.contracts import (
    DetectionObservationState,
    DetectionProvenance,
    DetectionResult,
    ModelIdentity,
    NormalizedDetection,
)
from app.inference.tracking import TemporalDetectionTracker
from app.schemas.live_result import BoundingBox

MODEL = ModelIdentity(
    model_id="fallback-fixture",
    architecture="YOLO11l",
    version="test",
)


def _detection_result(
    frame_id: int,
    timestamp_ms: int,
    detections: list[NormalizedDetection],
) -> DetectionResult:
    return DetectionResult(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        source_width=100,
        source_height=100,
        model=MODEL,
        taxonomy_version="detection-taxonomy-v1",
        detections=detections,
        inference_latency_ms=1,
        device="cpu",
        provenance_mode=DetectionProvenance.PRETRAINED_FALLBACK,
        source_frame_id=frame_id,
    )


def _car(frame_id: int, confidence: float = 0.6, x: float = 0.2) -> NormalizedDetection:
    return NormalizedDetection(
        detection_id=f"{frame_id}-car",
        application_class="car",
        source_class="motorcycle",
        source_class_id=3,
        confidence=confidence,
        source_confidence=confidence,
        bbox=BoundingBox(x=x, y=0.2, width=0.12, height=0.08),
        source_frame_id=frame_id,
    )


def test_tracking_requires_real_seed_then_decays_and_expires() -> None:
    tracker = TemporalDetectionTracker(track_ttl_ms=2_000)
    empty = tracker.update(
        _detection_result(0, 0, []),
        frame_id=0,
        timestamp_ms=0,
    )
    assert empty is not None
    assert empty.detections == []

    first = tracker.update(
        _detection_result(1, 1_000, [_car(1)]),
        frame_id=1,
        timestamp_ms=1_000,
    )
    assert first is not None
    track_id = first.detections[0].track_id
    assert track_id is not None
    assert first.detections[0].persistence == 1
    assert first.detections[0].observation_state is DetectionObservationState.DETECTED

    confirmed = tracker.update(
        _detection_result(2, 2_000, [_car(2, confidence=0.7, x=0.21)]),
        frame_id=2,
        timestamp_ms=2_000,
    )
    assert confirmed is not None
    assert confirmed.detections[0].track_id == track_id
    assert confirmed.detections[0].persistence == 2

    missed_once = tracker.update(
        _detection_result(3, 3_000, []),
        frame_id=3,
        timestamp_ms=3_000,
    )
    assert missed_once is not None
    persisted = missed_once.detections[0]
    assert persisted.track_id == track_id
    assert persisted.observation_state is DetectionObservationState.TRACK_PERSISTED
    assert persisted.track_confidence is not None
    assert confirmed.detections[0].track_confidence is not None
    assert persisted.track_confidence < confirmed.detections[0].track_confidence
    assert persisted.source_class == "motorcycle"
    assert persisted.source_class_id == 3
    assert persisted.source_confidence == 0.7
    assert persisted.source_frame_id == 2

    expired = tracker.update(
        _detection_result(4, 4_000, []),
        frame_id=4,
        timestamp_ms=4_000,
    )
    assert expired is not None
    assert expired.detections == []


def test_single_low_confidence_observation_is_not_carried_forward() -> None:
    tracker = TemporalDetectionTracker(track_ttl_ms=2_000)
    tracker.update(
        _detection_result(0, 0, [_car(0, confidence=0.3)]),
        frame_id=0,
        timestamp_ms=0,
    )

    missed = tracker.update(
        _detection_result(1, 1_000, []),
        frame_id=1,
        timestamp_ms=1_000,
    )

    assert missed is not None
    assert missed.detections == []
