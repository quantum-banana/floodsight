import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from app.core.config import Settings
from app.inference.contracts import (
    DetectionProvenance,
    DetectionResult,
    ModelIdentity,
    NormalizedDetection,
    SegmentationClassStatistic,
    SegmentationProvenance,
    SegmentationResult,
    encode_mask,
)
from app.inference.pipeline import InferencePipeline
from app.main import create_app
from app.schemas.live_result import BoundingBox, SourceMode
from app.schemas.model_status import (
    InferenceState,
    ModelOperationalMode,
    ModelState,
    ModelStatus,
    ModelStatusResponse,
)
from app.services.demo_incident import get_demo_snapshots
from app.services.inference_coordinator import InferenceCoordinator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_SCHEMA = json.loads(
    (PROJECT_ROOT / "shared/schemas/live-result.schema.json").read_text(encoding="utf-8")
)
FRAME_INTELLIGENCE_SCHEMA = json.loads(
    (PROJECT_ROOT / "shared/schemas/frame-intelligence.schema.json").read_text(encoding="utf-8")
)
MODEL = ModelIdentity(model_id="stub", architecture="stub", version="test")


class StubSegmentationAdapter:
    def infer(self, frame: np.ndarray, *, frame_id: int, timestamp_ms: int) -> SegmentationResult:
        height, width = frame.shape[:2]
        class_map = np.zeros((height, width), dtype=np.uint8)
        class_map[height // 4 : height // 2, width // 4 : width // 2] = 1
        statistics = [
            SegmentationClassStatistic(
                class_id=class_id,
                class_name=name,
                pixel_count=int(np.count_nonzero(class_map == class_id)),
                coverage_percent=float(
                    np.count_nonzero(class_map == class_id) * 100 / class_map.size
                ),
                mean_confidence=0.9 if np.any(class_map == class_id) else 0,
            )
            for class_id, name in ((0, "background_other"), (1, "water"), (15, "pool"))
        ]
        return SegmentationResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            source_width=width,
            source_height=height,
            model=MODEL,
            taxonomy_version="segmentation-taxonomy-v2",
            mask=encode_mask(class_map),
            class_statistics=statistics,
            inference_latency_ms=2,
            device="cpu",
            provenance_mode=SegmentationProvenance.REAL_MODEL,
        )


class StubDetectionAdapter:
    def infer(self, frame: np.ndarray, *, frame_id: int, timestamp_ms: int) -> DetectionResult:
        height, width = frame.shape[:2]
        return DetectionResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            source_width=width,
            source_height=height,
            model=MODEL,
            taxonomy_version="detection-taxonomy-v1",
            detections=[
                NormalizedDetection(
                    detection_id=f"person-{frame_id}",
                    application_class="person",
                    source_class="person",
                    source_class_id=0,
                    confidence=0.96,
                    bbox=BoundingBox(x=0.3, y=0.3, width=0.05, height=0.08),
                )
            ],
            inference_latency_ms=1,
            device="cpu",
            provenance_mode=DetectionProvenance.PRETRAINED_FALLBACK,
        )


class RoadTransitionSegmentationAdapter:
    def infer(self, frame: np.ndarray, *, frame_id: int, timestamp_ms: int) -> SegmentationResult:
        height, width = frame.shape[:2]
        class_map = np.full((height, width), 4, dtype=np.uint8)
        class_map[height // 4 : height // 2, width // 4 : width // 2] = 1
        if frame_id >= 2:
            class_map[height // 2 : 3 * height // 4, : width // 4] = 5
        statistics = [
            SegmentationClassStatistic(
                class_id=class_id,
                class_name=name,
                pixel_count=int(np.count_nonzero(class_map == class_id)),
                coverage_percent=float(
                    np.count_nonzero(class_map == class_id) * 100 / class_map.size
                ),
                mean_confidence=0.9 if np.any(class_map == class_id) else 0,
            )
            for class_id, name in (
                (0, "background_other"),
                (1, "water"),
                (4, "road_clear"),
                (5, "road_blocked"),
                (15, "pool"),
            )
        ]
        return SegmentationResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            source_width=width,
            source_height=height,
            model=MODEL,
            taxonomy_version="segmentation-taxonomy-v2",
            mask=encode_mask(class_map),
            class_statistics=statistics,
            inference_latency_ms=2,
            device="cpu",
            provenance_mode=SegmentationProvenance.REAL_MODEL,
        )


def _ready_status() -> ModelStatus:
    return ModelStatus(
        status=ModelState.READY,
        model="stub",
        mode=ModelOperationalMode.REAL,
        version="test",
        device="cpu",
        provenance_mode="REAL_MODEL",
    )


def test_end_to_end_stub_pipeline_serializes_backend_decisions() -> None:
    pipeline = InferencePipeline(
        Settings(
            inference_resolution=256,
            segmentation_cadence=2,
            detection_cadence=3,
        )
    )
    pipeline.segmentation_adapter = StubSegmentationAdapter()  # type: ignore[assignment]
    pipeline.detection_adapter = StubDetectionAdapter()  # type: ignore[assignment]
    pipeline._segmentation_status = _ready_status()
    pipeline._detection_status = _ready_status().model_copy(
        update={
            "mode": ModelOperationalMode.FALLBACK,
            "provenance_mode": "PRETRAINED_FALLBACK",
        }
    )
    pipeline._initialized = True

    result = pipeline.process(
        session_id="stub-session-1234567890",
        frame_bgr=np.zeros((32, 32, 3), dtype=np.uint8),
        frame_id=0,
        timestamp_ms=10_000,
        source_mode=SourceMode.VIDEO_FILE,
    )

    assert result is not None
    assert result.system_status.inference_state is InferenceState.LIVE
    assert result.segmentation.mask is not None
    assert result.zones[0].people_count == 1
    assert result.zones[0].grid_cells == ["B2"]
    assert result.statistics.damaged_buildings.value == 0
    assert result.scene_summary is not None
    assert result.scene_summary.pool_coverage_percent == 0
    assert result.zones[0].alerts[0].code == "POTENTIAL_STRANDED_PERSON"
    assert "PERSON_IN_HIGH_FLOOD_ZONE" in result.zones[0].alerts[0].reason_codes
    assert result.detections[0].model_provenance == "PRETRAINED_FALLBACK"
    assert result.system_status.detection_details is not None
    assert result.system_status.detection_details.mode is ModelOperationalMode.FALLBACK
    Draft202012Validator(LIVE_SCHEMA).validate(result.model_dump(mode="json"))

    reused = pipeline.process(
        session_id="stub-session-1234567890",
        frame_bgr=np.zeros((32, 32, 3), dtype=np.uint8),
        frame_id=1,
        timestamp_ms=10_100,
        source_mode=SourceMode.VIDEO_FILE,
    )
    assert reused is not None
    assert reused.evidence_frames is not None
    assert reused.evidence_frames.segmentation_source_frame_id == 0
    assert reused.evidence_frames.detection_source_frame_id == 0
    assert reused.evidence_frames.segmentation_reused is True
    assert reused.evidence_frames.detection_reused is True
    assert reused.zones[0].alerts[0].temporal_samples == 1
    Draft202012Validator(LIVE_SCHEMA).validate(reused.model_dump(mode="json"))


def test_pipeline_emits_explicit_event_when_primary_route_becomes_unsafe() -> None:
    pipeline = InferencePipeline(Settings(inference_resolution=256))
    pipeline.segmentation_adapter = RoadTransitionSegmentationAdapter()  # type: ignore[assignment]
    pipeline.detection_adapter = StubDetectionAdapter()  # type: ignore[assignment]
    pipeline._segmentation_status = _ready_status()
    pipeline._detection_status = _ready_status().model_copy(
        update={
            "mode": ModelOperationalMode.FALLBACK,
            "provenance_mode": "PRETRAINED_FALLBACK",
        }
    )
    pipeline._initialized = True

    results = [
        pipeline.process(
            session_id="route-transition-session",
            frame_bgr=np.zeros((32, 32, 3), dtype=np.uint8),
            frame_id=frame_id,
            timestamp_ms=20_000 + frame_id * 300,
            source_mode=SourceMode.VIDEO_FILE,
        )
        for frame_id in range(4)
    ]
    final = results[-1]

    assert final is not None
    assert final.route is not None
    assert final.route.changed_reason_code == "ROUTE_CHANGED_PRIMARY_ACCESS_UNSAFE"
    assert final.route.previous_edge_ids
    assert any(event.code == "ROUTE_CHANGED_PRIMARY_ACCESS_UNSAFE" for event in final.events)
    assert any(road.state.value == "BLOCKED" and not road.enabled for road in final.roads)


class SocketStubPipeline:
    def status(self) -> ModelStatusResponse:
        ready = _ready_status()
        return ModelStatusResponse(
            segmentation=ready,
            detection=ready.model_copy(deep=True),
            inference_state=InferenceState.LIVE,
        )

    def should_process(self, frame_id: int) -> bool:
        return True

    def process(self, **_: object):
        return get_demo_snapshots()[0]


def test_websocket_acknowledgement_precedes_ordered_intelligence_update() -> None:
    application = create_app()
    stub = SocketStubPipeline()
    application.state.inference_pipeline = stub
    application.state.inference_coordinator = InferenceCoordinator(stub)  # type: ignore[arg-type]
    image = np.full((8, 12, 3), 120, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    payload = encoded.tobytes()

    with TestClient(application) as client:
        session = client.post(
            "/api/ingest/sessions",
            json={"source_mode": "VIDEO_FILE", "media_origin": "USER_VIDEO_FILE"},
        ).json()
        with client.websocket_connect(
            f"/ws/ingest/sessions/{session['session_id']}/frames"
        ) as socket:
            socket.send_json(
                {
                    "type": "frame_metadata",
                    "frame_id": 1,
                    "captured_at_ms": 100,
                    "media_time_ms": 100,
                    "source_mode": "VIDEO_FILE",
                    "media_origin": "USER_VIDEO_FILE",
                    "mime_type": "image/jpeg",
                    "byte_length": len(payload),
                    "width": 12,
                    "height": 8,
                }
            )
            socket.send_bytes(payload)
            acknowledgement = socket.receive_json()
            intelligence = socket.receive_json()
        report_response = client.get(f"/api/ingest/sessions/{session['session_id']}/report")

    assert acknowledgement["type"] == "frame_result"
    assert acknowledgement["accepted"] is True
    assert intelligence["type"] == "frame_intelligence"
    assert intelligence["frame_id"] == 1
    assert intelligence["sequence"] == 0
    registry = Registry().with_resource(LIVE_SCHEMA["$id"], Resource.from_contents(LIVE_SCHEMA))
    Draft202012Validator(FRAME_INTELLIGENCE_SCHEMA, registry=registry).validate(intelligence)
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["generated_from_frame_id"] == 0
    assert report["priority_order"]
    assert report["reason_codes"]
    assert report["data_origin"] == "DERIVED_ANALYTIC"
    assert "personnel" in report["responsible_ai_statement"].lower()
    assert "authority" in report["responsible_ai_statement"].lower()
