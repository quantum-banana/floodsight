import asyncio
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from httpx import Response as HttpxResponse
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from starlette.websockets import WebSocketDisconnect

from app.inference.contracts import DetectorInferenceMode
from app.main import create_app
from app.schemas.ingestion import (
    MAX_VIDEO_DETECTED_CLASSES,
    MAX_VIDEO_PRIORITY_OBSERVATIONS,
    AggregateMetricAvailability,
    FrameMetadata,
    IngestionSessionCreate,
    IngestionSessionState,
    MediaOrigin,
    VideoAnalysisComplete,
)
from app.schemas.live_result import (
    AccessStatus,
    DataOrigin,
    DetectionCategory,
    EvidenceFrames,
    FeatureProvenance,
    RoadState,
    SceneSummary,
    SegmentationState,
    Severity,
    SourceMode,
)
from app.schemas.model_status import (
    InferenceState,
    ModelOperationalMode,
    ModelState,
    ModelStatus,
    ModelStatusResponse,
)
from app.services.demo_incident import get_demo_snapshots
from app.services.incident_reporting import build_video_analysis_report
from app.services.inference_coordinator import InferenceCoordinator
from app.services.video_analysis import VideoAnalysisAggregator

pytestmark = pytest.mark.anyio
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_SCHEMA = json.loads(
    (PROJECT_ROOT / "shared/schemas/live-result.schema.json").read_text(encoding="utf-8")
)
COMPLETION_SCHEMA = json.loads(
    (PROJECT_ROOT / "shared/schemas/video-analysis-complete.schema.json").read_text(
        encoding="utf-8"
    )
)


def _model_status(state: ModelState) -> ModelStatus:
    return ModelStatus(
        status=state,
        model="test-model" if state is ModelState.READY else None,
        mode=(
            ModelOperationalMode.REAL
            if state is ModelState.READY
            else ModelOperationalMode.UNAVAILABLE
        ),
        provenance_mode="REAL_MODEL" if state is ModelState.READY else None,
    )


def _status(
    *,
    segmentation: ModelState = ModelState.UNAVAILABLE,
    detection: ModelState = ModelState.READY,
) -> ModelStatusResponse:
    return ModelStatusResponse(
        segmentation=_model_status(segmentation),
        detection=_model_status(detection),
        inference_state=(
            InferenceState.LIVE
            if segmentation is ModelState.READY and detection is ModelState.READY
            else InferenceState.DEGRADED
            if ModelState.READY in (segmentation, detection)
            else InferenceState.MODEL_UNAVAILABLE
        ),
    )


def _with_status(snapshot, status: ModelStatusResponse):
    return snapshot.model_copy(
        deep=True,
        update={
            "system_status": snapshot.system_status.model_copy(
                update={
                    "segmentation_model": status.segmentation.status,
                    "detection_model": status.detection.status,
                    "inference_state": status.inference_state,
                    "segmentation_details": status.segmentation,
                    "detection_details": status.detection,
                }
            )
        },
    )


def test_summary_retains_an_earlier_priority_after_the_latest_frame_is_empty() -> None:
    status = _status()
    observed = _with_status(get_demo_snapshots()[-1], status)
    latest = observed.model_copy(
        deep=True,
        update={
            "frame_id": observed.frame_id + 1,
            "zones": [],
            "highest_priority_zone_id": None,
            "incident_severity": Severity.LOW,
            "route": None,
        },
    )
    aggregator = VideoAnalysisAggregator("video-analysis-session-0001")

    aggregator.add(observed, media_time_ms=1_000)
    aggregator.add(latest, media_time_ms=2_000)
    summary = aggregator.build_summary(
        frames_accepted=2,
        frames_dropped=0,
        model_status=status,
    )

    assert aggregator.latest_result is latest
    assert summary.highest_priority_zone_id == observed.zones[0].zone_id
    assert summary.incident_severity is Severity.CRITICAL
    assert summary.priorities[0].source_frame_id == observed.frame_id
    assert summary.priorities[0].media_time_ms == 1_000
    assert summary.priorities[0].segmentation_evidence_available is True
    assert summary.priorities[0].detection_evidence_available is True
    assert (
        summary.priorities[0].building_damage_count_availability
        is AggregateMetricAvailability.AVAILABLE
    )
    report = build_video_analysis_report(
        VideoAnalysisComplete(
            session_id=summary.session_id,
            summary=summary,
            latest_result=latest,
        )
    )
    assert report.analysis_scope == "WHOLE_VIDEO"
    assert report.highest_priority_zone_id == observed.zones[0].zone_id
    assert report.severity_established is True


def test_unavailable_segmentation_is_nullable_while_direct_detections_are_aggregated() -> None:
    status = _status()
    snapshot = _with_status(get_demo_snapshots()[-1], status)
    detections = [
        item.model_copy(
            update={
                "label": "person",
                "source_class": "person",
                "observation_state": "DETECTED",
            }
        )
        for item in snapshot.detections
        if item.category is DetectionCategory.PERSON
    ][:2]
    snapshot = snapshot.model_copy(
        update={
            "detections": detections,
            "zones": [],
            "highest_priority_zone_id": None,
            "evidence_frames": EvidenceFrames(
                detection_source_frame_id=snapshot.frame_id,
                segmentation_source_frame_id=None,
            ),
        }
    )
    aggregator = VideoAnalysisAggregator("video-analysis-session-0002")

    aggregator.add(snapshot, media_time_ms=5_000)
    summary = aggregator.build_summary(
        frames_accepted=1,
        frames_dropped=0,
        model_status=status,
    )

    assert summary.statistics.people_detected.value == 2
    assert summary.detected_classes[0].label == "person"
    assert summary.statistics.flooded_area_percent.value is None
    assert (
        summary.statistics.flooded_area_percent.availability
        is AggregateMetricAvailability.MODEL_UNAVAILABLE
    )
    assert summary.statistics.blocked_road_cells.value is None
    assert summary.statistics.building_damage_coverage_percent.value is None
    assert (
        summary.statistics.damaged_buildings.availability
        is AggregateMetricAvailability.NOT_SUPPORTED
    )


def test_multiframe_aggregation_uses_fresh_peaks_and_strongest_zone_observation() -> None:
    status = _status(segmentation=ModelState.READY)
    base = _with_status(get_demo_snapshots()[-1], status)
    person_template = next(
        item for item in base.detections if item.category is DetectionCategory.PERSON
    )
    vehicle_template = next(
        item for item in base.detections if item.category is DetectionCategory.VEHICLE
    )

    def detection(template, detection_id: str, track_id: str, state: str):
        return template.model_copy(
            update={
                "detection_id": detection_id,
                "track_id": track_id,
                "label": "person"
                if template.category is DetectionCategory.PERSON
                else "car",
                "source_class": "person"
                if template.category is DetectionCategory.PERSON
                else "car",
                "observation_state": state,
            }
        )

    def frame(
        frame_id: int,
        *,
        detections,
        detection_source: int,
        detection_reused: bool,
        segmentation_source: int,
        segmentation_reused: bool,
        flood: float,
        blocked: float,
        damage: float,
        priority: float,
    ):
        zone = base.zones[0].model_copy(
            update={
                "priority_score": priority,
                "severity": Severity.CRITICAL if priority >= 80 else Severity.MODERATE,
                "building_damage_count": 0,
                "data_origin": DataOrigin.DERIVED_ANALYTIC,
            }
        )
        return base.model_copy(
            deep=True,
            update={
                "frame_id": frame_id,
                "detections": detections,
                "zones": [zone],
                "highest_priority_zone_id": zone.zone_id,
                "segmentation": base.segmentation.model_copy(
                    update={"status": SegmentationState.READY}
                ),
                "statistics": base.statistics.model_copy(
                    update={
                        "flooded_area_percent": base.statistics.flooded_area_percent.model_copy(
                            update={"value": flood}
                        ),
                        "blocked_roads": base.statistics.blocked_roads.model_copy(
                            update={"value": blocked}
                        ),
                    }
                ),
                "scene_summary": SceneSummary(
                    water_flood_coverage_percent=flood,
                    pool_coverage_percent=0,
                    road_clear_coverage_percent=0,
                    road_flooded_coverage_percent=0,
                    road_blocked_coverage_percent=blocked,
                    building_damage_coverage_percent=damage,
                    provenance=[FeatureProvenance.SEGMENTATION],
                    data_origin=DataOrigin.DERIVED_ANALYTIC,
                ),
                "evidence_frames": EvidenceFrames(
                    detection_source_frame_id=detection_source,
                    segmentation_source_frame_id=segmentation_source,
                    detection_reused=detection_reused,
                    segmentation_reused=segmentation_reused,
                ),
            },
        )

    observed_person = detection(person_template, "p-1", "person-track-1", "DETECTED")
    persisted_person = detection(
        person_template, "p-persisted", "person-track-2", "TRACK_PERSISTED"
    )
    observed_vehicle = detection(vehicle_template, "v-1", "vehicle-track-1", "DETECTED")
    first = frame(
        1,
        detections=[observed_person, persisted_person, observed_vehicle],
        detection_source=1,
        detection_reused=False,
        segmentation_source=1,
        segmentation_reused=False,
        flood=20,
        blocked=1,
        damage=5,
        priority=40,
    )
    reused = frame(
        2,
        detections=[observed_person, persisted_person, observed_vehicle],
        detection_source=1,
        detection_reused=True,
        segmentation_source=1,
        segmentation_reused=True,
        flood=99,
        blocked=9,
        damage=99,
        priority=55,
    )
    second = frame(
        3,
        detections=[
            observed_person,
            detection(person_template, "p-2", "person-track-2", "DETECTED"),
            observed_vehicle,
            detection(vehicle_template, "v-2", "vehicle-track-2", "DETECTED"),
        ],
        detection_source=3,
        detection_reused=False,
        segmentation_source=3,
        segmentation_reused=False,
        flood=60,
        blocked=3,
        damage=15,
        priority=96,
    )
    detection_only_status = _status(segmentation=ModelState.UNAVAILABLE)
    strongest = _with_status(
        frame(
            4,
            detections=[observed_person],
            detection_source=4,
            detection_reused=False,
            segmentation_source=3,
            segmentation_reused=True,
            flood=95,
            blocked=8,
            damage=70,
            priority=99,
        ),
        detection_only_status,
    ).model_copy(
        update={
            "segmentation": base.segmentation.model_copy(
                update={"status": SegmentationState.NOT_CONFIGURED}
            ),
            "evidence_frames": EvidenceFrames(
                detection_source_frame_id=4,
                segmentation_source_frame_id=None,
                detection_reused=False,
                segmentation_reused=False,
            ),
        }
    )
    detection_only_zone = strongest.zones[0].model_copy(
        update={
            "flood_coverage_percent": 0,
            "building_damage_coverage_percent": 0,
            "pool_coverage_percent": 0,
            "road_condition": RoadState.UNKNOWN,
            "access_status": AccessStatus.UNKNOWN,
            "primary_reason": "Direct person evidence retained for this test observation",
            "reasons": [
                reason
                for reason in strongest.zones[0].reasons
                if reason.code == "PERSON_EVIDENCE"
            ],
        }
    )
    strongest = strongest.model_copy(
        update={"zones": [detection_only_zone], "route": None}
    )
    aggregator = VideoAnalysisAggregator("video-analysis-session-0004")

    aggregator.add(first, media_time_ms=1_000)
    aggregator.add(reused, media_time_ms=2_000)
    aggregator.add(second, media_time_ms=3_000)
    aggregator.add(strongest, media_time_ms=4_000)
    summary = aggregator.build_summary(
        frames_accepted=4,
        frames_dropped=0,
        model_status=status,
    )

    assert summary.statistics.people_detected.value == 2
    assert summary.statistics.people_detected.supporting_frame_count == 3
    assert summary.statistics.vehicles_detected.value == 2
    assert summary.statistics.flooded_area_percent.value == 60
    assert summary.statistics.flooded_area_percent.supporting_frame_count == 2
    assert summary.statistics.blocked_road_cells.value == 3
    assert summary.statistics.building_damage_coverage_percent.value == 15
    assert summary.priorities[0].zone.priority_score == 99
    assert summary.priorities[0].zone.rank == 1
    assert summary.priorities[0].source_frame_id == 4
    assert summary.priorities[0].supporting_update_count == 4
    assert summary.priorities[0].segmentation_evidence_available is False
    assert summary.priorities[0].detection_evidence_available is True
    assert (
        summary.priorities[0].building_damage_count_availability
        is AggregateMetricAvailability.NOT_SUPPORTED
    )


def test_equal_strength_priority_prefers_the_fully_evidenced_observation() -> None:
    ready = _status(segmentation=ModelState.READY)
    base = _with_status(get_demo_snapshots()[-1], ready)
    full = base.model_copy(
        deep=True,
        update={
            "frame_id": 10,
            "segmentation": base.segmentation.model_copy(
                update={"status": SegmentationState.READY}
            ),
        },
    )
    detection_only_status = _status(segmentation=ModelState.UNAVAILABLE)
    partial_zone = full.zones[0].model_copy(
        update={
            "flood_coverage_percent": 0,
            "building_damage_coverage_percent": 0,
            "pool_coverage_percent": 0,
            "road_condition": RoadState.UNKNOWN,
            "access_status": AccessStatus.UNKNOWN,
            "reasons": [
                reason for reason in full.zones[0].reasons if reason.code == "PERSON_EVIDENCE"
            ],
        }
    )
    partial = _with_status(
        full.model_copy(
            deep=True,
            update={
                "frame_id": 11,
                "zones": [partial_zone],
                "route": None,
                "segmentation": base.segmentation.model_copy(
                    update={"status": SegmentationState.NOT_CONFIGURED}
                ),
            },
        ),
        detection_only_status,
    )
    aggregator = VideoAnalysisAggregator("evidence-tie-session-0001")

    aggregator.add(full, media_time_ms=1_000)
    aggregator.add(partial, media_time_ms=1_100)
    summary = aggregator.build_summary(
        frames_accepted=2,
        frames_dropped=0,
        model_status=ready,
    )

    assert summary.priorities[0].source_frame_id == 10
    assert summary.priorities[0].segmentation_evidence_available is True
    assert summary.priorities[0].detection_evidence_available is True


def test_aggregator_is_bounded_across_batched_zones_classes_and_long_video() -> None:
    status = _status(segmentation=ModelState.READY)
    base = _with_status(get_demo_snapshots()[-1], status)
    base = base.model_copy(
        update={
            "segmentation": base.segmentation.model_copy(
                update={"status": SegmentationState.READY}
            )
        }
    )
    zone_template = base.zones[0].model_copy(
        update={
            "building_damage_count": 0,
            "data_origin": DataOrigin.DERIVED_ANALYTIC,
        }
    )

    def zones(start: int, count: int):
        return [
            zone_template.model_copy(
                update={
                    "zone_id": f"ZONE-{index:03d}",
                    "display_name": f"Zone {index:03d}",
                }
            )
            for index in range(start, start + count)
        ]

    first = base.model_copy(
        deep=True,
        update={
            "frame_id": 1,
            "zones": zones(0, MAX_VIDEO_PRIORITY_OBSERVATIONS),
            "evidence_frames": EvidenceFrames(
                detection_source_frame_id=1,
                segmentation_source_frame_id=1,
            ),
        },
    )
    detection_template = next(
        item for item in base.detections if item.category is DetectionCategory.VEHICLE
    )
    many_classes = [
        detection_template.model_copy(
            update={
                "detection_id": f"vehicle-{index}",
                "track_id": f"vehicle-track-{index}",
                "source_class": f"vehicle-class-{index}",
                "observation_state": "DETECTED",
            }
        )
        for index in range(MAX_VIDEO_DETECTED_CLASSES + 12)
    ]
    second = base.model_copy(
        deep=True,
        update={
            "frame_id": 2,
            "zones": zones(MAX_VIDEO_PRIORITY_OBSERVATIONS, 12),
            "detections": many_classes,
            "evidence_frames": EvidenceFrames(
                detection_source_frame_id=2,
                segmentation_source_frame_id=2,
            ),
        },
    )
    aggregator = VideoAnalysisAggregator("bounded-video-session-0001")
    aggregator.add(first, media_time_ms=100)
    aggregator.add(second, media_time_ms=200)

    for frame_id in range(3, 103):
        result = base.model_copy(
            deep=True,
            update={
                "frame_id": frame_id,
                "zones": [],
                "detections": [],
                "evidence_frames": EvidenceFrames(
                    detection_source_frame_id=frame_id,
                    segmentation_source_frame_id=frame_id,
                ),
            },
        )
        aggregator.add(result, media_time_ms=frame_id * 100)

    summary = aggregator.build_summary(
        frames_accepted=102,
        frames_dropped=0,
        model_status=status,
    )

    assert len(summary.priorities) == MAX_VIDEO_PRIORITY_OBSERVATIONS
    assert summary.priorities_truncated is True
    assert len(summary.detected_classes) == MAX_VIDEO_DETECTED_CLASSES
    assert summary.detected_classes_truncated is True
    assert aggregator._last_detection_source_frame_id == 102
    assert aggregator._last_segmentation_source_frame_id == 102
    assert not hasattr(aggregator, "_detection_source_frames")
    assert not hasattr(aggregator, "_segmentation_source_frames")
    report = build_video_analysis_report(
        VideoAnalysisComplete(
            session_id=summary.session_id,
            summary=summary,
            latest_result=aggregator.latest_result,
        )
    )
    assert report.priorities_truncated is True


class _DelayedPipeline:
    def __init__(self, release: threading.Event) -> None:
        self.release = release
        self.started = threading.Event()
        self.result = _with_status(get_demo_snapshots()[-1], _status())

    def should_process(self, frame_id: int) -> bool:
        return frame_id >= 0

    def process(self, **_: object):
        self.started.set()
        assert self.release.wait(timeout=2)
        return self.result

    def status(self) -> ModelStatusResponse:
        return _status()

    def close_session(self, session_id: str) -> None:
        del session_id


class _ObservedCoordinator(InferenceCoordinator):
    def __init__(self, pipeline) -> None:
        super().__init__(pipeline)
        self.finalize_started = threading.Event()

    async def finalize(self, *args, **kwargs):
        self.finalize_started.set()
        return await super().finalize(*args, **kwargs)


async def test_coordinator_finalize_drains_pending_work_and_is_idempotent() -> None:
    release = threading.Event()
    pipeline = _DelayedPipeline(release)
    coordinator = InferenceCoordinator(pipeline)  # type: ignore[arg-type]
    metadata = FrameMetadata(
        frame_id=4,
        captured_at_ms=10_000,
        media_time_ms=8_000,
        source_mode=SourceMode.VIDEO_FILE,
        media_origin="USER_VIDEO_FILE",
        mime_type="image/jpeg",
        byte_length=10,
        width=2,
        height=2,
    )

    async def publish(_message) -> None:
        return None

    submitted = coordinator.submit(
        session_id="video-analysis-session-0003",
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        metadata=metadata,
        callback=publish,
        detector_mode=DetectorInferenceMode.STANDARD,
    )
    assert submitted is True
    finalizing = asyncio.create_task(
        coordinator.finalize(
            "video-analysis-session-0003",
            frames_accepted=1,
            frames_dropped=0,
        )
    )
    await asyncio.to_thread(pipeline.started.wait, 1)
    assert finalizing.done() is False

    release.set()
    first = await finalizing
    second = await coordinator.finalize(
        "video-analysis-session-0003",
        frames_accepted=99,
        frames_dropped=99,
    )

    assert first == second
    assert first.summary.frames_analyzed == 1
    assert first.summary.frames_accepted == 1
    assert first.summary.last_media_time_ms == 8_000
    assert first.latest_result is not None


async def test_coordinator_finalizes_when_intelligence_transport_disconnects() -> None:
    release = threading.Event()
    release.set()
    pipeline = _DelayedPipeline(release)
    coordinator = InferenceCoordinator(pipeline)  # type: ignore[arg-type]
    metadata = FrameMetadata(
        frame_id=5,
        captured_at_ms=11_000,
        media_time_ms=9_000,
        source_mode=SourceMode.VIDEO_FILE,
        media_origin="USER_VIDEO_FILE",
        mime_type="image/jpeg",
        byte_length=10,
        width=2,
        height=2,
    )

    async def disconnected_publish(_message) -> None:
        raise WebSocketDisconnect(code=1006)

    assert coordinator.submit(
        session_id="video-analysis-session-transport-loss",
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        metadata=metadata,
        callback=disconnected_publish,
        detector_mode=DetectorInferenceMode.STANDARD,
    )

    completed = await coordinator.finalize(
        "video-analysis-session-transport-loss",
        frames_accepted=1,
        frames_dropped=0,
    )

    assert completed.state is IngestionSessionState.COMPLETE
    assert completed.summary.frames_analyzed == 1
    assert completed.summary.last_media_time_ms == 9_000
    assert completed.latest_result is not None


def test_completion_immediately_after_ack_includes_the_acknowledged_frame() -> None:
    application = create_app()
    release = threading.Event()
    pipeline = _DelayedPipeline(release)
    coordinator = _ObservedCoordinator(pipeline)  # type: ignore[arg-type]
    application.state.inference_pipeline = pipeline
    application.state.inference_coordinator = coordinator
    image = np.full((8, 12, 3), 120, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    payload = encoded.tobytes()
    completion_response: list[HttpxResponse] = []

    with TestClient(application) as local_client:
        session = local_client.post(
            "/api/ingest/sessions",
            json={"source_mode": "VIDEO_FILE", "media_origin": "USER_VIDEO_FILE"},
        ).json()
        with local_client.websocket_connect(
            f"/ws/ingest/sessions/{session['session_id']}/frames"
        ) as socket:
            socket.send_json(
                {
                    "type": "frame_metadata",
                    "frame_id": 0,
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
            assert acknowledgement["accepted"] is True
            assert pipeline.started.wait(timeout=1)

            def complete() -> None:
                completion_response.append(
                    local_client.post(
                        f"/api/ingest/sessions/{session['session_id']}/complete"
                    )
                )

            completion_thread = threading.Thread(target=complete)
            completion_thread.start()
            assert coordinator.finalize_started.wait(timeout=1)
            release.set()
            completion_thread.join(timeout=3)

    assert completion_thread.is_alive() is False
    assert len(completion_response) == 1
    response = completion_response[0]
    assert response.status_code == 200
    assert response.json()["summary"]["frames_analyzed"] == 1
    assert response.json()["summary"]["frames_accepted"] == 1


def test_acknowledged_work_survives_websocket_disconnect_until_completion() -> None:
    application = create_app()
    release = threading.Event()
    pipeline = _DelayedPipeline(release)
    coordinator = _ObservedCoordinator(pipeline)  # type: ignore[arg-type]
    application.state.inference_pipeline = pipeline
    application.state.inference_coordinator = coordinator
    image = np.full((8, 12, 3), 120, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    payload = encoded.tobytes()
    completion_response: list[HttpxResponse] = []

    with TestClient(application) as local_client:
        session = local_client.post(
            "/api/ingest/sessions",
            json={"source_mode": "VIDEO_FILE", "media_origin": "USER_VIDEO_FILE"},
        ).json()
        with local_client.websocket_connect(
            f"/ws/ingest/sessions/{session['session_id']}/frames"
        ) as socket:
            socket.send_json(
                {
                    "type": "frame_metadata",
                    "frame_id": 0,
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
            assert socket.receive_json()["accepted"] is True
            assert pipeline.started.wait(timeout=1)

        def complete() -> None:
            completion_response.append(
                local_client.post(
                    f"/api/ingest/sessions/{session['session_id']}/complete"
                )
            )

        completion_thread = threading.Thread(target=complete)
        completion_thread.start()
        assert coordinator.finalize_started.wait(timeout=1)
        release.set()
        completion_thread.join(timeout=3)

    assert completion_thread.is_alive() is False
    assert completion_response[0].status_code == 200
    assert completion_response[0].json()["summary"]["frames_accepted"] == 1
    assert completion_response[0].json()["summary"]["frames_analyzed"] == 1


def test_finalizing_session_rejects_new_frame_without_mutating_counters() -> None:
    application = create_app()
    release = threading.Event()
    pipeline = _DelayedPipeline(release)
    coordinator = _ObservedCoordinator(pipeline)  # type: ignore[arg-type]
    application.state.inference_pipeline = pipeline
    application.state.inference_coordinator = coordinator
    image = np.full((8, 12, 3), 120, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    payload = encoded.tobytes()
    completion_response: list[HttpxResponse] = []

    with TestClient(application) as local_client:
        session = local_client.post(
            "/api/ingest/sessions",
            json={"source_mode": "VIDEO_FILE", "media_origin": "USER_VIDEO_FILE"},
        ).json()
        with local_client.websocket_connect(
            f"/ws/ingest/sessions/{session['session_id']}/frames"
        ) as socket:
            socket.send_json(
                {
                    "type": "frame_metadata",
                    "frame_id": 0,
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
            assert socket.receive_json()["accepted"] is True
            assert pipeline.started.wait(timeout=1)

            def complete() -> None:
                completion_response.append(
                    local_client.post(
                        f"/api/ingest/sessions/{session['session_id']}/complete"
                    )
                )

            completion_thread = threading.Thread(target=complete)
            completion_thread.start()
            assert coordinator.finalize_started.wait(timeout=1)
            socket.send_json(
                {
                    "type": "frame_metadata",
                    "frame_id": 1,
                    "captured_at_ms": 200,
                    "media_time_ms": 200,
                    "source_mode": "VIDEO_FILE",
                    "media_origin": "USER_VIDEO_FILE",
                    "mime_type": "image/jpeg",
                    "byte_length": len(payload),
                    "width": 12,
                    "height": 8,
                }
            )
            rejected = socket.receive_json()
            assert rejected["accepted"] is False
            assert rejected["code"] == "analysis_closed"
            release.set()
            completion_thread.join(timeout=3)

        current = local_client.get(
            f"/api/ingest/sessions/{session['session_id']}"
        ).json()

    assert completion_thread.is_alive() is False
    assert completion_response[0].json()["summary"]["frames_accepted"] == 1
    assert completion_response[0].json()["summary"]["frames_analyzed"] == 1
    assert current["state"] == "COMPLETE"
    assert current["counters"]["frames_received"] == 1
    assert current["counters"]["frames_accepted"] == 1
    assert current["counters"]["protocol_errors"] == 0


async def test_zero_result_completion_is_explicit_available_and_idempotent(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/ingest/sessions",
        json={"source_mode": "VIDEO_FILE", "media_origin": "USER_VIDEO_FILE"},
    )
    session_id = created.json()["session_id"]

    first = await client.post(f"/api/ingest/sessions/{session_id}/complete")
    second = await client.post(f"/api/ingest/sessions/{session_id}/complete")
    session = await client.get(f"/api/ingest/sessions/{session_id}")
    report = await client.get(f"/api/ingest/sessions/{session_id}/report")

    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["state"] == "COMPLETE"
    assert first.json()["latest_result"] is None
    assert first.json()["summary"]["frames_analyzed"] == 0
    assert first.json()["summary"]["incident_severity"] is None
    assert session.json()["state"] == "COMPLETE"
    assert report.json()["analysis_scope"] == "WHOLE_VIDEO"
    assert report.json()["severity_established"] is False
    assert report.json()["explanation"] == (
        "No frames were analyzed; rescue priority was not established."
    )
    assert report.json()["access_summary"] == (
        "No frames were analyzed; relative access was not assessed."
    )
    assert (
        report.json()["aggregate_availability"]["flooded_area_percent"]
        == "MODEL_UNAVAILABLE"
    )
    registry = Registry().with_resource(
        LIVE_SCHEMA["$id"], Resource.from_contents(LIVE_SCHEMA)
    )
    Draft202012Validator(COMPLETION_SCHEMA, registry=registry).validate(first.json())


def test_complete_session_rejects_websocket_intake_without_reopening_it() -> None:
    application = create_app()

    with TestClient(application) as local_client:
        session = local_client.post(
            "/api/ingest/sessions",
            json={"source_mode": "VIDEO_FILE", "media_origin": "USER_VIDEO_FILE"},
        ).json()
        completed = local_client.post(
            f"/api/ingest/sessions/{session['session_id']}/complete"
        )
        assert completed.status_code == 200
        with local_client.websocket_connect(
            f"/ws/ingest/sessions/{session['session_id']}/frames"
        ) as socket:
            rejected = socket.receive_json()
        current = local_client.get(
            f"/api/ingest/sessions/{session['session_id']}"
        ).json()

    assert rejected["accepted"] is False
    assert rejected["code"] == "analysis_closed"
    assert current["state"] == "COMPLETE"
    assert current["counters"]["frames_received"] == 0
    assert current["counters"]["frames_accepted"] == 0


async def test_delete_is_rejected_while_analysis_is_finalizing() -> None:
    application = create_app()
    manager = application.state.ingestion_manager
    session = manager.create(
        IngestionSessionCreate(
            source_mode=SourceMode.VIDEO_FILE,
            media_origin=MediaOrigin.USER_VIDEO_FILE,
        )
    )
    record = manager.get_record(session.session_id)
    manager.touch(record, IngestionSessionState.FINALIZING)
    transport = ASGITransport(app=application)

    async with AsyncClient(transport=transport, base_url="http://testserver") as local_client:
        response = await local_client.delete(
            f"/api/ingest/sessions/{session.session_id}"
        )
        report = await local_client.get(
            f"/api/ingest/sessions/{session.session_id}/report"
        )
        retained = await local_client.get(f"/api/ingest/sessions/{session.session_id}")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "analysis_finalizing"
    assert report.status_code == 409
    assert report.json()["error"]["code"] == "analysis_finalizing"
    assert retained.json()["state"] == "FINALIZING"
