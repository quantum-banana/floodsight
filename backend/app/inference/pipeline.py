import base64
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.config import PROJECT_ROOT, Settings
from app.inference.contracts import (
    DetectionObservationState,
    DetectionProvenance,
    DetectionResult,
    DetectorInferenceMode,
    FusedScene,
    SegmentationProvenance,
    SegmentationResult,
    decode_mask,
)
from app.inference.fusion import SceneFusionEngine
from app.inference.model_registry import (
    ModelRegistry,
    ModelType,
    RegistryProvenance,
    ResolvedModel,
)
from app.inference.segformer import SegFormerAdapter
from app.inference.taxonomy import Taxonomy, load_taxonomy
from app.inference.tracking import TemporalDetectionTracker
from app.inference.yolo import YoloAdapter
from app.intelligence.contracts import GridCellEvidence
from app.intelligence.priority import PriorityEngine
from app.intelligence.routing import AccessibilityEngine, RoutingEngine
from app.intelligence.temporal import TemporalRoadTracker, TemporalZoneTracker
from app.intelligence.zones import VEHICLE_CLASSES, RescueZoneEngine
from app.schemas.live_result import (
    ApiState,
    CoordinateSpace,
    DataOrigin,
    Detection,
    DetectionCategory,
    EventCategory,
    EventSeverity,
    EvidenceFrames,
    IncidentEvent,
    IncidentMetadata,
    LiveResult,
    Metric,
    MetricUnit,
    Road,
    Route,
    SceneSummary,
    Segmentation,
    SegmentationClass,
    SegmentationMask,
    SegmentationState,
    Severity,
    SourceDimensions,
    SourceMode,
    Statistics,
    StreamState,
    SystemStatus,
    Zone,
)
from app.schemas.model_status import (
    InferenceState,
    ModelOperationalMode,
    ModelState,
    ModelStatus,
    ModelStatusResponse,
)


@dataclass(slots=True)
class _SessionState:
    tracker: TemporalZoneTracker
    road_tracker: TemporalRoadTracker
    object_tracker: TemporalDetectionTracker
    detector_mode: DetectorInferenceMode
    router: RoutingEngine = field(default_factory=RoutingEngine)
    last_segmentation: SegmentationResult | None = None
    last_detection: DetectionResult | None = None
    previous: LiveResult | None = None
    events: list[IncidentEvent] = field(default_factory=list)


class InferencePipeline:
    """Backend source of truth from transient frame through rescue decisions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        registry_path = Path(settings.model_registry_path)
        if not registry_path.is_absolute():
            registry_path = PROJECT_ROOT / registry_path
        self.registry_error: str | None = None
        try:
            self.registry = ModelRegistry(
                path=registry_path,
                project_root=PROJECT_ROOT,
                environment={
                    "FLOODSIGHT_SEGMENTATION_CHECKPOINT": settings.segmentation_checkpoint,
                    "FLOODSIGHT_DETECTION_CHECKPOINT": settings.detection_checkpoint,
                    "FLOODSIGHT_DETECTION_FALLBACK_CHECKPOINT": (
                        settings.detection_fallback_checkpoint
                    ),
                },
            )
        except (ValueError, OSError) as exc:
            self.registry = None
            self.registry_error = type(exc).__name__
        self.segmentation_taxonomy = load_taxonomy(
            PROJECT_ROOT / "shared/taxonomy/segmentation-taxonomy-v2.yaml",
            expected_version="segmentation-taxonomy-v2",
        )
        self.fusion = SceneFusionEngine(self.segmentation_taxonomy)
        self.zone_engine = RescueZoneEngine(self.segmentation_taxonomy)
        self.priority_engine = PriorityEngine()
        self.accessibility_engine = AccessibilityEngine()
        self.segmentation_adapter: SegFormerAdapter | None = None
        self.detection_adapter: YoloAdapter | None = None
        self._segmentation_status = _unavailable_status("Segmentation model is not loaded.")
        self._detection_status = _unavailable_status("Detection model is not loaded.")
        self._initialized = False
        self._loading = False
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionState] = {}

    def initialize(self) -> None:
        if self._initialized or self._loading:
            return
        self._loading = True
        try:
            if self.registry is None:
                self._segmentation_status = _error_status("Model registry could not be loaded.")
                self._detection_status = _error_status("Model registry could not be loaded.")
                return
            self.segmentation_adapter, self._segmentation_status = self._load_segmentation()
            self.detection_adapter, self._detection_status = self._load_detection()
        finally:
            self._loading = False
            self._initialized = True

    def _load_segmentation(self) -> tuple[SegFormerAdapter | None, ModelStatus]:
        records = self.registry.enabled(ModelType.SEGMENTATION) if self.registry else []
        if not records:
            return None, _unavailable_status("No enabled segmentation model record.")
        last_model = records[0]
        for model in records:
            last_model = model
            try:
                adapter = SegFormerAdapter(
                    model,
                    device=self.settings.inference_device,
                    precision=self.settings.inference_precision,
                    inference_resolution=self.settings.inference_resolution,
                )
                adapter.load()
                return adapter, _ready_status(model, adapter.runtime.device)
            except (FileNotFoundError, ImportError, RuntimeError, ValueError):
                continue
        return None, _unavailable_model_status(
            last_model, "Configured segmentation artifact or runtime is unavailable."
        )

    def _load_detection(self) -> tuple[YoloAdapter | None, ModelStatus]:
        records = self.registry.enabled(ModelType.DETECTION) if self.registry else []
        records.sort(
            key=lambda item: item.record.provenance is RegistryProvenance.PRETRAINED_FALLBACK
        )
        if not records:
            return None, _unavailable_status("No enabled detection model record.")
        last_model = records[0]
        for model in records:
            last_model = model
            try:
                adapter = YoloAdapter(
                    model,
                    device=self.settings.inference_device,
                    precision=self.settings.inference_precision,
                    inference_resolution=self.settings.detection_standard_resolution,
                    confidence_threshold=self.settings.detection_confidence_threshold,
                    iou_threshold=self.settings.detection_iou_threshold,
                    aerial_inference_resolution=self.settings.detection_aerial_resolution,
                    aerial_tile_overlap=self.settings.detection_aerial_tile_overlap,
                    aerial_fusion_iou_threshold=(
                        self.settings.detection_aerial_fusion_iou_threshold
                    ),
                    aerial_high_recall_resolution=(
                        self.settings.detection_aerial_high_recall_resolution
                    ),
                    aerial_high_recall_tile_overlap=(
                        self.settings.detection_aerial_high_recall_tile_overlap
                    ),
                    aerial_high_recall_person_confidence=(
                        self.settings.detection_aerial_high_recall_person_confidence
                    ),
                    segformer_reinspection_enabled=(
                        self.settings.detection_segformer_reinspection_enabled
                    ),
                    segformer_reinspection_padding=(
                        self.settings.detection_segformer_reinspection_padding
                    ),
                    segformer_reinspection_min_pixels=(
                        self.settings.detection_segformer_reinspection_min_pixels
                    ),
                )
                adapter.load()
                return adapter, _ready_status(model, adapter.runtime.device)
            except (FileNotFoundError, ImportError, RuntimeError, ValueError):
                continue
        return None, _unavailable_model_status(
            last_model, "Configured detection artifact or runtime is unavailable."
        )

    def status(self) -> ModelStatusResponse:
        if self._loading:
            state = InferenceState.MODEL_LOADING
        elif self.registry_error:
            state = InferenceState.ERROR
        else:
            ready = sum(
                status.status is ModelState.READY
                for status in (self._segmentation_status, self._detection_status)
            )
            state = (
                InferenceState.LIVE
                if ready == 2
                else InferenceState.DEGRADED
                if ready == 1
                else InferenceState.MODEL_UNAVAILABLE
            )
        return ModelStatusResponse(
            segmentation=self._segmentation_status,
            detection=self._detection_status,
            inference_state=state,
        )

    def can_process(self) -> bool:
        return self._initialized and (
            self.segmentation_adapter is not None or self.detection_adapter is not None
        )

    def close_session(self, session_id: str) -> None:
        """Release temporal and routing state when an ingestion session is forgotten."""
        self._sessions.pop(session_id, None)

    def should_process(self, frame_id: int) -> bool:
        return frame_id % self.settings.inference_frame_stride == 0 and self.can_process()

    def process(
        self,
        *,
        session_id: str,
        frame_bgr: NDArray[np.uint8],
        frame_id: int,
        timestamp_ms: int,
        source_mode: SourceMode,
        detector_mode: DetectorInferenceMode = DetectorInferenceMode.STANDARD,
    ) -> LiveResult | None:
        if not self.should_process(frame_id):
            return None
        detector_mode = DetectorInferenceMode(detector_mode)
        state = self._sessions.setdefault(
            session_id,
            _SessionState(
                tracker=TemporalZoneTracker(
                    window_ms=self.settings.temporal_window_ms,
                    track_ttl_ms=self.settings.temporal_track_ttl_ms,
                    urgent_person_confidence=self.settings.urgent_person_confidence,
                ),
                road_tracker=TemporalRoadTracker(window_ms=self.settings.temporal_window_ms),
                object_tracker=TemporalDetectionTracker(
                    track_ttl_ms=self.settings.temporal_track_ttl_ms
                ),
                detector_mode=detector_mode,
            ),
        )
        if state.detector_mode is not detector_mode:
            state.detector_mode = detector_mode
            state.last_detection = None
            state.object_tracker.reset()
        with self._lock:
            segmentation = self._run_segmentation(state, frame_bgr, frame_id, timestamp_ms)
            raw_detection = self._run_detection(
                state,
                frame_bgr,
                frame_id,
                timestamp_ms,
                detector_mode,
                segmentation,
            )
            detection = (
                state.object_tracker.update(
                    raw_detection,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    fresh_observation=(
                        raw_detection is not None and not raw_detection.reused_from_previous
                    ),
                )
                if detector_mode is DetectorInferenceMode.AERIAL_HIGH_RECALL
                else raw_detection
            )
        height, width = frame_bgr.shape[:2]
        scene = self.fusion.fuse(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            source_width=width,
            source_height=height,
            segmentation=segmentation,
            detection=detection,
        )
        raw_cells = self.zone_engine.cells(scene)
        cells = state.road_tracker.update(
            raw_cells,
            timestamp_ms,
            fresh_observation=(segmentation is not None and not segmentation.reused_from_previous),
        )
        candidates = self.zone_engine.candidates(
            cells,
            timestamp_ms,
            person_observation_fresh=(
                raw_detection is not None
                and not raw_detection.reused_from_previous
                and not any(
                    item.application_class == "person"
                    and item.observation_state is DetectionObservationState.TRACK_PERSISTED
                    for item in (detection.detections if detection is not None else [])
                )
            ),
        )
        operational = state.tracker.update(candidates, timestamp_ms)
        zones = self.priority_engine.prioritize(operational)
        graph = self.accessibility_engine.build(cells)
        roads = self.accessibility_engine.as_live_roads(graph)
        route = None
        alternatives = []
        if zones:
            route, alternatives = state.router.route(
                graph, zone_id=zones[0].zone_id, target_cells=zones[0].grid_cells
            )
        origin = _result_origin(segmentation, detection)
        events = self._events(state, zones, roads, route, timestamp_ms, origin)
        model_status = self.status()
        result = LiveResult(
            incident_id=f"LIVE-{session_id[:12]}",
            incident=IncidentMetadata(
                incident_id=f"LIVE-{session_id[:12]}",
                title="Live Frame Intelligence",
                location_label="Normalized image-space assessment",
                started_at_ms=timestamp_ms,
                coordinate_space=CoordinateSpace.NORMALIZED_IMAGE,
                data_origin=origin,
            ),
            frame_id=frame_id,
            snapshot_index=frame_id,
            snapshot_count=frame_id + 1,
            timestamp_ms=timestamp_ms,
            source_mode=source_mode,
            coordinate_space=CoordinateSpace.NORMALIZED_IMAGE,
            data_origin=origin,
            stream_state=StreamState.PLAYING,
            incident_severity=zones[0].severity if zones else Severity.LOW,
            highest_priority_zone_id=zones[0].zone_id if zones else None,
            system_status=SystemStatus(
                api=ApiState.OPERATIONAL,
                segmentation_model=model_status.segmentation.status,
                detection_model=model_status.detection.status,
                inference_state=model_status.inference_state,
                segmentation_details=model_status.segmentation,
                detection_details=model_status.detection,
            ),
            statistics=_statistics(scene, cells, zones, detection, origin),
            detections=_live_detections(detection),
            segmentation=_live_segmentation(segmentation, self.segmentation_taxonomy),
            roads=roads,
            zones=zones,
            events=events,
            route=route,
            route_alternatives=alternatives,
            scene_summary=_scene_summary(scene, cells, origin),
            source_dimensions=SourceDimensions(width=width, height=height),
            evidence_frames=EvidenceFrames(
                segmentation_source_frame_id=(
                    segmentation.source_frame_id if segmentation else None
                ),
                detection_source_frame_id=detection.source_frame_id if detection else None,
                segmentation_reused=(segmentation.reused_from_previous if segmentation else False),
                detection_reused=detection.reused_from_previous if detection else False,
            ),
        )
        state.previous = result
        return result

    def _run_segmentation(
        self,
        state: _SessionState,
        frame: NDArray[np.uint8],
        frame_id: int,
        timestamp_ms: int,
    ) -> SegmentationResult | None:
        adapter = self.segmentation_adapter
        if adapter is None:
            return None
        previous = state.last_segmentation
        should_run = frame_id % self.settings.segmentation_cadence == 0 or previous is None
        if (
            not should_run
            and previous
            and previous.source_width == frame.shape[1]
            and (previous.source_height == frame.shape[0])
        ):
            return previous.model_copy(
                update={
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp_ms,
                    "source_frame_id": (
                        previous.source_frame_id
                        if previous.source_frame_id is not None
                        else previous.frame_id
                    ),
                    "reused_from_previous": True,
                }
            )
        try:
            result = adapter.infer(frame, frame_id=frame_id, timestamp_ms=timestamp_ms)
        except (RuntimeError, ValueError):
            self.segmentation_adapter = None
            self._segmentation_status = _error_status("Segmentation inference failed.")
            return None
        state.last_segmentation = result
        self._segmentation_status = self._segmentation_status.model_copy(
            update={
                "latency_ms": result.inference_latency_ms,
                "last_successful_inference_ms": timestamp_ms,
                "device": result.device,
            }
        )
        return result

    def _run_detection(
        self,
        state: _SessionState,
        frame: NDArray[np.uint8],
        frame_id: int,
        timestamp_ms: int,
        detector_mode: DetectorInferenceMode,
        segmentation: SegmentationResult | None,
    ) -> DetectionResult | None:
        adapter = self.detection_adapter
        if adapter is None:
            return None
        previous = state.last_detection
        should_run = frame_id % self.settings.detection_cadence == 0 or previous is None
        if (
            not should_run
            and previous
            and previous.source_width == frame.shape[1]
            and (previous.source_height == frame.shape[0])
        ):
            return previous.model_copy(
                update={
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp_ms,
                    "source_frame_id": (
                        previous.source_frame_id
                        if previous.source_frame_id is not None
                        else previous.frame_id
                    ),
                    "reused_from_previous": True,
                }
            )
        try:
            result = adapter.infer(
                frame,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                detector_mode=detector_mode,
                segmentation=segmentation,
            )
        except (RuntimeError, ValueError):
            self.detection_adapter = None
            self._detection_status = _error_status("Detection inference failed.")
            return None
        state.last_detection = result
        self._detection_status = self._detection_status.model_copy(
            update={
                "latency_ms": result.inference_latency_ms,
                "last_successful_inference_ms": timestamp_ms,
                "device": result.device,
            }
        )
        return result

    def _events(
        self,
        state: _SessionState,
        zones: list[Zone],
        roads: list[Road],
        route: Route | None,
        timestamp_ms: int,
        origin: DataOrigin,
    ) -> list[IncidentEvent]:
        previous = state.previous
        messages: list[tuple[EventSeverity, EventCategory, str]] = []
        if zones and (previous is None or zones[0].zone_id != previous.highest_priority_zone_id):
            messages.append(
                (
                    EventSeverity.WARNING,
                    EventCategory.PRIORITY,
                    f"{zones[0].display_name} entered the highest rescue-priority position.",
                )
            )
        previous_people = previous.statistics.people_detected.value if previous else 0
        current_people = sum(zone.people_count for zone in zones)
        if current_people > previous_people:
            messages.append(
                (
                    EventSeverity.CRITICAL,
                    EventCategory.DETECTION,
                    "New person evidence escalated without temporal delay.",
                )
            )
        previous_alerts = (
            {
                zone.zone_id
                for zone in previous.zones
                if any(alert.code == "POTENTIAL_STRANDED_PERSON" for alert in zone.alerts)
            }
            if previous
            else set()
        )
        current_alerts = {
            zone.zone_id
            for zone in zones
            if any(alert.code == "POTENTIAL_STRANDED_PERSON" for alert in zone.alerts)
        }
        for zone_id in sorted(current_alerts - previous_alerts):
            messages.append(
                (
                    EventSeverity.CRITICAL,
                    EventCategory.DETECTION,
                    f"POTENTIAL_STRANDED_PERSON raised in {zone_id}; trained-personnel "
                    "review is required.",
                )
            )
        previous_blocked = (
            {road.road_id for road in previous.roads if not road.enabled} if previous else set()
        )
        current_blocked = {road.road_id for road in roads if not road.enabled}
        for edge_id in sorted(current_blocked - previous_blocked):
            messages.append(
                (EventSeverity.WARNING, EventCategory.ACCESS, f"{edge_id} became unavailable.")
            )
        if route is not None and route.changed_reason:
            messages.append((EventSeverity.WARNING, EventCategory.ROUTE, route.changed_reason))
        for offset, (severity, category, message) in enumerate(messages):
            state.events.append(
                IncidentEvent(
                    event_id=f"LIVE-{timestamp_ms}-{offset}",
                    timestamp_ms=timestamp_ms,
                    severity=severity,
                    category=category,
                    message=message,
                    data_origin=origin,
                    code=(
                        route.changed_reason_code
                        if category is EventCategory.ROUTE and route is not None
                        else "POTENTIAL_STRANDED_PERSON"
                        if category is EventCategory.DETECTION
                        and message.startswith("POTENTIAL_STRANDED_PERSON")
                        else None
                    ),
                )
            )
        state.events = state.events[-20:]
        return list(reversed(state.events))


def _ready_status(model: ResolvedModel, device: str) -> ModelStatus:
    provenance = model.record.provenance
    mode = (
        ModelOperationalMode.FALLBACK
        if provenance is RegistryProvenance.PRETRAINED_FALLBACK
        else ModelOperationalMode.SIMULATED
        if provenance is RegistryProvenance.SIMULATED
        else ModelOperationalMode.REAL
    )
    return ModelStatus(
        status=ModelState.READY,
        model=model.record.model_id,
        mode=mode,
        version=model.record.version,
        device=device,
        provenance_mode=provenance.value,
        message=(
            "Pretrained fallback; not the final VisDrone model."
            if mode is ModelOperationalMode.FALLBACK
            else "Model loaded from the local registry."
        ),
    )


def _unavailable_model_status(model: ResolvedModel, message: str) -> ModelStatus:
    return ModelStatus(
        status=ModelState.UNAVAILABLE,
        model=model.record.model_id,
        mode=ModelOperationalMode.UNAVAILABLE,
        version=model.record.version,
        provenance_mode=model.record.provenance.value,
        message=message,
    )


def _unavailable_status(message: str) -> ModelStatus:
    return ModelStatus(
        status=ModelState.UNAVAILABLE,
        model=None,
        mode=ModelOperationalMode.UNAVAILABLE,
        message=message,
    )


def _error_status(message: str) -> ModelStatus:
    return ModelStatus(
        status=ModelState.ERROR,
        model=None,
        mode=ModelOperationalMode.UNAVAILABLE,
        message=message,
    )


def _result_origin(
    segmentation: SegmentationResult | None, detection: DetectionResult | None
) -> DataOrigin:
    modes = [item.provenance_mode for item in (segmentation, detection) if item is not None]
    if modes and all(mode.value == "SIMULATED" for mode in modes):
        return DataOrigin.DEMO_SIMULATED
    return DataOrigin.DERIVED_ANALYTIC


def _statistics(
    scene: FusedScene,
    cells: list[GridCellEvidence],
    zones: list[object],
    detection: DetectionResult | None,
    origin: DataOrigin,
) -> Statistics:
    semantics = {item.class_name: item.coverage_percent for item in scene.semantic_evidence}
    flood = sum(semantics.get(name, 0.0) for name in ("water", "road_flooded", "building_flooded"))
    people = sum(item.application_class == "person" for item in scene.detections)
    vehicles = sum(item.application_class in VEHICLE_CLASSES for item in scene.detections)
    blocked_edges = sum(cell.road_state.value == "BLOCKED" for cell in cells)
    semantic_confidence = max((item.confidence for item in scene.semantic_evidence), default=0.0)
    detection_confidence = max((item.confidence for item in scene.detections), default=0.0)
    return Statistics(
        flooded_area_percent=Metric(
            value=min(100.0, flood),
            unit=MetricUnit.PERCENT,
            confidence=semantic_confidence or None,
            data_origin=origin,
        ),
        people_detected=Metric(
            value=people,
            unit=MetricUnit.COUNT,
            confidence=detection_confidence or None,
            data_origin=origin,
        ),
        vehicles_detected=Metric(
            value=vehicles,
            unit=MetricUnit.COUNT,
            confidence=detection_confidence or None,
            data_origin=origin,
        ),
        blocked_roads=Metric(
            value=blocked_edges,
            unit=MetricUnit.COUNT,
            confidence=semantic_confidence or None,
            data_origin=DataOrigin.DERIVED_ANALYTIC,
        ),
        damaged_buildings=Metric(
            value=0,
            unit=MetricUnit.COUNT,
            confidence=None,
            data_origin=DataOrigin.DERIVED_ANALYTIC,
        ),
    )


def _live_detections(result: DetectionResult | None) -> list[Detection]:
    if result is None:
        return []
    model_origin = (
        DataOrigin.DEMO_SIMULATED
        if result.provenance_mode is DetectionProvenance.SIMULATED
        else DataOrigin.REAL_ML_OUTPUT
    )
    return [
        Detection(
            detection_id=item.detection_id,
            category=DetectionCategory.PERSON
            if item.application_class == "person"
            else DetectionCategory.VEHICLE
            if item.application_class in VEHICLE_CLASSES
            else DetectionCategory.OTHER,
            label=item.application_class,
            confidence=item.confidence,
            bbox=item.bbox,
            data_origin=(
                DataOrigin.DERIVED_ANALYTIC
                if item.observation_state is DetectionObservationState.TRACK_PERSISTED
                else model_origin
            ),
            source_class=item.source_class,
            source_class_id=item.source_class_id,
            source_confidence=item.source_confidence,
            detection_confidence=item.confidence,
            track_id=item.track_id,
            track_confidence=item.track_confidence,
            persistence=item.persistence,
            observation_state=item.observation_state.value,
            source_frame_id=item.source_frame_id,
            model_id=result.model.model_id,
            model_provenance=result.provenance_mode.value,
        )
        for item in result.detections
    ]


def _live_segmentation(result: SegmentationResult | None, taxonomy: Taxonomy) -> Segmentation:
    if result is None:
        return Segmentation(
            status=SegmentationState.NOT_CONFIGURED, classes=[], regions=[], mask=None
        )
    origin = (
        DataOrigin.DEMO_SIMULATED
        if result.provenance_mode is SegmentationProvenance.SIMULATED
        else DataOrigin.REAL_ML_OUTPUT
    )
    colors = {item.class_id: item.color for item in taxonomy.classes}
    mask = decode_mask(result.mask)
    scale = min(1.0, 320 / max(mask.shape))
    if scale < 1:
        mask = cv2.resize(
            mask,
            (round(mask.shape[1] * scale), round(mask.shape[0] * scale)),
            interpolation=cv2.INTER_NEAREST,
        )
    palette = np.zeros((256, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        palette[class_id] = color
    rgb = palette[mask]
    success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Unable to encode segmentation overlay")
    return Segmentation(
        status=SegmentationState.READY,
        classes=[
            SegmentationClass(
                class_id=item.class_id,
                label=item.class_name,
                coverage_percent=item.coverage_percent,
                confidence=item.mean_confidence,
                data_origin=origin,
                color=list(colors[item.class_id]),
            )
            for item in result.class_statistics
        ],
        regions=[],
        mask=SegmentationMask(
            width=mask.shape[1],
            height=mask.shape[0],
            data=base64.b64encode(encoded.tobytes()).decode("ascii"),
        ),
    )


def _scene_summary(
    scene: FusedScene, cells: list[GridCellEvidence], origin: DataOrigin
) -> SceneSummary:
    semantics = {item.class_name: item.coverage_percent for item in scene.semantic_evidence}

    def average(attribute: str) -> float:
        return sum(getattr(cell, attribute) for cell in cells) / max(1, len(cells))

    flood_names = ("water", "road_flooded", "building_flooded")
    return SceneSummary(
        water_flood_coverage_percent=min(
            100.0,
            sum(semantics.get(name, 0.0) for name in flood_names),
        ),
        pool_coverage_percent=semantics.get("pool", 0.0),
        road_clear_coverage_percent=semantics.get("road_clear", 0.0),
        road_flooded_coverage_percent=semantics.get("road_flooded", 0.0),
        road_blocked_coverage_percent=semantics.get("road_blocked", 0.0),
        building_damage_coverage_percent=average("building_damage_coverage_percent"),
        provenance=scene.provenance,
        data_origin=origin,
    )
