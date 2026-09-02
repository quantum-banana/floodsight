from __future__ import annotations

import time
from dataclasses import dataclass

from app.schemas.ingestion import (
    MAX_VIDEO_DETECTED_CLASSES,
    MAX_VIDEO_PRIORITY_OBSERVATIONS,
    AggregateMetric,
    AggregateMetricAggregation,
    AggregateMetricAvailability,
    DetectedClassFinding,
    VideoAnalysisStatistics,
    VideoAnalysisSummary,
    VideoPriorityObservation,
)
from app.schemas.live_result import (
    DataOrigin,
    Detection,
    DetectionCategory,
    LiveResult,
    MetricUnit,
    Route,
    SegmentationState,
    Severity,
    Zone,
    ZoneReason,
)
from app.schemas.model_status import ModelState, ModelStatusResponse

MAX_PRIORITY_OBSERVATIONS = MAX_VIDEO_PRIORITY_OBSERVATIONS
MAX_DETECTED_CLASSES = MAX_VIDEO_DETECTED_CLASSES
_SEVERITY_ORDER = {
    Severity.LOW: 0,
    Severity.MODERATE: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


@dataclass(slots=True)
class _PeakMetric:
    value: float = 0
    confidence: float | None = None
    frame_count: int = 0

    def observe(self, value: float, confidence: float | None) -> None:
        self.frame_count += 1
        if value > self.value:
            self.value = value
            self.confidence = confidence
        elif value == self.value and confidence is not None:
            self.confidence = max(self.confidence or 0, confidence)


@dataclass(slots=True)
class _ClassFinding:
    label: str
    category: DetectionCategory
    peak_count: int = 0
    max_confidence: float = 0
    frame_count: int = 0


@dataclass(slots=True)
class _PriorityObservation:
    zone: Zone
    frame_id: int
    media_time_ms: int
    route: Route | None
    segmentation_evidence_available: bool
    detection_evidence_available: bool
    building_damage_count_availability: AggregateMetricAvailability
    update_count: int = 1


class VideoAnalysisAggregator:
    """Bounded whole-video findings built only from emitted backend intelligence."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.frames_analyzed = 0
        self.first_frame_id: int | None = None
        self.last_frame_id: int | None = None
        self.first_media_time_ms: int | None = None
        self.last_media_time_ms: int | None = None
        self.latest_result: LiveResult | None = None
        self._last_detection_source_frame_id: int | None = None
        self._last_segmentation_source_frame_id: int | None = None
        self._people = _PeakMetric()
        self._vehicles = _PeakMetric()
        self._flood = _PeakMetric()
        self._blocked_cells = _PeakMetric()
        self._damage_coverage = _PeakMetric()
        self._classes: dict[str, _ClassFinding] = {}
        self._detected_classes_truncated = False
        self._priorities: dict[str, _PriorityObservation] = {}
        self._priorities_truncated = False

    def add(self, result: LiveResult, *, media_time_ms: int) -> None:
        self.frames_analyzed += 1
        if self.first_frame_id is None:
            self.first_frame_id = result.frame_id
            self.first_media_time_ms = media_time_ms
        self.last_frame_id = result.frame_id
        self.last_media_time_ms = media_time_ms
        self.latest_result = result
        self._add_detection_evidence(result)
        self._add_segmentation_evidence(result)
        self._add_priorities(result, media_time_ms)

    def build_summary(
        self,
        *,
        frames_accepted: int,
        frames_dropped: int,
        model_status: ModelStatusResponse,
    ) -> VideoAnalysisSummary:
        priorities = self._ordered_priorities()
        highest = priorities[0].zone if priorities else None
        return VideoAnalysisSummary(
            session_id=self.session_id,
            generated_at_ms=int(time.time() * 1_000),
            frames_accepted=frames_accepted,
            frames_analyzed=self.frames_analyzed,
            frames_dropped=frames_dropped,
            first_analyzed_frame_id=self.first_frame_id,
            last_analyzed_frame_id=self.last_frame_id,
            first_media_time_ms=self.first_media_time_ms,
            last_media_time_ms=self.last_media_time_ms,
            statistics=VideoAnalysisStatistics(
                flooded_area_percent=self._segmentation_metric(
                    self._flood, MetricUnit.PERCENT, model_status
                ),
                people_detected=self._detection_metric(self._people, model_status),
                vehicles_detected=self._detection_metric(self._vehicles, model_status),
                blocked_road_cells=self._segmentation_metric(
                    self._blocked_cells, MetricUnit.COUNT, model_status
                ),
                damaged_buildings=AggregateMetric(
                    value=None,
                    unit=MetricUnit.COUNT,
                    availability=AggregateMetricAvailability.NOT_SUPPORTED,
                    aggregation=AggregateMetricAggregation.NOT_APPLICABLE,
                    supporting_frame_count=0,
                    confidence=None,
                ),
                building_damage_coverage_percent=self._segmentation_metric(
                    self._damage_coverage, MetricUnit.PERCENT, model_status
                ),
            ),
            detected_classes=self._detected_class_findings(),
            detected_classes_truncated=self._detected_classes_truncated,
            priorities=priorities,
            priorities_truncated=self._priorities_truncated,
            highest_priority_zone_id=highest.zone_id if highest else None,
            incident_severity=highest.severity if highest else None,
            segmentation_status=model_status.segmentation,
            detection_status=model_status.detection,
            inference_state=model_status.inference_state,
            responsible_ai_statement=(
                "FloodSight summarizes sampled model observations for decision support. "
                "Counts are peak simultaneous detections, not unique people across the video; "
                "trained emergency personnel must verify findings and retain response authority."
            ),
            data_origin=DataOrigin.DERIVED_ANALYTIC,
        )

    def _add_detection_evidence(self, result: LiveResult) -> None:
        evidence = result.evidence_frames
        if result.system_status.detection_model is not ModelState.READY:
            return
        if evidence is not None and evidence.detection_reused:
            return
        source_frame_id = (
            evidence.detection_source_frame_id
            if evidence is not None and evidence.detection_source_frame_id is not None
            else result.frame_id
        )
        # Source IDs are monotonic within an ingestion session. Retaining only the
        # last accepted ID prevents cadence reuse without one stored ID per frame.
        if (
            self._last_detection_source_frame_id is not None
            and source_frame_id <= self._last_detection_source_frame_id
        ):
            return
        self._last_detection_source_frame_id = source_frame_id

        direct = [
            item for item in result.detections if item.observation_state != "TRACK_PERSISTED"
        ]
        direct = _deduplicate_detections(direct)
        people = [item for item in direct if item.category is DetectionCategory.PERSON]
        vehicles = [item for item in direct if item.category is DetectionCategory.VEHICLE]
        self._people.observe(len(people), _max_confidence(people))
        self._vehicles.observe(len(vehicles), _max_confidence(vehicles))

        grouped: dict[tuple[str, DetectionCategory], list[Detection]] = {}
        for item in direct:
            if item.category not in (DetectionCategory.PERSON, DetectionCategory.VEHICLE):
                continue
            grouped.setdefault((item.source_class or item.label, item.category), []).append(item)
        for (label, category), items in grouped.items():
            finding = self._classes.setdefault(label, _ClassFinding(label=label, category=category))
            finding.peak_count = max(finding.peak_count, len(items))
            finding.max_confidence = max(finding.max_confidence, _max_confidence(items) or 0)
            finding.frame_count += 1
        while len(self._classes) > MAX_DETECTED_CLASSES:
            weakest = min(
                self._classes.values(),
                key=lambda item: (item.peak_count, item.max_confidence, item.label),
            )
            self._classes.pop(weakest.label, None)
            self._detected_classes_truncated = True

    def _add_segmentation_evidence(self, result: LiveResult) -> None:
        evidence = result.evidence_frames
        if result.system_status.segmentation_model is not ModelState.READY:
            return
        if evidence is not None and evidence.segmentation_reused:
            return
        source_frame_id = (
            evidence.segmentation_source_frame_id
            if evidence is not None and evidence.segmentation_source_frame_id is not None
            else result.frame_id
        )
        if (
            self._last_segmentation_source_frame_id is not None
            and source_frame_id <= self._last_segmentation_source_frame_id
        ):
            return
        self._last_segmentation_source_frame_id = source_frame_id

        summary = result.scene_summary
        flood_value = (
            summary.water_flood_coverage_percent
            if summary is not None
            else result.statistics.flooded_area_percent.value
        )
        damage_value = summary.building_damage_coverage_percent if summary is not None else 0
        self._flood.observe(flood_value, _semantic_confidence(result, _FLOOD_CLASSES))
        self._blocked_cells.observe(
            result.statistics.blocked_roads.value,
            _semantic_confidence(result, {"road_blocked"}),
        )
        self._damage_coverage.observe(
            damage_value,
            _semantic_confidence(result, _DAMAGE_CLASSES),
        )

    def _add_priorities(self, result: LiveResult, media_time_ms: int) -> None:
        for zone in result.zones:
            if zone.stale:
                continue
            segmentation_evidence_available = _segmentation_evidence_available(
                result, zone
            )
            detection_evidence_available = _detection_evidence_available(result, zone)
            route = (
                result.route
                if result.route and result.route.target_zone_id == zone.zone_id
                else None
            )
            current = self._priorities.get(zone.zone_id)
            if current is None:
                self._priorities[zone.zone_id] = _PriorityObservation(
                    zone=zone,
                    frame_id=result.frame_id,
                    media_time_ms=media_time_ms,
                    route=route,
                    segmentation_evidence_available=segmentation_evidence_available,
                    detection_evidence_available=detection_evidence_available,
                    building_damage_count_availability=(
                        _building_damage_count_availability(zone)
                    ),
                )
            else:
                current.update_count += 1
                if _zone_strength(
                    zone,
                    result.frame_id,
                    segmentation_evidence_available=segmentation_evidence_available,
                    detection_evidence_available=detection_evidence_available,
                ) > _zone_strength(
                    current.zone,
                    current.frame_id,
                    segmentation_evidence_available=(
                        current.segmentation_evidence_available
                    ),
                    detection_evidence_available=current.detection_evidence_available,
                ):
                    current.zone = zone
                    current.frame_id = result.frame_id
                    current.media_time_ms = media_time_ms
                    current.route = route
                    current.segmentation_evidence_available = (
                        segmentation_evidence_available
                    )
                    current.detection_evidence_available = detection_evidence_available
                    current.building_damage_count_availability = (
                        _building_damage_count_availability(zone)
                    )
        while len(self._priorities) > MAX_PRIORITY_OBSERVATIONS:
            weakest_id = min(
                self._priorities,
                key=lambda zone_id: _zone_strength(
                    self._priorities[zone_id].zone,
                    self._priorities[zone_id].frame_id,
                    segmentation_evidence_available=(
                        self._priorities[zone_id].segmentation_evidence_available
                    ),
                    detection_evidence_available=(
                        self._priorities[zone_id].detection_evidence_available
                    ),
                ),
            )
            self._priorities.pop(weakest_id)
            self._priorities_truncated = True

    def _ordered_priorities(self) -> list[VideoPriorityObservation]:
        ordered = sorted(
            self._priorities.values(),
            key=lambda item: _zone_strength(
                item.zone,
                item.frame_id,
                segmentation_evidence_available=item.segmentation_evidence_available,
                detection_evidence_available=item.detection_evidence_available,
            ),
            reverse=True,
        )
        return [
            VideoPriorityObservation(
                zone=_zone_for_summary(
                    item.zone,
                    detection_evidence_available=item.detection_evidence_available,
                ).model_copy(update={"rank": rank}),
                source_frame_id=item.frame_id,
                media_time_ms=item.media_time_ms,
                supporting_update_count=item.update_count,
                segmentation_evidence_available=item.segmentation_evidence_available,
                detection_evidence_available=item.detection_evidence_available,
                building_damage_count_availability=(
                    item.building_damage_count_availability
                ),
                associated_route=item.route,
            )
            for rank, item in enumerate(ordered, start=1)
        ]

    def _detection_metric(
        self, peak: _PeakMetric, model_status: ModelStatusResponse
    ) -> AggregateMetric:
        availability = _metric_availability(
            peak.frame_count,
            model_status.detection.status,
        )
        return AggregateMetric(
            value=peak.value if availability is AggregateMetricAvailability.AVAILABLE else None,
            unit=MetricUnit.COUNT,
            availability=availability,
            aggregation=AggregateMetricAggregation.PEAK_SIMULTANEOUS_DIRECT_DETECTIONS,
            supporting_frame_count=peak.frame_count,
            confidence=peak.confidence,
        )

    def _segmentation_metric(
        self,
        peak: _PeakMetric,
        unit: MetricUnit,
        model_status: ModelStatusResponse,
    ) -> AggregateMetric:
        availability = _metric_availability(
            peak.frame_count,
            model_status.segmentation.status,
        )
        return AggregateMetric(
            value=peak.value if availability is AggregateMetricAvailability.AVAILABLE else None,
            unit=unit,
            availability=availability,
            aggregation=AggregateMetricAggregation.PEAK_FRESH_SEGMENTATION,
            supporting_frame_count=peak.frame_count,
            confidence=peak.confidence,
        )

    def _detected_class_findings(self) -> list[DetectedClassFinding]:
        return [
            DetectedClassFinding(
                label=item.label,
                category=item.category,
                peak_simultaneous_count=item.peak_count,
                max_confidence=item.max_confidence,
                supporting_frame_count=item.frame_count,
            )
            for item in sorted(
                self._classes.values(),
                key=lambda finding: (-finding.peak_count, -finding.max_confidence, finding.label),
            )
        ]


_FLOOD_CLASSES = {"water", "road_flooded", "building_flooded"}
_DAMAGE_CLASSES = {
    "building_minor_damage",
    "building_major_damage",
    "building_destroyed",
}


def _metric_availability(
    frame_count: int, model_state: ModelState
) -> AggregateMetricAvailability:
    if frame_count:
        return AggregateMetricAvailability.AVAILABLE
    if model_state is not ModelState.READY:
        return AggregateMetricAvailability.MODEL_UNAVAILABLE
    return AggregateMetricAvailability.NO_ANALYZED_FRAMES


def _deduplicate_detections(detections: list[Detection]) -> list[Detection]:
    unique: dict[str, Detection] = {}
    for item in detections:
        key = item.track_id or item.detection_id
        previous = unique.get(key)
        if previous is None or item.confidence > previous.confidence:
            unique[key] = item
    return list(unique.values())


def _max_confidence(detections: list[Detection]) -> float | None:
    return max((item.confidence for item in detections), default=None)


def _semantic_confidence(result: LiveResult, classes: set[str]) -> float | None:
    return max(
        (item.confidence for item in result.segmentation.classes if item.label in classes),
        default=None,
    )


def _segmentation_evidence_available(result: LiveResult, zone: Zone) -> bool:
    current_payload_available = result.segmentation.status in {
        SegmentationState.READY,
        SegmentationState.SIMULATED,
    }
    retained_zone_evidence = (
        zone.flood_coverage_percent > 0
        or zone.pool_coverage_percent > 0
        or zone.building_damage_coverage_percent > 0
        or zone.road_condition.value != "UNKNOWN"
        or zone.access_status.value != "UNKNOWN"
        or any(
            reason.code
            not in {
                "PERSON_EVIDENCE",
                "NO_PERSON_EVIDENCE",
                "PERSON_EVIDENCE_UNAVAILABLE",
            }
            and reason.code != "SEGMENTATION_EVIDENCE_UNAVAILABLE"
            for reason in zone.reasons
        )
    )
    return current_payload_available or retained_zone_evidence


def _detection_evidence_available(result: LiveResult, zone: Zone) -> bool:
    if result.system_status.detection_model is ModelState.READY:
        return True
    if any(item.data_origin is DataOrigin.DEMO_SIMULATED for item in result.detections):
        return True
    return (
        zone.people_count > 0
        or zone.vehicle_count > 0
        or any(
            reason.code in {"PERSON_EVIDENCE", "NO_PERSON_EVIDENCE"}
            for reason in zone.reasons
        )
    )


def _building_damage_count_availability(zone: Zone) -> AggregateMetricAvailability:
    if zone.data_origin is DataOrigin.DEMO_SIMULATED:
        return AggregateMetricAvailability.AVAILABLE
    return AggregateMetricAvailability.NOT_SUPPORTED


def _zone_for_summary(zone: Zone, *, detection_evidence_available: bool) -> Zone:
    if detection_evidence_available:
        return zone
    unsupported = [reason for reason in zone.reasons if reason.code == "NO_PERSON_EVIDENCE"]
    if not unsupported:
        return zone
    retained = [reason for reason in zone.reasons if reason.code != "NO_PERSON_EVIDENCE"]
    qualifier = ZoneReason(
        code="PERSON_EVIDENCE_UNAVAILABLE",
        label="Person evidence unavailable",
        description=(
            "No detector evidence was available for this observation; "
            "zero is not a measured person count"
        ),
        contribution=0,
        data_origin=DataOrigin.DERIVED_ANALYTIC,
    )
    primary_reason = zone.primary_reason
    if primary_reason in {reason.description for reason in unsupported}:
        primary_reason = (
            max(retained, key=lambda reason: reason.contribution).description
            if retained
            else qualifier.description
        )
    return zone.model_copy(
        update={
            "primary_reason": primary_reason,
            "reasons": [qualifier, *retained],
        }
    )


def _zone_strength(
    zone: Zone,
    frame_id: int,
    *,
    segmentation_evidence_available: bool,
    detection_evidence_available: bool,
) -> tuple[float, int, float, int, int]:
    return (
        zone.priority_score,
        _SEVERITY_ORDER[zone.severity],
        zone.confidence,
        int(segmentation_evidence_available) + int(detection_evidence_available),
        frame_id,
    )
