import time

from app.schemas.incident import IncidentReport
from app.schemas.ingestion import AggregateMetric, VideoAnalysisComplete
from app.schemas.live_result import DataOrigin, LiveResult, Metric, Severity, Statistics


def build_incident_report(snapshot: LiveResult) -> IncidentReport:
    highest = min(snapshot.zones, key=lambda zone: zone.rank) if snapshot.zones else None
    segmentation = snapshot.system_status.segmentation_details
    detection = snapshot.system_status.detection_details
    route = snapshot.route
    return IncidentReport(
        incident_id=snapshot.incident_id,
        title=snapshot.incident.title,
        generated_at_ms=int(time.time() * 1_000),
        severity=snapshot.incident_severity,
        statistics=snapshot.statistics,
        critical_zone_count=sum(zone.severity.value == "CRITICAL" for zone in snapshot.zones),
        highest_priority_zone_id=highest.zone_id if highest else None,
        highest_priority_zone_name=highest.display_name if highest else None,
        explanation=(
            highest.primary_reason
            if highest
            else "No operational rescue zone is currently supported by available evidence."
        ),
        access_summary=(
            route.access_summary
            if route
            else "No relative tactical route is currently available."
        ),
        responsible_ai_statement=(
            "FloodSight provides evidence-backed decision support. Trained emergency personnel "
            "must verify observations and retain response authority."
        ),
        data_origin=DataOrigin.DERIVED_ANALYTIC,
        generated_from_frame_id=snapshot.frame_id,
        priority_order=[
            zone.zone_id for zone in sorted(snapshot.zones, key=lambda zone: zone.rank)
        ],
        reason_codes=list(
            dict.fromkeys(reason.code for zone in snapshot.zones for reason in zone.reasons)
        ),
        route=route,
        model_provenance={
            "segmentation": segmentation.provenance_mode if segmentation else "UNAVAILABLE",
            "detection": detection.provenance_mode if detection else "UNAVAILABLE",
        },
    )


def build_video_analysis_report(completed: VideoAnalysisComplete) -> IncidentReport:
    summary = completed.summary
    priorities = summary.priorities
    highest = priorities[0] if priorities else None
    route = highest.associated_route if highest else None
    latest = completed.latest_result
    statistics = summary.statistics
    return IncidentReport(
        incident_id=(latest.incident_id if latest else f"LIVE-{summary.session_id[:12]}"),
        title=(latest.incident.title if latest else "Completed Video Analysis"),
        generated_at_ms=summary.generated_at_ms,
        severity=summary.incident_severity or Severity.LOW,
        statistics=Statistics(
            flooded_area_percent=_report_metric(statistics.flooded_area_percent),
            people_detected=_report_metric(statistics.people_detected),
            vehicles_detected=_report_metric(statistics.vehicles_detected),
            blocked_roads=_report_metric(statistics.blocked_road_cells),
            damaged_buildings=_report_metric(statistics.damaged_buildings),
        ),
        critical_zone_count=sum(
            item.zone.severity is Severity.CRITICAL for item in priorities
        ),
        highest_priority_zone_id=highest.zone.zone_id if highest else None,
        highest_priority_zone_name=highest.zone.display_name if highest else None,
        explanation=(
            highest.zone.primary_reason
            if highest
            else "No frames were analyzed; rescue priority was not established."
            if summary.frames_analyzed == 0
            else "No operational rescue zone was supported by analyzed video evidence."
        ),
        access_summary=(
            (
                "Historical relative image-space route observed at "
                f"video {highest.media_time_ms} ms (frame {highest.source_frame_id}): "
                f"{route.access_summary} Current conditions require trained-personnel verification."
            )
            if route
            else "No frames were analyzed; relative access was not assessed."
            if summary.frames_analyzed == 0
            else "No relative tactical route was retained in the completed video analysis."
        ),
        responsible_ai_statement=summary.responsible_ai_statement,
        data_origin=DataOrigin.DERIVED_ANALYTIC,
        generated_from_frame_id=summary.last_analyzed_frame_id,
        priority_order=[item.zone.zone_id for item in priorities],
        reason_codes=list(
            dict.fromkeys(
                reason.code for item in priorities for reason in item.zone.reasons
            )
        ),
        route=route,
        model_provenance={
            "segmentation": summary.segmentation_status.provenance_mode or "UNAVAILABLE",
            "detection": summary.detection_status.provenance_mode or "UNAVAILABLE",
        },
        analysis_scope="WHOLE_VIDEO",
        aggregate_availability={
            "flooded_area_percent": statistics.flooded_area_percent.availability.value,
            "people_detected": statistics.people_detected.availability.value,
            "vehicles_detected": statistics.vehicles_detected.availability.value,
            "blocked_road_cells": statistics.blocked_road_cells.availability.value,
            "damaged_buildings": statistics.damaged_buildings.availability.value,
            "building_damage_coverage_percent": (
                statistics.building_damage_coverage_percent.availability.value
            ),
        },
        severity_established=summary.incident_severity is not None,
        priorities_truncated=summary.priorities_truncated,
    )


def _report_metric(metric: AggregateMetric) -> Metric:
    return Metric(
        value=metric.value or 0,
        unit=metric.unit,
        confidence=metric.confidence,
        data_origin=DataOrigin.DERIVED_ANALYTIC,
    )
