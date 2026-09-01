import time

from app.schemas.incident import IncidentReport
from app.schemas.live_result import DataOrigin, LiveResult


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
