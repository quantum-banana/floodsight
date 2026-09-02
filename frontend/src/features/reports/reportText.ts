import type { IncidentReport } from "../../types/api";
import { formatTimestamp } from "../../utils/format";

export function buildReportText(report: IncidentReport): string {
  const stats = report.statistics;
  const wholeVideo = report.analysis_scope === "WHOLE_VIDEO";
  const value = (key: string, availableValue: string | number) => (
    !wholeVideo || report.aggregate_availability?.[key] === "AVAILABLE"
      ? availableValue
      : "Unavailable"
  );
  return [
    "FLOODSIGHT INCIDENT REPORT",
    `Analysis scope: ${wholeVideo ? "Whole video" : "Latest frame"}`,
    `Incident: ${report.incident_id} — ${report.title}`,
    `Generated: ${formatTimestamp(report.generated_at_ms)} UTC`,
    `Generated from frame: ${report.generated_from_frame_id ?? "not available"}`,
    `Data origin: ${report.data_origin}`,
    "",
    `Incident severity: ${report.severity_established === false ? "Not established" : report.severity}`,
    `${wholeVideo ? "Peak flood coverage" : "Flood coverage"}: ${value("flooded_area_percent", `${stats.flooded_area_percent.value}%`)}`,
    `${wholeVideo ? "Peak people" : "People"}: ${value("people_detected", stats.people_detected.value)}`,
    `${wholeVideo ? "Peak vehicles" : "Vehicles"}: ${value("vehicles_detected", stats.vehicles_detected.value)}`,
    `${wholeVideo ? "Peak blocked grid cells" : "Blocked roads"}: ${value("blocked_road_cells", stats.blocked_roads.value)}`,
    `Damaged buildings: ${value("damaged_buildings", stats.damaged_buildings.value)}`,
    `Critical zones: ${report.critical_zone_count}`,
    `Highest priority: ${report.highest_priority_zone_name ?? "None"}`,
    `Priority order: ${report.priority_order?.join(", ") || "None"}`,
    ...(wholeVideo && report.priorities_truncated
      ? ["Priority coverage: strongest retained observations only; additional observations were omitted."]
      : []),
    `Reason codes: ${report.reason_codes?.join(", ") || "None"}`,
    `Explanation: ${report.explanation}`,
    `Relative access: ${wholeVideo ? "Historical sampled-video observation; verify current conditions. " : ""}${report.access_summary}`,
    "",
    "RESPONSIBLE AI",
    report.responsible_ai_statement,
  ].join("\n");
}
