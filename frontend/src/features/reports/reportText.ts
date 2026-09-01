import type { IncidentReport } from "../../types/api";
import { formatTimestamp } from "../../utils/format";

export function buildReportText(report: IncidentReport): string {
  const stats = report.statistics;
  return [
    "FLOODSIGHT INCIDENT REPORT",
    `Incident: ${report.incident_id} — ${report.title}`,
    `Generated: ${formatTimestamp(report.generated_at_ms)} UTC`,
    `Generated from frame: ${report.generated_from_frame_id ?? "not available"}`,
    `Data origin: ${report.data_origin}`,
    "",
    `Incident severity: ${report.severity}`,
    `Flood coverage: ${stats.flooded_area_percent.value}%`,
    `People: ${stats.people_detected.value}`,
    `Vehicles: ${stats.vehicles_detected.value}`,
    `Blocked roads: ${stats.blocked_roads.value}`,
    `Damaged buildings: ${stats.damaged_buildings.value}`,
    `Critical zones: ${report.critical_zone_count}`,
    `Highest priority: ${report.highest_priority_zone_name ?? "None"}`,
    `Priority order: ${report.priority_order?.join(", ") || "None"}`,
    `Reason codes: ${report.reason_codes?.join(", ") || "None"}`,
    `Explanation: ${report.explanation}`,
    `Relative access: ${report.access_summary}`,
    "",
    "RESPONSIBLE AI",
    report.responsible_ai_statement,
  ].join("\n");
}
