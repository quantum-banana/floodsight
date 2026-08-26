import type { LiveResult } from "../../types/liveResult";
import { formatTimestamp } from "../../utils/format";

export function buildReportText(snapshot: LiveResult): string {
  const highest = snapshot.zones.find((zone) => zone.zone_id === snapshot.highest_priority_zone_id);
  const stats = snapshot.statistics;
  const criticalZones = snapshot.zones.filter((zone) => zone.severity === "CRITICAL").length;
  return [
    "FLOODSIGHT INCIDENT REPORT",
    `Incident: ${snapshot.incident_id} — ${snapshot.incident.title}`,
    `Generated from observation: ${formatTimestamp(snapshot.timestamp_ms)} UTC`,
    `Data origin: ${snapshot.data_origin}`,
    "",
    `Incident severity: ${snapshot.incident_severity}`,
    `Simulated flood coverage: ${stats.flooded_area_percent.value}%`,
    `People: ${stats.people_detected.value}`,
    `Vehicles: ${stats.vehicles_detected.value}`,
    `Blocked roads: ${stats.blocked_roads.value}`,
    `Damaged buildings: ${stats.damaged_buildings.value}`,
    `Critical zones: ${criticalZones}`,
    `Highest priority: ${highest ? `${highest.display_name} — ${highest.priority_score}/100` : "None"}`,
    `Explanation: ${highest?.primary_reason ?? "No rescue zones supplied."}`,
    `Relative access: ${snapshot.route?.access_summary ?? "No relative route supplied in this snapshot."}`,
    "",
    "RESPONSIBLE AI",
    "FloodSight is decision support. This simulated report requires human review before any operational action.",
  ].join("\n");
}
