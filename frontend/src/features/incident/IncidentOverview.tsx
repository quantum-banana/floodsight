import { OriginBadge } from "../../components/OriginBadge";
import type { ConnectionState } from "../../hooks/useDemoIncident";
import type { LiveResult } from "../../types/liveResult";
import { formatPercent, formatTimestamp, severityLabel } from "../../utils/format";

interface IncidentOverviewProps {
  snapshot: LiveResult;
  connectionState: ConnectionState;
  embedded?: boolean;
}

export function IncidentOverview({ snapshot, connectionState, embedded = false }: IncidentOverviewProps) {
  const stats = snapshot.statistics;
  const highest = snapshot.zones.find((zone) => zone.zone_id === snapshot.highest_priority_zone_id);
  const statItems = [
    ["Flood coverage", formatPercent(stats.flooded_area_percent.value)],
    ["People", stats.people_detected.value],
    ["Vehicles", stats.vehicles_detected.value],
    ["Blocked roads", stats.blocked_roads.value],
    ["Highest priority", highest?.display_name ?? "None"],
    ["Incident severity", severityLabel(snapshot.incident_severity)],
  ] as const;

  return (
    <section className={embedded ? "incident-summary" : "command-panel incident-summary"} aria-labelledby="overview-heading">
      <header>
        <h2 id="overview-heading">Incident</h2>
        <strong className={`incident-severity severity-${snapshot.incident_severity.toLowerCase()}`}>{severityLabel(snapshot.incident_severity)}</strong>
      </header>
      <dl className="incident-stat-list">
        {statItems.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
      </dl>
      <details className="incident-meta">
        <summary className="disclosure-summary">Incident details</summary>
        <dl>
          <div><dt>Last observation</dt><dd>{formatTimestamp(snapshot.timestamp_ms)} UTC</dd></div>
          <div><dt>Stream</dt><dd>{connectionState.replaceAll("_", " ").toUpperCase()}</dd></div>
          <div><dt>Damaged buildings</dt><dd>{stats.damaged_buildings.value}</dd></div>
          <div><dt>Coordinate space</dt><dd>{snapshot.coordinate_space}</dd></div>
          <div><dt>Data origin</dt><dd><OriginBadge origin={snapshot.data_origin} compact /></dd></div>
        </dl>
      </details>
    </section>
  );
}
