import { Icon } from "../../components/Icon";
import { OriginBadge } from "../../components/OriginBadge";
import type { ConnectionState } from "../../hooks/useDemoIncident";
import type { LiveResult } from "../../types/liveResult";
import { formatTimestamp, severityLabel } from "../../utils/format";

interface IncidentOverviewProps {
  snapshot: LiveResult;
  connectionState: ConnectionState;
}

const severityClasses = {
  LOW: "text-emerald-700 border-emerald-500/25 bg-emerald-50",
  MODERATE: "text-amber-700 border-amber-500/25 bg-amber-50",
  HIGH: "text-orange-700 border-orange-500/25 bg-orange-50",
  CRITICAL: "text-rose-700 border-rose-500/30 bg-rose-50",
};

export function IncidentOverview({ snapshot, connectionState }: IncidentOverviewProps) {
  const stats = snapshot.statistics;
  const simulated = snapshot.data_origin === "DEMO_SIMULATED";
  const highest = snapshot.zones.find((zone) => zone.zone_id === snapshot.highest_priority_zone_id);
  const statItems = [
    { label: "Flood coverage", value: `${stats.flooded_area_percent.value}%`, icon: "water" as const },
    { label: "People", value: stats.people_detected.value, icon: "people" as const },
    { label: "Vehicles", value: stats.vehicles_detected.value, icon: "vehicle" as const },
    { label: "Blocked roads", value: stats.blocked_roads.value, icon: "road" as const },
    { label: "Highest priority", value: highest?.display_name ?? "None", icon: "alert" as const },
    { label: "Incident severity", value: severityLabel(snapshot.incident_severity), icon: "activity" as const },
  ];

  return (
    <aside className="command-panel min-w-0 overflow-hidden" aria-labelledby="overview-heading">
      <div className="panel-heading min-h-0 py-3">
        <div><h2 id="overview-heading" className="panel-title">Incident overview</h2><p className="panel-subtitle">{simulated ? "Latest simulated observation" : `Backend intelligence · Frame ${snapshot.frame_id}`}</p></div>
        <span className={`rounded-lg border px-2.5 py-1 text-[0.62rem] font-bold tracking-[0.12em] uppercase ${severityClasses[snapshot.incident_severity]}`}>{severityLabel(snapshot.incident_severity)}</span>
      </div>
      <div className="grid grid-cols-2 gap-px bg-sky-950/10">
        {statItems.map(({ label, value, icon }) => (
          <OverviewStatistic key={label} label={label} value={value} icon={icon} />
        ))}
      </div>
      <details className="border-t border-sky-950/10 px-4 py-1">
        <summary className="disclosure-summary">Incident details</summary>
        <dl className="space-y-3 pb-4 pt-2 text-xs">
          <div className="flex items-center justify-between gap-4"><dt className="text-slate-600">Last observation</dt><dd className="font-mono text-slate-300">{formatTimestamp(snapshot.timestamp_ms)} UTC</dd></div>
          <div className="flex items-center justify-between gap-4"><dt className="text-slate-600">Stream state</dt><dd className="font-semibold text-cyan-200">{connectionState.replaceAll("_", " ").toUpperCase()}</dd></div>
          <div className="flex items-center justify-between gap-4"><dt className="text-slate-600">Damaged buildings</dt><dd className="font-mono text-slate-300">{stats.damaged_buildings.value}</dd></div>
          <div className="flex items-center justify-between gap-4"><dt className="text-slate-600">Coordinate space</dt><dd className="text-right text-slate-300">Relative tactical</dd></div>
          <div className="flex items-center justify-between gap-4"><dt className="text-slate-600">Data origin</dt><dd><OriginBadge origin={snapshot.data_origin} compact /></dd></div>
        </dl>
      </details>
    </aside>
  );
}

function OverviewStatistic({ label, value, icon }: { label: string; value: string | number; icon: "water" | "people" | "vehicle" | "road" | "alert" | "activity" }) {
  return <div className="metric-cell p-3"><div className="flex items-center gap-2 text-slate-500"><Icon name={icon} className="h-3.5 w-3.5" /><span className="text-[0.58rem] font-semibold tracking-[0.08em] uppercase">{label}</span></div><p className="mt-1.5 truncate font-mono text-lg font-semibold text-slate-100" title={String(value)}>{value}</p></div>;
}
