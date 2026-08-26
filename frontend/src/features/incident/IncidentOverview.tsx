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
  LOW: "text-emerald-300 border-emerald-400/20 bg-emerald-400/[0.07]",
  MODERATE: "text-yellow-300 border-yellow-400/20 bg-yellow-400/[0.07]",
  HIGH: "text-orange-300 border-orange-400/20 bg-orange-400/[0.07]",
  CRITICAL: "text-rose-300 border-rose-400/25 bg-rose-400/[0.08]",
};

export function IncidentOverview({ snapshot, connectionState }: IncidentOverviewProps) {
  const stats = snapshot.statistics;
  const highest = snapshot.zones.find((zone) => zone.zone_id === snapshot.highest_priority_zone_id);
  const statItems = [
    { label: "Flood coverage", value: `${stats.flooded_area_percent.value}%`, icon: "water" as const },
    { label: "People", value: stats.people_detected.value, icon: "people" as const },
    { label: "Vehicles", value: stats.vehicles_detected.value, icon: "vehicle" as const },
    { label: "Blocked roads", value: stats.blocked_roads.value, icon: "road" as const },
    { label: "Damaged buildings", value: stats.damaged_buildings.value, icon: "building" as const },
    { label: "Highest priority", value: highest?.display_name ?? "None", icon: "alert" as const },
  ];

  return (
    <aside className="command-panel min-w-0" aria-labelledby="overview-heading">
      <div className="panel-heading">
        <div><h2 id="overview-heading" className="panel-title">Incident overview</h2><p className="panel-subtitle">Latest simulated observation</p></div>
        <span className={`rounded-lg border px-2.5 py-1 text-[0.66rem] font-bold tracking-[0.14em] uppercase ${severityClasses[snapshot.incident_severity]}`}>{severityLabel(snapshot.incident_severity)}</span>
      </div>
      <div className="grid grid-cols-2 gap-px bg-white/[0.06]">
        {statItems.map(({ label, value, icon }) => (
          <OverviewStatistic key={label} label={label} value={value} icon={icon} />
        ))}
      </div>
      <dl className="space-y-3 border-t border-white/[0.06] p-4 text-xs">
        <div className="flex items-center justify-between gap-4"><dt className="text-slate-600">Last observation</dt><dd className="font-mono text-slate-300">{formatTimestamp(snapshot.timestamp_ms)} UTC</dd></div>
        <div className="flex items-center justify-between gap-4"><dt className="text-slate-600">Stream state</dt><dd className="font-semibold text-cyan-200">{connectionState.replaceAll("_", " ").toUpperCase()}</dd></div>
        <div className="flex items-center justify-between gap-4"><dt className="text-slate-600">Coordinate space</dt><dd className="text-right text-slate-300">Relative tactical</dd></div>
        <div className="flex items-center justify-between gap-4"><dt className="text-slate-600">Data origin</dt><dd><OriginBadge origin={snapshot.data_origin} compact /></dd></div>
      </dl>
    </aside>
  );
}

function OverviewStatistic({ label, value, icon }: { label: string; value: string | number; icon: "water" | "people" | "vehicle" | "road" | "building" | "alert" }) {
  return <div className="bg-[#0b171e] p-4"><div className="flex items-center gap-2 text-slate-500"><Icon name={icon} className="h-3.5 w-3.5" /><span className="text-[0.63rem] font-semibold tracking-[0.11em] uppercase">{label}</span></div><p className="mt-2 font-mono text-xl font-semibold text-slate-100">{value}</p></div>;
}
