import { Icon } from "../../components/Icon";
import type { Route, Zone } from "../../types/liveResult";
import { formatConfidence } from "../../utils/format";

interface PriorityCardProps {
  zone: Zone;
  route: Route | null;
  primary?: boolean;
  selected: boolean;
  onSelect: () => void;
}

const severityStyle = {
  LOW: "border-l-emerald-500",
  MODERATE: "border-l-amber-500",
  HIGH: "border-l-orange-500",
  CRITICAL: "border-l-rose-500",
};

const severityBadge = {
  LOW: "border-emerald-500/25 bg-emerald-50 text-emerald-700",
  MODERATE: "border-amber-500/25 bg-amber-50 text-amber-700",
  HIGH: "border-orange-500/25 bg-orange-50 text-orange-700",
  CRITICAL: "border-rose-500/25 bg-rose-50 text-rose-700",
};

export function PriorityCard({ zone, route, primary = false, selected, onSelect }: PriorityCardProps) {
  const strandedAlert = zone.alerts?.find((alert) => alert.code === "POTENTIAL_STRANDED_PERSON");
  const damageValue = zone.building_damage_count > 0
    ? `${zone.building_damage_count} structures`
    : `${zone.building_damage_coverage_percent ?? 0}%`;
  const zoneRoute = route?.target_zone_id === zone.zone_id ? route : null;

  if (primary) {
    const routeSummary = zoneRoute
      ? ["BASE", ...(zoneRoute.edge_ids ?? []), zone.display_name.toUpperCase()].join(" → ")
      : "No recommended route in this update";
    return (
      <article className={`priority-card priority-card-primary border-l-[3px] ${severityStyle[zone.severity]} ${selected ? "priority-card-selected" : ""}`} data-zone-id={zone.zone_id}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <span className={`inline-flex rounded-full border px-2.5 py-1 text-[0.6rem] font-extrabold tracking-[0.14em] uppercase ${severityBadge[zone.severity]}`}>{zone.severity}</span>
            <h3 className="mt-3 text-lg font-semibold text-slate-900">{zone.display_name}</h3>
            <p className="mt-0.5 font-mono text-[0.62rem] tracking-[0.08em] text-slate-600 uppercase">{zone.zone_id} · Rank #{zone.rank}</p>
          </div>
          <div className="text-right">
            <span className="block font-mono text-4xl font-semibold leading-none text-slate-900">{zone.priority_score}</span>
            <span className="mt-1 block text-[0.58rem] font-bold tracking-[0.09em] text-slate-600 uppercase">Rescue priority</span>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2">
          <DecisionFact label="People" value={zone.people_count} />
          <DecisionFact label="Flood" value={`${zone.flood_coverage_percent}%`} />
          <DecisionFact label="Access" value={zone.access_status} />
          <DecisionFact label="Damage" value={damageValue} />
        </div>

        {strandedAlert && (
          <div className="mt-3 flex items-start gap-2.5 rounded-xl border border-rose-400/25 bg-rose-50/80 p-3" aria-label="Potential stranded person alert">
            <Icon name="people" className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
            <div><p className="text-[0.66rem] font-extrabold tracking-[0.08em] text-rose-700 uppercase">Potential stranded person</p><p className="mt-1 text-[0.65rem] text-rose-700">{zone.zone_id} · Human {strandedAlert.person_evidence} · Flood {strandedAlert.flood_exposure} · Access {strandedAlert.primary_access}</p></div>
          </div>
        )}

        <section className="mt-4" aria-label="Priority evidence">
          <p className="text-[0.64rem] font-extrabold tracking-[0.12em] text-slate-700 uppercase">Why priority #1?</p>
          <ul className="mt-2 space-y-1.5 text-[0.72rem] leading-5 text-slate-500">
            {(zone.reasons.length ? zone.reasons.slice(0, 3).map((reason) => reason.label) : [zone.primary_reason]).map((reason) => <li key={reason} className="flex gap-2"><span className="text-cyan-600">•</span><span>{reason}</span></li>)}
          </ul>
        </section>

        <section className="mt-4 rounded-xl border border-sky-800/12 bg-sky-50/75 p-3" aria-label="Recommended route">
          <div className="flex items-center gap-2 text-[0.62rem] font-extrabold tracking-[0.1em] text-sky-800 uppercase"><Icon name="route" className="h-3.5 w-3.5" />Recommended route</div>
          <p className="mt-2 font-mono text-[0.68rem] leading-5 text-slate-700">{routeSummary}</p>
          {zoneRoute && <p className="mt-1 text-[0.64rem] leading-4 text-slate-500">{zoneRoute.label}</p>}
        </section>

        <div className="mt-3 flex items-center justify-between gap-2 border-t border-sky-950/10 pt-3">
          <span className="text-[0.62rem] text-slate-500">Confidence {formatConfidence(zone.confidence)}</span>
          <button type="button" onClick={onSelect} className="inline-flex min-h-8 items-center gap-1 rounded-lg px-2 text-[0.68rem] font-semibold text-cyan-700 hover:bg-cyan-100/70 focus-visible:outline-2 focus-visible:outline-cyan-700">View zone <Icon name="chevron" className="h-3 w-3" /></button>
        </div>
      </article>
    );
  }

  return (
    <article className={`priority-card border-l-2 ${severityStyle[zone.severity]} ${selected ? "priority-card-selected" : ""}`} data-zone-id={zone.zone_id}>
      <div className="flex items-start gap-3">
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-sky-950/10 bg-white/60 font-mono text-xs font-bold text-slate-200">#{zone.rank}</div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-3">
            <div><h3 className="text-sm font-semibold text-white">{zone.display_name}</h3><p className="mt-0.5 font-mono text-[0.58rem] tracking-wide text-slate-600 uppercase">{zone.zone_id} · {zone.severity} · {zone.access_status}</p></div>
            <div className="text-right"><span className="font-mono text-xl font-semibold text-white">{zone.priority_score}</span><span className="text-[0.58rem] text-slate-600"> /100</span></div>
          </div>
          <p className="mt-2 line-clamp-2 text-xs leading-5 text-slate-400">{zone.primary_reason}</p>
          {strandedAlert && <p className="mt-2 text-[0.62rem] font-bold tracking-[0.07em] text-rose-600 uppercase">Potential stranded person · {strandedAlert.person_evidence} evidence</p>}
          <div className="mt-2 flex items-center justify-between gap-2 border-t border-sky-950/10 pt-2">
            <span className="inline-flex items-center gap-1.5 text-[0.62rem] text-slate-500"><Icon name="road" className="h-3 w-3" />{zone.road_condition === "UNKNOWN" ? "UNCERTAIN" : zone.road_condition.replaceAll("_", " ")}</span>
            <button type="button" onClick={onSelect} className="inline-flex min-h-8 items-center gap-1 rounded-lg px-2 text-[0.68rem] font-semibold text-cyan-700 hover:bg-cyan-100/70 focus-visible:outline-2 focus-visible:outline-cyan-700">View zone <Icon name="chevron" className="h-3 w-3" /></button>
          </div>
        </div>
      </div>
    </article>
  );
}

function DecisionFact({ label, value }: { label: string; value: string | number }) {
  return <div className="metric-cell rounded-lg border border-sky-950/8 px-2.5 py-2"><p className="text-[0.56rem] font-bold tracking-[0.08em] text-slate-600 uppercase">{label}</p><p className="mt-1 truncate font-mono text-xs font-semibold text-slate-800" title={String(value)}>{value}</p></div>;
}
