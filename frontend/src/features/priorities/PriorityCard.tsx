import { Icon } from "../../components/Icon";
import type { Zone } from "../../types/liveResult";
import { formatConfidence } from "../../utils/format";

interface PriorityCardProps {
  zone: Zone;
  selected: boolean;
  onSelect: () => void;
}

const severityStyle = {
  LOW: "border-l-emerald-400",
  MODERATE: "border-l-yellow-400",
  HIGH: "border-l-orange-400",
  CRITICAL: "border-l-rose-400",
};

export function PriorityCard({ zone, selected, onSelect }: PriorityCardProps) {
  const damageValue = zone.building_damage_count > 0
    ? zone.building_damage_count
    : `${zone.building_damage_coverage_percent ?? 0}%`;
  return (
    <article className={`priority-card border-l-2 ${severityStyle[zone.severity]} ${selected ? "priority-card-selected" : ""}`} data-zone-id={zone.zone_id}>
      <div className="flex items-start gap-3">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl border border-white/[0.07] bg-white/[0.035] font-mono text-sm font-bold text-slate-200">#{zone.rank}</div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-3">
            <div><h3 className="text-sm font-semibold text-white">{zone.display_name}</h3><p className="mt-0.5 font-mono text-[0.58rem] tracking-wide text-slate-600 uppercase">{zone.zone_id}</p><p className="mt-1 text-[0.65rem] font-bold tracking-[0.14em] text-slate-500 uppercase">{zone.severity} · {zone.access_status}</p></div>
            <div className="text-right"><span className="font-mono text-2xl font-semibold text-white">{zone.priority_score}</span><span className="text-[0.62rem] text-slate-600"> /100</span></div>
          </div>
          <p className="mt-2 line-clamp-2 text-xs leading-5 text-slate-400">{zone.primary_reason}</p>
          <div className="mt-3 grid grid-cols-4 gap-1.5 text-center">
            <MiniStat label="People" value={zone.people_count} />
            <MiniStat label="Flood" value={`${zone.flood_coverage_percent}%`} />
            <MiniStat label={zone.building_damage_count > 0 ? "Buildings" : "Damage cov."} value={damageValue} />
            <MiniStat label="Confidence" value={formatConfidence(zone.confidence)} />
          </div>
          <div className="mt-3 flex items-center justify-between gap-2 border-t border-white/[0.05] pt-3">
            <span className="inline-flex items-center gap-1.5 text-[0.65rem] text-slate-500"><Icon name="road" className="h-3 w-3" />{zone.road_condition.replaceAll("_", " ")}</span>
            <button type="button" onClick={onSelect} className="inline-flex min-h-8 items-center gap-1 rounded-lg px-2 text-[0.68rem] font-semibold text-cyan-300 hover:bg-cyan-300/[0.07] focus-visible:outline-2 focus-visible:outline-cyan-300">View zone <Icon name="chevron" className="h-3 w-3" /></button>
          </div>
        </div>
      </div>
    </article>
  );
}

function MiniStat({ label, value }: { label: string; value: string | number }) {
  return <div className="rounded-lg bg-white/[0.025] px-1 py-2"><p className="font-mono text-xs font-semibold text-slate-200">{value}</p><p className="mt-0.5 truncate text-[0.55rem] tracking-wide text-slate-600 uppercase">{label}</p></div>;
}
