import { EmptyState } from "../command-center/CommandStates";
import type { Zone } from "../../types/liveResult";
import { PriorityCard } from "./PriorityCard";

interface PriorityListProps {
  zones: Zone[];
  selectedZoneId: string | null;
  onSelectZone: (zoneId: string) => void;
}

export function PriorityList({ zones, selectedZoneId, onSelectZone }: PriorityListProps) {
  const ordered = [...zones].sort((a, b) => a.rank - b.rank);
  const simulated = zones.every((zone) => zone.data_origin === "DEMO_SIMULATED");
  return (
    <section className="command-panel min-w-0" aria-labelledby="priorities-heading">
      <div className="panel-heading"><div><h2 id="priorities-heading" className="panel-title">Rescue priorities</h2><p className="panel-subtitle">{simulated ? "Supplied deterministic ranking · Updates by snapshot" : "Backend-computed urgency · Confidence shown separately"}</p></div><span className="rounded-full bg-white/[0.04] px-2.5 py-1 font-mono text-[0.65rem] text-slate-500">{zones.length} ZONES</span></div>
      <div className="priority-list max-h-[34rem] space-y-2 overflow-y-auto p-3">
        {ordered.length ? ordered.map((zone) => <PriorityCard key={zone.zone_id} zone={zone} selected={selectedZoneId === zone.zone_id} onSelect={() => onSelectZone(zone.zone_id)} />) : <EmptyState label={simulated ? "No rescue zones in this simulated snapshot." : "No evidence-supported rescue zones in the current frame."} />}
      </div>
    </section>
  );
}
