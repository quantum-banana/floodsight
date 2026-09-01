import { EmptyState } from "../command-center/CommandStates";
import type { Route, Zone } from "../../types/liveResult";
import { PriorityCard } from "./PriorityCard";

interface PriorityListProps {
  zones: Zone[];
  route: Route | null;
  selectedZoneId: string | null;
  onSelectZone: (zoneId: string) => void;
}

export function PriorityList({ zones, route, selectedZoneId, onSelectZone }: PriorityListProps) {
  const ordered = [...zones].sort((a, b) => a.rank - b.rank);
  const simulated = zones.every((zone) => zone.data_origin === "DEMO_SIMULATED");
  const [primary, ...additional] = ordered;
  return (
    <section className="command-panel min-w-0 overflow-hidden" aria-labelledby="priorities-heading">
      <div className="panel-heading min-h-0 py-3"><div><h2 id="priorities-heading" className="panel-title">Rescue priority</h2><p className="panel-subtitle">{simulated ? "Supplied deterministic ranking" : "Backend-computed urgency · Confidence separate"}</p></div><span className="rounded-full bg-sky-950/5 px-2.5 py-1 font-mono text-[0.62rem] text-slate-500">{zones.length} ZONES</span></div>
      <div className="priority-list p-3">
        {primary ? (
          <>
            <PriorityCard zone={primary} route={route} primary selected={selectedZoneId === primary.zone_id} onSelect={() => onSelectZone(primary.zone_id)} />
            {additional.length > 0 && (
              <details className="mt-2 border-t border-sky-950/10 px-1 pt-1">
                <summary className="disclosure-summary">Additional rescue zones <span className="font-mono">{additional.length}</span></summary>
                <div className="space-y-2 pb-1 pt-1">
                  {additional.map((zone) => <PriorityCard key={zone.zone_id} zone={zone} route={route} selected={selectedZoneId === zone.zone_id} onSelect={() => onSelectZone(zone.zone_id)} />)}
                </div>
              </details>
            )}
          </>
        ) : <EmptyState label={simulated ? "No rescue zones in this simulated snapshot." : "No evidence-supported rescue zones in the current frame."} />}
      </div>
    </section>
  );
}
