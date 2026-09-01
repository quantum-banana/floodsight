import { EmptyState } from "../command-center/CommandStates";
import type { Route, Zone } from "../../types/liveResult";
import { PriorityCard } from "./PriorityCard";

interface PriorityListProps {
  zones: Zone[];
  route: Route | null;
  selectedZoneId: string | null;
  onSelectZone: (zoneId: string) => void;
  embedded?: boolean;
}

export function PriorityList({ zones, route, selectedZoneId, onSelectZone, embedded = false }: PriorityListProps) {
  const ordered = [...zones].sort((a, b) => a.rank - b.rank);
  const simulated = zones.every((zone) => zone.data_origin === "DEMO_SIMULATED");
  const [primary, ...additional] = ordered;

  return (
    <section className={embedded ? "priority-section" : "command-panel priority-section"} aria-labelledby="priorities-heading">
      <header className="priority-section-heading">
        <h2 id="priorities-heading">What needs attention first?</h2>
        <span>{zones.length} zones</span>
      </header>
      {primary ? (
        <>
          <PriorityCard zone={primary} route={route} primary selected={selectedZoneId === primary.zone_id} onSelect={() => onSelectZone(primary.zone_id)} />
          {additional.length > 0 && (
            <details className="additional-zones">
              <summary className="disclosure-summary">Other rescue zones <span>{additional.length}</span></summary>
              <div>{additional.map((zone) => <PriorityCard key={zone.zone_id} zone={zone} route={route} selected={selectedZoneId === zone.zone_id} onSelect={() => onSelectZone(zone.zone_id)} />)}</div>
            </details>
          )}
        </>
      ) : <EmptyState label={simulated ? "No rescue zones in this scenario." : "No evidence-supported rescue zones."} />}
    </section>
  );
}
