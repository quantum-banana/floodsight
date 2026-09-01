import { Icon } from "../../components/Icon";
import type { Route, Zone } from "../../types/liveResult";
import { formatConfidence, formatPercent } from "../../utils/format";

interface PriorityCardProps {
  zone: Zone;
  route: Route | null;
  primary?: boolean;
  selected: boolean;
  onSelect: () => void;
}

export function PriorityCard({ zone, route, primary = false, selected, onSelect }: PriorityCardProps) {
  const strandedAlert = zone.alerts?.find((alert) => alert.code === "POTENTIAL_STRANDED_PERSON");
  const damageValue = zone.building_damage_count > 0 ? `${zone.building_damage_count} structures` : `${zone.building_damage_coverage_percent ?? 0}%`;
  const zoneRoute = route?.target_zone_id === zone.zone_id ? route : null;

  if (primary) {
    const routeSummary = zoneRoute
      ? ["Base", ...(zoneRoute.edge_ids ?? []), zone.display_name].join(" → ")
      : "No route in this update";

    return (
      <article className={`priority-decision severity-${zone.severity.toLowerCase()} ${selected ? "priority-decision-selected" : ""}`} data-zone-id={zone.zone_id}>
        <header className="priority-decision-header">
          <div>
            <span className="priority-severity">{zone.severity}</span>
            <h3>{zone.display_name}</h3>
            <p>{zone.zone_id} · Rank {zone.rank}</p>
          </div>
          <div className="priority-score"><strong>{zone.priority_score}</strong><span>Rescue priority</span></div>
        </header>

        <dl className="priority-facts">
          <DecisionFact label="People" value={zone.people_count} />
          <DecisionFact label="Flooded" value={formatPercent(zone.flood_coverage_percent)} />
          <DecisionFact label="Access" value={zone.access_status} />
          <DecisionFact label="Damage" value={damageValue} />
        </dl>

        {strandedAlert && (
          <div className="priority-alert" aria-label="Potential stranded person alert">
            <Icon name="people" />
            <div><strong>Potential stranded person</strong><span>{zone.zone_id} · Human {strandedAlert.person_evidence} · Flood {strandedAlert.flood_exposure} · Access {strandedAlert.primary_access}</span></div>
          </div>
        )}

        <section className="priority-reasons" aria-label="Priority evidence">
          <h4>Why this zone?</h4>
          <ul>{(zone.reasons.length ? zone.reasons.slice(0, 3).map((reason) => reason.label) : [zone.primary_reason]).map((reason) => <li key={reason}>{reason}</li>)}</ul>
        </section>

        <section className="priority-route" aria-label="Recommended route">
          <div><Icon name="route" /><h4>Recommended route</h4></div>
          <p>{routeSummary}</p>
          {zoneRoute && <span>{zoneRoute.label}</span>}
        </section>

        <footer className="priority-decision-footer">
          <span>Confidence {formatConfidence(zone.confidence)}</span>
          <button type="button" onClick={onSelect} className="command-button command-button-ghost" aria-label="View zone">Details <Icon name="chevron" /></button>
        </footer>
      </article>
    );
  }

  return (
    <article className={`priority-secondary severity-${zone.severity.toLowerCase()} ${selected ? "priority-secondary-selected" : ""}`} data-zone-id={zone.zone_id}>
      <div><span>{zone.severity}</span><h3>{zone.display_name}</h3><p>{zone.primary_reason}</p></div>
      <div className="priority-secondary-score"><strong>{zone.priority_score}</strong><button type="button" onClick={onSelect} aria-label="View zone"><Icon name="chevron" /></button></div>
    </article>
  );
}

function DecisionFact({ label, value }: { label: string; value: string | number }) {
  return <div><dt>{label}</dt><dd title={String(value)}>{value}</dd></div>;
}
