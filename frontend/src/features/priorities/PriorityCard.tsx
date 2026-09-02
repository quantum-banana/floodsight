import { Icon } from "../../components/Icon";
import type { VideoPriorityObservation } from "../../types/ingestion";
import type { Route, Zone } from "../../types/liveResult";
import { formatConfidence, formatPercent } from "../../utils/format";

interface PriorityCardProps {
  zone: Zone;
  route: Route | null;
  primary?: boolean;
  selected: boolean;
  onSelect: () => void;
  observation?: VideoPriorityObservation | null;
  segmentationEvidenceAvailable?: boolean;
  detectionEvidenceAvailable?: boolean;
}

const formatMediaTime = (mediaTimeMs: number) => {
  const totalSeconds = Math.max(0, Math.floor(mediaTimeMs / 1_000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
};

export function PriorityCard({ zone, route, primary = false, selected, onSelect, observation = null, segmentationEvidenceAvailable = true, detectionEvidenceAvailable = true }: PriorityCardProps) {
  const hasSegmentationEvidence = observation?.segmentation_evidence_available ?? segmentationEvidenceAvailable;
  const hasDetectionEvidence = observation?.detection_evidence_available ?? detectionEvidenceAvailable;
  const strandedAlert = zone.alerts?.find((alert) => alert.code === "POTENTIAL_STRANDED_PERSON");
  const damageValue = !hasSegmentationEvidence
    ? "Unavailable"
    : observation?.building_damage_count_availability === "AVAILABLE" && zone.building_damage_count > 0
      ? `${zone.building_damage_count} structures`
      : `${zone.building_damage_coverage_percent ?? 0}%`;
  const zoneRoute = route?.target_zone_id === zone.zone_id ? route : null;

  if (primary) {
    const routeSummary = zoneRoute
      ? ["Base", ...(zoneRoute.edge_ids ?? []), zone.display_name].join(" → ")
      : observation === null
        ? "No route in this update"
        : "No relative route was retained for this video observation";

    return (
      <article className={`priority-decision severity-${zone.severity.toLowerCase()} ${selected ? "priority-decision-selected" : ""}`} data-zone-id={zone.zone_id}>
        <header className="priority-decision-header">
          <div>
            <span className="priority-severity">{zone.severity}</span>
            <h3>{zone.display_name}</h3>
            <p>{zone.zone_id} · Rank {zone.rank}</p>
            {observation && <p className="historical-priority-observation">Observed at {formatMediaTime(observation.media_time_ms)} · source frame {observation.source_frame_id}</p>}
          </div>
          <div className="priority-score"><strong>{zone.priority_score}</strong><span>Rescue priority</span></div>
        </header>

        <dl className="priority-facts">
          <DecisionFact label="People" value={hasDetectionEvidence ? zone.people_count : "Unavailable"} />
          <DecisionFact label="Flooded" value={hasSegmentationEvidence ? formatPercent(zone.flood_coverage_percent) : "Unavailable"} />
          <DecisionFact label="Access" value={hasSegmentationEvidence ? zone.access_status : "Unavailable"} />
          <DecisionFact label="Damage" value={damageValue} />
        </dl>

        {strandedAlert && hasDetectionEvidence && (
          <div className="priority-alert" aria-label="Potential stranded person alert">
            <Icon name="people" />
            <div><strong>Potential stranded person</strong><span>{zone.zone_id} · Human {strandedAlert.person_evidence} · Flood {hasSegmentationEvidence ? strandedAlert.flood_exposure : "UNAVAILABLE"} · Access {strandedAlert.primary_access}</span></div>
          </div>
        )}

        <section className="priority-reasons" aria-label="Priority evidence">
          <h4>Why this zone?</h4>
          <ul>{(zone.reasons.length ? zone.reasons.slice(0, 3).map((reason) => reason.label) : [zone.primary_reason]).map((reason) => <li key={reason}>{reason}</li>)}</ul>
        </section>

        <section className="priority-route" aria-label={observation === null ? "Recommended route" : "Historical relative route observation"}>
          <div><Icon name="route" /><h4>{observation === null ? "Recommended route" : `Relative route observed at ${formatMediaTime(observation.media_time_ms)} · source frame ${observation.source_frame_id}`}</h4></div>
          <p>{routeSummary}</p>
          {zoneRoute && <span>{zoneRoute.label}{observation !== null ? " · Historical image-relative evidence; verify current access." : ""}</span>}
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
      <div><span>{zone.severity}</span><h3>{zone.display_name}</h3>{observation && <p className="historical-priority-observation">Observed at {formatMediaTime(observation.media_time_ms)} · source frame {observation.source_frame_id}</p>}<p>{zone.primary_reason}</p></div>
      <div className="priority-secondary-score"><strong>{zone.priority_score}</strong><button type="button" onClick={onSelect} aria-label="View zone"><Icon name="chevron" /></button></div>
    </article>
  );
}

function DecisionFact({ label, value }: { label: string; value: string | number }) {
  return <div><dt>{label}</dt><dd title={String(value)}>{value}</dd></div>;
}
