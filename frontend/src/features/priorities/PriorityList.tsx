import { EmptyState } from "../command-center/CommandStates";
import type { VideoPriorityObservation } from "../../types/ingestion";
import type { DataOrigin, Route, SystemStatus, Zone } from "../../types/liveResult";
import { PriorityCard } from "./PriorityCard";

interface PriorityListProps {
  zones: Zone[];
  route: Route | null;
  dataOrigin: DataOrigin;
  systemStatus: SystemStatus;
  selectedZoneId: string | null;
  onSelectZone: (zoneId: string) => void;
  scope?: "CURRENT_FRAME" | "WHOLE_VIDEO";
  analyzedFrameCount?: number;
  priorityObservations?: readonly VideoPriorityObservation[];
  prioritiesTruncated?: boolean;
  embedded?: boolean;
}

const emptyPriorityMessage = (
  dataOrigin: DataOrigin,
  status: SystemStatus,
  scope: "CURRENT_FRAME" | "WHOLE_VIDEO",
  analyzedFrameCount: number,
) => {
  if (dataOrigin === "DEMO_SIMULATED") return "No rescue zones in this simulated scenario.";
  if (scope === "WHOLE_VIDEO" && analyzedFrameCount === 0) {
    return "No frames were analyzed, so rescue priorities could not be established.";
  }
  const unavailableEvidence = [
    status.segmentation_model !== "ready" ? "flood/road/damage segmentation" : null,
    status.detection_model !== "ready" ? "people/vehicle detection" : null,
  ].filter((item): item is string => item !== null);
  if (unavailableEvidence.length) {
    if (scope === "WHOLE_VIDEO") {
      return `Rescue priorities could not be established conclusively across the analyzed video because ${unavailableEvidence.join(" and ")} ${unavailableEvidence.length === 1 ? "was" : "were"} unavailable; missing evidence was not simulated.`;
    }
    return `No evidence-supported rescue zones. ${unavailableEvidence.join(" and ")} unavailable; missing evidence was not simulated.`;
  }
  return scope === "WHOLE_VIDEO"
    ? "No evidence-supported rescue zones were found across the analyzed video."
    : "No evidence-supported rescue zones in the current frames.";
};

export function PriorityList({
  zones,
  route,
  dataOrigin,
  systemStatus,
  selectedZoneId,
  onSelectZone,
  scope = "CURRENT_FRAME",
  analyzedFrameCount = 0,
  priorityObservations = [],
  prioritiesTruncated = false,
  embedded = false,
}: PriorityListProps) {
  const ordered = [...zones].sort((a, b) => a.rank - b.rank);
  const [primary, ...additional] = ordered;
  const observationsByZone = new Map(
    priorityObservations.map((observation) => [observation.zone.zone_id, observation]),
  );
  const defaultSegmentationEvidence = dataOrigin === "DEMO_SIMULATED" || systemStatus.segmentation_model === "ready";
  const defaultDetectionEvidence = dataOrigin === "DEMO_SIMULATED" || systemStatus.detection_model === "ready";

  return (
    <section className={embedded ? "priority-section" : "command-panel priority-section"} aria-labelledby="priorities-heading">
      <header className="priority-section-heading">
        <h2 id="priorities-heading">What needs attention first?</h2>
        <span>{scope === "WHOLE_VIDEO" ? `${zones.length} priorities · ${analyzedFrameCount} frames` : `${zones.length} zones`}</span>
      </header>
      {scope === "WHOLE_VIDEO" && prioritiesTruncated && (
        <p className="priority-truncation-note">Showing the strongest {zones.length} retained priority observations; additional observations were omitted.</p>
      )}
      {primary ? (
        <>
          <PriorityCard zone={primary} route={route} primary selected={selectedZoneId === primary.zone_id} onSelect={() => onSelectZone(primary.zone_id)} observation={observationsByZone.get(primary.zone_id)} segmentationEvidenceAvailable={defaultSegmentationEvidence} detectionEvidenceAvailable={defaultDetectionEvidence} />
          {additional.length > 0 && (
            <details className="additional-zones">
              <summary className="disclosure-summary">Other rescue zones <span>{additional.length}</span></summary>
              <div>{additional.map((zone) => <PriorityCard key={zone.zone_id} zone={zone} route={route} selected={selectedZoneId === zone.zone_id} onSelect={() => onSelectZone(zone.zone_id)} observation={observationsByZone.get(zone.zone_id)} segmentationEvidenceAvailable={defaultSegmentationEvidence} detectionEvidenceAvailable={defaultDetectionEvidence} />)}</div>
            </details>
          )}
        </>
      ) : <EmptyState label={emptyPriorityMessage(dataOrigin, systemStatus, scope, analyzedFrameCount)} />}
    </section>
  );
}
