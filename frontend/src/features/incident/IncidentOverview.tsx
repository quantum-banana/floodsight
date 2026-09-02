import { OriginBadge } from "../../components/OriginBadge";
import type { ConnectionState } from "../../hooks/useDemoIncident";
import type { AggregateMetric, VideoAnalysisSummary } from "../../types/ingestion";
import type { LiveResult } from "../../types/liveResult";
import { formatConfidence, formatPercent, formatTimestamp, severityLabel } from "../../utils/format";

interface IncidentOverviewProps {
  snapshot: LiveResult | null;
  connectionState: ConnectionState;
  summary?: VideoAnalysisSummary | null;
  embedded?: boolean;
}

const aggregateValue = (metric: AggregateMetric) => {
  if (metric.availability !== "AVAILABLE" || metric.value === null) return "Unavailable";
  return metric.unit === "percent" ? formatPercent(metric.value) : metric.value;
};

const displayClassLabel = (label: string) => label.replaceAll("_", " ");

export function IncidentOverview({ snapshot, connectionState, summary = null, embedded = false }: IncidentOverviewProps) {
  if (summary) return <FinalVideoOverview summary={summary} embedded={embedded} />;
  if (!snapshot) return null;
  const stats = snapshot.statistics;
  const highest = snapshot.zones.find((zone) => zone.zone_id === snapshot.highest_priority_zone_id);
  const simulated = snapshot.data_origin === "DEMO_SIMULATED";
  const segmentationAvailable = simulated || ["ready", "simulated"].includes(snapshot.segmentation.status);
  const detectionAvailable = simulated || snapshot.system_status.detection_model === "ready";
  const statItems = [
    ["Flood coverage", segmentationAvailable ? formatPercent(stats.flooded_area_percent.value) : "Unavailable"],
    ["People", detectionAvailable ? stats.people_detected.value : "Unavailable"],
    ["Vehicles", detectionAvailable ? stats.vehicles_detected.value : "Unavailable"],
    ["Blocked roads", segmentationAvailable ? stats.blocked_roads.value : "Unavailable"],
    ["Highest priority", highest?.display_name ?? "None"],
    ["Incident severity", severityLabel(snapshot.incident_severity)],
  ] as const;

  return (
    <section className={embedded ? "incident-summary" : "command-panel incident-summary"} aria-labelledby="overview-heading">
      <header>
        <h2 id="overview-heading">Incident</h2>
        <strong className={`incident-severity severity-${snapshot.incident_severity.toLowerCase()}`}>{severityLabel(snapshot.incident_severity)}</strong>
      </header>
      <dl className="incident-stat-list">
        {statItems.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
      </dl>
      <details className="incident-meta">
        <summary className="disclosure-summary">Incident details</summary>
        <dl>
          <div><dt>Last observation</dt><dd>{formatTimestamp(snapshot.timestamp_ms)} UTC</dd></div>
          <div><dt>Stream</dt><dd>{connectionState.replaceAll("_", " ").toUpperCase()}</dd></div>
          <div><dt>Damaged buildings</dt><dd>{simulated ? stats.damaged_buildings.value : "Not supported"}</dd></div>
          <div><dt>Coordinate space</dt><dd>{snapshot.coordinate_space}</dd></div>
          <div><dt>Data origin</dt><dd><OriginBadge origin={snapshot.data_origin} compact /></dd></div>
        </dl>
      </details>
    </section>
  );
}

function FinalVideoOverview({ summary, embedded }: { summary: VideoAnalysisSummary; embedded: boolean }) {
  const stats = summary.statistics;
  const highest = summary.priorities.find(
    (item) => item.zone.zone_id === summary.highest_priority_zone_id,
  ) ?? summary.priorities[0];
  const severity = summary.incident_severity;
  const evidenceIncomplete = summary.segmentation_status.status !== "ready"
    || summary.detection_status.status !== "ready";
  const highestPriorityValue = highest
    ? `${highest.zone.display_name} · ${highest.zone.priority_score}`
    : summary.frames_analyzed === 0 || evidenceIncomplete
      ? "Not established"
      : "None found";
  const objectEmptyMessage = summary.frames_analyzed === 0
    ? "No frames were analyzed, so object findings could not be established."
    : stats.people_detected.availability !== "AVAILABLE" && stats.vehicles_detected.availability !== "AVAILABLE"
      ? "Object detection was unavailable, so object findings could not be established."
      : "No supported object classes were detected in analyzed frames.";
  const statItems = [
    ["Peak flood coverage", aggregateValue(stats.flooded_area_percent)],
    ["Peak people", aggregateValue(stats.people_detected)],
    ["Peak vehicles", aggregateValue(stats.vehicles_detected)],
    ["Peak blocked grid cells", aggregateValue(stats.blocked_road_cells)],
    ["Peak damage coverage", aggregateValue(stats.building_damage_coverage_percent)],
    ["Highest priority", highestPriorityValue],
    ["Incident severity", severity ? severityLabel(severity) : "Not established"],
    ["Frames analyzed", summary.frames_analyzed],
  ] as const;

  return (
    <section className={embedded ? "incident-summary final-video-summary" : "command-panel incident-summary final-video-summary"} aria-labelledby="overview-heading">
      <header>
        <div>
          <span className="final-findings-label">FINAL VIDEO FINDINGS</span>
          <h2 id="overview-heading">What the analysis found</h2>
        </div>
        <OriginBadge origin={summary.data_origin} compact />
      </header>
      <dl className="incident-stat-list final-stat-list">
        {statItems.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
      </dl>

      <section className="detected-class-findings" aria-labelledby="objects-found-heading">
        <header>
          <h3 id="objects-found-heading">Objects found</h3>
          <span>Peak simultaneous detections</span>
        </header>
        {summary.detected_classes.length ? (
          <ul>
            {summary.detected_classes.map((finding) => (
              <li key={`${finding.category}-${finding.label}`}>
                <div><strong>{displayClassLabel(finding.label)}</strong><span>{finding.category.toLowerCase()}</span></div>
                <div><strong>{finding.peak_simultaneous_count}</strong><span>{formatConfidence(finding.max_confidence)} confidence</span></div>
              </li>
            ))}
          </ul>
        ) : <p>{objectEmptyMessage}</p>}
        {summary.detected_classes_truncated && (
          <p className="priority-truncation-note">Showing the strongest retained object classes; additional class findings were omitted.</p>
        )}
      </section>

      <details className="incident-meta">
        <summary className="disclosure-summary">Analysis details</summary>
        <dl>
          <div><dt>Generated</dt><dd>{formatTimestamp(summary.generated_at_ms)} UTC</dd></div>
          <div><dt>Accepted frames</dt><dd>{summary.frames_accepted}</dd></div>
          <div><dt>Dropped inference frames</dt><dd>{summary.frames_dropped}</dd></div>
          <div><dt>Analyzed frame range</dt><dd>{summary.first_analyzed_frame_id ?? "—"}–{summary.last_analyzed_frame_id ?? "—"}</dd></div>
          <div><dt>Segmentation</dt><dd>{summary.segmentation_status.status.toUpperCase()}</dd></div>
          <div><dt>Detection</dt><dd>{summary.detection_status.status.toUpperCase()}</dd></div>
          <div><dt>Data origin</dt><dd><OriginBadge origin={summary.data_origin} compact /></dd></div>
        </dl>
      </details>
      <p className="final-responsible-note">{summary.responsible_ai_statement}</p>
    </section>
  );
}
