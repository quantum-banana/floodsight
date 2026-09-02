import { useCallback, useEffect, useMemo, useState } from "react";

import { ConnectionIndicator } from "../../components/ConnectionIndicator";
import { Icon } from "../../components/Icon";
import { WaterAmbience } from "../../components/WaterAmbience";
import { useDemoIncident, type ConnectionState } from "../../hooks/useDemoIncident";
import { useFrameIngestion } from "../../hooks/useFrameIngestion";
import { useMediaSource, type MediaSourceState } from "../../hooks/useMediaSource";
import type { DetectorInferenceMode, IngestionMetrics } from "../../types/ingestion";
import type { IncidentMetadata, SystemStatus } from "../../types/liveResult";
import { EventTimeline } from "../events/EventTimeline";
import { ApplicationHeader } from "../incident/ApplicationHeader";
import { IncidentOverview } from "../incident/IncidentOverview";
import { SimulationControls } from "../incident/SimulationControls";
import { MediaSourceSelector } from "../media/MediaSourceSelector";
import { ObservationPanel } from "../observation/ObservationPanel";
import { PriorityList } from "../priorities/PriorityList";
import { ZoneDetailsDrawer } from "../priorities/ZoneDetailsDrawer";
import { IncidentReportModal } from "../reports/IncidentReportModal";
import { TacticalMap } from "../tactical-map/TacticalMap";
import { DEFAULT_LAYERS, type LayerKey, type LayerState } from "../tactical-map/layers";
import { CommandLoadingState, CommandOfflineState } from "./CommandStates";

interface CommandCenterProps {
  demoMode?: boolean;
}

const actualConnectionState = (state: string, mediaState: MediaSourceState): ConnectionState => {
  if (mediaState === "PAUSED") return "paused";
  if (mediaState === "COMPLETE") return "complete";
  if (state === "streaming") return "connected";
  if (state === "creating_session" || state === "connecting") return "connecting";
  if (state === "offline" || state === "malformed") return state;
  return "complete";
};

const operationalLabel = (state: string, mediaState: MediaSourceState, hasIntelligence: boolean) => {
  if (state === "offline") return "OFFLINE";
  if (state === "malformed" || mediaState === "ERROR") return "DEGRADED";
  if (mediaState === "PAUSED") return "PAUSED";
  if (mediaState === "COMPLETE") return "FINAL";
  if (mediaState === "PLAYING") return hasIntelligence ? "LIVE" : "ANALYSING";
  if (state === "creating_session" || state === "connecting") return "ANALYSING";
  return "READY";
};

export function CommandCenter({ demoMode = false }: CommandCenterProps) {
  const demo = useDemoIncident(demoMode);
  const media = useMediaSource(demoMode ? "SIMULATION" : "VIDEO_FILE");
  const [detectorMode, setDetectorMode] = useState<DetectorInferenceMode>("STANDARD");
  const ingestion = useFrameIngestion({
    videoElement: media.videoElement,
    sourceMode: media.mode === "SIMULATION" ? null : media.mode,
    mediaOrigin: media.mediaOrigin,
    detectorMode,
    sourceReady: media.readyForIngestion,
    captureActive: media.isPlaying,
    sourceComplete: media.mode === "VIDEO_FILE" && media.state === "COMPLETE",
    sourceGeneration: media.generation,
  });
  const [layers, setLayers] = useState<LayerState>(DEFAULT_LAYERS);
  const [selectedZoneId, setSelectedZoneId] = useState<string | null>(null);
  const [reportOpen, setReportOpen] = useState(false);

  const usingSimulation = demoMode && media.mode === "SIMULATION";
  const snapshot = usingSimulation ? demo.snapshot : ingestion.intelligence;
  const finalSummary = !usingSimulation && ingestion.completionState === "COMPLETE"
    ? ingestion.completion?.summary ?? null
    : null;
  const finalPriority = finalSummary?.priorities.find(
    (item) => item.zone.zone_id === finalSummary.highest_priority_zone_id,
  ) ?? finalSummary?.priorities[0] ?? null;
  const railZones = useMemo(
    () => finalSummary
      ? finalSummary.priorities.map((item) => item.zone)
      : snapshot?.zones ?? [],
    [finalSummary, snapshot],
  );
  const railRoute = finalSummary ? finalPriority?.associated_route ?? null : snapshot?.route ?? null;
  const railSystemStatus: SystemStatus | null = finalSummary ? {
    api: "operational",
    segmentation_model: finalSummary.segmentation_status.status,
    detection_model: finalSummary.detection_status.status,
    inference_state: finalSummary.inference_state,
    segmentation_details: finalSummary.segmentation_status,
    detection_details: finalSummary.detection_status,
  } : snapshot?.system_status ?? null;
  const headerIncident: IncidentMetadata = snapshot?.incident ?? {
    incident_id: ingestion.metrics.sessionId ? `LIVE-${ingestion.metrics.sessionId.slice(0, 12)}` : "LIVE-PENDING",
    title: "Live Frame Intelligence",
    location_label: "Normalized image-space assessment",
    started_at_ms: 0,
    coordinate_space: "NORMALIZED_IMAGE",
    data_origin: "DERIVED_ANALYTIC",
  };
  const connectionState = usingSimulation
    ? demo.connectionState
    : actualConnectionState(ingestion.metrics.connectionState, media.state);
  const connectionLabel = usingSimulation
    ? undefined
    : ingestion.completionState === "FINALIZING"
      ? "FINALISING"
      : operationalLabel(ingestion.metrics.connectionState, media.state, Boolean(snapshot));
  const selectedZone = useMemo(
    () => railZones.find((zone) => zone.zone_id === selectedZoneId) ?? null,
    [railZones, selectedZoneId],
  );
  const selectedPriorityObservation = useMemo(
    () => finalSummary?.priorities.find((item) => item.zone.zone_id === selectedZoneId) ?? null,
    [finalSummary, selectedZoneId],
  );
  const selectedZoneIsSimulated = selectedZone?.data_origin === "DEMO_SIMULATED";
  const selectedZoneHasSegmentationEvidence = selectedZoneIsSimulated
    || snapshot?.segmentation.status === "ready"
    || snapshot?.segmentation.status === "simulated";
  const selectedZoneHasDetectionEvidence = selectedZoneIsSimulated
    || snapshot?.system_status.detection_model === "ready";
  const {
    mode: mediaMode,
    state: mediaState,
    videoElement: mediaVideoElement,
    seekToPausedTime,
  } = media;

  useEffect(() => {
    if (
      mediaMode !== "VIDEO_FILE"
      || mediaState !== "COMPLETE"
      || ingestion.completionState !== "COMPLETE"
      || !mediaVideoElement
      || ingestion.completion?.summary.last_media_time_ms === null
      || ingestion.completion?.summary.last_media_time_ms === undefined
    ) {
      return;
    }
    const analyzedTime = Math.max(0, ingestion.completion.summary.last_media_time_ms / 1_000);
    seekToPausedTime(analyzedTime);
  }, [ingestion.completion, ingestion.completionState, mediaMode, mediaState, mediaVideoElement, seekToPausedTime]);

  const toggleLayer = useCallback((layer: LayerKey) => {
    setLayers((current) => ({ ...current, [layer]: !current[layer] }));
  }, []);
  const focusMap = useCallback(() => {
    document.getElementById("tactical-map")?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

  if (demoMode && demo.connectionState === "loading" && !demo.snapshot) return <CommandLoadingState />;
  if (demoMode && (demo.connectionState === "offline" || demo.connectionState === "disconnected") && !demo.snapshot) {
    return <CommandOfflineState message={demo.error ?? "Unable to load the demo scenario."} onRetry={demo.retry} />;
  }
  if (demoMode && (!demo.snapshot || !demo.detail)) {
    return <CommandOfflineState title="Scenario unavailable" message="The backend returned no deterministic incident state." onRetry={demo.retry} />;
  }

  const degradedDemo = ["offline", "disconnected", "malformed"].includes(demo.connectionState);

  return (
    <div className="arctic-shell min-h-screen text-slate-100">
      <WaterAmbience />
      <ApplicationHeader incident={headerIncident} connectionState={connectionState} connectionLabel={connectionLabel} demoMode={demoMode} onOpenReport={() => setReportOpen(true)} />

      <main className="relative z-10 mx-auto max-w-[1680px] px-3 py-3 sm:px-4 lg:px-5">
        <div aria-hidden="true" className="command-grid-bg pointer-events-none fixed inset-0 opacity-20" />

        <div className="command-layout">
          <div className="intelligence-column">
            {demoMode ? (
              <section className="demo-control-bar" aria-label="Demo scenario controls">
                <div><span>Demo scenario</span><ConnectionIndicator state={demo.connectionState} /></div>
                <SimulationControls state={demo.connectionState} snapshotIndex={snapshot?.snapshot_index ?? 0} snapshotCount={snapshot?.snapshot_count ?? 1} onStart={demo.start} onPause={demo.pause} onResume={demo.resume} onReset={demo.reset} />
              </section>
            ) : (
              <MediaSourceSelector
                media={media}
                ingestion={ingestion}
                detectorMode={detectorMode}
                onDetectorModeChange={setDetectorMode}
              />
            )}

            {media.error && <StatusBanner tone="critical" message={media.error} />}
            {ingestion.metrics.lastError && !usingSimulation && <StatusBanner tone={ingestion.metrics.connectionState === "offline" ? "critical" : "warning"} message={ingestion.metrics.lastError} action={ingestion.retry} />}
            {!usingSimulation && ingestion.completionState === "FINALIZING" && <StatusBanner tone="neutral" message="Video ended. Finalizing whole-video findings after pending inference completes…" />}
            {usingSimulation && demo.connectionState === "reconnecting" && <StatusBanner tone="warning" message="Scenario reconnecting…" />}
            {usingSimulation && degradedDemo && <StatusBanner tone="critical" message={`${demo.error ?? "Scenario unavailable."} Last valid intelligence retained.`} action={demo.retry} />}
            {usingSimulation && demo.connectionState === "paused" && <StatusBanner tone="neutral" message="Scenario paused." />}

            <ObservationPanel
              snapshot={snapshot}
              layers={layers}
              selectedZoneId={selectedZoneId}
              onToggleLayer={toggleLayer}
              onSelectZone={setSelectedZoneId}
              mediaMode={media.mode}
              mediaState={media.state}
              mediaOrigin={media.mediaOrigin}
              mediaVideoSrc={media.videoSrc}
              mediaStream={media.stream}
              bindVideoElement={media.bindVideoElement}
              onMediaLoadedMetadata={media.onLoadedMetadata}
              onMediaEnded={media.onEnded}
              ingestion={ingestion}
            />
          </div>

          {(snapshot || finalSummary) && railSystemStatus ? (
            <aside className="command-panel command-rail" aria-label="Incident command intelligence">
              <PriorityList
                embedded
                zones={railZones}
                route={railRoute}
                dataOrigin={finalSummary?.data_origin ?? snapshot?.data_origin ?? "DERIVED_ANALYTIC"}
                systemStatus={railSystemStatus}
                selectedZoneId={selectedZoneId}
                onSelectZone={setSelectedZoneId}
                scope={finalSummary ? "WHOLE_VIDEO" : "CURRENT_FRAME"}
                analyzedFrameCount={finalSummary?.frames_analyzed ?? 0}
                priorityObservations={finalSummary?.priorities}
                prioritiesTruncated={finalSummary?.priorities_truncated ?? false}
              />
              <IncidentOverview embedded snapshot={snapshot} summary={finalSummary} connectionState={connectionState} />
            </aside>
          ) : <PendingIntelligence metrics={ingestion.metrics} mediaState={media.state} />}
        </div>

        {snapshot && (
          <div className="secondary-intelligence">
            <TacticalMap snapshot={snapshot} layers={layers} selectedZoneId={selectedZoneId} onSelectZone={setSelectedZoneId} />
            <EventTimeline events={snapshot.events} />
          </div>
        )}

        <footer className="command-footer">Decision support · Human authority retained</footer>
      </main>

      <ZoneDetailsDrawer
        zone={selectedZone}
        observation={selectedPriorityObservation}
        segmentationEvidenceAvailable={selectedZoneHasSegmentationEvidence}
        detectionEvidenceAvailable={selectedZoneHasDetectionEvidence}
        buildingDamageCountAvailability={selectedZoneIsSimulated ? "AVAILABLE" : "NOT_SUPPORTED"}
        onClose={() => setSelectedZoneId(null)}
        onFocusMap={focusMap}
      />
      {reportOpen && <IncidentReportModal snapshot={snapshot} sessionId={usingSimulation ? null : ingestion.metrics.sessionId} reportRevision={usingSimulation ? undefined : ingestion.completionState} open onClose={() => setReportOpen(false)} />}
    </div>
  );
}

function pendingIntelligenceState(metrics: IngestionMetrics, mediaState: MediaSourceState) {
  if (metrics.analysisStatus === "MODEL_UNAVAILABLE") {
    return {
      label: "MODEL UNAVAILABLE",
      tone: "critical" as const,
      title: "Inference models unavailable",
      description: "Bounding boxes require a configured detection model; rescue priorities require supported detection and/or segmentation evidence. FloodSight has not substituted simulated values.",
    };
  }
  if (metrics.analysisStatus === "ERROR" || metrics.connectionState === "offline" || metrics.connectionState === "malformed") {
    return {
      label: "ANALYSIS UNAVAILABLE",
      tone: "critical" as const,
      title: "Live intelligence unavailable",
      description: "FloodSight could not produce a verified intelligence update for this media.",
    };
  }
  if (metrics.analysisStatus === "MODEL_LOADING") {
    return {
      label: "MODELS LOADING",
      tone: "warning" as const,
      title: "Preparing inference models",
      description: "Bounding boxes and rescue priorities will appear after the configured models are ready and a frame is analysed.",
    };
  }
  if (mediaState === "PAUSED") {
    return {
      label: "ANALYSIS PAUSED",
      tone: "warning" as const,
      title: "Waiting for the first intelligence update",
      description: "Resume the media to submit frames for analysis.",
    };
  }
  return {
    label: "AWAITING INTELLIGENCE",
    tone: "neutral" as const,
    title: "Rescue priority",
    description: "Intelligence appears as media is analysed.",
  };
}

function PendingIntelligence({ metrics, mediaState }: { metrics: IngestionMetrics; mediaState: MediaSourceState }) {
  const pending = pendingIntelligenceState(metrics, mediaState);
  const showRecoveryLinks = pending.tone === "critical";
  return (
    <aside className="command-panel command-rail command-rail-pending" aria-label="Intelligence pending">
      <div>
        <span className={`ready-mark ready-mark-${pending.tone}`}><i />{pending.label}</span>
        <h2>{pending.title}</h2>
        <p>{pending.description}</p>
        {metrics.analysisStatus !== "AWAITING_FRAME" && <p className="pending-model-status">{metrics.modelStatus}</p>}
        {mediaState === "PAUSED" && (
          <p className="pending-paused-guidance">
            Analysis is paused before the first intelligence update. Resume the media to submit a frame.
          </p>
        )}
        {showRecoveryLinks && (
          <nav className="pending-actions" aria-label="Unavailable intelligence actions">
            <a className="command-button command-button-secondary" href="/system">Open system status</a>
            <a className="command-button command-button-ghost" href="/demo">Open DEMO_SIMULATED replay</a>
          </nav>
        )}
      </div>
      <dl className="incident-stat-list" aria-label="Incident metrics awaiting intelligence">
        {["Flood coverage", "People", "Vehicles", "Blocked roads", "Highest priority", "Incident severity"].map((label) => <div key={label}><dt>{label}</dt><dd>—</dd></div>)}
      </dl>
    </aside>
  );
}

function StatusBanner({ tone, message, action }: { tone: "warning" | "critical" | "neutral" | "success"; message: string; action?: () => void }) {
  return <div role="status" className={`status-banner status-banner-${tone}`}><span><Icon name={tone === "critical" ? "alert" : tone === "success" ? "check" : "activity"} />{message}</span>{action && <button type="button" onClick={action}>Retry</button>}</div>;
}
