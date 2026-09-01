import { useCallback, useMemo, useState } from "react";

import { ConnectionIndicator } from "../../components/ConnectionIndicator";
import { Icon } from "../../components/Icon";
import { WaterAmbience } from "../../components/WaterAmbience";
import { useDemoIncident, type ConnectionState } from "../../hooks/useDemoIncident";
import { useFrameIngestion } from "../../hooks/useFrameIngestion";
import { useMediaSource, type MediaSourceState } from "../../hooks/useMediaSource";
import type { IncidentMetadata } from "../../types/liveResult";
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
  if (state === "streaming") return "connected";
  if (state === "creating_session" || state === "connecting") return "connecting";
  if (state === "offline" || state === "malformed") return state;
  return "complete";
};

const operationalLabel = (state: string, mediaState: MediaSourceState, hasIntelligence: boolean) => {
  if (state === "offline") return "OFFLINE";
  if (state === "malformed" || mediaState === "ERROR") return "DEGRADED";
  if (mediaState === "PAUSED") return "PAUSED";
  if (mediaState === "PLAYING") return hasIntelligence ? "LIVE" : "ANALYSING";
  if (state === "creating_session" || state === "connecting") return "ANALYSING";
  return "READY";
};

export function CommandCenter({ demoMode = false }: CommandCenterProps) {
  const demo = useDemoIncident(demoMode);
  const media = useMediaSource(demoMode ? "SIMULATION" : "VIDEO_FILE");
  const ingestion = useFrameIngestion({
    videoElement: media.videoElement,
    sourceMode: media.mode === "SIMULATION" ? null : media.mode,
    mediaOrigin: media.mediaOrigin,
    sourceReady: media.readyForIngestion,
    captureActive: media.isPlaying,
    sourceGeneration: media.generation,
  });
  const [layers, setLayers] = useState<LayerState>(DEFAULT_LAYERS);
  const [selectedZoneId, setSelectedZoneId] = useState<string | null>(null);
  const [reportOpen, setReportOpen] = useState(false);

  const usingSimulation = demoMode && media.mode === "SIMULATION";
  const snapshot = usingSimulation ? demo.snapshot : ingestion.intelligence;
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
    : operationalLabel(ingestion.metrics.connectionState, media.state, Boolean(snapshot));
  const selectedZone = useMemo(
    () => snapshot?.zones.find((zone) => zone.zone_id === selectedZoneId) ?? null,
    [snapshot, selectedZoneId],
  );

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
            ) : <MediaSourceSelector media={media} ingestion={ingestion} />}

            {media.error && <StatusBanner tone="critical" message={media.error} />}
            {ingestion.metrics.lastError && !usingSimulation && <StatusBanner tone={ingestion.metrics.connectionState === "offline" ? "critical" : "warning"} message={ingestion.metrics.lastError} action={ingestion.retry} />}
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

          {snapshot ? (
            <aside className="command-panel command-rail" aria-label="Incident command intelligence">
              <PriorityList embedded zones={snapshot.zones} route={snapshot.route} selectedZoneId={selectedZoneId} onSelectZone={setSelectedZoneId} />
              <IncidentOverview embedded snapshot={snapshot} connectionState={connectionState} />
            </aside>
          ) : <PendingIntelligence />}
        </div>

        {snapshot && (
          <div className="secondary-intelligence">
            <TacticalMap snapshot={snapshot} layers={layers} selectedZoneId={selectedZoneId} onSelectZone={setSelectedZoneId} />
            <EventTimeline events={snapshot.events} />
          </div>
        )}

        <footer className="command-footer">Decision support · Human authority retained</footer>
      </main>

      <ZoneDetailsDrawer zone={selectedZone} onClose={() => setSelectedZoneId(null)} onFocusMap={focusMap} />
      {reportOpen && <IncidentReportModal snapshot={snapshot} sessionId={usingSimulation ? null : ingestion.metrics.sessionId} open onClose={() => setReportOpen(false)} />}
    </div>
  );
}

function PendingIntelligence() {
  return (
    <aside className="command-panel command-rail command-rail-pending" aria-label="Intelligence pending">
      <div><span className="ready-mark"><i />Ready</span><h2>Rescue priority</h2><p>Intelligence appears as media is analysed.</p></div>
      <dl className="incident-stat-list" aria-label="Incident metrics awaiting intelligence">
        {["Flood coverage", "People", "Vehicles", "Blocked roads", "Highest priority", "Incident severity"].map((label) => <div key={label}><dt>{label}</dt><dd>—</dd></div>)}
      </dl>
    </aside>
  );
}

function StatusBanner({ tone, message, action }: { tone: "warning" | "critical" | "neutral" | "success"; message: string; action?: () => void }) {
  return <div role="status" className={`status-banner status-banner-${tone}`}><span><Icon name={tone === "critical" ? "alert" : tone === "success" ? "check" : "activity"} />{message}</span>{action && <button type="button" onClick={action}>Retry</button>}</div>;
}
