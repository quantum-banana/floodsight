import { useCallback, useMemo, useState } from "react";

import { ConnectionIndicator } from "../../components/ConnectionIndicator";
import { Icon } from "../../components/Icon";
import { useDemoIncident } from "../../hooks/useDemoIncident";
import { useFrameIngestion } from "../../hooks/useFrameIngestion";
import { useMediaSource } from "../../hooks/useMediaSource";
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

export function CommandCenter() {
  const demo = useDemoIncident();
  const media = useMediaSource();
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

  const selectedZone = useMemo(
    () => demo.snapshot?.zones.find((zone) => zone.zone_id === selectedZoneId) ?? null,
    [demo.snapshot, selectedZoneId],
  );

  const toggleLayer = useCallback((layer: LayerKey) => {
    setLayers((current) => ({ ...current, [layer]: !current[layer] }));
  }, []);

  const focusMap = useCallback(() => {
    document.getElementById("tactical-map")?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

  if (demo.connectionState === "loading" && !demo.snapshot) return <CommandLoadingState />;
  if ((demo.connectionState === "offline" || demo.connectionState === "disconnected") && !demo.snapshot) {
    return <CommandOfflineState message={demo.error ?? "Unable to load the deterministic incident."} onRetry={demo.retry} />;
  }
  if (!demo.snapshot || !demo.detail) {
    return <CommandOfflineState title="Incident payload unavailable" message="The backend returned no deterministic incident state." onRetry={demo.retry} />;
  }

  const snapshot = demo.snapshot;
  const degraded = ["offline", "disconnected", "malformed"].includes(demo.connectionState);

  return (
    <div className="min-h-screen bg-[#060e13] text-slate-100">
      <ApplicationHeader incident={snapshot.incident} connectionState={demo.connectionState} onOpenReport={() => setReportOpen(true)} />
      <main className="relative mx-auto max-w-[1600px] px-3 py-3 sm:px-4 lg:px-5">
        <div aria-hidden="true" className="command-grid-bg pointer-events-none fixed inset-0 opacity-30" />

        <MediaSourceSelector media={media} ingestion={ingestion} />

        {media.mode === "SIMULATION" && <section className="relative mb-3 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/[0.06] bg-[#0a161d]/88 px-3 py-2.5 backdrop-blur sm:px-4" aria-label="Simulation status and controls">
          <div className="flex items-center gap-3"><div className="sm:hidden"><ConnectionIndicator state={demo.connectionState} /></div><div className="hidden sm:block"><p className="text-[0.62rem] font-semibold tracking-[0.15em] text-slate-600 uppercase">Deterministic incident replay</p><p className="mt-0.5 text-xs text-slate-400">Six fixed snapshots · No ML inference</p></div></div>
          <SimulationControls state={demo.connectionState} snapshotIndex={snapshot.snapshot_index} snapshotCount={snapshot.snapshot_count} onStart={demo.start} onPause={demo.pause} onResume={demo.resume} onReset={demo.reset} />
        </section>}

        {media.error && <StatusBanner tone="critical" message={media.error} />}
        {ingestion.metrics.lastError && media.mode !== "SIMULATION" && (
          <StatusBanner
            tone={ingestion.metrics.connectionState === "offline" ? "critical" : "warning"}
            message={ingestion.metrics.lastError}
            action={ingestion.retry}
          />
        )}

        {demo.connectionState === "reconnecting" && <StatusBanner tone="warning" message="Stream interrupted. Reconnecting to the next deterministic snapshot…" />}
        {degraded && <StatusBanner tone="critical" message={`${demo.error ?? "The stream is unavailable."} Current values are retained from the last valid backend message; no local values were substituted.`} action={demo.retry} />}
        {demo.connectionState === "paused" && <StatusBanner tone="neutral" message="Simulation paused. The current backend snapshot is held until you resume." />}
        {demo.connectionState === "complete" && <StatusBanner tone="success" message="Deterministic replay complete. Reset or resume to replay the scenario." />}

        <div className="relative grid min-w-0 gap-3 xl:grid-cols-[minmax(0,1.7fr)_22rem]">
          <ObservationPanel snapshot={snapshot} layers={layers} selectedZoneId={selectedZoneId} onToggleLayer={toggleLayer} onSelectZone={setSelectedZoneId} mediaMode={media.mode} mediaOrigin={media.mediaOrigin} mediaVideoSrc={media.videoSrc} mediaStream={media.stream} bindVideoElement={media.bindVideoElement} onMediaLoadedMetadata={media.onLoadedMetadata} onMediaEnded={media.onEnded} ingestion={ingestion} />
          <IncidentOverview snapshot={snapshot} connectionState={demo.connectionState} />
        </div>

        <div className="relative mt-3 grid min-w-0 gap-3 lg:grid-cols-[23rem_minmax(0,1fr)] xl:grid-cols-[25rem_minmax(0,1fr)]">
          <PriorityList zones={snapshot.zones} selectedZoneId={selectedZoneId} onSelectZone={setSelectedZoneId} />
          <TacticalMap snapshot={snapshot} layers={layers} selectedZoneId={selectedZoneId} onToggleLayer={toggleLayer} onSelectZone={setSelectedZoneId} />
        </div>

        <div className="relative mt-3"><EventTimeline events={snapshot.events} /></div>
        <footer className="relative flex flex-wrap items-center justify-between gap-3 px-1 py-5 text-[0.62rem] tracking-[0.12em] text-slate-700 uppercase"><span>Decision support · Human authority retained</span><span>Phase 2 · Browser frame ingestion · No inference</span></footer>
      </main>

      <ZoneDetailsDrawer zone={selectedZone} onClose={() => setSelectedZoneId(null)} onFocusMap={focusMap} />
      <IncidentReportModal snapshot={snapshot} open={reportOpen} onClose={() => setReportOpen(false)} />
    </div>
  );
}

function StatusBanner({ tone, message, action }: { tone: "warning" | "critical" | "neutral" | "success"; message: string; action?: () => void }) {
  const styles = { warning: "border-amber-300/15 bg-amber-300/[0.05] text-amber-100", critical: "border-rose-400/15 bg-rose-400/[0.05] text-rose-100", neutral: "border-cyan-300/15 bg-cyan-300/[0.04] text-cyan-100", success: "border-emerald-400/15 bg-emerald-400/[0.04] text-emerald-100" }[tone];
  return <div role="status" className={`relative mb-3 flex items-center justify-between gap-3 rounded-xl border px-4 py-3 text-xs ${styles}`}><span className="flex items-center gap-2"><Icon name={tone === "critical" ? "alert" : tone === "success" ? "check" : "activity"} />{message}</span>{action && <button type="button" onClick={action} className="shrink-0 font-semibold text-cyan-300 underline decoration-cyan-300/30 underline-offset-4">Retry</button>}</div>;
}
