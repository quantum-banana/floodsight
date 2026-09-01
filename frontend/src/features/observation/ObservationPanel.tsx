import { useState } from "react";

import { Icon } from "../../components/Icon";
import { OriginBadge } from "../../components/OriginBadge";
import type { FrameIngestionController } from "../../hooks/useFrameIngestion";
import type { MediaSourceMode } from "../../hooks/useMediaSource";
import type { MediaOrigin } from "../../types/ingestion";
import type { LiveResult } from "../../types/liveResult";
import { formatTimestamp } from "../../utils/format";
import { IngestionStatusStrip } from "../media/IngestionStatusStrip";
import { LayerControls } from "../tactical-map/LayerControls";
import type { LayerKey, LayerState } from "../tactical-map/layers";
import { ModelStatusPanel } from "./ModelStatusPanel";
import { OverlayRenderer } from "./OverlayRenderer";

interface ObservationPanelProps {
  snapshot: LiveResult;
  layers: LayerState;
  selectedZoneId: string | null;
  onToggleLayer: (layer: LayerKey) => void;
  onSelectZone: (zoneId: string) => void;
  mediaMode: MediaSourceMode;
  mediaOrigin: MediaOrigin | null;
  mediaVideoSrc: string | null;
  mediaStream: MediaStream | null;
  bindVideoElement: (element: HTMLVideoElement | null) => void;
  onMediaLoadedMetadata: () => void;
  onMediaEnded: () => void;
  ingestion: FrameIngestionController;
}

export function ObservationPanel(props: ObservationPanelProps) {
  const [segmentationOpacity, setSegmentationOpacity] = useState(0.42);
  if (props.mediaMode !== "SIMULATION") {
    return (
      <ActualObservation
        {...props}
        segmentationOpacity={segmentationOpacity}
        onSegmentationOpacityChange={setSegmentationOpacity}
      />
    );
  }
  return <SimulatedObservation {...props} />;
}

interface ActualObservationProps extends ObservationPanelProps {
  segmentationOpacity: number;
  onSegmentationOpacityChange: (value: number) => void;
}

function ActualObservation({
  layers,
  selectedZoneId,
  onToggleLayer,
  onSelectZone,
  mediaMode,
  mediaOrigin,
  mediaVideoSrc,
  mediaStream,
  bindVideoElement,
  onMediaLoadedMetadata,
  onMediaEnded,
  ingestion,
  segmentationOpacity,
  onSegmentationOpacityChange,
}: ActualObservationProps) {
  const intelligence = ingestion.intelligence;
  const isSimulatedFallback = intelligence?.data_origin === "DEMO_SIMULATED";
  return (
    <section className="command-panel min-w-0 overflow-hidden" aria-labelledby="observation-heading">
      <div className="panel-heading flex-wrap gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="status-pulse" aria-hidden="true" />
            <h2 id="observation-heading" className="panel-title">Actual media and intelligence</h2>
          </div>
          <p className="panel-subtitle">Browser-local {mediaMode === "VIDEO_FILE" ? "video file" : "camera stream"} · backend-owned analysis</p>
        </div>
        <span className="media-origin-badge" aria-label={`Media origin: ${mediaOrigin}`}>{mediaOrigin}</span>
      </div>

      <div className="border-b border-white/[0.06] px-4 py-2.5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <LayerControls layers={layers} onToggle={onToggleLayer} compact />
          <label className="flex items-center gap-2 text-[0.62rem] text-slate-500">
            Mask opacity
            <input
              aria-label="Segmentation mask opacity"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={segmentationOpacity}
              onChange={(event) => onSegmentationOpacityChange(Number(event.target.value))}
              className="w-24 accent-cyan-400"
            />
          </label>
        </div>
      </div>

      {intelligence && <ModelStatusPanel status={intelligence.system_status} />}

      <div className="actual-media-scene relative aspect-video min-h-64 overflow-hidden bg-black">
        <video
          ref={bindVideoElement}
          src={mediaMode === "VIDEO_FILE" ? (mediaVideoSrc ?? undefined) : undefined}
          muted={mediaMode === "WEBCAM"}
          playsInline
          preload="metadata"
          onLoadedMetadata={onMediaLoadedMetadata}
          onEnded={onMediaEnded}
          className="h-full w-full object-contain"
          aria-label={mediaMode === "VIDEO_FILE" ? "Selected local video preview" : "Webcam preview"}
        />
        {intelligence && (
          <OverlayRenderer
            snapshot={intelligence}
            layers={layers}
            selectedZoneId={selectedZoneId}
            onSelectZone={onSelectZone}
            showBase={false}
            segmentationOpacity={segmentationOpacity}
            simulated={isSimulatedFallback}
          />
        )}
        {!mediaVideoSrc && !mediaStream && (
          <div className="absolute inset-0 grid place-content-center px-6 text-center">
            <Icon name={mediaMode === "WEBCAM" ? "eye" : "play"} className="mx-auto h-8 w-8 text-cyan-300/50" />
            <p className="mt-3 text-sm font-semibold text-slate-300">{mediaMode === "WEBCAM" ? "Start the camera to request permission" : "Choose a local video to preview"}</p>
            <p className="mt-1 text-xs text-slate-600">Original media never leaves the browser.</p>
          </div>
        )}
        <div className={`absolute top-3 left-3 rounded-md border bg-[#06110e]/85 px-2.5 py-1 font-mono text-[0.62rem] tracking-wider uppercase ${
          isSimulatedFallback
            ? "border-amber-300/20 text-amber-200"
            : intelligence
              ? "border-emerald-300/20 text-emerald-300"
              : "border-white/10 text-slate-400"
        }`}>
          {isSimulatedFallback
            ? "SIMULATED FALLBACK"
            : intelligence
              ? `BACKEND INTELLIGENCE · FRAME ${intelligence.frame_id}`
              : ingestion.metrics.analysisStatus.replaceAll("_", " ")}
        </div>
      </div>

      {intelligence && <SemanticLegend snapshot={intelligence} />}
      <details className="border-t border-sky-950/10 px-3 py-1">
        <summary className="disclosure-summary">Inference diagnostics</summary>
        <div className="-mx-3 border-t border-sky-950/10">
          <IngestionStatusStrip metrics={ingestion.metrics} />
          {intelligence?.evidence_frames && <EvidenceFrameStatus snapshot={intelligence} />}
        </div>
      </details>
      {!intelligence && (
        <div className="border-t border-white/[0.06] bg-white/[0.02] px-4 py-3 text-xs leading-5 text-slate-400">
          {ingestion.metrics.analysisStatus === "MODEL_UNAVAILABLE"
            ? "No inference overlay is shown because the configured model artifacts are unavailable."
            : "Awaiting the first backend-computed intelligence update. No local analytics are substituted."}
        </div>
      )}
    </section>
  );
}

function EvidenceFrameStatus({ snapshot }: { snapshot: LiveResult }) {
  const evidence = snapshot.evidence_frames;
  if (!evidence) return null;
  const describe = (source: number | null, reused: boolean) => source === null
    ? "unavailable"
    : `frame ${source}${reused ? " · cached by configured cadence" : " · current inference"}`;
  return (
    <div className="grid gap-2 border-t border-white/[0.06] px-4 py-3 font-mono text-[0.62rem] text-slate-500 sm:grid-cols-2" aria-label="Evidence frame freshness">
      <span>Segmentation: {describe(evidence.segmentation_source_frame_id, evidence.segmentation_reused)}</span>
      <span>Detection: {describe(evidence.detection_source_frame_id, evidence.detection_reused)}</span>
    </div>
  );
}

function SimulatedObservation({
  snapshot,
  layers,
  selectedZoneId,
  onToggleLayer,
  onSelectZone,
}: ObservationPanelProps) {
  return (
    <section className="command-panel min-w-0 overflow-hidden" aria-labelledby="observation-heading">
      <div className="panel-heading flex-wrap gap-3">
        <div>
          <div className="flex items-center gap-2"><span className="status-pulse" aria-hidden="true" /><h2 id="observation-heading" className="panel-title">Simulated sensor view</h2></div>
          <p className="panel-subtitle">Normalized scene evidence · Snapshot {snapshot.snapshot_index + 1}</p>
        </div>
        <div className="flex items-center gap-2"><span className="font-mono text-[0.65rem] text-slate-500">{formatTimestamp(snapshot.timestamp_ms)} UTC</span><OriginBadge origin={snapshot.data_origin} compact /></div>
      </div>
      <div className="border-b border-white/[0.06] px-4 py-2.5"><LayerControls layers={layers} onToggle={onToggleLayer} compact /></div>
      <ModelStatusPanel status={snapshot.system_status} />
      <div className="sensor-scene relative aspect-[16/8.7] min-h-64 overflow-hidden bg-[#08141a]">
        <OverlayRenderer snapshot={snapshot} layers={layers} selectedZoneId={selectedZoneId} onSelectZone={onSelectZone} />
        <div aria-hidden="true" className="sensor-scan-line" />
        <div className="pointer-events-none absolute right-3 bottom-3 flex items-center gap-2 rounded-lg border border-white/[0.08] bg-[#071016]/80 px-2.5 py-1.5 font-mono text-[0.6rem] tracking-wider text-slate-500 uppercase backdrop-blur"><Icon name="eye" className="h-3 w-3 text-cyan-300" /> Synthetic geometry only</div>
      </div>
    </section>
  );
}

function SemanticLegend({ snapshot }: { snapshot: LiveResult }) {
  const classes = snapshot.segmentation.classes.filter((item) => item.coverage_percent > 0);
  if (!classes.length) return null;
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-2 border-t border-white/[0.06] px-4 py-3 text-[0.62rem] text-slate-500" aria-label="Segmentation class legend">
      {classes.map((item) => {
        const rgb = item.color ?? [148, 163, 184];
        return (
          <span key={`${item.class_id ?? "class"}-${item.label}`} className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: `rgb(${rgb.join(",")})` }} />
            {item.label.replaceAll("_", " ")} · {item.coverage_percent.toFixed(1)}%
          </span>
        );
      })}
    </div>
  );
}
