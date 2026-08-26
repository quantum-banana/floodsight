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

export function ObservationPanel({
  snapshot,
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
}: ObservationPanelProps) {
  if (mediaMode !== "SIMULATION") {
    return (
      <section className="command-panel min-w-0 overflow-hidden" aria-labelledby="observation-heading">
        <div className="panel-heading flex-wrap gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="status-pulse" aria-hidden="true" />
              <h2 id="observation-heading" className="panel-title">Actual media preview</h2>
            </div>
            <p className="panel-subtitle">
              Browser-local {mediaMode === "VIDEO_FILE" ? "video file" : "camera stream"} · frame ingestion only
            </p>
          </div>
          <span className="media-origin-badge" aria-label={`Media origin: ${mediaOrigin}`}>
            {mediaOrigin}
          </span>
        </div>

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
          {!mediaVideoSrc && !mediaStream && (
            <div className="absolute inset-0 grid place-content-center px-6 text-center">
              <Icon name={mediaMode === "WEBCAM" ? "eye" : "play"} className="mx-auto h-8 w-8 text-cyan-300/50" />
              <p className="mt-3 text-sm font-semibold text-slate-300">
                {mediaMode === "WEBCAM" ? "Start the camera to request permission" : "Choose a local video to preview"}
              </p>
              <p className="mt-1 text-xs text-slate-600">Original media never leaves the browser.</p>
            </div>
          )}
          <div className="absolute top-3 left-3 rounded-md border border-emerald-300/20 bg-[#06110e]/85 px-2.5 py-1 font-mono text-[0.62rem] tracking-wider text-emerald-300 uppercase">
            No simulated overlay
          </div>
        </div>

        <IngestionStatusStrip metrics={ingestion.metrics} />
        <div className="border-t border-amber-300/15 bg-amber-300/[0.045] px-4 py-3 text-xs leading-5 text-amber-100">
          <strong>SIMULATED ANALYTICS — NOT DERIVED FROM CURRENT VIDEO.</strong>{" "}
          Only frame decode and basic quality measurements are derived from this media.
        </div>
      </section>
    );
  }

  return (
    <section className="command-panel min-w-0 overflow-hidden" aria-labelledby="observation-heading">
      <div className="panel-heading flex-wrap gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="status-pulse" aria-hidden="true" />
            <h2 id="observation-heading" className="panel-title">Simulated sensor view</h2>
          </div>
          <p className="panel-subtitle">Normalized scene evidence · Snapshot {snapshot.snapshot_index + 1}</p>
        </div>
        <div className="flex items-center gap-2"><span className="font-mono text-[0.65rem] text-slate-500">{formatTimestamp(snapshot.timestamp_ms)} UTC</span><OriginBadge origin={snapshot.data_origin} compact /></div>
      </div>

      <div className="border-b border-white/[0.06] px-4 py-2.5"><LayerControls layers={layers} onToggle={onToggleLayer} compact /></div>

      <div className="sensor-scene relative aspect-[16/8.7] min-h-64 overflow-hidden bg-[#08141a]">
        <OverlayRenderer snapshot={snapshot} layers={layers} selectedZoneId={selectedZoneId} onSelectZone={onSelectZone} />
        <div aria-hidden="true" className="sensor-scan-line" />
        <div className="pointer-events-none absolute right-3 bottom-3 flex items-center gap-2 rounded-lg border border-white/[0.08] bg-[#071016]/80 px-2.5 py-1.5 font-mono text-[0.6rem] tracking-wider text-slate-500 uppercase backdrop-blur">
          <Icon name="eye" className="h-3 w-3 text-cyan-300" /> Synthetic geometry only
        </div>
      </div>
    </section>
  );
}
