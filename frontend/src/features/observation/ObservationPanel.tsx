import { useState } from "react";

import { Icon } from "../../components/Icon";
import type { FrameIngestionController } from "../../hooks/useFrameIngestion";
import type { MediaSourceMode, MediaSourceState } from "../../hooks/useMediaSource";
import type { MediaOrigin } from "../../types/ingestion";
import type { LiveResult } from "../../types/liveResult";
import { formatTimestamp } from "../../utils/format";
import { IngestionStatusStrip } from "../media/IngestionStatusStrip";
import { LayerControls } from "../tactical-map/LayerControls";
import type { LayerKey, LayerState } from "../tactical-map/layers";
import { ModelStatusDetails, ModelStatusPills } from "./ModelStatusPanel";
import { OverlayRenderer } from "./OverlayRenderer";

interface ObservationPanelProps {
  snapshot: LiveResult | null;
  layers: LayerState;
  selectedZoneId: string | null;
  onToggleLayer: (layer: LayerKey) => void;
  onSelectZone: (zoneId: string) => void;
  mediaMode: MediaSourceMode;
  mediaState: MediaSourceState;
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
    return <ActualObservation {...props} segmentationOpacity={segmentationOpacity} onSegmentationOpacityChange={setSegmentationOpacity} />;
  }
  if (!props.snapshot) return null;
  return <SimulatedObservation {...props} snapshot={props.snapshot} />;
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
  mediaState,
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
  const stateLabel = mediaState === "PAUSED"
    ? "PAUSED"
    : intelligence
    ? "LIVE"
    : ingestion.metrics.connectionState === "offline"
      ? "OFFLINE"
      : ingestion.metrics.connectionState === "idle"
          ? "READY"
          : "ANALYSING";

  return (
    <section className="command-panel canvas-panel" aria-labelledby="observation-heading">
      <h2 id="observation-heading" className="sr-only">Actual media and intelligence</h2>
      <div className="actual-media-scene canvas-frame relative aspect-video min-h-64 overflow-hidden bg-black">
        <video
          ref={bindVideoElement}
          src={mediaMode === "VIDEO_FILE" ? (mediaVideoSrc ?? undefined) : undefined}
          muted={mediaMode === "WEBCAM"}
          playsInline
          preload="metadata"
          onLoadedMetadata={onMediaLoadedMetadata}
          onEnded={onMediaEnded}
          className="h-full w-full object-contain"
          aria-label={mediaMode === "VIDEO_FILE" ? "Selected local video preview" : "Live camera preview"}
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

        <div className="canvas-toolbar">
          <LayerControls layers={layers} onToggle={onToggleLayer} />
          <label className="opacity-tool" title="Segmentation mask opacity">
            <Icon name="water" />
            <span className="sr-only">Mask opacity</span>
            <input
              aria-label="Segmentation mask opacity"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={segmentationOpacity}
              onChange={(event) => onSegmentationOpacityChange(Number(event.target.value))}
            />
          </label>
          {intelligence && <ModelStatusPills status={intelligence.system_status} />}
        </div>

        {!mediaVideoSrc && !mediaStream && (
          <div className="canvas-placeholder">
            <Icon name={mediaMode === "WEBCAM" ? "eye" : "play"} />
            <p>{mediaMode === "WEBCAM" ? "Start live camera" : "Choose a video"}</p>
          </div>
        )}

        <span className={`canvas-state canvas-state-${stateLabel.toLowerCase()}`} aria-label={`Actual media state: ${stateLabel}`}>
          <span />{stateLabel}
        </span>
        {mediaOrigin && <span className="sr-only" aria-label={`Media origin: ${mediaOrigin}`}>{mediaOrigin}</span>}
        {intelligence && <SemanticLegend snapshot={intelligence} />}
      </div>

      {intelligence ? (
        <details id="model-details" className="canvas-details">
          <summary className="disclosure-summary">Details</summary>
          <div className="canvas-details-content">
            <section aria-labelledby="model-detail-heading">
              <h3 id="model-detail-heading">Model</h3>
              <ModelStatusDetails status={intelligence.system_status} />
            </section>
            <section aria-labelledby="session-detail-heading">
              <h3 id="session-detail-heading">Session and diagnostics</h3>
              <IngestionStatusStrip metrics={ingestion.metrics} />
              {intelligence.evidence_frames && <EvidenceFrameStatus snapshot={intelligence} />}
            </section>
          </div>
        </details>
      ) : (
        <div className="canvas-awaiting" role="status">Awaiting intelligence</div>
      )}
    </section>
  );
}

function EvidenceFrameStatus({ snapshot }: { snapshot: LiveResult }) {
  const evidence = snapshot.evidence_frames;
  if (!evidence) return null;
  const describe = (source: number | null, reused: boolean) => source === null
    ? "unavailable"
    : `frame ${source}${reused ? " · cached by cadence" : " · current inference"}`;
  return (
    <div className="evidence-freshness" aria-label="Evidence frame freshness">
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
}: ObservationPanelProps & { snapshot: LiveResult }) {
  return (
    <section className="command-panel canvas-panel" aria-labelledby="observation-heading">
      <h2 id="observation-heading" className="sr-only">Demo scenario intelligence</h2>
      <div className="sensor-scene canvas-frame relative aspect-[16/8.7] min-h-64 overflow-hidden">
        <OverlayRenderer snapshot={snapshot} layers={layers} selectedZoneId={selectedZoneId} onSelectZone={onSelectZone} />
        <div className="canvas-toolbar">
          <LayerControls layers={layers} onToggle={onToggleLayer} />
          <ModelStatusPills status={snapshot.system_status} />
        </div>
        <span className="canvas-state canvas-state-demo">Demo · {formatTimestamp(snapshot.timestamp_ms)} UTC</span>
      </div>
      <details id="model-details" className="canvas-details">
        <summary className="disclosure-summary">Details</summary>
        <div className="canvas-details-content"><section><h3>Model</h3><ModelStatusDetails status={snapshot.system_status} /></section></div>
      </details>
    </section>
  );
}

function SemanticLegend({ snapshot }: { snapshot: LiveResult }) {
  const classes = snapshot.segmentation.classes.filter((item) => item.coverage_percent > 0);
  if (!classes.length) return null;
  return (
    <div className="semantic-legend" aria-label="Segmentation class legend">
      {classes.map((item) => {
        const rgb = item.color ?? [148, 163, 184];
        return <span key={`${item.class_id ?? "class"}-${item.label}`}><i style={{ backgroundColor: `rgb(${rgb.join(",")})` }} />{item.label.replaceAll("_", " ")} {item.coverage_percent.toFixed(1)}%</span>;
      })}
    </div>
  );
}
