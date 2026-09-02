import { Icon } from "../../components/Icon";
import type { FrameIngestionController } from "../../hooks/useFrameIngestion";
import type { MediaSourceController, MediaSourceMode } from "../../hooks/useMediaSource";
import type { DetectorInferenceMode } from "../../types/ingestion";

interface MediaSourceSelectorProps {
  media: MediaSourceController;
  ingestion: FrameIngestionController;
  detectorMode: DetectorInferenceMode;
  onDetectorModeChange: (mode: DetectorInferenceMode) => void;
}

const modes: Array<{ mode: MediaSourceMode; label: string }> = [
  { mode: "VIDEO_FILE", label: "Video" },
  { mode: "WEBCAM", label: "Live camera" },
];

const detectorModes: Array<{
  mode: DetectorInferenceMode;
  label: string;
  description: string;
}> = [
  { mode: "STANDARD", label: "Standard", description: "Full frame" },
  { mode: "AERIAL", label: "Aerial", description: "Tiled" },
  { mode: "AERIAL_HIGH_RECALL", label: "High recall", description: "Tiled + tracking" },
];

const detectorModeLabel = (mode: DetectorInferenceMode) =>
  detectorModes.find((item) => item.mode === mode)?.label ?? "High recall";

const formatBytes = (bytes: number) => bytes >= 1024 * 1024
  ? `${(bytes / 1024 / 1024).toFixed(1)} MB`
  : `${Math.max(1, Math.round(bytes / 1024))} KB`;

const operationalState = (state: MediaSourceController["state"]) => ({
  IDLE: "READY",
  PREPARING: "ANALYSING",
  READY: "READY",
  PLAYING: "LIVE",
  PAUSED: "PAUSED",
  COMPLETE: "FINAL",
  STOPPED: "READY",
  ERROR: "DEGRADED",
})[state];

export function MediaSourceSelector({
  media,
  ingestion,
  detectorMode,
  onDetectorModeChange,
}: MediaSourceSelectorProps) {
  const canStart = media.mode === "WEBCAM" || (
    media.mode === "VIDEO_FILE" && Boolean(media.videoSrc) && media.state !== "PREPARING"
  );
  const canRetry = ingestion.metrics.connectionState === "offline" || ingestion.metrics.connectionState === "malformed";
  const hasActiveSource = Boolean(media.fileInfo || media.stream) || !["IDLE", "STOPPED"].includes(media.state);
  const completionFinalizing = ingestion.completionState === "FINALIZING";
  const finalizingLockMessage = "Whole-video findings are still finalizing. Wait for completion before changing or stopping the source.";
  const detectorModeLocked = !["IDLE", "READY", "STOPPED"].includes(media.state)
    || ["FINALIZING", "COMPLETE", "ERROR"].includes(ingestion.completionState);
  const detectorLockMessage = "Detection profile is locked for this analysis. Stop or choose a new video to change it.";

  return (
    <section className="source-control" aria-label="Media source">
      <div className="source-segment" role="group" aria-label="Media source selector">
        {modes.map((item) => (
          <button
            key={item.mode}
            type="button"
            aria-pressed={media.mode === item.mode}
            className={media.mode === item.mode ? "source-segment-active" : ""}
            disabled={completionFinalizing}
            title={completionFinalizing ? finalizingLockMessage : undefined}
            onClick={() => media.selectMode(item.mode)}
          >
            {item.mode === "VIDEO_FILE" ? <Icon name="play" /> : <Icon name="eye" />}
            {item.label}
          </button>
        ))}
      </div>

      <div className="source-actions" aria-label="Actual media controls">
        <details
          className={`detector-profile ${detectorModeLocked ? "detector-profile-locked" : ""}`}
          onToggle={(event) => {
            if (detectorModeLocked) event.currentTarget.open = false;
          }}
        >
          <summary
            aria-label={`Detection profile: ${detectorModeLabel(detectorMode)}`}
            aria-disabled={detectorModeLocked}
            title={detectorModeLocked ? detectorLockMessage : undefined}
            onClick={(event) => {
              if (detectorModeLocked) event.preventDefault();
            }}
          >
            <Icon name="eye" />
            Detection: {detectorModeLabel(detectorMode)}
          </summary>
          <div className="detector-profile-menu" role="group" aria-label="Detector inference mode">
            {detectorModes.map((item) => (
              <button
                key={item.mode}
                type="button"
                aria-pressed={detectorMode === item.mode}
                disabled={detectorModeLocked}
                title={detectorModeLocked ? detectorLockMessage : undefined}
                onClick={() => onDetectorModeChange(item.mode)}
              >
                <span>{item.label}</span>
                <small>{item.description}</small>
              </button>
            ))}
          </div>
        </details>

        {media.mode === "VIDEO_FILE" && (
          <>
            {media.fileInfo && (
              <span className="source-filename" title={`${media.fileInfo.name} · ${formatBytes(media.fileInfo.size)}`}>
                {media.fileInfo.name}
              </span>
            )}
            <label
              className={`command-button ${completionFinalizing ? "cursor-not-allowed opacity-50" : "cursor-pointer"} ${media.fileInfo ? "command-button-ghost" : "command-button-primary"}`}
              htmlFor="video-file-input"
              aria-label={media.fileInfo ? "Change video" : "Choose video"}
              aria-disabled={completionFinalizing}
              title={completionFinalizing ? finalizingLockMessage : undefined}
              onClick={(event) => {
                if (completionFinalizing) event.preventDefault();
              }}
            >
              <Icon name={media.fileInfo ? "reset" : "play"} />
              {media.fileInfo ? "Change" : "Choose video"}
            </label>
            <input
              id="video-file-input"
              className="sr-only"
              type="file"
              disabled={completionFinalizing}
              accept="video/mp4,video/webm,video/ogg,video/*"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) media.selectFile(file);
                event.target.value = "";
              }}
            />
          </>
        )}

        {hasActiveSource && <span className={`source-state source-state-${operationalState(media.state).toLowerCase()}`}><span />{operationalState(media.state)}</span>}

        {media.state === "PLAYING" ? (
          <button type="button" className="command-icon-button" onClick={media.pause} aria-label="Pause media"><Icon name="pause" /></button>
        ) : media.state === "PAUSED" ? (
          <button type="button" className="command-button command-button-primary" onClick={() => void media.resume()}><Icon name="play" />Resume</button>
        ) : canStart ? (
          <button
            type="button"
            className="command-button command-button-primary"
            disabled={completionFinalizing}
            title={completionFinalizing ? finalizingLockMessage : undefined}
            onClick={() => void media.start()}
          >
            <Icon name="play" />{media.mode === "WEBCAM" ? "Start" : "Analyse"}
          </button>
        ) : null}

        {hasActiveSource && (
          <button type="button" className="command-icon-button" disabled={completionFinalizing} onClick={media.stop} aria-label="Stop media" title={completionFinalizing ? finalizingLockMessage : "Stop"}><Icon name="reset" /></button>
        )}
        {canRetry && <button type="button" className="command-button command-button-secondary" onClick={ingestion.retry}>Retry</button>}
      </div>
    </section>
  );
}
