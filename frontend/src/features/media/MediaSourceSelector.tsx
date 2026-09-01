import { Icon } from "../../components/Icon";
import type { FrameIngestionController } from "../../hooks/useFrameIngestion";
import type { MediaSourceController, MediaSourceMode } from "../../hooks/useMediaSource";

interface MediaSourceSelectorProps {
  media: MediaSourceController;
  ingestion: FrameIngestionController;
}

const modes: Array<{ mode: MediaSourceMode; label: string }> = [
  { mode: "VIDEO_FILE", label: "Video" },
  { mode: "WEBCAM", label: "Live camera" },
];

const formatBytes = (bytes: number) => bytes >= 1024 * 1024
  ? `${(bytes / 1024 / 1024).toFixed(1)} MB`
  : `${Math.max(1, Math.round(bytes / 1024))} KB`;

const operationalState = (state: MediaSourceController["state"]) => ({
  IDLE: "READY",
  PREPARING: "ANALYSING",
  READY: "READY",
  PLAYING: "LIVE",
  PAUSED: "PAUSED",
  STOPPED: "READY",
  ERROR: "DEGRADED",
})[state];

export function MediaSourceSelector({ media, ingestion }: MediaSourceSelectorProps) {
  const canStart = media.mode === "WEBCAM" || (
    media.mode === "VIDEO_FILE" && Boolean(media.videoSrc) && media.state !== "PREPARING"
  );
  const canRetry = ingestion.metrics.connectionState === "offline" || ingestion.metrics.connectionState === "malformed";
  const hasActiveSource = Boolean(media.fileInfo || media.stream) || !["IDLE", "STOPPED"].includes(media.state);

  return (
    <section className="source-control" aria-label="Media source">
      <div className="source-segment" role="group" aria-label="Media source selector">
        {modes.map((item) => (
          <button
            key={item.mode}
            type="button"
            aria-pressed={media.mode === item.mode}
            className={media.mode === item.mode ? "source-segment-active" : ""}
            onClick={() => media.selectMode(item.mode)}
          >
            {item.mode === "VIDEO_FILE" ? <Icon name="play" /> : <Icon name="eye" />}
            {item.label}
          </button>
        ))}
      </div>

      <div className="source-actions" aria-label="Actual media controls">
        {media.mode === "VIDEO_FILE" && (
          <>
            {media.fileInfo && (
              <span className="source-filename" title={`${media.fileInfo.name} · ${formatBytes(media.fileInfo.size)}`}>
                {media.fileInfo.name}
              </span>
            )}
            <label
              className={`command-button cursor-pointer ${media.fileInfo ? "command-button-ghost" : "command-button-primary"}`}
              htmlFor="video-file-input"
              aria-label={media.fileInfo ? "Change video" : "Choose video"}
            >
              <Icon name={media.fileInfo ? "reset" : "play"} />
              {media.fileInfo ? "Change" : "Choose video"}
            </label>
            <input
              id="video-file-input"
              className="sr-only"
              type="file"
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
            onClick={() => void media.start()}
          >
            <Icon name="play" />{media.mode === "WEBCAM" ? "Start" : "Analyse"}
          </button>
        ) : null}

        {hasActiveSource && (
          <button type="button" className="command-icon-button" onClick={media.stop} aria-label="Stop media" title="Stop"><Icon name="reset" /></button>
        )}
        {canRetry && <button type="button" className="command-button command-button-secondary" onClick={ingestion.retry}>Retry</button>}
      </div>
    </section>
  );
}
