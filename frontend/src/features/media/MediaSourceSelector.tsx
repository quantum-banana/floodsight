import { Icon } from "../../components/Icon";
import type { FrameIngestionController } from "../../hooks/useFrameIngestion";
import type {
  MediaSourceController,
  MediaSourceMode,
} from "../../hooks/useMediaSource";

interface MediaSourceSelectorProps {
  media: MediaSourceController;
  ingestion: FrameIngestionController;
}

const modes: Array<{ mode: MediaSourceMode; label: string; detail: string }> = [
  { mode: "SIMULATION", label: "Simulation", detail: "Deterministic replay" },
  { mode: "VIDEO_FILE", label: "Video file", detail: "Local browser media" },
  { mode: "WEBCAM", label: "Webcam", detail: "Device camera" },
];

const formatBytes = (bytes: number) =>
  bytes >= 1024 * 1024
    ? `${(bytes / 1024 / 1024).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} KB`;

export function MediaSourceSelector({ media, ingestion }: MediaSourceSelectorProps) {
  const canStart =
    media.mode === "WEBCAM" ||
    (media.mode === "VIDEO_FILE" && Boolean(media.videoSrc) && media.state !== "PREPARING");
  const actualMode = media.mode !== "SIMULATION";
  const canRetry =
    ingestion.metrics.connectionState === "offline" ||
    ingestion.metrics.connectionState === "malformed";

  return (
    <section className="media-source-panel" aria-labelledby="media-source-heading">
      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-3">
        <div className="mr-1 min-w-36">
          <p id="media-source-heading" className="section-label">Media source</p>
          <p className="mt-0.5 text-[0.64rem] text-slate-600">Choose one active input</p>
        </div>
        <div className="media-mode-tabs" role="group" aria-label="Media source selector">
          {modes.map((item) => (
            <button
              key={item.mode}
              type="button"
              aria-pressed={media.mode === item.mode}
              className={`media-mode-tab ${media.mode === item.mode ? "media-mode-tab-active" : ""}`}
              onClick={() => media.selectMode(item.mode)}
            >
              <span>{item.label}</span>
              <small>{item.detail}</small>
            </button>
          ))}
        </div>
      </div>

      {media.mode === "VIDEO_FILE" && (
        <div className="flex flex-wrap items-center justify-end gap-2">
          <label className="command-button command-button-secondary cursor-pointer" htmlFor="video-file-input">
            <Icon name="play" /> Choose video
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
          {media.fileInfo && (
            <span className="max-w-56 truncate text-[0.68rem] text-slate-400" title={media.fileInfo.name}>
              {media.fileInfo.name} · {formatBytes(media.fileInfo.size)}
              {media.fileInfo.width && media.fileInfo.height
                ? ` · ${media.fileInfo.width}×${media.fileInfo.height}`
                : ""}
              {media.fileInfo.durationSeconds !== null
                ? ` · ${media.fileInfo.durationSeconds.toFixed(1)} s`
                : ""}
            </span>
          )}
        </div>
      )}

      {actualMode && (
        <div className="flex flex-wrap items-center justify-end gap-2" aria-label="Actual media controls">
          <span className="rounded-full border border-white/[0.08] px-2.5 py-1 font-mono text-[0.6rem] text-slate-400">
            SOURCE {media.state}
          </span>
          {media.state === "PLAYING" ? (
            <button type="button" className="command-control" onClick={media.pause}>
              <Icon name="pause" /> Pause
            </button>
          ) : media.state === "PAUSED" ? (
            <button type="button" className="command-control" onClick={() => void media.resume()}>
              <Icon name="play" /> Resume
            </button>
          ) : (
            <button
              type="button"
              className="command-control"
              disabled={!canStart}
              onClick={() => void media.start()}
            >
              <Icon name="play" /> {media.mode === "WEBCAM" ? "Start camera" : "Play"}
            </button>
          )}
          <button
            type="button"
            className="command-control"
            disabled={media.state === "IDLE"}
            onClick={media.stop}
          >
            <Icon name="reset" /> Stop / reset
          </button>
          {canRetry && (
            <button type="button" className="command-control" onClick={ingestion.retry}>
              Retry ingestion
            </button>
          )}
        </div>
      )}
    </section>
  );
}
