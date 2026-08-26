import type { IngestionMetrics } from "../../types/ingestion";

export function IngestionStatusStrip({ metrics }: { metrics: IngestionMetrics }) {
  const qualityTone = metrics.latestQualityState?.includes("BLURRY")
    ? "text-amber-200"
    : "text-emerald-300";
  const items = [
    ["Connection", metrics.connectionState.replaceAll("_", " ").toUpperCase()],
    ["Server session", metrics.sessionState ?? "NONE"],
    ["Captured", metrics.capturedFrames.toString()],
    ["Acknowledged", metrics.acknowledgedFrames.toString()],
    ["Client dropped", metrics.clientDroppedFrames.toString()],
    ["Capture FPS", `${metrics.measuredFps.toFixed(1)} / ${metrics.requestedFps.toFixed(1)}`],
    ["Processing", metrics.latestProcessingMs === null ? "—" : `${metrics.latestProcessingMs.toFixed(1)} ms`],
    ["Dimensions", metrics.latestDimensions ?? "—"],
  ];
  return (
    <div className="ingestion-status-grid" aria-label="Frame ingestion status">
      {items.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
      <div>
        <span>Frame quality</span>
        <strong className={qualityTone}>{metrics.latestQualityState ?? "Awaiting frame"}</strong>
      </div>
    </div>
  );
}
