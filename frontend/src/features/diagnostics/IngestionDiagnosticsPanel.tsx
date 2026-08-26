import { loadIngestionDiagnostics } from "../../services/ingestionDiagnostics";

const showNumber = (value: number | null, suffix = "") =>
  value === null ? "—" : `${value.toFixed(value % 1 === 0 ? 0 : 1)}${suffix}`;

export function IngestionDiagnosticsPanel() {
  const metrics = loadIngestionDiagnostics();
  const items = metrics
    ? [
        ["Source mode", metrics.sourceMode ?? "INACTIVE"],
        ["Media origin", metrics.mediaOrigin ?? "NONE"],
        ["Session", metrics.sessionId ? `${metrics.sessionId.slice(0, 8)}…` : "NONE"],
        ["Server state", metrics.sessionState ?? "NONE"],
        ["Connection", metrics.connectionState.toUpperCase()],
        ["Requested FPS", metrics.requestedFps.toFixed(1)],
        ["Measured FPS", metrics.measuredFps.toFixed(1)],
        ["Received / accepted", `${metrics.acknowledgedFrames} / ${metrics.acknowledgedFrames - metrics.rejectedFrames}`],
        ["Rejected", metrics.rejectedFrames.toString()],
        ["Client dropped", metrics.clientDroppedFrames.toString()],
        ["Latest frame", metrics.latestFrameId?.toString() ?? "—"],
        ["Dimensions", metrics.latestDimensions ?? "—"],
        ["Blur score", showNumber(metrics.latestBlurScore)],
        ["Luminance", showNumber(metrics.latestLuminance)],
        ["Processing", showNumber(metrics.latestProcessingMs, " ms")],
        ["Last error", metrics.lastError ?? "NONE"],
        ["Models", metrics.modelStatus],
        ["Frame quality", metrics.latestQualityState ?? "AWAITING FRAME"],
        ["Analysis", metrics.analysisStatus],
        ["Metric origin", "DERIVED_ANALYTIC"],
      ]
    : [
        ["Source mode", "INACTIVE"],
        ["Connection", "IDLE"],
        ["Models", "NOT_CONFIGURED"],
        ["Analysis", "DEMO_SIMULATED"],
      ];

  return (
    <section className="mt-6 overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0c1820]/85" aria-labelledby="ingestion-diagnostics-heading">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.07] px-5 py-4 sm:px-6">
        <div>
          <p className="text-[0.68rem] font-semibold tracking-[0.2em] text-cyan-400 uppercase">Phase 2 transport</p>
          <h2 id="ingestion-diagnostics-heading" className="mt-1 text-lg font-semibold text-white">Frame ingestion diagnostics</h2>
        </div>
        <span className="media-origin-badge">DERIVED_ANALYTIC</span>
      </div>
      <div className="grid gap-px bg-white/[0.055] sm:grid-cols-2 lg:grid-cols-4">
        {items.map(([label, value]) => (
          <div key={label} className="min-w-0 bg-[#0b161d] px-5 py-4">
            <p className="text-[0.6rem] font-semibold tracking-[0.12em] text-slate-600 uppercase">{label}</p>
            <p className="mt-1.5 truncate font-mono text-xs text-slate-300" title={value}>{value}</p>
          </div>
        ))}
      </div>
      <div className="border-t border-amber-300/10 bg-amber-300/[0.025] px-5 py-3 text-xs text-slate-500 sm:px-6">
        Metrics contain no raw frame bytes. Frame quality is derived from decoded media; incident analysis remains DEMO_SIMULATED.
      </div>
    </section>
  );
}
