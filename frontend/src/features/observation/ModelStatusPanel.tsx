import type { ModelStatus, SystemStatus } from "../../types/liveResult";

const tone = (status?: ModelStatus | null) => {
  if (status?.status === "ready" && status.mode === "REAL") return "text-emerald-300";
  if (status?.status === "ready") return "text-amber-200";
  if (status?.status === "loading") return "text-cyan-200";
  return "text-rose-200";
};

function ModelLine({ label, status }: { label: string; status?: ModelStatus | null }) {
  const latency = status?.latency_ms == null ? "—" : `${status.latency_ms.toFixed(1)} ms`;
  return (
    <div className="min-w-0 rounded-lg border border-white/[0.06] bg-black/15 px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[0.58rem] font-semibold tracking-wider text-slate-600 uppercase">{label}</span>
        <strong className={`text-[0.62rem] tracking-wide ${tone(status)}`}>{status?.mode ?? "UNAVAILABLE"}</strong>
      </div>
      <p className="mt-1 truncate font-mono text-[0.62rem] text-slate-400" title={status?.model ?? undefined}>
        {status?.model ?? "No model loaded"} · {status?.device ?? "—"} · {latency}
      </p>
    </div>
  );
}

export function ModelStatusPanel({ status }: { status: SystemStatus }) {
  return (
    <div className="grid gap-2 border-t border-white/[0.06] bg-[#08141a] p-3 sm:grid-cols-2" aria-label="Inference model status">
      <ModelLine label="Segmentation" status={status.segmentation_details} />
      <ModelLine label="Detection" status={status.detection_details} />
    </div>
  );
}
