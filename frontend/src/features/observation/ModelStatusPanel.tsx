import type { ModelStatus, SystemStatus } from "../../types/liveResult";

const modeTone = (status?: ModelStatus | null) => {
  if (status?.status === "ready" && status.mode === "REAL") return "model-pill-real";
  if (status?.status === "ready") return "model-pill-fallback";
  if (status?.status === "loading") return "model-pill-loading";
  return "model-pill-unavailable";
};

function ModePill({ label, status }: { label: string; status?: ModelStatus | null }) {
  const mode = status?.mode ?? "UNAVAILABLE";
  return (
    <span className={`model-pill ${modeTone(status)}`} aria-label={`${label} model state: ${mode}`}>
      <span aria-hidden="true" className="model-pill-dot" />
      {label} {mode}
    </span>
  );
}

function ModelLine({ label, status }: { label: string; status?: ModelStatus | null }) {
  const latency = status?.latency_ms == null ? "—" : `${status.latency_ms.toFixed(1)} ms`;
  return (
    <div className="detail-cell min-w-0 rounded-xl border border-sky-950/10 px-3 py-2.5">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[0.6rem] font-semibold tracking-wider text-slate-600 uppercase">{label}</span>
        <strong className="text-[0.62rem] tracking-wide text-slate-700">{status?.mode ?? "UNAVAILABLE"}</strong>
      </div>
      <p className="mt-1 truncate font-mono text-[0.64rem] text-slate-400" title={status?.model ?? undefined}>{status?.model ?? "No model loaded"}</p>
      <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[0.6rem]">
        <div><dt className="text-slate-600">Version</dt><dd className="truncate font-mono text-slate-400">{status?.version ?? "—"}</dd></div>
        <div><dt className="text-slate-600">Device</dt><dd className="truncate font-mono text-slate-400">{status?.device ?? "—"}</dd></div>
        <div><dt className="text-slate-600">Latency</dt><dd className="font-mono text-slate-400">{latency}</dd></div>
        <div><dt className="text-slate-600">Provenance</dt><dd className="truncate font-mono text-slate-400">{status?.provenance_mode ?? "—"}</dd></div>
      </dl>
      {status?.message && <p className="mt-2 text-[0.62rem] leading-4 text-slate-500">{status.message}</p>}
    </div>
  );
}

export function ModelStatusPanel({ status }: { status: SystemStatus }) {
  return (
    <section className="border-t border-sky-950/10 bg-white/45 px-3 py-2.5" aria-label="Inference model status">
      <div className="flex flex-wrap items-center gap-2">
        <span className="mr-1 text-[0.58rem] font-bold tracking-[0.12em] text-slate-600 uppercase">Models</span>
        <ModePill label="SEG" status={status.segmentation_details} />
        <ModePill label="DET" status={status.detection_details} />
      </div>
      <details className="mt-1">
        <summary className="disclosure-summary">Model details</summary>
        <div className="grid gap-2 pb-1 sm:grid-cols-2">
          <ModelLine label="Segmentation" status={status.segmentation_details} />
          <ModelLine label="Detection" status={status.detection_details} />
        </div>
      </details>
    </section>
  );
}
