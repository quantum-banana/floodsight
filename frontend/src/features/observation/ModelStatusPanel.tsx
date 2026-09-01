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
      {label}<span aria-hidden="true" className="model-pill-dot" />{mode}
    </span>
  );
}

function ModelDetail({ label, status }: { label: string; status?: ModelStatus | null }) {
  const latency = status?.latency_ms == null ? "—" : `${status.latency_ms.toFixed(1)} ms`;
  return (
    <section className="model-detail">
      <div>
        <p>{label}</p>
        <strong title={status?.model ?? undefined}>{status?.model ?? "No model loaded"}</strong>
      </div>
      <dl>
        <div><dt>Mode</dt><dd>{status?.mode ?? "UNAVAILABLE"}</dd></div>
        <div><dt>Version</dt><dd>{status?.version ?? "—"}</dd></div>
        <div><dt>Device</dt><dd>{status?.device ?? "—"}</dd></div>
        <div><dt>Latency</dt><dd>{latency}</dd></div>
        <div><dt>Provenance</dt><dd>{status?.provenance_mode ?? "—"}</dd></div>
      </dl>
      {status?.message && <p className="model-detail-message">{status.message}</p>}
    </section>
  );
}

export function ModelStatusPills({ status }: { status: SystemStatus }) {
  return (
    <div className="model-pills" aria-label="Inference model status">
      <ModePill label="SEG" status={status.segmentation_details} />
      <ModePill label="DET" status={status.detection_details} />
    </div>
  );
}

export function ModelStatusDetails({ status }: { status: SystemStatus }) {
  return (
    <div className="model-details-grid">
      <ModelDetail label="Segmentation" status={status.segmentation_details} />
      <ModelDetail label="Detection" status={status.detection_details} />
    </div>
  );
}

export function ModelStatusPanel({ status }: { status: SystemStatus }) {
  return (
    <section className="model-status-panel">
      <ModelStatusPills status={status} />
      <details id="model-details">
        <summary className="disclosure-summary">Model details</summary>
        <ModelStatusDetails status={status} />
      </details>
    </section>
  );
}
