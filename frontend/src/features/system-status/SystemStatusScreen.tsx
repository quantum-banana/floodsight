import { OriginBadge } from "../../components/OriginBadge";
import { StatusCard } from "../../components/StatusCard";
import type { SystemSnapshot } from "../../types/api";
import type { ModelStatus } from "../../types/liveResult";
import { IngestionDiagnosticsPanel } from "../diagnostics/IngestionDiagnosticsPanel";

interface SystemStatusScreenProps {
  snapshot: SystemSnapshot;
}

const iconClassName = "h-5 w-5";

function InterfaceIcon() {
  return (
    <svg aria-hidden="true" className={iconClassName} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <rect x="3" y="4" width="18" height="14" rx="2" />
      <path d="M8 21h8M12 18v3M7 9h4M7 13h7" />
    </svg>
  );
}

function ApiIcon() {
  return (
    <svg aria-hidden="true" className={iconClassName} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <ellipse cx="12" cy="5" rx="7" ry="3" />
      <path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" />
    </svg>
  );
}

function ModelIcon() {
  return (
    <svg aria-hidden="true" className={iconClassName} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M12 3 4.5 7.2 12 11.5l7.5-4.3L12 3Z" />
      <path d="m4.5 12.2 7.5 4.3 7.5-4.3M4.5 17.2l7.5 4.3 7.5-4.3" />
    </svg>
  );
}

const formatModelState = (state: string) =>
  state === "not_configured" ? "Not configured" : state.replaceAll("_", " ");

const modelTone = (model: ModelStatus): "online" | "pending" | "offline" =>
  model.status === "ready" ? "online" : model.status === "loading" ? "pending" : "offline";

const modelDetail = (model: ModelStatus) => [
  model.model ?? "No model loaded",
  model.version,
  model.device,
  model.latency_ms === null ? null : `${model.latency_ms.toFixed(1)} ms`,
].filter(Boolean).join(" · ");

export function SystemStatusScreen({ snapshot }: SystemStatusScreenProps) {
  const { health, models, sample } = snapshot;
  const stats = sample.statistics;
  const statisticItems = [
    ["Flooded area", `${stats.flooded_area_percent.value}%`],
    ["People", stats.people_detected.value.toString()],
    ["Vehicles", stats.vehicles_detected.value.toString()],
    ["Blocked roads", stats.blocked_roads.value.toString()],
    ["Damaged structures", stats.damaged_buildings.value.toString()],
  ];

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#071016] text-slate-100">
      <div aria-hidden="true" className="absolute inset-0 bg-[radial-gradient(circle_at_12%_5%,rgba(17,113,138,0.18),transparent_31%),radial-gradient(circle_at_89%_24%,rgba(20,184,166,0.08),transparent_26%)]" />
      <div aria-hidden="true" className="grid-overlay absolute inset-0 opacity-30" />

      <div className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col px-5 py-6 sm:px-8 lg:px-10">
        <header className="flex flex-wrap items-center justify-between gap-5 border-b border-white/[0.08] pb-6">
          <div className="flex items-center gap-3.5">
            <div className="relative grid h-11 w-11 place-items-center rounded-xl border border-cyan-300/20 bg-cyan-300/[0.08] text-cyan-300 shadow-[0_0_35px_rgba(34,211,238,0.08)]">
              <svg aria-hidden="true" className="h-6 w-6" viewBox="0 0 32 32" fill="none">
                <path d="M6 11.5c5.2-4.7 14.8-4.7 20 0" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                <path d="M8 17c4.1-3.5 11.9-3.5 16 0" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity=".75" />
                <path d="M10 22c3.2-2.4 8.8-2.4 12 0" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity=".5" />
              </svg>
              <span className="absolute -right-1 -bottom-1 h-2.5 w-2.5 rounded-full border-2 border-[#071016] bg-emerald-400" />
            </div>
            <div>
              <p className="text-xl font-bold tracking-[0.14em] text-white uppercase">FloodSight</p>
              <p className="mt-0.5 text-xs tracking-[0.06em] text-slate-500">From Drone Pixels to Rescue Decisions</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <a href="/" className="command-button command-button-secondary">Command center</a>
            <span className="hidden text-[0.68rem] font-semibold tracking-[0.18em] text-slate-600 uppercase sm:inline">Application integration diagnostics</span>
            <span className="rounded-full border border-emerald-400/20 bg-emerald-400/[0.08] px-3 py-1.5 text-[0.68rem] font-bold tracking-[0.16em] text-emerald-300 uppercase">System online</span>
          </div>
        </header>

        <main className="flex-1 py-10 lg:py-14">
          <section className="max-w-3xl">
            <p className="text-xs font-bold tracking-[0.24em] text-cyan-400 uppercase">Development readiness</p>
            <h1 className="mt-4 text-3xl font-semibold tracking-[-0.035em] text-white sm:text-4xl lg:text-5xl">Command infrastructure is connected.</h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-400 sm:text-base">The interface, API, model adapters, temporal intelligence, rescue-zone scoring, and relative routing share validated contracts. Model availability below reflects the currently resolved local artifacts.</p>
          </section>

          <section aria-label="System status" className="mt-9 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatusCard label="Frontend" value="Operational" detail="React interface rendered" tone="online" icon={<InterfaceIcon />} />
            <StatusCard label="Backend connection" value="Connected" detail={`${health.service} · v${health.version}`} tone="online" icon={<ApiIcon />} />
            <StatusCard label="Segmentation model" value={`${models.segmentation.mode} · ${formatModelState(models.segmentation.status)}`} detail={modelDetail(models.segmentation)} tone={modelTone(models.segmentation)} icon={<ModelIcon />} />
            <StatusCard label="Detection model" value={`${models.detection.mode} · ${formatModelState(models.detection.status)}`} detail={modelDetail(models.detection)} tone={modelTone(models.detection)} icon={<ModelIcon />} />
          </section>

          <IngestionDiagnosticsPanel />

          <section className="mt-6 overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0c1820]/85 shadow-[0_24px_70px_rgba(0,0,0,0.28)] backdrop-blur">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.07] px-5 py-4 sm:px-6">
              <div>
                <p className="text-[0.68rem] font-semibold tracking-[0.2em] text-slate-500 uppercase">Contract preview</p>
                <div className="mt-1 flex items-baseline gap-3">
                  <h2 className="text-lg font-semibold text-white">Incident {sample.incident_id}</h2>
                  <span className="text-xs text-slate-600">Frame {sample.frame_id}</span>
                </div>
              </div>
              <OriginBadge origin="DEMO_SIMULATED" />
            </div>

            <div className="grid gap-px bg-white/[0.06] sm:grid-cols-2 lg:grid-cols-5">
              {statisticItems.map(([label, value]) => (
                <div key={label} className="bg-[#0b161d] px-5 py-5 sm:px-6">
                  <p className="text-[0.68rem] font-medium tracking-[0.12em] text-slate-600 uppercase">{label}</p>
                  <p className="mt-2 font-mono text-2xl font-semibold tracking-tight text-slate-200">{value}</p>
                </div>
              ))}
            </div>

            <div className="flex items-start gap-3 border-t border-white/[0.07] bg-amber-300/[0.025] px-5 py-4 text-sm text-slate-500 sm:px-6">
              <svg aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M12 9v4m0 4h.01M10.3 4.3 2.8 17a2 2 0 0 0 1.7 3h15a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0Z" /></svg>
              <p>Preview values exercise the shared data contract. They are simulated and are not model detections or human-verified observations.</p>
            </div>
          </section>
        </main>

        <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-white/[0.07] py-5 text-[0.68rem] tracking-[0.12em] text-slate-600 uppercase">
          <span>Decision support · Human authority retained</span>
          <span>API v{health.version} · REST connected</span>
        </footer>
      </div>
    </div>
  );
}
