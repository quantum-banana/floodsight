import { Icon } from "../../components/Icon";
import { WaterAmbience, WaveLoader } from "../../components/WaterAmbience";

export function CommandLoadingState() {
  return (
    <main className="arctic-shell command-state" aria-live="polite" aria-label="Loading FloodSight command centre">
      <WaterAmbience />
      <section className="relative z-10 text-center">
        <p className="text-lg font-bold tracking-[0.18em] text-slate-900 uppercase">FloodSight</p>
        <WaveLoader />
        <p className="eyebrow mt-5">Initialising Decision Intelligence</p>
        <h1 className="mt-3 text-2xl font-semibold text-white">Loading deterministic incident</h1>
        <p className="mt-2 text-sm text-slate-500">Segmentation · Detection · Connecting command centre…</p>
      </section>
    </main>
  );
}

interface CommandOfflineStateProps {
  title?: string;
  message: string;
  onRetry: () => void;
}

export function CommandOfflineState({
  title = "Command link unavailable",
  message,
  onRetry,
}: CommandOfflineStateProps) {
  return (
    <main className="arctic-shell command-state px-5">
      <WaterAmbience />
      <section role="alert" className="command-panel relative z-10 max-w-lg p-8 text-center sm:p-10">
        <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl border border-rose-400/25 bg-rose-50 text-rose-600"><Icon name="alert" className="h-6 w-6" /></span>
        <p className="eyebrow mt-6 text-rose-600">Backend offline</p>
        <h1 className="mt-3 text-2xl font-semibold text-white">{title}</h1>
        <p className="mt-3 text-sm leading-6 text-slate-400">{message}</p>
        <p className="mt-2 text-xs leading-5 text-slate-600">No local incident values have been substituted.</p>
        <button type="button" onClick={onRetry} className="command-button command-button-primary mt-7">Retry connection</button>
      </section>
    </main>
  );
}

export function EmptyState({ label }: { label: string }) {
  return (
    <div className="empty-state grid min-h-36 place-items-center rounded-xl border border-dashed p-6 text-center text-sm text-slate-600">
      {label}
    </div>
  );
}
