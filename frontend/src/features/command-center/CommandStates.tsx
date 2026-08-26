import { Icon } from "../../components/Icon";

export function CommandLoadingState() {
  return (
    <main className="command-state" aria-live="polite" aria-label="Loading FloodSight command centre">
      <div className="radar-loader" aria-hidden="true"><span /></div>
      <p className="eyebrow mt-6">FloodSight command link</p>
      <h1 className="mt-3 text-2xl font-semibold text-white">Loading deterministic incident</h1>
      <p className="mt-2 text-sm text-slate-500">Validating the Phase 1 backend contract…</p>
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
    <main className="command-state px-5">
      <section role="alert" className="command-panel max-w-lg p-8 text-center sm:p-10">
        <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl border border-rose-400/20 bg-rose-400/[0.08] text-rose-300"><Icon name="alert" className="h-6 w-6" /></span>
        <p className="eyebrow mt-6 text-rose-300">Backend offline</p>
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
    <div className="grid min-h-36 place-items-center rounded-xl border border-dashed border-white/[0.09] bg-white/[0.015] p-6 text-center text-sm text-slate-600">
      {label}
    </div>
  );
}
