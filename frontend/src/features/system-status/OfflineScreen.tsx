import { WaterAmbience } from "../../components/WaterAmbience";

interface OfflineScreenProps {
  message: string;
  onRetry: () => void;
}

export function OfflineScreen({ message, onRetry }: OfflineScreenProps) {
  return (
    <main className="arctic-shell command-state p-6 text-slate-100">
      <WaterAmbience />
      <section role="alert" className="command-panel relative z-10 w-full max-w-lg p-8 text-center sm:p-10">
        <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl border border-rose-400/25 bg-rose-50 text-rose-600">
          <svg aria-hidden="true" className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M5.6 5.6A9 9 0 0 0 3 12m15.4-6.4A9 9 0 0 1 21 12M8.5 9.2A5 5 0 0 0 7 12m8.5-2.8A5 5 0 0 1 17 12m-7 4a2 2 0 1 1 4 0 2 2 0 0 1-4 0ZM3 3l18 18" /></svg>
        </div>
        <p className="mt-6 text-xs font-bold tracking-[0.22em] text-rose-600 uppercase">Backend offline</p>
        <h1 className="mt-3 text-2xl font-semibold">Command link unavailable</h1>
        <p className="mt-3 text-sm leading-6 text-slate-400">{message}</p>
        <p className="mt-2 text-xs leading-5 text-slate-600">Start the FastAPI service and verify the configured backend URL, then retry.</p>
        <button type="button" onClick={onRetry} className="command-button command-button-primary mt-7">Retry connection</button>
      </section>
    </main>
  );
}
