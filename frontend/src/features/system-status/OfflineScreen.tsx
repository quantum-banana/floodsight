interface OfflineScreenProps {
  message: string;
  onRetry: () => void;
}

export function OfflineScreen({ message, onRetry }: OfflineScreenProps) {
  return (
    <main className="relative grid min-h-screen place-items-center overflow-hidden bg-[#071016] p-6 text-slate-100">
      <div aria-hidden="true" className="absolute inset-0 bg-[radial-gradient(circle_at_50%_42%,rgba(251,113,133,0.09),transparent_32%)]" />
      <section role="alert" className="relative w-full max-w-lg rounded-3xl border border-rose-400/15 bg-[#0c1820]/90 p-8 text-center shadow-[0_30px_90px_rgba(0,0,0,0.4)] sm:p-10">
        <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl border border-rose-400/20 bg-rose-400/[0.08] text-rose-300">
          <svg aria-hidden="true" className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M5.6 5.6A9 9 0 0 0 3 12m15.4-6.4A9 9 0 0 1 21 12M8.5 9.2A5 5 0 0 0 7 12m8.5-2.8A5 5 0 0 1 17 12m-7 4a2 2 0 1 1 4 0 2 2 0 0 1-4 0ZM3 3l18 18" /></svg>
        </div>
        <p className="mt-6 text-xs font-bold tracking-[0.22em] text-rose-300 uppercase">Backend offline</p>
        <h1 className="mt-3 text-2xl font-semibold">Command link unavailable</h1>
        <p className="mt-3 text-sm leading-6 text-slate-400">{message}</p>
        <p className="mt-2 text-xs leading-5 text-slate-600">Start the FastAPI service and verify the configured backend URL, then retry.</p>
        <button type="button" onClick={onRetry} className="mt-7 inline-flex min-h-11 items-center justify-center rounded-xl border border-cyan-300/20 bg-cyan-300/[0.1] px-5 text-sm font-semibold text-cyan-200 transition hover:border-cyan-300/40 hover:bg-cyan-300/[0.15] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300">Retry connection</button>
      </section>
    </main>
  );
}

