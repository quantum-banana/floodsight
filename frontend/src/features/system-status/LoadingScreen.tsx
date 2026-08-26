export function LoadingScreen() {
  return (
    <main className="relative grid min-h-screen place-items-center overflow-hidden bg-[#071016] p-6 text-slate-100">
      <div aria-hidden="true" className="absolute inset-0 bg-[radial-gradient(circle_at_50%_40%,rgba(34,211,238,0.1),transparent_34%)]" />
      <section aria-live="polite" aria-label="Connecting to FloodSight backend" className="relative text-center">
        <div className="mx-auto grid h-16 w-16 place-items-center rounded-2xl border border-cyan-300/15 bg-cyan-300/[0.05]">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-700 border-t-cyan-300" />
        </div>
        <p className="mt-6 text-xs font-bold tracking-[0.24em] text-cyan-300 uppercase">FloodSight</p>
        <h1 className="mt-3 text-xl font-semibold">Establishing command link</h1>
        <p className="mt-2 text-sm text-slate-500">Checking API and model readiness…</p>
      </section>
    </main>
  );
}

