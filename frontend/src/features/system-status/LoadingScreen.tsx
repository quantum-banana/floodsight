import { WaterAmbience, WaveLoader } from "../../components/WaterAmbience";

export function LoadingScreen() {
  return (
    <main className="arctic-shell command-state p-6 text-slate-100">
      <WaterAmbience />
      <section aria-live="polite" aria-label="Connecting to FloodSight backend" className="relative z-10 text-center">
        <p className="text-lg font-bold tracking-[0.18em] text-slate-900 uppercase">FloodSight</p>
        <WaveLoader />
        <p className="eyebrow mt-5">Command centre</p>
        <h1 className="mt-3 text-xl font-semibold">Preparing intelligence…</h1>
        <p className="mt-2 text-sm text-slate-500">Segmentation / Detection / Operations</p>
      </section>
    </main>
  );
}
