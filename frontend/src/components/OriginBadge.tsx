interface OriginBadgeProps {
  origin: "DEMO_SIMULATED";
}

export function OriginBadge({ origin }: OriginBadgeProps) {
  return (
    <div
      aria-label={`Data origin: ${origin}`}
      className="inline-flex items-center gap-2 rounded-full border border-amber-300/20 bg-amber-300/[0.08] px-3 py-1.5 text-[0.68rem] font-bold tracking-[0.18em] text-amber-200 uppercase"
    >
      <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-amber-300" />
      Demo / Simulated
    </div>
  );
}

