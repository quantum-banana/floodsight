import type { DataOrigin } from "../types/liveResult";
import { formatOrigin } from "../utils/format";

interface OriginBadgeProps {
  origin: DataOrigin;
  compact?: boolean;
}

export function OriginBadge({ origin, compact = false }: OriginBadgeProps) {
  const simulated = origin === "DEMO_SIMULATED";
  return (
    <span
      aria-label={`Data origin: ${origin}`}
      className={`inline-flex items-center gap-2 rounded-full border font-bold tracking-[0.15em] uppercase ${
        compact ? "px-2.5 py-1 text-[0.6rem]" : "px-3 py-1.5 text-[0.66rem]"
      } ${
        simulated
          ? "border-amber-300/25 bg-amber-300/[0.09] text-amber-200"
          : "border-cyan-300/25 bg-cyan-300/[0.08] text-cyan-200"
      }`}
    >
      <span
        aria-hidden="true"
        className={`h-1.5 w-1.5 rounded-full ${simulated ? "bg-amber-300" : "bg-cyan-300"}`}
      />
      {formatOrigin(origin)}
    </span>
  );
}

