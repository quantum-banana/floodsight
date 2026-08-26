import type { ReactNode } from "react";

type StatusTone = "online" | "pending" | "offline" | "neutral";

interface StatusCardProps {
  label: string;
  value: string;
  detail: string;
  tone: StatusTone;
  icon: ReactNode;
}

const toneStyles: Record<StatusTone, { dot: string; value: string; glow: string }> = {
  online: {
    dot: "bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.75)]",
    value: "text-emerald-300",
    glow: "from-emerald-400/12",
  },
  pending: {
    dot: "bg-amber-400 shadow-[0_0_12px_rgba(251,191,36,0.7)]",
    value: "text-amber-300",
    glow: "from-amber-400/10",
  },
  offline: {
    dot: "bg-rose-400 shadow-[0_0_12px_rgba(251,113,133,0.7)]",
    value: "text-rose-300",
    glow: "from-rose-400/10",
  },
  neutral: {
    dot: "bg-slate-500",
    value: "text-slate-300",
    glow: "from-slate-400/8",
  },
};

export function StatusCard({ label, value, detail, tone, icon }: StatusCardProps) {
  const styles = toneStyles[tone];

  return (
    <section
      aria-label={`${label}: ${value}`}
      className="group relative overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0c1820]/85 p-5 shadow-[0_20px_50px_rgba(0,0,0,0.24)] backdrop-blur"
    >
      <div
        aria-hidden="true"
        className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${styles.glow} via-transparent to-transparent opacity-80`}
      />
      <div className="relative flex items-start justify-between gap-4">
        <div>
          <p className="text-[0.68rem] font-semibold tracking-[0.2em] text-slate-500 uppercase">
            {label}
          </p>
          <div className="mt-3 flex items-center gap-2.5">
            <span aria-hidden="true" className={`h-2 w-2 rounded-full ${styles.dot}`} />
            <p className={`text-base font-semibold ${styles.value}`}>{value}</p>
          </div>
          <p className="mt-2 text-sm leading-5 text-slate-500">{detail}</p>
        </div>
        <div className="rounded-xl border border-white/[0.08] bg-white/[0.04] p-2.5 text-slate-400 transition-colors group-hover:text-cyan-300">
          {icon}
        </div>
      </div>
    </section>
  );
}

