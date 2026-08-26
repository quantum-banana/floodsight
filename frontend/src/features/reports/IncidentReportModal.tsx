import { useEffect, useRef, useState } from "react";

import { Icon } from "../../components/Icon";
import { OriginBadge } from "../../components/OriginBadge";
import type { LiveResult } from "../../types/liveResult";
import { formatTimestamp, severityLabel } from "../../utils/format";
import { buildReportText } from "./reportText";

interface IncidentReportModalProps {
  snapshot: LiveResult | null;
  open: boolean;
  onClose: () => void;
}

export function IncidentReportModal({ snapshot, open, onClose }: IncidentReportModalProps) {
  const [copied, setCopied] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, open]);

  if (!open || !snapshot) return null;
  const stats = snapshot.statistics;
  const highest = snapshot.zones.find((zone) => zone.zone_id === snapshot.highest_priority_zone_id);
  const criticalZones = snapshot.zones.filter((zone) => zone.severity === "CRITICAL").length;

  const copyReport = async () => {
    try {
      await navigator.clipboard.writeText(buildReportText(snapshot));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1_500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center overflow-y-auto bg-black/70 p-3 backdrop-blur-sm sm:p-6" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section role="dialog" aria-modal="true" aria-labelledby="report-title" className="modal-enter my-auto w-full max-w-3xl overflow-hidden rounded-2xl border border-white/[0.1] bg-[#0a171e] shadow-[0_40px_100px_rgba(0,0,0,.55)]">
        <header className="flex items-start justify-between gap-4 border-b border-white/[0.07] p-5 sm:p-6"><div><div className="flex items-center gap-2 text-cyan-300"><Icon name="report" /><span className="text-xs font-bold tracking-[0.18em] uppercase">FloodSight</span></div><h2 id="report-title" className="mt-3 text-xl font-semibold text-white sm:text-2xl">Incident report</h2><p className="mt-1 text-xs text-slate-500">{snapshot.incident_id} · {snapshot.incident.title}</p></div><button ref={closeRef} type="button" onClick={onClose} className="command-icon-button" aria-label="Close incident report"><Icon name="close" /></button></header>
        <div className="max-h-[68vh] overflow-y-auto p-5 sm:p-6">
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-300/10 bg-amber-300/[0.035] p-3"><div><p className="text-[0.6rem] tracking-wider text-slate-600 uppercase">Generated from observation</p><p className="mt-1 font-mono text-xs text-slate-300">{formatTimestamp(snapshot.timestamp_ms)} UTC</p></div><OriginBadge origin={snapshot.data_origin} /></div>
          <div className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-4"><ReportMetric label="Severity" value={severityLabel(snapshot.incident_severity)} /><ReportMetric label="Flood coverage" value={`${stats.flooded_area_percent.value}%`} /><ReportMetric label="People" value={stats.people_detected.value} /><ReportMetric label="Vehicles" value={stats.vehicles_detected.value} /><ReportMetric label="Blocked roads" value={stats.blocked_roads.value} /><ReportMetric label="Damaged buildings" value={stats.damaged_buildings.value} /><ReportMetric label="Critical zones" value={criticalZones} /><ReportMetric label="Top priority" value={highest?.display_name ?? "None"} /></div>
          <div className="mt-5 grid gap-3 sm:grid-cols-2"><ReportSection title="Priority explanation" text={highest?.primary_reason ?? "No rescue zones supplied."} /><ReportSection title="Relative access summary" text={snapshot.route?.access_summary ?? "No relative route supplied in this snapshot."} /></div>
          <div className="mt-5 rounded-xl border border-cyan-300/10 bg-cyan-300/[0.035] p-4"><p className="section-label text-cyan-300">Responsible AI</p><p className="mt-2 text-xs leading-5 text-slate-400">FloodSight is decision support. This simulated report requires human review before any operational action.</p></div>
        </div>
        <footer className="flex flex-wrap justify-end gap-2 border-t border-white/[0.07] p-4 sm:px-6"><button type="button" onClick={() => void copyReport()} className="command-button command-button-secondary"><Icon name={copied ? "check" : "clipboard"} />{copied ? "Copied" : "Copy report text"}</button><button type="button" onClick={() => window.print()} className="command-button command-button-secondary"><Icon name="print" />Print</button><button type="button" onClick={onClose} className="command-button command-button-primary">Close</button></footer>
      </section>
    </div>
  );
}

function ReportMetric({ label, value }: { label: string; value: string | number }) {
  return <div className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-3"><p className="font-mono text-base font-semibold text-slate-100">{value}</p><p className="mt-1 text-[0.58rem] tracking-wider text-slate-600 uppercase">{label}</p></div>;
}

function ReportSection({ title, text }: { title: string; text: string }) {
  return <section className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4"><h3 className="section-label">{title}</h3><p className="mt-2 text-xs leading-5 text-slate-400">{text}</p></section>;
}
