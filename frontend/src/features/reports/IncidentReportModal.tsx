import { useEffect, useRef, useState } from "react";

import { Icon } from "../../components/Icon";
import { OriginBadge } from "../../components/OriginBadge";
import { getDemoIncidentReport } from "../../services/api";
import { getLiveIncidentReport } from "../../services/ingestionApi";
import type { IncidentReport } from "../../types/api";
import type { LiveResult } from "../../types/liveResult";
import { formatTimestamp, severityLabel } from "../../utils/format";
import { buildReportText } from "./reportText";

interface IncidentReportModalProps {
  snapshot: LiveResult | null;
  sessionId: string | null;
  reportRevision?: string;
  open: boolean;
  onClose: () => void;
}

export function IncidentReportModal({
  snapshot,
  sessionId,
  reportRevision = "current",
  open,
  onClose,
}: IncidentReportModalProps) {
  const incidentId = snapshot?.incident_id ?? null;
  const requestKey = sessionId ? `live:${sessionId}:${reportRevision}` : `demo:${incidentId ?? "none"}`;
  const [loaded, setLoaded] = useState<{ key: string; report: IncidentReport } | null>(null);
  const [failure, setFailure] = useState<{ key: string; message: string } | null>(null);
  const [copied, setCopied] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);
  const report = loaded?.key === requestKey ? loaded.report : null;
  const error = failure?.key === requestKey ? failure.message : null;

  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, open]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const request = sessionId
      ? getLiveIncidentReport(sessionId)
      : incidentId
        ? getDemoIncidentReport(incidentId)
        : Promise.reject(new Error("No backend intelligence is available to report."));
    void request
      .then((nextReport) => {
        if (!cancelled) setLoaded({ key: requestKey, report: nextReport });
      })
      .catch((requestError: unknown) => {
        if (!cancelled) {
          setFailure({
            key: requestKey,
            message: requestError instanceof Error
              ? requestError.message
              : "Unable to generate the backend incident report.",
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [incidentId, open, requestKey, sessionId]);

  if (!open) return null;

  const copyReport = async () => {
    if (!report) return;
    try {
      await navigator.clipboard.writeText(buildReportText(report));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1_500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center overflow-y-auto bg-black/70 p-3 backdrop-blur-sm sm:p-6" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section role="dialog" aria-modal="true" aria-labelledby="report-title" className="modal-enter my-auto w-full max-w-3xl overflow-hidden rounded-2xl border border-white/[0.1] bg-[#0a171e] shadow-[0_40px_100px_rgba(0,0,0,.55)]">
        <header className="flex items-start justify-between gap-4 border-b border-white/[0.07] p-5 sm:p-6">
          <div>
            <div className="flex items-center gap-2 text-cyan-300"><Icon name="report" /><span className="text-xs font-bold tracking-[0.18em] uppercase">FloodSight</span></div>
            <h2 id="report-title" className="mt-3 text-xl font-semibold text-white sm:text-2xl">Incident report</h2>
            <p className="mt-1 text-xs text-slate-500">{report?.incident_id ?? snapshot?.incident_id ?? "Pending backend state"} · {report?.title ?? snapshot?.incident.title ?? "Live intelligence"}</p>
          </div>
          <button ref={closeRef} type="button" onClick={onClose} className="command-icon-button" aria-label="Close incident report"><Icon name="close" /></button>
        </header>

        <div className="max-h-[68vh] overflow-y-auto p-5 sm:p-6">
          {!report && !error && <ReportState message="Generating report from backend-owned intelligence…" />}
          {error && <ReportState message={error} error />}
          {report && <ReportBody report={report} />}
        </div>

        <footer className="flex flex-wrap justify-end gap-2 border-t border-white/[0.07] p-4 sm:px-6">
          <button type="button" disabled={!report} onClick={() => void copyReport()} className="command-button command-button-secondary disabled:cursor-not-allowed disabled:opacity-40"><Icon name={copied ? "check" : "clipboard"} />{copied ? "Copied" : "Copy report text"}</button>
          <button type="button" disabled={!report} onClick={() => window.print()} className="command-button command-button-secondary disabled:cursor-not-allowed disabled:opacity-40"><Icon name="print" />Print</button>
          <button type="button" onClick={onClose} className="command-button command-button-primary">Close</button>
        </footer>
      </section>
    </div>
  );
}

function ReportBody({ report }: { report: IncidentReport }) {
  const stats = report.statistics;
  const wholeVideo = report.analysis_scope === "WHOLE_VIDEO";
  const available = (key: string) => !wholeVideo || report.aggregate_availability?.[key] === "AVAILABLE";
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-300/10 bg-amber-300/[0.035] p-3">
        <div><p className="text-[0.6rem] tracking-wider text-slate-600 uppercase">{wholeVideo ? "Whole video findings" : "Backend generated"}</p><p className="mt-1 font-mono text-xs text-slate-300">{formatTimestamp(report.generated_at_ms)} UTC · {wholeVideo ? "Last analyzed frame" : "Frame"} {report.generated_from_frame_id ?? "—"}</p></div>
        <OriginBadge origin={report.data_origin} />
      </div>
      <div className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <ReportMetric label="Severity" value={report.severity_established === false ? "Not established" : severityLabel(report.severity)} />
        <ReportMetric label={wholeVideo ? "Peak flood coverage" : "Flood coverage"} value={available("flooded_area_percent") ? `${stats.flooded_area_percent.value}%` : "Unavailable"} />
        <ReportMetric label={wholeVideo ? "Peak people" : "People"} value={available("people_detected") ? stats.people_detected.value : "Unavailable"} />
        <ReportMetric label={wholeVideo ? "Peak vehicles" : "Vehicles"} value={available("vehicles_detected") ? stats.vehicles_detected.value : "Unavailable"} />
        <ReportMetric label={wholeVideo ? "Peak blocked grid cells" : "Blocked roads"} value={available("blocked_road_cells") ? stats.blocked_roads.value : "Unavailable"} />
        <ReportMetric label="Damaged buildings" value={available("damaged_buildings") ? stats.damaged_buildings.value : "Unavailable"} />
        <ReportMetric label="Critical zones" value={report.critical_zone_count} />
        <ReportMetric label="Top priority" value={report.highest_priority_zone_name ?? "None"} />
      </div>
      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <ReportSection title="Priority explanation" text={report.explanation} />
        <ReportSection
          title={wholeVideo ? "Historical relative access observation" : "Relative access summary"}
          text={wholeVideo ? `Observed during sampled video; verify current conditions. ${report.access_summary}` : report.access_summary}
        />
      </div>
      {wholeVideo && report.priorities_truncated && (
        <p role="note" className="mt-5 rounded-xl border border-amber-300/15 bg-amber-300/[0.05] p-3 text-xs leading-5 text-amber-100">Only the strongest retained priority observations are shown; additional observations were omitted.</p>
      )}
      {report.model_provenance && (
        <div className="mt-5 rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
          <p className="section-label">Model provenance</p>
          <p className="mt-2 font-mono text-xs text-slate-400">Segmentation: {report.model_provenance.segmentation ?? "UNAVAILABLE"} · Detection: {report.model_provenance.detection ?? "UNAVAILABLE"}</p>
        </div>
      )}
      <div className="mt-5 rounded-xl border border-cyan-300/10 bg-cyan-300/[0.035] p-4"><p className="section-label text-cyan-300">Responsible AI</p><p className="mt-2 text-xs leading-5 text-slate-400">{report.responsible_ai_statement}</p></div>
    </>
  );
}

function ReportState({ message, error = false }: { message: string; error?: boolean }) {
  return <div role={error ? "alert" : "status"} className={`rounded-xl border p-6 text-center text-sm ${error ? "border-rose-400/15 text-rose-200" : "border-white/[0.07] text-slate-400"}`}>{message}</div>;
}

function ReportMetric({ label, value }: { label: string; value: string | number }) {
  return <div className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-3"><p className="font-mono text-base font-semibold text-slate-100">{value}</p><p className="mt-1 text-[0.58rem] tracking-wider text-slate-600 uppercase">{label}</p></div>;
}

function ReportSection({ title, text }: { title: string; text: string }) {
  return <section className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4"><h3 className="section-label">{title}</h3><p className="mt-2 text-xs leading-5 text-slate-400">{text}</p></section>;
}
