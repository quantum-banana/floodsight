import { useEffect, useRef } from "react";

import { Icon } from "../../components/Icon";
import { OriginBadge } from "../../components/OriginBadge";
import type { Zone } from "../../types/liveResult";
import { formatConfidence, formatPercent, formatTimestamp } from "../../utils/format";

interface ZoneDetailsDrawerProps {
  zone: Zone | null;
  onClose: () => void;
  onFocusMap: () => void;
}

export function ZoneDetailsDrawer({ zone, onClose, onFocusMap }: ZoneDetailsDrawerProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!zone) return;
    closeButtonRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, zone]);

  if (!zone) return null;
  const strandedAlert = zone.alerts?.find(
    (alert) => alert.code === "POTENTIAL_STRANDED_PERSON",
  );

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/55 backdrop-blur-[2px]" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <aside role="dialog" aria-modal="true" aria-labelledby="zone-drawer-title" className="drawer-enter flex h-full w-full max-w-md flex-col border-l border-white/[0.1] bg-[#09151c] shadow-[-30px_0_80px_rgba(0,0,0,.45)]">
        <div className="flex items-start justify-between gap-4 border-b border-white/[0.07] p-5">
          <div><p className="eyebrow">Rescue zone detail</p><h2 id="zone-drawer-title" className="mt-2 text-xl font-semibold text-white">{zone.display_name}</h2><p className="mt-1 font-mono text-[0.66rem] text-cyan-300">{zone.zone_id}</p><p className="mt-1 text-xs text-slate-500">Rank #{zone.rank} · Updated {formatTimestamp(zone.updated_at_ms)} UTC</p></div>
          <button ref={closeButtonRef} type="button" onClick={onClose} className="command-icon-button" aria-label="Close zone details"><Icon name="close" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-5">
          <div className="grid grid-cols-3 gap-2">
            <DrawerMetric label="Priority" value={`${zone.priority_score}/100`} />
            <DrawerMetric label="Category" value={zone.severity} />
            <DrawerMetric label="Confidence" value={formatConfidence(zone.confidence)} />
          </div>
          {strandedAlert && (
            <section
              className="mt-4 rounded-xl border border-rose-300/25 bg-rose-300/[0.07] p-4"
              aria-label="Potential stranded person alert"
            >
              <p className="text-[0.62rem] font-bold tracking-[0.14em] text-rose-200 uppercase">
                Potential stranded person
              </p>
              <p className="mt-1 text-sm font-semibold text-white">{zone.display_name}</p>
              <dl className="mt-3 grid grid-cols-3 gap-2">
                <AlertFact label="Person evidence" value={strandedAlert.person_evidence} />
                <AlertFact label="Flood exposure" value={strandedAlert.flood_exposure} />
                <AlertFact label="Primary access" value={strandedAlert.primary_access} />
              </dl>
              <p className="mt-3 text-[0.68rem] leading-5 text-slate-400">
                Model-driven potential-risk signal only. Trained emergency personnel must verify.
              </p>
              <ul className="mt-2 space-y-1 font-mono text-[0.62rem] text-rose-100/80">
                {strandedAlert.reason_codes.map((code) => <li key={code}>{code}</li>)}
              </ul>
            </section>
          )}
          <dl className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-white/[0.06] bg-white/[0.06]">
            <Fact label="People" value={zone.people_count} /><Fact label="Vehicles" value={zone.vehicle_count} /><Fact label="Flood coverage" value={formatPercent(zone.flood_coverage_percent)} /><Fact label="Building instances" value={zone.building_damage_count} /><Fact label="Damage coverage" value={formatPercent(zone.building_damage_coverage_percent ?? 0)} /><Fact label="Pool coverage" value={formatPercent(zone.pool_coverage_percent ?? 0)} /><Fact label="Road condition" value={zone.road_condition === "UNKNOWN" ? "UNCERTAIN" : zone.road_condition} /><Fact label="Access status" value={zone.access_status} /><Fact label="Grid cells" value={zone.grid_cells?.join(", ") || "Not supplied"} /><Fact label="Temporal samples" value={zone.temporal_samples ?? 1} /><Fact label="Temporal state" value={zone.stale ? "STALE" : "CURRENT"} />
          </dl>
          <section className="mt-6" aria-labelledby="reason-heading"><h3 id="reason-heading" className="section-label">Priority explanation</h3><p className="mt-2 text-sm leading-6 text-slate-300">{zone.primary_reason}</p><div className="mt-4 space-y-4">{zone.reasons.map((reason) => <div key={reason.code}><div className="flex items-center justify-between gap-3 text-xs"><span className="font-medium text-slate-300">{reason.label}</span><span className="font-mono text-slate-200">+{reason.contribution}</span></div><div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-white/[0.05]"><div className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-cyan-300" style={{ width: `${reason.contribution}%` }} /></div><p className="mt-1.5 text-[0.68rem] text-slate-600">{reason.description}</p></div>)}</div></section>
          <div className="mt-6 flex items-center justify-between gap-3 rounded-xl border border-amber-300/10 bg-amber-300/[0.035] p-3"><span className="text-[0.66rem] text-slate-500">Supplied contribution total: <strong className="text-slate-300">{zone.reasons.reduce((sum, item) => sum + item.contribution, 0)}</strong></span><OriginBadge origin={zone.data_origin} compact /></div>
        </div>
        <div className="grid grid-cols-2 gap-2 border-t border-white/[0.07] p-4"><button type="button" onClick={onFocusMap} className="command-button command-button-primary"><Icon name="focus" />Focus on map</button><button type="button" onClick={onClose} className="command-button command-button-secondary">Close</button></div>
      </aside>
    </div>
  );
}

function DrawerMetric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-3 text-center"><p className="font-mono text-sm font-semibold text-white">{value}</p><p className="mt-1 text-[0.58rem] tracking-wider text-slate-600 uppercase">{label}</p></div>;
}

function Fact({ label, value }: { label: string; value: string | number }) {
  return <div className="bg-[#0b171e] p-3"><dt className="text-[0.6rem] tracking-wide text-slate-600 uppercase">{label}</dt><dd className="mt-1 text-xs font-medium text-slate-300">{value}</dd></div>;
}

function AlertFact({ label, value }: { label: string; value: string }) {
  return <div><dt className="text-[0.55rem] tracking-wide text-slate-500 uppercase">{label}</dt><dd className="mt-1 text-[0.66rem] font-bold text-rose-100">{value}</dd></div>;
}
