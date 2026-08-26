import { ConnectionIndicator } from "../../components/ConnectionIndicator";
import { Icon } from "../../components/Icon";
import { OriginBadge } from "../../components/OriginBadge";
import type { ConnectionState } from "../../hooks/useDemoIncident";
import type { IncidentMetadata } from "../../types/liveResult";

interface ApplicationHeaderProps {
  incident: IncidentMetadata;
  connectionState: ConnectionState;
  onOpenReport: () => void;
}

export function ApplicationHeader({
  incident,
  connectionState,
  onOpenReport,
}: ApplicationHeaderProps) {
  return (
    <header className="command-header sticky top-0 z-30 border-b border-white/[0.07] bg-[#071016]/94 backdrop-blur-xl">
      <div className="mx-auto flex min-h-16 max-w-[1600px] items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <div className="flex min-w-0 items-center gap-3 sm:gap-5">
          <a href="/" className="flex shrink-0 items-center gap-2.5 rounded-lg focus-visible:outline-2 focus-visible:outline-cyan-300">
            <span className="grid h-9 w-9 place-items-center rounded-lg border border-cyan-300/20 bg-cyan-300/[0.07] text-cyan-300">
              <Icon name="water" className="h-5 w-5" />
            </span>
            <span className="hidden text-sm font-bold tracking-[0.16em] text-white uppercase sm:inline">FloodSight</span>
          </a>
          <span aria-hidden="true" className="hidden h-7 w-px bg-white/[0.08] md:block" />
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-semibold text-cyan-300">{incident.incident_id}</span>
              <span className="hidden h-1 w-1 rounded-full bg-slate-700 sm:block" />
              <h1 className="truncate text-sm font-semibold text-slate-100 sm:text-base">{incident.title}</h1>
            </div>
            <p className="mt-0.5 hidden truncate text-[0.68rem] tracking-wide text-slate-500 md:block">{incident.location_label}</p>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <div className="hidden xl:block"><OriginBadge origin={incident.data_origin} compact /></div>
          <div className="hidden sm:block"><ConnectionIndicator state={connectionState} /></div>
          <button type="button" onClick={onOpenReport} className="command-button command-button-primary" aria-label="Open incident report">
            <Icon name="report" />
            <span className="hidden lg:inline">Incident report</span>
          </button>
          <a href="/system" className="command-icon-button" aria-label="Open system diagnostics" title="System diagnostics">
            <Icon name="diagnostics" />
          </a>
        </div>
      </div>
    </header>
  );
}
