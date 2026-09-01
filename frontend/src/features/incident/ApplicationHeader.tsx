import { ConnectionIndicator } from "../../components/ConnectionIndicator";
import { Icon } from "../../components/Icon";
import type { ConnectionState } from "../../hooks/useDemoIncident";
import type { IncidentMetadata } from "../../types/liveResult";

interface ApplicationHeaderProps {
  incident: IncidentMetadata;
  connectionState: ConnectionState;
  connectionLabel?: string;
  demoMode: boolean;
  onOpenReport: () => void;
}

export function ApplicationHeader({
  incident,
  connectionState,
  connectionLabel,
  demoMode,
  onOpenReport,
}: ApplicationHeaderProps) {
  const pending = incident.incident_id === "LIVE-PENDING";

  return (
    <header className="command-header sticky top-0 z-30">
      <div className="mx-auto flex h-14 max-w-[1680px] items-center justify-between gap-4 px-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-4 lg:gap-7">
          <a href="/" className="brand-lockup" aria-label="FloodSight home">
            <span className="brand-mark"><Icon name="water" className="h-[1.1rem] w-[1.1rem]" /></span>
            <span>FloodSight</span>
          </a>
          <span aria-hidden="true" className="header-rule" />
          <div className="min-w-0 leading-tight">
            <h1 className="truncate text-sm font-semibold text-slate-900">{pending ? "Flood response" : incident.title}</h1>
            <p className="mt-0.5 truncate text-[0.66rem] text-slate-500">{pending ? "New incident" : incident.incident_id}{demoMode ? " · Demo scenario" : ""}</p>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          <div className="hidden sm:block"><ConnectionIndicator state={connectionState} label={connectionLabel} /></div>
          <button type="button" onClick={onOpenReport} className="command-button command-button-secondary" aria-label="Open incident report">
            <Icon name="report" />
            <span className="hidden md:inline">Report</span>
          </button>
          <details className="header-menu">
            <summary className="command-icon-button" aria-label="Open navigation menu" title="More">
              <span aria-hidden="true" className="header-menu-dots">•••</span>
            </summary>
            <nav aria-label="FloodSight navigation" className="header-menu-popover">
              <a href="#model-details">Model details</a>
              <a href="/diagnostics">Diagnostics</a>
              <a href="/system">System status</a>
              {demoMode && <><span className="header-menu-divider" /><a href="/">Exit demo</a></>}
            </nav>
          </details>
        </div>
      </div>
    </header>
  );
}
