import { Icon } from "../../components/Icon";
import type { IncidentEvent } from "../../types/liveResult";
import { formatTimestamp } from "../../utils/format";
import { EmptyState } from "../command-center/CommandStates";

interface EventTimelineProps {
  events: IncidentEvent[];
}

const eventTone = {
  INFO: "border-cyan-500/25 bg-cyan-50 text-cyan-800",
  WARNING: "border-amber-500/25 bg-amber-50 text-amber-800",
  CRITICAL: "border-rose-500/25 bg-rose-50 text-rose-700",
};

export function EventTimeline({ events }: EventTimelineProps) {
  const uniqueEvents = [...new Map(events.map((event) => [event.event_id, event])).values()]
    .sort((a, b) => b.timestamp_ms - a.timestamp_ms)
    .slice(0, 16);
  const latest = uniqueEvents.slice(0, 4);
  const older = uniqueEvents.slice(4);

  return (
    <section className="command-panel min-w-0 overflow-hidden" aria-labelledby="events-heading">
      <div className="panel-heading"><div><div className="flex items-center gap-2"><Icon name="activity" className="h-4 w-4 text-cyan-300" /><h2 id="events-heading" className="panel-title">Incident timeline</h2></div><p className="panel-subtitle">Latest meaningful events · Newest first</p></div><span className="font-mono text-[0.62rem] text-slate-600">{uniqueEvents.length} EVENTS</span></div>
      <div className="p-2.5 sm:p-3">
        {latest.length ? (
          <>
            <EventList events={latest} />
            {older.length > 0 && (
              <details className="mt-1 border-t border-sky-950/10 px-2 pt-1">
                <summary className="disclosure-summary">Older events <span className="font-mono">{older.length}</span></summary>
                <EventList events={older} />
              </details>
            )}
          </>
        ) : <EmptyState label="No incident events in this snapshot." />}
      </div>
    </section>
  );
}

function EventList({ events }: { events: IncidentEvent[] }) {
  return (
    <ol className="divide-y divide-sky-950/8">
      {events.map((event) => (
        <li key={event.event_id} className="grid grid-cols-[4.5rem_1fr] gap-3 px-2 py-3 sm:grid-cols-[4.5rem_5.5rem_1fr]">
          <time dateTime={new Date(event.timestamp_ms).toISOString()} className="font-mono text-[0.66rem] text-slate-500">{formatTimestamp(event.timestamp_ms)}</time>
          <span className={`hidden w-fit rounded-md border px-2 py-0.5 text-[0.56rem] font-bold tracking-wider uppercase sm:inline ${eventTone[event.severity]}`}>{event.category}</span>
          <p className="text-xs leading-5 text-slate-300">{event.message}{event.code && <code className="mt-1 block text-[0.58rem] text-slate-500">{event.code}</code>}</p>
        </li>
      ))}
    </ol>
  );
}
