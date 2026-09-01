import { Icon } from "../../components/Icon";
import type { IncidentEvent } from "../../types/liveResult";
import { formatTimestamp } from "../../utils/format";
import { EmptyState } from "../command-center/CommandStates";

interface EventTimelineProps {
  events: IncidentEvent[];
}

const eventTone = {
  INFO: "border-cyan-300/25 bg-cyan-300/[0.08] text-cyan-300",
  WARNING: "border-amber-300/25 bg-amber-300/[0.08] text-amber-300",
  CRITICAL: "border-rose-400/25 bg-rose-400/[0.08] text-rose-300",
};

export function EventTimeline({ events }: EventTimelineProps) {
  const uniqueEvents = [...new Map(events.map((event) => [event.event_id, event])).values()]
    .sort((a, b) => b.timestamp_ms - a.timestamp_ms)
    .slice(0, 16);

  return (
    <section className="command-panel min-w-0" aria-labelledby="events-heading">
      <div className="panel-heading"><div><div className="flex items-center gap-2"><Icon name="activity" className="h-4 w-4 text-cyan-300" /><h2 id="events-heading" className="panel-title">Live incident events</h2></div><p className="panel-subtitle">Newest first · Stable backend event IDs</p></div><span className="font-mono text-[0.65rem] text-slate-600">{uniqueEvents.length} EVENTS</span></div>
      <div className="max-h-64 overflow-y-auto p-2 sm:p-3">
        {uniqueEvents.length ? <ol className="divide-y divide-white/[0.05]">{uniqueEvents.map((event) => <li key={event.event_id} className="grid grid-cols-[4.5rem_1fr] gap-3 px-2 py-3 sm:grid-cols-[5.5rem_6.5rem_1fr]"><time dateTime={new Date(event.timestamp_ms).toISOString()} className="font-mono text-[0.68rem] text-slate-500">{formatTimestamp(event.timestamp_ms)}</time><span className={`hidden w-fit rounded-md border px-2 py-0.5 text-[0.58rem] font-bold tracking-wider uppercase sm:inline ${eventTone[event.severity]}`}>{event.category}</span><p className="text-xs leading-5 text-slate-300">{event.message}{event.code && <code className="mt-1 block text-[0.58rem] text-slate-500">{event.code}</code>}</p></li>)}</ol> : <EmptyState label="No incident events in this snapshot." />}
      </div>
    </section>
  );
}
