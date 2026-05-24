"use client";

import type { SimEvent } from "@/lib/simulation/history";

// A scrollable, clickable history of what happened in a run. Clicking an entry
// hands its snapshot back so the caller can rewind to that moment. Newest first,
// because that's what the reader just did and wants to undo.
export function EventLogPanel<TSnapshot>({
  events,
  onRestore,
  emptyHint = "Step, click, or flip a control to start the log.",
}: {
  events: SimEvent<TSnapshot>[];
  onRestore: (event: SimEvent<TSnapshot>) => void;
  emptyHint?: string;
}) {
  return (
    <div className="ring-foreground/10 max-h-28 overflow-y-auto rounded-lg ring-1">
      {events.length === 0 ? (
        <p className="text-muted-foreground px-2 py-1.5 text-xs">{emptyHint}</p>
      ) : (
        <ul className="text-xs">
          {[...events].reverse().map((event) => (
            <li key={event.id}>
              <button
                type="button"
                className="text-muted-foreground hover:bg-foreground/5 w-full px-2 py-1 text-left font-mono"
                onClick={() => onRestore(event)}
              >
                {event.id}. {event.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
