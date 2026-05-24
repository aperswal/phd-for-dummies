// A bounded, append-only log of what happened in a simulation. Each entry
// carries a snapshot the reader can rewind to, which is most of the value of an
// intervention: do something, then jump back to the moment just before it. The
// snapshot type is generic so a sim can record whatever fully describes a moment
// (a parameter set, a full state, a seed plus an input cursor).
export interface SimEvent<TSnapshot> {
  id: number;
  tick: number;
  label: string;
  snapshot: TSnapshot;
}

export interface EventLog<TSnapshot> {
  events: SimEvent<TSnapshot>[];
  nextEventId: number;
}

export function emptyLog<TSnapshot>(): EventLog<TSnapshot> {
  return { events: [], nextEventId: 1 };
}

const DEFAULT_LIMIT = 60;

export function record<TSnapshot>(
  log: EventLog<TSnapshot>,
  tick: number,
  label: string,
  snapshot: TSnapshot,
  limit = DEFAULT_LIMIT,
): EventLog<TSnapshot> {
  const event: SimEvent<TSnapshot> = {
    id: log.nextEventId,
    tick,
    label,
    snapshot,
  };
  return {
    events: [...log.events, event].slice(-limit),
    nextEventId: log.nextEventId + 1,
  };
}
