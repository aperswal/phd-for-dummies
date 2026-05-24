// A small, fast, seeded pseudo-random generator. Threading a seeded rng through
// a simulation instead of calling Math.random is what makes a run reproducible,
// so "what just happened?" can be replayed and scrubbed. Same seed, same stream.
export type Rng = () => number;

// One step of the mulberry32 recurrence as a pure function: given the
// generator's internal state, return the next value in [0, 1) and the next
// state. A reducer can thread `state` through immutable sim state and stay
// deterministic without a stateful closure, which is what makes a stepped
// simulation replayable.
export function nextRandom(state: number): { value: number; state: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, state: a };
}

export function mulberry32(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    const next = nextRandom(state);
    state = next.state;
    return next.value;
  };
}
