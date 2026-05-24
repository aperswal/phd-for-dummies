import { describe, expect, it } from "vitest";

import { N_ACTIONS, N_STATES } from "@/lib/simulation/mdp";

import {
  actualImprovement,
  improvementBound,
  initialState,
  safeStep,
  status,
  step,
  type Intervention,
  type SimState,
} from "@/components/visualizations/approximately-optimal-approximate-reinforcement-learning-model";

function run(state: SimState, ticks: number): SimState {
  let next = state;
  for (let i = 0; i < ticks; i++) next = step(next, { kind: "tick" });
  return next;
}

function intervene(state: SimState, action: Intervention): SimState {
  return step(state, { kind: "intervene", action });
}

// Walks a run and reports how many rounds eta_mu fell. Stops early if the run
// reaches its break point so a finished run doesn't pad the count with no-ops.
function countDrops(start: SimState, ticks: number): number {
  let cur = start;
  let drops = 0;
  for (let i = 0; i < ticks; i++) {
    const prev = cur;
    cur = step(cur, { kind: "tick" });
    if (cur.steps === prev.steps) break;
    if (cur.etaMu < prev.etaMu - 1e-9) drops++;
  }
  return drops;
}

describe("conservative policy iteration on a gridworld", () => {
  it("reproduces a run exactly from the same seed and inputs", () => {
    const a = run(initialState("greedy"), 60);
    const b = run(initialState("greedy"), 60);
    expect(b.etaMu).toBe(a.etaMu);
    expect(b.rng).toBe(a.rng);
    expect(b.pi).toEqual(a.pi);
  });

  it("keeps every policy a valid distribution after the mixture update", () => {
    const end = run(initialState("greedy"), 80);
    for (let s = 0; s < N_STATES; s++) {
      const sum = (end.pi[s] ?? []).reduce((acc, x) => acc + x, 0);
      expect(sum).toBeCloseTo(1, 9);
      for (let a = 0; a < N_ACTIONS; a++) {
        expect(end.pi[s]?.[a]).toBeGreaterThanOrEqual(-1e-12);
      }
    }
  });

  it("conservative steps improve eta_mu every round and never drop (Theorem 4.1)", () => {
    expect(countDrops(initialState("conservative"), 400)).toBe(0);
  });

  it("conservative climbs near optimal then stops at the break point", () => {
    const end = run(initialState("conservative"), 400);
    expect(status(end).fractionMu).toBeGreaterThan(0.9);
    expect(end.done).toBe(true);
    expect(end.policyAdvantage).toBeLessThanOrEqual(0.012 + 1e-9);
  });

  it("the safe step stays inside Theorem 4.1's guaranteed-improvement region", () => {
    const mid = run(initialState("conservative"), 15);
    const alpha = safeStep(mid);
    expect(alpha).toBeGreaterThan(0);
    // The guarantee is positive at the safe step, and the actual gain clears it.
    const bound = improvementBound(mid, alpha);
    expect(bound).toBeGreaterThan(0);
    expect(actualImprovement(mid, alpha)).toBeGreaterThanOrEqual(bound - 1e-9);
  });

  it("the guarantee turns negative once the step is too big", () => {
    const mid = run(initialState("conservative"), 15);
    // alpha = 1 is far past the zero crossing on this discounting, so the
    // guaranteed improvement is negative: committing fully is not safe.
    expect(improvementBound(mid, 1)).toBeLessThan(0);
  });

  it("greedy commitment degrades: eta_mu drops on some rounds and never settles", () => {
    const start = initialState("greedy");
    expect(countDrops(start, 150)).toBeGreaterThan(0);
    expect(run(start, 150).done).toBe(false);
  });

  it("recovers and climbs monotonically when switched from greedy to conservative mid-run", () => {
    const lurching = run(initialState("greedy"), 60);
    const before = status(lurching).fractionMu;
    const switched = intervene(lurching, {
      type: "setUpdate",
      value: "conservative",
    });
    expect(countDrops(switched, 300)).toBe(0);
    expect(status(run(switched, 300)).fractionMu).toBeGreaterThanOrEqual(
      before - 1e-9,
    );
  });

  it("a start-only restart distribution leaves an optimal-visited state uncovered (infinite mismatch)", () => {
    const end = run(initialState("restart"), 500);
    expect(end.mismatch).toBe(Infinity);
    expect(end.done).toBe(true);
  });

  it("a uniform restart keeps the distribution mismatch finite", () => {
    const end = run(initialState("conservative"), 100);
    expect(Number.isFinite(end.mismatch)).toBe(true);
    expect(end.mismatch).toBeGreaterThan(1);
  });

  it("switching the restart from start-only to uniform makes the mismatch finite", () => {
    const stalled = run(initialState("restart"), 120);
    expect(stalled.mismatch).toBe(Infinity);
    const switched = intervene(stalled, {
      type: "setRestart",
      value: "uniform",
    });
    expect(Number.isFinite(switched.mismatch)).toBe(true);
  });

  it("resets the policy to uniform on demand", () => {
    const learned = run(initialState("conservative"), 100);
    const reset = intervene(learned, { type: "resetPolicy" });
    expect(reset.steps).toBe(0);
    expect(reset.pi[reset.mdp.start]).toEqual([0.25, 0.25, 0.25, 0.25]);
  });

  it("logs the moment a greedy round drops performance", () => {
    const end = run(initialState("greedy"), 80);
    expect(
      end.log.events.some((event) => event.label.includes("dropped")),
    ).toBe(true);
  });
});
