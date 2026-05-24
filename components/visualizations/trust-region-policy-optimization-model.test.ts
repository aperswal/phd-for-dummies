import { describe, expect, it } from "vitest";

import {
  initialState,
  N_ACTIONS,
  N_STATES,
  status,
  step,
  type Intervention,
  type SimState,
} from "@/components/visualizations/trust-region-policy-optimization-model";

function run(state: SimState, ticks: number): SimState {
  let next = state;
  for (let i = 0; i < ticks; i++) next = step(next, { kind: "tick" });
  return next;
}

function intervene(state: SimState, action: Intervention): SimState {
  return step(state, { kind: "intervene", action });
}

// Counts rounds where the true return fell while the surrogate predicted a gain,
// the overshoot the trust region exists to prevent. Stops early if the run
// settles so a finished run doesn't pad the count.
function countOvershoots(start: SimState, ticks: number): number {
  let cur = start;
  let count = 0;
  for (let i = 0; i < ticks; i++) {
    const prev = cur;
    cur = step(cur, { kind: "tick" });
    if (cur.steps === prev.steps) break;
    if (cur.surrogateGain > 1e-4 && cur.realGain < -1e-4) count++;
  }
  return count;
}

describe("trust region policy optimization on a gridworld", () => {
  it("reproduces a run exactly from the same seed and inputs", () => {
    const a = run(initialState("overshoot"), 50);
    const b = run(initialState("overshoot"), 50);
    expect(b.eta).toBe(a.eta);
    expect(b.theta).toEqual(a.theta);
  });

  it("keeps every policy a valid distribution after each update", () => {
    const end = run(initialState("trust-region"), 60);
    for (let s = 0; s < N_STATES; s++) {
      const sum = (end.pi[s] ?? []).reduce((acc, x) => acc + x, 0);
      // Terminal states hold an all-zero row; decision states sum to 1.
      expect(sum === 0 || Math.abs(sum - 1) < 1e-9).toBe(true);
      for (let a = 0; a < N_ACTIONS; a++) {
        expect(end.pi[s]?.[a]).toBeGreaterThanOrEqual(-1e-12);
      }
    }
  });

  it("every trust-region step keeps the measured mean KL inside delta", () => {
    let cur = initialState("trust-region");
    for (let i = 0; i < 80; i++) {
      const prev = cur;
      cur = step(cur, { kind: "tick" });
      if (cur.steps === prev.steps) break;
      expect(cur.appliedKL).toBeLessThanOrEqual(cur.params.delta + 1e-9);
    }
  });

  it("the trust-region climb reaches near optimal and settles", () => {
    const end = run(initialState("trust-region"), 300);
    expect(status(end).fraction).toBeGreaterThan(0.9);
    expect(end.done).toBe(true);
  });

  it("the trust-region run almost never overshoots", () => {
    // The KL budget keeps the surrogate honest, so a true-return drop while the
    // surrogate predicts a gain is rare to absent.
    expect(
      countOvershoots(initialState("trust-region"), 300),
    ).toBeLessThanOrEqual(1);
  });

  it("the unconstrained step overshoots: surrogate up, true return down", () => {
    // A big plain step with no KL limit steps past where the surrogate is
    // trustworthy, so on some rounds the surrogate predicts a gain the true
    // return does not deliver.
    expect(countOvershoots(initialState("overshoot"), 120)).toBeGreaterThan(0);
  });

  it("the penalty form crawls: tiny KL steps and slow progress", () => {
    const tr = run(initialState("trust-region"), 25);
    const pen = run(initialState("penalty"), 25);
    // After the same number of rounds the penalty form's per-step KL is far
    // smaller and it has covered far less ground toward optimal.
    expect(pen.appliedKL).toBeLessThan(tr.appliedKL);
    expect(status(pen).fraction).toBeLessThan(status(tr).fraction);
  });

  it("recovers when switched from overshoot to trust region mid-run", () => {
    const lurching = run(initialState("overshoot"), 40);
    const switched = intervene(lurching, {
      type: "setUpdate",
      value: "trustRegion",
    });
    const before = status(switched).fraction;
    const after = status(run(switched, 200)).fraction;
    expect(after).toBeGreaterThan(before);
    expect(countOvershoots(switched, 200)).toBeLessThanOrEqual(1);
  });

  it("a wider delta lets the policy move farther per step than a tight one", () => {
    const tight = run(initialState("trust-region"), 5);
    const wide = run(
      intervene(initialState("trust-region"), { type: "setDelta", value: 0.2 }),
      5,
    );
    expect(wide.appliedKL).toBeGreaterThan(tight.appliedKL);
  });

  it("logs an overshoot moment in the unconstrained run", () => {
    const end = run(initialState("overshoot"), 120);
    expect(end.log.events.some((event) => event.label.includes("true"))).toBe(
      true,
    );
  });

  it("resets the policy to uniform on demand", () => {
    const learned = run(initialState("trust-region"), 80);
    const reset = intervene(learned, { type: "resetPolicy" });
    expect(reset.steps).toBe(0);
    const start = reset.pi[reset.mdp.start] ?? [];
    for (let a = 0; a < N_ACTIONS; a++) {
      expect(start[a]).toBeCloseTo(0.25, 9);
    }
  });
});
