import { describe, expect, it } from "vitest";

import {
  globalPeak,
  initialState,
  status,
  step,
  type Intervention,
  type SimState,
} from "@/components/visualizations/statistical-gradient-following-model";

function run(state: SimState, ticks: number): SimState {
  let next = state;
  for (let i = 0; i < ticks; i++) next = step(next, { kind: "tick" });
  return next;
}

function intervene(state: SimState, action: Intervention): SimState {
  return step(state, { kind: "intervene", action });
}

describe("REINFORCE Gaussian unit", () => {
  it("reproduces a run exactly from the same seed and inputs", () => {
    const a = run(initialState("climb", 7), 50);
    const b = run(initialState("climb", 7), 50);
    expect(b.mu).toEqual(a.mu);
    expect(b.sigma).toBe(a.sigma);
    expect(b.rng).toBe(a.rng);
    expect(b.baseline).toBe(a.baseline);
  });

  it("climbs the hill it cannot see and settles on the peak", () => {
    const start = initialState("climb", 3);
    const startScore = status(start);
    const end = run(start, 600);
    const endScore = status(end);

    expect(endScore.expected).toBeGreaterThan(startScore.expected);
    expect(endScore.fractionOfBest).toBeGreaterThan(0.85);
    expect(endScore.atGlobalPeak).toBe(true);
  });

  it("narrows its exploration width as it converges", () => {
    const start = initialState("climb", 3);
    const end = run(start, 600);
    expect(end.sigma).toBeLessThan(start.sigma);
    expect(end.sigma).toBeLessThan(0.09);
  });

  it("moves along the true gradient while climbing (Theorem 1)", () => {
    // After enough batches to build up an averaged step but before convergence,
    // the averaged mu-step should point the same way as the true gradient.
    const mid = run(initialState("climb", 3), 30);
    expect(status(mid).gradientCos).toBeGreaterThan(0);
  });

  it("keeps the exploration width fixed when sigma is pinned", () => {
    const pinned = intervene(initialState("climb", 3), {
      type: "setSigmaMode",
      value: "fixed",
    });
    const end = run(pinned, 200);
    expect(Math.abs(end.sigma - pinned.sigma)).toBeLessThan(1e-9);
  });

  it("gets trapped on the lesser peak and never finds the taller one", () => {
    const end = run(initialState("trapped", 5), 500);
    const score = status(end);
    expect(score.atGlobalPeak).toBe(false);
    expect(score.trappedOnLesserPeak).toBe(true);
    expect(score.fractionOfBest).toBeLessThan(0.85);
  });

  it("escapes the trap once the aim is dropped onto the taller slope", () => {
    const trapped = run(initialState("trapped", 5), 500);
    const peak = globalPeak(trapped.landscape);
    const relocated = intervene(trapped, {
      type: "moveMu",
      pos: { x: peak.center.x - 0.12, y: peak.center.y - 0.12 },
    });
    expect(relocated.avgUpdateMu).toEqual({ x: 0, y: 0 });
    const reopened = intervene(relocated, { type: "setSigma", value: 0.2 });
    const end = run(reopened, 600);
    expect(status(end).atGlobalPeak).toBe(true);
  });

  it("logs the moment it reaches the peak so the run can be rewound", () => {
    const end = run(initialState("climb", 3), 600);
    expect(
      end.log.events.some((event) => event.label.includes("reached the peak")),
    ).toBe(true);
  });
});
