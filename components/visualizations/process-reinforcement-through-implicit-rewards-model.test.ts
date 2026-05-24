import { describe, expect, it } from "vitest";

import {
  initialState,
  policyDistribution,
  step,
  type Intervention,
  type PrimeState,
} from "@/components/visualizations/process-reinforcement-through-implicit-rewards-model";

function run(state: PrimeState, ticks: number): PrimeState {
  let next = state;
  for (let i = 0; i < ticks; i++) next = step(next, { kind: "tick" });
  return next;
}

function intervene(state: PrimeState, action: Intervention): PrimeState {
  return step(state, { kind: "intervene", action });
}

describe("PRIME implicit-reward training loop", () => {
  it("reproduces a run exactly from the same seed and inputs", () => {
    const a = run(initialState("online"), 40);
    const b = run(initialState("online"), 40);
    expect(b.policyLogits).toEqual(a.policyLogits);
    expect(b.prmPref).toEqual(a.prmPref);
    expect(b.rng).toBe(a.rng);
    expect(b.trueAccuracy).toBe(a.trueAccuracy);
  });

  it("the policy distribution always stays a valid simplex", () => {
    let state = run(initialState("online"), 60);
    const sum = (s: PrimeState) =>
      policyDistribution(s).correct +
      policyDistribution(s).shortcut +
      policyDistribution(s).wrong;
    expect(sum(state)).toBeCloseTo(1, 6);
    state = run(intervene(state, { type: "injectShortcut" }), 30);
    expect(sum(state)).toBeCloseTo(1, 6);
    expect(policyDistribution(state).correct).toBeGreaterThanOrEqual(0);
  });

  it("online PRIME drives true accuracy up and keeps reward coupled to it (happy path)", () => {
    const end = run(initialState("online"), 120);
    expect(end.trueAccuracy).toBeGreaterThan(0.7);
    // Reward and true accuracy track each other: the gap stays small.
    expect(Math.abs(end.hackGap)).toBeLessThan(0.2);
    // The online PRM stays calibrated, ranking the sound step over the shortcut.
    expect(end.prmAccuracy).toBeGreaterThan(0.5);
  });

  it("freezing the PRM and tempting the shortcut opens a reward-hacking gap (Figure 5 failure)", () => {
    const online = run(initialState("online"), 120);
    const frozen = run(initialState("frozen"), 8);
    const hacked = run(intervene(frozen, { type: "injectShortcut" }), 60);

    // Frozen PRM accuracy ends up worse than the online run's, the paper's
    // central claim that the offline PRM degrades while the online one holds.
    expect(hacked.prmAccuracy).toBeLessThan(online.prmAccuracy);
    // Reported reward outruns true accuracy: that positive gap is the hack.
    expect(hacked.hackGap).toBeGreaterThan(0.1);
    // And true accuracy genuinely suffers versus the online run.
    expect(hacked.trueAccuracy).toBeLessThan(online.trueAccuracy);
  });

  it("a frozen PRM never moves its preferences; an online PRM does", () => {
    const startFrozen = initialState("frozen");
    const frozen = run(startFrozen, 30);
    expect(frozen.prmPref).toEqual(startFrozen.prmPref);

    const startOnline = initialState("online");
    const online = run(startOnline, 30);
    expect(online.prmPref).not.toEqual(startOnline.prmPref);
  });

  it("from-scratch PRM starts at zero reward and the online update teaches it to rank correctly", () => {
    const start = initialState("scratch");
    expect(start.prmPref.correct).toBe(0);
    expect(start.prmPref.shortcut).toBe(0);
    const trained = run(start, 120);
    // After online training the PRM ranks the sound step above the shortcut.
    expect(trained.prmPref.correct).toBeGreaterThan(trained.prmPref.shortcut);
    expect(trained.prmAccuracy).toBeGreaterThan(0.5);
  });

  it("the event log records steps and rewind restores the earlier state exactly", () => {
    const start = initialState("online");
    const mid = run(start, 10);
    const snapshot = mid.log.events[mid.log.events.length - 1]?.snapshot;
    expect(snapshot).toBeDefined();
    if (!snapshot) return;
    const further = run(mid, 20);
    expect(further.tick).toBe(30);
    const restored = intervene(further, {
      type: "restore",
      snapshot,
      label: "rewind",
    });
    expect(restored.policyLogits).toEqual(snapshot.policyLogits);
    expect(restored.prmPref).toEqual(snapshot.prmPref);
    expect(restored.rng).toBe(snapshot.rng);
  });

  it("beta scales the process reward's pull without breaking the simplex", () => {
    const lowBeta = run(
      intervene(initialState("online"), { type: "setBeta", value: 0.1 }),
      60,
    );
    const highBeta = run(
      intervene(initialState("online"), { type: "setBeta", value: 1.2 }),
      60,
    );
    expect(lowBeta.trueAccuracy).toBeGreaterThan(0);
    expect(highBeta.trueAccuracy).toBeGreaterThan(0);
  });
});
