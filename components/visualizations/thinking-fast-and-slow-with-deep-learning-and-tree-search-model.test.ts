import { describe, expect, it } from "vitest";

import {
  advance,
  getPreset,
  initialState,
  isDone,
  MAX_ITERATIONS,
  MOHEX_ELO,
  readout,
  step,
  type ExitState,
} from "./thinking-fast-and-slow-with-deep-learning-and-tree-search-model";

function run(state: ExitState, iterations: number): ExitState {
  let current = state;
  for (let i = 0; i < iterations; i++) current = advance(current);
  return current;
}

describe("the ExIt loop invariants", () => {
  it("keeps the expert at least as strong as the apprentice every iteration", () => {
    const final = run(initialState("exit"), 40);
    for (const point of final.history) {
      expect(point.expert).toBeGreaterThanOrEqual(point.apprentice - 1e-6);
    }
  });

  it("is deterministic given the same preset and seed", () => {
    const a = run(initialState("exit", 11), 30);
    const b = run(initialState("exit", 11), 30);
    expect(b.history).toEqual(a.history);
    expect(b.apprentice).toBe(a.apprentice);
  });

  it("climbs the apprentice past MoHeX on the happy path", () => {
    const final = run(initialState("exit"), 60);
    expect(final.apprentice).toBeGreaterThan(MOHEX_ELO);
    expect(readout(final).beatsMohex).toBe(true);
    expect(readout(final).iterationsToMohex).not.toBeNull();
  });
});

describe("the search gain depends on intuition", () => {
  it("shrinks when the apprentice prior is turned off", () => {
    const withPrior = readout(initialState("exit")).searchGain;
    const off = step(initialState("exit"), {
      kind: "intervene",
      action: { type: "toggleApprenticePrior" },
    });
    expect(readout(off).searchGain).toBeLessThan(withPrior);
  });

  it("grows when the value network is turned on", () => {
    const base = readout(initialState("exit")).searchGain;
    const value = step(initialState("exit"), {
      kind: "intervene",
      action: { type: "toggleValueNet" },
    });
    expect(readout(value).searchGain).toBeGreaterThan(base);
  });

  it("reaches MoHeX in fewer iterations with a value network than without", () => {
    const policy = run(initialState("exit"), 80);
    const value = run(initialState("value"), 80);
    const policyAt = readout(policy).iterationsToMohex;
    const valueAt = readout(value).iterationsToMohex;
    expect(policyAt).not.toBeNull();
    expect(valueAt).not.toBeNull();
    expect(valueAt ?? Infinity).toBeLessThanOrEqual(policyAt ?? 0);
  });
});

describe("REINFORCE versus ExIt", () => {
  it("stalls far below where ExIt finishes over the same iterations", () => {
    const exit = run(initialState("exit"), 50);
    const reinforce = run(initialState("reinforce"), 50);
    expect(exit.apprentice).toBeGreaterThan(reinforce.apprentice + 200);
  });

  it("keeps expert and apprentice equal in REINFORCE because there is no planner", () => {
    const final = run(initialState("reinforce"), 20);
    for (const point of final.history) {
      expect(point.expert).toBe(point.apprentice);
    }
  });
});

describe("cost-sensitive targets", () => {
  it("closes more of the gap per iteration with TPT than with CAT", () => {
    const tpt = run(initialState("exit"), 6).apprentice;
    const catStart = step(initialState("exit"), {
      kind: "intervene",
      action: { type: "setTarget", value: "CAT" },
    });
    const cat = run(catStart, 6).apprentice;
    expect(tpt).toBeGreaterThan(cat);
  });
});

describe("terminal state", () => {
  it("is not done at the start", () => {
    expect(isDone(initialState("exit"))).toBe(false);
  });

  it("is done once the run reaches MAX_ITERATIONS", () => {
    const final = run(initialState("exit"), MAX_ITERATIONS);
    expect(isDone(final)).toBe(true);
  });

  it("is not done one step before MAX_ITERATIONS", () => {
    const penultimate = run(initialState("exit"), MAX_ITERATIONS - 1);
    expect(isDone(penultimate)).toBe(false);
  });
});

describe("the step reducer and interventions", () => {
  it("advances one iteration per tick and records it", () => {
    const next = step(initialState("exit"), { kind: "tick" });
    expect(next.iteration).toBe(1);
    expect(next.log.events.at(-1)?.label).toContain("iter 1");
  });

  it("switching to REINFORCE mid-run flattens the climb", () => {
    const midway = run(initialState("exit"), 10);
    const switched = step(midway, {
      kind: "intervene",
      action: { type: "setMode", value: "reinforce" },
    });
    const before = switched.apprentice;
    const after = run(switched, 15).apprentice;
    expect(Math.abs(after - before)).toBeLessThan(220);
  });

  it("rederives the expert the instant simulations change", () => {
    const more = step(initialState("noprior"), {
      kind: "intervene",
      action: { type: "setSimulations", value: 30000 },
    });
    const fewer = step(initialState("noprior"), {
      kind: "intervene",
      action: { type: "setSimulations", value: 1000 },
    });
    expect(more.expert).toBeGreaterThan(fewer.expert);
  });

  it("loads a preset's parameters", () => {
    const loaded = step(initialState("exit"), {
      kind: "intervene",
      action: { type: "loadPreset", id: "value" },
    });
    expect(loaded.params).toEqual(getPreset("value").params);
    expect(loaded.iteration).toBe(0);
  });

  it("restores a logged moment so the reader can rewind", () => {
    const five = run(initialState("exit"), 5);
    const snap = five.log.events.at(-1);
    expect(snap).toBeDefined();
    const moved = run(five, 10);
    if (!snap) throw new Error("no snapshot");
    const back = step(moved, {
      kind: "intervene",
      action: { type: "restore", snapshot: snap.snapshot, label: "rewind" },
    });
    expect(back.iteration).toBe(snap.snapshot.iteration);
    expect(back.apprentice).toBe(snap.snapshot.apprentice);
  });
});
