import { describe, expect, it } from "vitest";

import {
  getPreset,
  initialState,
  recommendationCorrect,
  recommendedMove,
  rootChildren,
  rootValue,
  step,
  trueBestMove,
  type SimState,
} from "./efficient-selectivity-and-backup-operators-model";

function run(state: SimState, ticks: number): SimState {
  let working = state;
  for (let i = 0; i < ticks; i++) working = step(working, { kind: "tick" });
  return working;
}

describe("Monte-Carlo tree search core", () => {
  it("creates one tree node per root move plus the root", () => {
    const state = initialState("coulom-mix");
    expect(rootChildren(state)).toHaveLength(4);
  });

  it("is deterministic given the same seed and input list", () => {
    const a = run(initialState("coulom-mix"), 200);
    const b = run(initialState("coulom-mix"), 200);
    expect(rootValue(b)).toBeCloseTo(rootValue(a), 10);
    expect(recommendedMove(b)).toBe(recommendedMove(a));
  });

  it("counts exactly one simulation per tick", () => {
    const state = run(initialState("coulom-mix"), 50);
    expect(state.simulations).toBe(50);
  });

  it("conserves the visit count: the root sees every simulation", () => {
    const state = run(initialState("coulom-mix"), 120);
    const root = rootChildren(state).reduce((sum, c) => sum + c.visits, 0);
    // Every simulation passes through one root child on its way down.
    expect(root).toBe(state.simulations);
  });
});

describe("selectivity keeps every move alive (Coulom) vs prunes (pruning)", () => {
  it("Coulom's selectivity gives every move a non-zero share of simulations", () => {
    const state = run(initialState("coulom-mix"), 400);
    for (const child of rootChildren(state)) {
      expect(child.visits).toBeGreaterThan(0);
    }
  });

  it("finds the hidden gem: the noisy high-value move wins under Coulom", () => {
    const state = run(initialState("coulom-mix"), 600);
    expect(recommendedMove(state)).toBe(trueBestMove(state));
    expect(recommendationCorrect(state)).toBe(true);
  });

  it("progressive pruning can starve the gem and recommend a worse move", () => {
    const state = run(initialState("pruning-trap"), 600);
    // The trap move (index 2) is the true best, but pruning cut it off.
    expect(state.pruned).toContain(trueBestMove(state));
    expect(recommendationCorrect(state)).toBe(false);
  });
});

describe("backup operators", () => {
  it("max backup overestimates the root value relative to mix", () => {
    const mixState = run(initialState("coulom-mix"), 500);
    const maxState = run(initialState("max-overestimate"), 500);
    expect(rootValue(maxState)).toBeGreaterThan(rootValue(mixState));
  });

  it("recomputes node values when the operator changes mid-run", () => {
    const before = run(initialState("coulom-mix"), 300);
    const beforeValue = rootValue(before);
    const after = step(before, {
      kind: "intervene",
      action: { type: "setBackup", value: "max" },
    });
    expect(after.simulations).toBe(before.simulations);
    expect(rootValue(after)).not.toBeCloseTo(beforeValue, 4);
  });
});

describe("the step reducer and interventions", () => {
  it("loads a preset and resets the run", () => {
    const state = step(run(initialState("coulom-mix"), 100), {
      kind: "intervene",
      action: { type: "loadPreset", id: "pruning-trap" },
    });
    expect(state.simulations).toBe(0);
    expect(state.params).toEqual(getPreset("pruning-trap").params);
  });

  it("a burst runs many simulations in one intervention", () => {
    const state = step(initialState("coulom-mix"), {
      kind: "intervene",
      action: { type: "burst", count: 64 },
    });
    expect(state.simulations).toBe(64);
  });

  it("restoring a logged snapshot rewinds the simulation count", () => {
    const grown = run(initialState("coulom-mix"), 80);
    const logged = step(grown, {
      kind: "intervene",
      action: { type: "setBackup", value: "mean" },
    });
    const snapshotEvent = logged.log.at(-1);
    expect(snapshotEvent).toBeDefined();
    if (!snapshotEvent) return;
    const further = run(logged, 40);
    const restored = step(further, {
      kind: "intervene",
      action: {
        type: "restore",
        snapshot: snapshotEvent.snapshot,
        label: "rewind",
      },
    });
    expect(restored.simulations).toBe(snapshotEvent.snapshot.simulations);
  });

  it("changing the tree resets and rebuilds the root children", () => {
    const state = step(run(initialState("coulom-mix"), 50), {
      kind: "intervene",
      action: { type: "setGame", id: "clear" },
    });
    expect(state.simulations).toBe(0);
    expect(state.params.gameId).toBe("clear");
    const grown = run(state, 300);
    expect(recommendedMove(grown)).toBe(trueBestMove(grown));
  });
});
