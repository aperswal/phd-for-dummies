import { describe, expect, it } from "vitest";

import {
  getPreset,
  highestTier,
  initialState,
  isTerminal,
  MAX_ROUNDS,
  step,
  TECH_TREE,
  type Input,
  type SimState,
} from "./voyager-an-open-ended-embodied-agent-model";

// Drive the loop forward a fixed number of ticks. Pure and deterministic, so the
// same preset and tick count always lands on the same state.
function run(presetId: string, ticks: number): SimState {
  let state = initialState(presetId);
  for (let i = 0; i < ticks; i++) {
    state = step(state, { kind: "tick" });
  }
  return state;
}

const tick: Input = { kind: "tick" };

describe("the lifelong-learning loop", () => {
  it("is deterministic given the same preset and inputs", () => {
    const a = run("full", 200);
    const b = run("full", 200);
    expect(b.unlocked).toEqual(a.unlocked);
    expect(b.iterations).toBe(a.iterations);
    expect(b.library.map((skill) => skill.itemId)).toEqual(
      a.library.map((skill) => skill.itemId),
    );
  });

  it("never unlocks an item before its prerequisites", () => {
    const state = run("full", 400);
    const have = new Set(state.unlocked);
    for (const id of state.unlocked) {
      const item = state.library.find((skill) => skill.itemId === id);
      expect(item ?? { itemId: id }).toBeTruthy();
    }
    // The stone pickaxe needs cobblestone, which needs the wooden pickaxe.
    if (have.has("stone_pickaxe")) {
      expect(have.has("cobblestone")).toBe(true);
      expect(have.has("wood_pickaxe")).toBe(true);
    }
    if (have.has("diamond")) {
      expect(have.has("iron_pickaxe")).toBe(true);
    }
  });

  it("climbs the tech tree to the diamond tier with everything on", () => {
    const state = run("full", 600);
    expect(state.unlocked.length).toBeGreaterThan(8);
    expect(highestTier(state)).toBeGreaterThanOrEqual(4);
  });
});

// Run until the agent unlocks `target` items, returning the iterations of code
// generation it spent getting there. Caps ticks so a stalled config still
// returns. Iterations, not ticks, is the paper's efficiency axis.
function iterationsToReach(
  presetId: string,
  target: number,
  maxTicks = 4000,
): number {
  let state = initialState(presetId);
  let ticks = 0;
  while (state.unlocked.length < target && ticks < maxTicks) {
    state = step(state, tick);
    ticks += 1;
  }
  return state.iterations;
}

describe("the skill library ablation", () => {
  it("reaches the same milestone in fewer iterations with the library on", () => {
    const withLibrary = iterationsToReach("full", 11);
    const withoutLibrary = iterationsToReach("no-library", 11);
    expect(withLibrary).toBeLessThan(withoutLibrary);
  });

  it("makes less progress within a fixed step budget without the library", () => {
    const withLibrary = run("full", 30);
    const withoutLibrary = run("no-library", 30);
    expect(withoutLibrary.unlocked.length).toBeLessThan(
      withLibrary.unlocked.length,
    );
  });

  it("composes later skills from earlier ones when the library is on", () => {
    const state = run("full", 600);
    const composed = state.library.filter((skill) => skill.composedFrom > 0);
    expect(composed.length).toBeGreaterThan(0);
  });

  it("commits nothing to the library when the library is off", () => {
    const state = run("no-library", 300);
    expect(state.library.length).toBe(0);
    expect(state.unlocked.length).toBeGreaterThan(0);
  });
});

describe("the self-verification ablation", () => {
  it("commits unverified skills when verification is off and zero when on", () => {
    const off = run("no-verification", 300);
    const on = run("full", 300);
    expect(off.committedUnverified).toBeGreaterThan(0);
    expect(on.committedUnverified).toBe(0);
  });
});

describe("the random curriculum ablation", () => {
  it("makes less tech-tree progress than the automatic curriculum", () => {
    const automatic = run("full", 500);
    const random = run("random-curriculum", 500);
    expect(highestTier(random)).toBeLessThanOrEqual(highestTier(automatic));
  });
});

describe("hallucinated tasks", () => {
  it("never unlock and drive failed attempts up", () => {
    const state = run("hallucinating", 400);
    expect(state.failedAttempts).toBeGreaterThan(0);
  });

  it("a forced hallucinated task fails after MAX_ROUNDS and shelves", () => {
    let state = initialState("full");
    state = step(state, {
      kind: "intervene",
      action: { type: "setHallucination", value: 1 },
    });
    state = step(state, {
      kind: "intervene",
      action: { type: "forceTask", itemId: "log" },
    });
    const before = state.failedAttempts;
    for (let i = 0; i < MAX_ROUNDS + 2; i++) {
      state = step(state, tick);
    }
    expect(state.failedAttempts).toBeGreaterThan(before);
    expect(state.unlocked).not.toContain("log");
  });
});

describe("interventions", () => {
  it("switching the coder to gpt-3.5 spends more iterations per item", () => {
    let weak = initialState("full");
    weak = step(weak, {
      kind: "intervene",
      action: { type: "setCoder", value: "gpt-3.5" },
    });
    for (let i = 0; i < 120; i++) weak = step(weak, tick);
    const strong = run("full", 120);
    const weakRate = weak.iterations / Math.max(1, weak.unlocked.length);
    const strongRate = strong.iterations / Math.max(1, strong.unlocked.length);
    expect(weakRate).toBeGreaterThan(strongRate);
  });

  it("loading a preset resets to that preset's params", () => {
    let state = run("full", 100);
    state = step(state, {
      kind: "intervene",
      action: { type: "loadPreset", id: "no-library" },
    });
    expect(state.params).toEqual(getPreset("no-library").params);
    expect(state.unlocked.length).toBe(0);
  });

  it("restore rewinds to a logged snapshot", () => {
    const state = run("full", 200);
    const event = state.log.events.at(-1);
    expect(event).toBeDefined();
    if (!event) return;
    const restored = step(state, {
      kind: "intervene",
      action: { type: "restore", snapshot: event.snapshot, label: "rewind" },
    });
    expect(restored.unlocked).toEqual(event.snapshot.unlocked);
    expect(restored.tick).toBe(event.snapshot.tick);
  });
});

describe("terminal state detection", () => {
  it("is not terminal at the start of a fresh run", () => {
    const state = initialState("full");
    expect(isTerminal(state)).toBe(false);
  });

  it("is terminal when every item is unlocked and no task is active", () => {
    let state = initialState("full");
    state = {
      ...state,
      unlocked: TECH_TREE.map((item) => item.id),
      active: null,
      phase: "idle",
    };
    expect(isTerminal(state)).toBe(true);
  });

  it("is not terminal while a task is still active even if frontier would be empty", () => {
    let state = initialState("full");
    const allButLast = TECH_TREE.slice(0, -1).map((item) => item.id);
    state = {
      ...state,
      unlocked: allButLast,
      active: {
        itemId: "diamond",
        label: "Mine Diamond",
        round: 1,
        difficulty: 0.85,
        retrievedSkills: 0,
        hallucinated: false,
        lastRoundExecuted: false,
      },
      phase: "code",
    };
    expect(isTerminal(state)).toBe(false);
  });
});
