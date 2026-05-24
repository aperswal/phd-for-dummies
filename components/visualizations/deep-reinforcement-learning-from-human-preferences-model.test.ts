import { describe, expect, it } from "vitest";

import {
  GOAL,
  initialState,
  preferProbability,
  PRESETS,
  step,
  trueReward,
  type Clip,
  type Input,
  type Intervention,
  type SimState,
} from "./deep-reinforcement-learning-from-human-preferences-model";

function tick(state: SimState, n: number): SimState {
  let s = state;
  for (let i = 0; i < n; i++) s = step(s, { kind: "tick" });
  return s;
}

function intervene(state: SimState, action: Intervention): SimState {
  return step(state, { kind: "intervene", action });
}

describe("true reward", () => {
  it("pays at the goal and costs a little everywhere else", () => {
    expect(trueReward(GOAL)).toBeGreaterThan(0);
    expect(trueReward(0)).toBeLessThan(0);
  });
});

describe("Bradley-Terry preference probability", () => {
  it("rises above one half when clip a sums to more predicted reward", () => {
    const predictor = { cell: new Array<number>(25).fill(0) };
    predictor.cell[GOAL] = 2;
    const a: Clip = { id: 1, cells: [GOAL, GOAL, GOAL] };
    const b: Clip = { id: 2, cells: [0, 1, 2] };
    expect(preferProbability(predictor, a, b)).toBeGreaterThan(0.9);
  });

  it("is exactly one half when both clips score the same", () => {
    const predictor = { cell: new Array<number>(25).fill(0.3) };
    const a: Clip = { id: 1, cells: [0, 1, 2] };
    const b: Clip = { id: 2, cells: [5, 6, 7] };
    expect(preferProbability(predictor, a, b)).toBeCloseTo(0.5, 6);
  });
});

describe("happy path with the synthetic oracle", () => {
  it("drives the learned reward toward the true reward over time", () => {
    const start = initialState("oracle");
    const before = start.alignment;
    const after = tick(start, 400);
    expect(after.labelCount).toBeGreaterThan(50);
    expect(after.alignment).toBeGreaterThan(before);
    expect(after.alignment).toBeGreaterThan(0.5);
  });

  it("recovers most of the optimal policy under clean oracle labels", () => {
    const after = tick(initialState("oracle"), 600);
    expect(after.optimalShare).toBeGreaterThan(0.6);
  });

  it("drives the Bradley-Terry loss down as the database grows", () => {
    const early = tick(initialState("oracle"), 60);
    const late = tick(early, 400);
    expect(late.loss).toBeLessThan(early.loss);
  });
});

describe("determinism", () => {
  it("reproduces the same run from the same seed and inputs", () => {
    const a = tick(initialState("oracle"), 300);
    const b = tick(initialState("oracle"), 300);
    expect(b.alignment).toBeCloseTo(a.alignment, 10);
    expect(b.labelCount).toBe(a.labelCount);
    expect(b.rhat).toEqual(a.rhat);
  });
});

describe("offline-only intervention", () => {
  it("stops gathering labels after the opening batch", () => {
    const after = tick(initialState("offline"), 600);
    expect(after.labelCount).toBeLessThanOrEqual(12);
  });

  it("ends up less aligned than the online oracle given the same horizon", () => {
    const online = tick(initialState("oracle"), 600);
    const offline = tick(initialState("offline"), 600);
    expect(offline.alignment).toBeLessThan(online.alignment);
  });
});

describe("noisy human intervention", () => {
  it("reaches lower alignment than clean labels over the same horizon", () => {
    const clean = tick(initialState("oracle"), 600);
    const noisy = tick(initialState("noisy"), 600);
    expect(noisy.alignment).toBeLessThan(clean.alignment);
  });
});

describe("manual labeling flow", () => {
  it("pauses on a query when you are the labeler and resumes after a label", () => {
    let state = initialState("you");
    while (state.pending === null && state.step < 50) {
      state = step(state, { kind: "tick" });
    }
    expect(state.pending).not.toBeNull();
    const before = state.labelCount;
    state = intervene(state, { type: "label", prefer: "a" });
    expect(state.pending).toBeNull();
    expect(state.labelCount).toBe(before + 1);
  });
});

describe("event log and rewind", () => {
  it("records labels and can restore an earlier snapshot", () => {
    const after = tick(initialState("oracle"), 200);
    expect(after.log.events.length).toBeGreaterThan(0);
    const target = after.log.events[5];
    expect(target).toBeDefined();
    if (!target) return;
    const restored = intervene(after, {
      type: "restore",
      snapshot: target.snapshot,
      label: "rewind",
    });
    expect(restored.labelCount).toBe(target.snapshot.labelCount);
    expect(restored.step).toBe(target.snapshot.step);
  });
});

describe("presets", () => {
  it("exposes a happy path and the offline failure case", () => {
    const ids = PRESETS.map((p) => p.id);
    expect(ids).toContain("oracle");
    expect(ids).toContain("offline");
  });
});

describe("model stays valid under any input", () => {
  it("never produces NaN alignment regardless of intervention order", () => {
    const inputs: Input[] = [
      { kind: "intervene", action: { type: "setMislabel", value: 0.5 } },
      { kind: "tick" },
      { kind: "intervene", action: { type: "toggleOnline" } },
      { kind: "tick" },
      { kind: "intervene", action: { type: "setQueryEvery", value: 1 } },
      { kind: "tick" },
      { kind: "intervene", action: { type: "setLabeler", value: "you" } },
      { kind: "tick" },
    ];
    let state = initialState("oracle");
    for (const input of inputs) state = step(state, input);
    expect(Number.isFinite(state.alignment)).toBe(true);
    expect(Number.isFinite(state.loss)).toBe(true);
  });
});
