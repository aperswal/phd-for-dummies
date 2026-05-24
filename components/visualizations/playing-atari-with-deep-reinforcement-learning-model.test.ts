import { describe, expect, it } from "vitest";

import {
  ACTION_COUNT,
  featurize,
  FEATURE_DIM,
  getPreset,
  initialState,
  meanReturn,
  PRESETS,
  qValues,
  step,
  type SimState,
} from "./playing-atari-with-deep-reinforcement-learning-model";

function run(state: SimState, ticks: number): SimState {
  let next = state;
  for (let i = 0; i < ticks; i++) next = step(next, { kind: "tick" });
  return next;
}

describe("featurize", () => {
  it("produces a one-hot-plus-relation vector of the declared length", () => {
    const features = featurize({ ballCol: 1, ballRow: 0, paddleCol: 2 });
    expect(features).toHaveLength(FEATURE_DIM);
    // paddle col, ball col, ball row, relation tile, and bias each fire.
    expect(features.filter((v) => v > 0).length).toBeGreaterThanOrEqual(5);
  });

  it("shares the relation tile across distinct states", () => {
    // Two different states where the ball sits right of the paddle share the
    // relation tile, which is the aliasing the deadly triad needs.
    const a = featurize({ ballCol: 3, ballRow: 0, paddleCol: 1 });
    const b = featurize({ ballCol: 4, ballRow: 0, paddleCol: 2 });
    // The last-but-one block is the relation tile; both set the "right" bin.
    const relIndexRight = FEATURE_DIM - 2;
    expect(a[relIndexRight]).toBeGreaterThan(0);
    expect(b[relIndexRight]).toBeGreaterThan(0);
  });

  it("is deterministic for the same state", () => {
    const a = featurize({ ballCol: 3, ballRow: 2, paddleCol: 1 });
    const b = featurize({ ballCol: 3, ballRow: 2, paddleCol: 1 });
    expect(a).toEqual(b);
  });
});

describe("determinism", () => {
  it("reproduces the exact same run for the same seed and inputs", () => {
    const a = run(initialState("stable"), 200);
    const b = run(initialState("stable"), 200);
    expect(a.metrics.maxAbsQ).toBeCloseTo(b.metrics.maxAbsQ, 10);
    expect(a.metrics.step).toBe(b.metrics.step);
    expect(a.weights).toEqual(b.weights);
  });
});

describe("happy path: experience replay keeps the value estimate bounded", () => {
  it("stays stable over a long run", () => {
    const state = run(initialState("stable"), 1500);
    expect(state.metrics.diverged).toBe(false);
    expect(Number.isFinite(state.metrics.maxAbsQ)).toBe(true);
    expect(state.metrics.maxAbsQ).toBeLessThan(50);
  });

  it("learns to catch better than chance", () => {
    const early = run(initialState("stable"), 120);
    const late = run(early, 3000);
    expect(meanReturn(late.metrics.returnsWindow)).toBeGreaterThan(
      meanReturn(early.metrics.returnsWindow),
    );
  });

  it("the q-network reports one value per action", () => {
    const state = run(initialState("stable"), 50);
    const q = qValues(state.weights, featurize(state.game));
    expect(q).toHaveLength(ACTION_COUNT);
  });
});

describe("failure case: turning replay off destabilizes learning", () => {
  it("the diverge preset blows the value estimate up", () => {
    const state = run(initialState("diverge"), 1500);
    expect(state.metrics.diverged).toBe(true);
  });

  it("toggling replay off mid-run on a stable run destabilizes it", () => {
    const stable = run(initialState("stable"), 800);
    expect(stable.metrics.diverged).toBe(false);
    const stableMax = stable.metrics.maxAbsQ;

    const keepReplay = run(stable, 1500);
    const droppedReplay = run(
      step(stable, { kind: "intervene", action: { type: "toggleReplay" } }),
      1500,
    );
    // Keeping replay on stays bounded; dropping it at the same learning rate
    // lets the value estimate run far past where it sat and past the calm run.
    expect(keepReplay.metrics.diverged).toBe(false);
    expect(droppedReplay.metrics.maxAbsQ).toBeGreaterThan(stableMax * 5);
    expect(droppedReplay.metrics.maxAbsQ).toBeGreaterThan(
      keepReplay.metrics.maxAbsQ * 5,
    );
  });
});

describe("interventions are recorded and rewindable", () => {
  it("logs an event when a parameter changes", () => {
    const state = step(run(initialState("stable"), 30), {
      kind: "intervene",
      action: { type: "setEpsilon", value: 0.5 },
    });
    expect(state.log.events.length).toBe(1);
    expect(state.params.epsilon).toBeCloseTo(0.5, 10);
  });

  it("restore rebuilds an earlier moment deterministically", () => {
    const warmed = run(initialState("stable"), 80);
    const snapshot = {
      params: { ...warmed.params },
      step: warmed.metrics.step,
    };
    const moved = step(warmed, {
      kind: "intervene",
      action: { type: "setLearningRate", value: 0.2 },
    });
    const restored = step(moved, {
      kind: "intervene",
      action: { type: "restore", snapshot, label: "rewind" },
    });
    expect(restored.metrics.step).toBe(warmed.metrics.step);
    expect(restored.metrics.maxAbsQ).toBeCloseTo(warmed.metrics.maxAbsQ, 8);
  });
});

describe("presets", () => {
  it("every preset id resolves and carries a full param set", () => {
    for (const preset of PRESETS) {
      expect(getPreset(preset.id).id).toBe(preset.id);
      expect(preset.params.bufferCapacity).toBeGreaterThan(0);
    }
  });
});
