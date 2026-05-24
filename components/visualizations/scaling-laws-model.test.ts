import { describe, expect, it } from "vitest";

import {
  buildCurve,
  computeAllocation,
  getPreset,
  initialState,
  lossFromCompute,
  lossFromData,
  lossFromParams,
  lossFromParamsAndData,
  optimalParams,
  overfitPenalty,
  POINT_COUNT,
  step,
  type ScalingParams,
} from "./scaling-laws-model";

const base: ScalingParams = {
  axisId: "compute",
  seed: 7,
  logCompute: 1,
  allocation: 1,
};

describe("the single-factor power laws", () => {
  it("fall monotonically as the scale factor grows", () => {
    expect(lossFromParams(1e9)).toBeLessThan(lossFromParams(1e6));
    expect(lossFromData(1e10)).toBeLessThan(lossFromData(1e8));
    expect(lossFromCompute(1)).toBeLessThan(lossFromCompute(1e-4));
  });

  it("are straight lines in log-log space (constant slope = the exponent)", () => {
    const a = Math.log10(lossFromParams(1e6));
    const b = Math.log10(lossFromParams(1e7));
    const c = Math.log10(lossFromParams(1e8));
    expect(b - a).toBeCloseTo(c - b, 6);
    // The slope of log L vs log N is -alpha_N = -0.076.
    expect(a - b).toBeCloseTo(0.076, 3);
  });
});

describe("the combined law L(N, D)", () => {
  it("recovers L(N) as the data goes to infinity", () => {
    const finite = lossFromParamsAndData(1e7, Number.POSITIVE_INFINITY);
    expect(finite).toBeGreaterThan(0);
    expect(Number.isFinite(finite)).toBe(true);
    // Adding more data can only help, never hurt.
    expect(lossFromParamsAndData(1e7, 1e11)).toBeGreaterThanOrEqual(finite);
  });

  it("shows overfitting grow as the dataset shrinks relative to the model", () => {
    const roomy = overfitPenalty(1e7, 1e11);
    const cramped = overfitPenalty(1e7, 1e7);
    expect(cramped).toBeGreaterThan(roomy);
    expect(roomy).toBeGreaterThanOrEqual(0);
  });
});

describe("the optimal compute allocation", () => {
  it("grows the optimal model size roughly as C^0.73", () => {
    const small = optimalParams(1);
    const big = optimalParams(10);
    const ratio = Math.log10(big / small);
    expect(ratio).toBeCloseTo(0.73, 2);
  });

  it("puts the realized loss at its floor when the split is optimal", () => {
    const optimal = computeAllocation({ ...base, allocation: 1 });
    const starved = computeAllocation({ ...base, allocation: 0.18 });
    const bloated = computeAllocation({ ...base, allocation: 1.8 });
    expect(optimal.excess).toBeLessThan(starved.excess);
    expect(optimal.excess).toBeLessThan(bloated.excess);
    expect(optimal.excess).toBeGreaterThanOrEqual(0);
  });

  it("makes a starved (too-small) model worse than the optimal split", () => {
    const optimal = computeAllocation({ ...base, allocation: 1 });
    const starved = computeAllocation({ ...base, allocation: 0.18 });
    expect(starved.realizedLoss).toBeGreaterThan(optimal.realizedLoss);
  });
});

describe("the curve builder", () => {
  it("reveals more points as the reveal threshold advances", () => {
    const early = buildCurve("compute", -6, 7, POINT_COUNT);
    const late = buildCurve("compute", 2, 7, POINT_COUNT);
    expect(late.length).toBeGreaterThan(early.length);
  });

  it("is deterministic for a given seed and exact for seed 0", () => {
    const a = buildCurve("params", 9, 7, POINT_COUNT);
    const b = buildCurve("params", 9, 7, POINT_COUNT);
    expect(b.map((p) => p.loss)).toEqual(a.map((p) => p.loss));
    const clean = buildCurve("params", 9, 0, POINT_COUNT);
    const first = clean[0];
    expect(first).toBeDefined();
    if (first) {
      expect(first.loss).toBeCloseTo(lossFromParams(first.x), 9);
    }
  });
});

describe("the step reducer", () => {
  it("draws the curve further along the axis on each tick", () => {
    let state = initialState("params-law");
    const start = state.revealLog;
    state = step(state, { kind: "tick" });
    expect(state.revealLog).toBeGreaterThan(start);
  });

  it("resets the reveal to the axis start when the axis changes", () => {
    let state = initialState("params-law");
    state = step(state, { kind: "tick" });
    state = step(state, { kind: "tick" });
    state = step(state, {
      kind: "intervene",
      action: { type: "setAxis", id: "data" },
    });
    expect(state.revealLog).toBe(7);
    expect(state.log.events.at(-1)?.label).toContain("axis");
  });

  it("loads a preset by id", () => {
    const state = step(initialState("params-law"), {
      kind: "intervene",
      action: { type: "loadPreset", id: "small-model" },
    });
    expect(state.params).toEqual(getPreset("small-model").params);
  });

  it("changes the allocation through an intervention and worsens the loss", () => {
    const before = computeAllocation(base);
    const state = step(
      { ...initialState("optimal-split"), params: { ...base } },
      { kind: "intervene", action: { type: "setAllocation", value: 0.18 } },
    );
    const after = computeAllocation(state.params);
    expect(after.realizedLoss).toBeGreaterThan(before.realizedLoss);
  });
});
