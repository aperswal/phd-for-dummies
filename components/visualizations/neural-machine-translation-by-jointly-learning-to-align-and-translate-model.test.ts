import { describe, expect, it } from "vitest";

import {
  ANNOTATION_DIM,
  computeAlignment,
  getPair,
  getPreset,
  initialState,
  PAIRS,
  step,
  type AlignParams,
} from "./neural-machine-translation-by-jointly-learning-to-align-and-translate-model";

const base: AlignParams = {
  pairId: "monotonic",
  step: 0,
  bottleneck: false,
  sharpness: 4,
  seed: 0,
};

describe("soft-alignment attention", () => {
  it("produces weights that form a probability distribution over the source", () => {
    const { weights, source } = computeAlignment(base);
    expect(weights.length).toBe(source.length);
    const total = weights.reduce((sum, value) => sum + value, 0);
    expect(total).toBeCloseTo(1, 6);
    expect(weights.every((value) => value >= 0)).toBe(true);
  });

  it("is deterministic given the same parameters", () => {
    const a = computeAlignment(base);
    const b = computeAlignment({ ...base });
    expect(b.weights).toEqual(a.weights);
    expect(b.context).toEqual(a.context);
  });

  it("returns a context vector of the annotation dimension", () => {
    const { context } = computeAlignment(base);
    expect(context.length).toBe(ANNOTATION_DIM);
  });

  it("aligns each target word to its diagonal source word in a monotonic sentence", () => {
    const pair = getPair("monotonic");
    for (let i = 0; i < pair.target.length; i++) {
      const { argmax } = computeAlignment({ ...base, step: i });
      expect(argmax).toBe(i);
    }
  });

  it("jumps ahead to the reordered noun when writing 'zone' (Figure 3a)", () => {
    const pair = getPair("reorder");
    const zoneStep = pair.target.indexOf("zone");
    const { weights, argmax } = computeAlignment({
      ...base,
      pairId: "reorder",
      step: zoneStep,
      sharpness: 5,
    });
    const areaIndex = pair.source.indexOf("Area");
    expect(argmax).toBe(areaIndex);
    // It reads 'Area' harder than the adjacent adjective 'Economic'.
    const economicIndex = pair.source.indexOf("Economic");
    expect(weights[areaIndex] ?? 0).toBeGreaterThan(
      weights[economicIndex] ?? 0,
    );
  });
});

describe("the fixed-vector bottleneck", () => {
  it("collapses the read to a flat blur, the same at every decoding step", () => {
    const early = computeAlignment({
      ...base,
      pairId: "long",
      step: 1,
      bottleneck: true,
    });
    const late = computeAlignment({
      ...base,
      pairId: "long",
      step: 8,
      bottleneck: true,
    });
    expect(late.weights).toEqual(early.weights);
    const n = early.weights.length;
    for (const weight of early.weights) {
      expect(weight).toBeCloseTo(1 / n, 6);
    }
  });

  it("loses the alignment that soft attention keeps on a long sentence", () => {
    const longTail = 8;
    const attended = computeAlignment({
      ...base,
      pairId: "long",
      step: longTail,
    });
    const bottlenecked = computeAlignment({
      ...base,
      pairId: "long",
      step: longTail,
      bottleneck: true,
    });
    expect(attended.maxWeight).toBeGreaterThan(bottlenecked.maxWeight);
    expect(bottlenecked.entropy).toBeGreaterThan(attended.entropy);
  });
});

describe("the alignment-network gain", () => {
  it("sharpens the read toward a hard alignment as it grows", () => {
    const soft = computeAlignment({
      ...base,
      pairId: "reorder",
      step: 1,
      sharpness: 1,
    });
    const sharp = computeAlignment({
      ...base,
      pairId: "reorder",
      step: 1,
      sharpness: 8,
    });
    expect(sharp.maxWeight).toBeGreaterThan(soft.maxWeight);
    expect(soft.entropy).toBeGreaterThan(sharp.entropy);
  });
});

describe("the step reducer", () => {
  it("advances the decoder one target word per tick and wraps around", () => {
    const n = getPair("monotonic").target.length;
    let state = {
      ...initialState("monotonic"),
      params: { ...base, step: n - 1 },
    };
    state = step(state, { kind: "tick" });
    expect(state.params.step).toBe(0);
    expect(state.log.events.at(-1)?.label).toContain("write");
  });

  it("turns the bottleneck on through an intervention and flattens the read", () => {
    const state = step(initialState("long"), {
      kind: "intervene",
      action: { type: "toggleBottleneck" },
    });
    expect(state.params.bottleneck).toBe(true);
    const { weights } = computeAlignment(state.params);
    const n = weights.length;
    for (const weight of weights) {
      expect(weight).toBeCloseTo(1 / n, 6);
    }
  });

  it("clamps the step when switching to a shorter sentence", () => {
    let state = {
      ...initialState("long"),
      params: { ...base, pairId: "long", step: 9 },
    };
    state = step(state, {
      kind: "intervene",
      action: { type: "setPair", id: "reorder" },
    });
    const reorderLength = getPair("reorder").target.length;
    expect(state.params.step).toBeLessThan(reorderLength);
  });

  it("loads a preset by id", () => {
    const state = step(initialState("monotonic"), {
      kind: "intervene",
      action: { type: "loadPreset", id: "bottleneck" },
    });
    expect(state.params).toEqual(getPreset("bottleneck").params);
  });
});

describe("sentence catalog", () => {
  it("keeps source, target, annotations, and queries the same length in every pair", () => {
    for (const pair of PAIRS) {
      expect(pair.target.length).toBe(pair.source.length);
      expect(pair.annotations.length).toBe(pair.source.length);
      expect(pair.queries.length).toBe(pair.target.length);
    }
  });
});
