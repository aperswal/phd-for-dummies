import { describe, expect, it } from "vitest";

import {
  computeAttention,
  getPreset,
  getSentence,
  HEADS,
  initialState,
  positionalEncoding,
  step,
  type AttentionParams,
} from "./attention-is-all-you-need-model";

const base: AttentionParams = {
  sentenceId: "animal",
  headIndex: 0,
  queryIndex: 7,
  scaling: true,
  causalMask: false,
  magnitude: 5,
};

describe("scaled dot-product attention", () => {
  it("produces weights that form a probability distribution", () => {
    const { weights } = computeAttention(base);
    const total = weights.reduce((sum, value) => sum + value, 0);
    expect(total).toBeCloseTo(1, 6);
    expect(weights.every((value) => value >= 0)).toBe(true);
  });

  it("is deterministic given the same parameters", () => {
    const a = computeAttention(base);
    const b = computeAttention({ ...base });
    expect(b.weights).toEqual(a.weights);
    expect(b.output).toEqual(a.output);
  });

  it("resolves the pronoun 'it' to 'animal', not the nearer 'street'", () => {
    const { tokens, weights, queryIndex } = computeAttention(base);
    expect(tokens[queryIndex]).toBe("it");
    const argmax = weights.indexOf(Math.max(...weights));
    expect(tokens[argmax]).toBe("animal");
    expect(weights[tokens.indexOf("animal")] ?? 0).toBeGreaterThan(
      weights[tokens.indexOf("street")] ?? 0,
    );
  });

  it("resolves 'it' to the only animate noun in a different sentence", () => {
    const result = computeAttention({
      ...base,
      sentenceId: "cat",
      queryIndex: 4,
    });
    const argmax = result.weights.indexOf(Math.max(...result.weights));
    expect(result.tokens[result.queryIndex]).toBe("it");
    expect(result.tokens[argmax]).toBe("cat");
  });
});

describe("the sqrt(d_k) scaling", () => {
  it("spreads the weight that large scores would otherwise collapse", () => {
    const saturated = computeAttention({
      ...base,
      headIndex: 0,
      queryIndex: 6,
      scaling: false,
      magnitude: 16,
    });
    const scaled = computeAttention({
      ...base,
      headIndex: 0,
      queryIndex: 6,
      scaling: true,
      magnitude: 16,
    });
    expect(saturated.maxWeight).toBeGreaterThan(scaled.maxWeight);
    expect(scaled.entropy).toBeGreaterThan(saturated.entropy);
  });
});

describe("the causal mask", () => {
  it("zeroes every position past the query and still sums to one", () => {
    const queryIndex = 6;
    const { weights } = computeAttention({
      ...base,
      causalMask: true,
      queryIndex,
    });
    for (let j = queryIndex + 1; j < weights.length; j++) {
      expect(weights[j]).toBe(0);
    }
    const total = weights.reduce((sum, value) => sum + value, 0);
    expect(total).toBeCloseTo(1, 6);
  });
});

describe("positional encoding", () => {
  it("matches the sin/cos formula at position 0", () => {
    const pe = positionalEncoding(0);
    expect(pe[0]).toBeCloseTo(0, 6);
    expect(pe[1]).toBeCloseTo(1, 6);
  });

  it("drives the neighbor head to weight an adjacent word over a far one", () => {
    const { weights, queryIndex } = computeAttention({
      ...base,
      headIndex: 1,
      queryIndex: 5,
      magnitude: 8,
    });
    expect(weights[queryIndex - 1] ?? 0).toBeGreaterThan(weights[0] ?? 0);
    expect(weights[queryIndex + 1] ?? 0).toBeGreaterThan(
      weights[weights.length - 1] ?? 0,
    );
  });
});

describe("the step reducer", () => {
  it("advances the query one word per tick and wraps around", () => {
    const n = getSentence("animal").tokens.length;
    let state = {
      ...initialState("meaning"),
      params: { ...base, queryIndex: n - 1 },
    };
    state = step(state, { kind: "tick" });
    expect(state.params.queryIndex).toBe(0);
    expect(state.log.events.at(-1)?.label).toContain("query");
  });

  it("turns masking on through an intervention and zeroes the future", () => {
    const state = step(initialState("meaning"), {
      kind: "intervene",
      action: { type: "toggleMask" },
    });
    expect(state.params.causalMask).toBe(true);
    const { weights, queryIndex } = computeAttention(state.params);
    for (let j = queryIndex + 1; j < weights.length; j++) {
      expect(weights[j]).toBe(0);
    }
  });

  it("loads a preset by id", () => {
    const state = step(initialState("meaning"), {
      kind: "intervene",
      action: { type: "loadPreset", id: "saturation" },
    });
    expect(state.params).toEqual(getPreset("saturation").params);
  });
});

describe("head catalog", () => {
  it("keeps every head's query and key projections the same width (d_k)", () => {
    for (const head of HEADS) {
      expect(head.qDirs.length).toBe(head.kDirs.length);
    }
  });
});
