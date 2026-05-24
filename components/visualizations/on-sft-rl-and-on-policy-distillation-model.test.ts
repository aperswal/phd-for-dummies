import { describe, expect, it } from "vitest";

import {
  computeGeometry,
  getPreset,
  initialState,
  methodName,
  step,
  type GeometryParams,
  type Method,
} from "@/components/visualizations/on-sft-rl-and-on-policy-distillation-model";

function paramsFor(
  method: Method,
  overrides: Partial<GeometryParams> = {},
): GeometryParams {
  return {
    method,
    batchSize: 120,
    rewardBias: 1,
    pivotStrength: 1,
    clipping: false,
    spread: 0.9,
    seed: 7,
    ...overrides,
  };
}

describe("computeGeometry determinism and invariants", () => {
  it("is deterministic for the same params", () => {
    const a = computeGeometry(paramsFor("rl"));
    const b = computeGeometry(paramsFor("rl"));
    expect(b.update).toEqual(a.update);
    expect(b.gradients.length).toBe(a.gradients.length);
  });

  it("produces one gradient per token, plus the pivot for OPSD", () => {
    expect(
      computeGeometry(paramsFor("sft", { batchSize: 50 })).gradients,
    ).toHaveLength(50);
    // OPSD appends a single extra pivot token to the diffuse fan.
    expect(
      computeGeometry(paramsFor("opsd", { batchSize: 50 })).gradients,
    ).toHaveLength(51);
  });

  it("clamps an absurd batch size into the valid range", () => {
    expect(
      computeGeometry(paramsFor("rl", { batchSize: 5000 })).gradients.length,
    ).toBeLessThanOrEqual(220);
    expect(
      computeGeometry(paramsFor("rl", { batchSize: 1 })).gradients.length,
    ).toBeGreaterThanOrEqual(2);
  });
});

describe("RL: sparse but saved by destructive interference", () => {
  it("a large batch cancels noise into a small, trustworthy update", () => {
    const big = computeGeometry(paramsFor("rl", { batchSize: 200, seed: 7 }));
    expect(big.collapsed).toBe(false);
    expect(big.updateMagnitude).toBeLessThan(big.meanMagnitude);
  });

  it("the averaged update shrinks relative to the cloud as the batch grows", () => {
    const small = computeGeometry(paramsFor("rl", { batchSize: 8, seed: 4 }));
    const big = computeGeometry(paramsFor("rl", { batchSize: 200, seed: 4 }));
    const smallRatio = small.updateMagnitude / small.meanMagnitude;
    const bigRatio = big.updateMagnitude / big.meanMagnitude;
    expect(bigRatio).toBeLessThan(smallRatio);
  });
});

describe("SFT: dense, biased, but spread out", () => {
  it("does not cancel: the update is a large fraction of the per-token magnitude", () => {
    const result = computeGeometry(paramsFor("sft", { batchSize: 120 }));
    // A coherent fan keeps most of its length after averaging, unlike RL noise.
    expect(result.updateMagnitude / result.meanMagnitude).toBeGreaterThan(0.4);
  });

  it("a wider spread is more diffuse than a narrow one", () => {
    const wide = computeGeometry(paramsFor("sft", { spread: 1.2, seed: 9 }));
    const narrow = computeGeometry(paramsFor("sft", { spread: 0.15, seed: 9 }));
    expect(wide.diffuseness).toBeGreaterThan(narrow.diffuseness);
  });
});

describe("OPSD: a single pivot token dominates the loss", () => {
  it("the pivot makes the cloud highly concentrated", () => {
    const sft = computeGeometry(paramsFor("sft", { batchSize: 60 }));
    const opsd = computeGeometry(
      paramsFor("opsd", { batchSize: 60, pivotStrength: 100 }),
    );
    expect(opsd.concentration).toBeGreaterThan(sft.concentration * 3);
  });

  it("without clipping a strong pivot collapses the update", () => {
    const result = computeGeometry(
      paramsFor("opsd", {
        batchSize: 60,
        pivotStrength: 100,
        clipping: false,
        seed: 5,
      }),
    );
    expect(result.collapsed).toBe(true);
  });

  it("turning the per-token KL clip on rescues the same collapse", () => {
    const unclipped = computeGeometry(
      paramsFor("opsd", {
        batchSize: 60,
        pivotStrength: 100,
        clipping: false,
        seed: 5,
      }),
    );
    const clipped = computeGeometry(
      paramsFor("opsd", {
        batchSize: 60,
        pivotStrength: 100,
        clipping: true,
        seed: 5,
      }),
    );
    expect(unclipped.collapsed).toBe(true);
    expect(clipped.collapsed).toBe(false);
    expect(clipped.updateMagnitude).toBeLessThan(unclipped.updateMagnitude);
  });
});

describe("step reducer", () => {
  it("a tick advances the seed and the tick count and logs a redraw", () => {
    const start = initialState("rl-cancels");
    const after = step(start, { kind: "tick" });
    expect(after.tick).toBe(1);
    expect(after.params.seed).toBe(start.params.seed + 1);
    expect(after.log.events.at(-1)?.label).toContain("redraw");
  });

  it("toggling clipping flips the flag and records it", () => {
    const start = initialState("opsd-collapse");
    const after = step(start, {
      kind: "intervene",
      action: { type: "toggleClipping" },
    });
    expect(after.params.clipping).toBe(!start.params.clipping);
    expect(after.log.events.at(-1)?.label).toContain("clip");
  });

  it("the opsd-collapse preset collapses and the opsd-clipped preset does not", () => {
    expect(computeGeometry(getPreset("opsd-collapse").params).collapsed).toBe(
      true,
    );
    expect(computeGeometry(getPreset("opsd-clipped").params).collapsed).toBe(
      false,
    );
  });

  it("restore returns to a recorded parameter set", () => {
    const start = initialState("rl-cancels");
    const snapshot = start.params;
    const moved = step(start, {
      kind: "intervene",
      action: { type: "setMethod", method: "opsd" },
    });
    const restored = step(moved, {
      kind: "intervene",
      action: { type: "restore", params: snapshot, label: "rewind" },
    });
    expect(restored.params.method).toBe("rl");
  });

  it("exposes readable method names", () => {
    expect(methodName("opsd")).toBe("OPSD");
    expect(methodName("rl")).toBe("RL");
  });
});
