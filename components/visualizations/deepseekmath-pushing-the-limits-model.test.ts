import { describe, expect, it } from "vitest";

import {
  ANSWER_KINDS,
  initialState,
  policyProbs,
  PRESETS,
  step,
  type Input,
  type SimState,
} from "./deepseekmath-pushing-the-limits-model";

function tick(state: SimState, times: number): SimState {
  let next = state;
  const input: Input = { kind: "tick" };
  for (let i = 0; i < times; i++) next = step(next, input);
  return next;
}

function correctMass(state: SimState): number {
  const probs = policyProbs(state);
  return probs.reduce(
    (sum, p, i) => sum + (ANSWER_KINDS[i]?.correct ? p : 0),
    0,
  );
}

describe("GRPO model invariants", () => {
  it("keeps the policy a valid distribution after many steps", () => {
    const state = tick(initialState("sharpen"), 80);
    const probs = policyProbs(state);
    const total = probs.reduce((sum, p) => sum + p, 0);
    expect(total).toBeCloseTo(1, 6);
    for (const p of probs) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(p)).toBe(true);
    }
  });

  it("is deterministic given the same seed and inputs", () => {
    const a = tick(initialState("sharpen"), 40);
    const b = tick(initialState("sharpen"), 40);
    expect(policyProbs(a)).toEqual(policyProbs(b));
  });

  it("normalizes rewards so the group advantages sum to about zero", () => {
    const state = tick(initialState("sharpen"), 5);
    const sumAdv = state.lastGroup.reduce((sum, s) => sum + s.advantage, 0);
    expect(Math.abs(sumAdv)).toBeLessThan(1e-6);
  });
});

describe("GRPO learns from a group baseline", () => {
  it("shifts mass onto correct kinds on the healthy run", () => {
    const start = initialState("sharpen");
    const end = tick(start, 60);
    expect(correctMass(end)).toBeGreaterThan(correctMass(start) + 0.1);
  });

  it("lifts Maj@K far more than Pass@K, the paper's finding", () => {
    const start = initialState("maj-not-pass");
    const end = tick(start, 80);
    const startPoint = start.history[start.history.length - 1];
    const endPoint = end.history[end.history.length - 1];
    if (!startPoint || !endPoint) throw new Error("missing history");
    const majGain = endPoint.majAtK - startPoint.majAtK;
    const passGain = endPoint.passAtK - startPoint.passAtK;
    expect(majGain).toBeGreaterThan(0.05);
    expect(majGain).toBeGreaterThan(passGain * 2);
  });
});

describe("interventions change the outcome under the paper's rules", () => {
  it("chases a wrong answer when the reward model is hacked", () => {
    const start = initialState("reward-hack");
    const hackedKind = start.params.hackKind;
    expect(hackedKind).toBeGreaterThanOrEqual(0);
    const end = tick(start, 80);
    const probs = policyProbs(end);
    const hackedProb = probs[hackedKind] ?? 0;
    // The over-rewarded wrong kind ends up the most likely answer.
    expect(hackedProb).toBe(Math.max(...probs));
    expect(ANSWER_KINDS[hackedKind]?.correct).toBe(false);
    expect(correctMass(end)).toBeLessThan(correctMass(start));
  });

  it("cannot lift correct mass with no headroom to sample from", () => {
    const start = initialState("no-headroom");
    const end = tick(start, 80);
    // Reweighting alone cannot raise an answer the policy almost never samples.
    expect(correctMass(end)).toBeLessThan(0.2);
  });

  it("gives zero advantage when every sampled reward ties", () => {
    // Force a state where all sampled kinds are wrong with identical reward by
    // collapsing the policy onto a single wrong kind, then stepping: the group is
    // uniform, so every advantage is zero and the policy does not move from the
    // reward term.
    const base = initialState("sharpen");
    const collapsed: SimState = {
      ...base,
      logits: [-20, -20, 8, -20, -20],
      refLogits: [-20, -20, 8, -20, -20],
      params: { ...base.params, klCoef: 0, rewardNoise: 0 },
    };
    const before = policyProbs(collapsed);
    const after = tick(collapsed, 1);
    const moved = policyProbs(after).reduce(
      (max, p, i) => Math.max(max, Math.abs(p - (before[i] ?? 0))),
      0,
    );
    expect(moved).toBeLessThan(1e-6);
    expect(after.lastStats.groupStd).toBeCloseTo(0, 6);
  });

  it("dividing by std rescales advantages versus mean-only", () => {
    const base = initialState("sharpen");
    const withStd = step({ ...base }, { kind: "tick" });
    const meanOnly = step(
      { ...base, params: { ...base.params, normalizeStd: false } },
      { kind: "tick" },
    );
    const stdSpread = Math.max(
      ...withStd.lastGroup.map((s) => Math.abs(s.advantage)),
    );
    const meanSpread = Math.max(
      ...meanOnly.lastGroup.map((s) => Math.abs(s.advantage)),
    );
    expect(stdSpread).not.toBeCloseTo(meanSpread, 4);
  });

  it("restore returns the simulation to an earlier moment", () => {
    const start = initialState("sharpen");
    const mid = tick(start, 10);
    const later = tick(mid, 10);
    const restored = step(later, {
      kind: "intervene",
      action: { type: "restore", snapshot: mid, label: "rewind" },
    });
    expect(restored.step).toBe(mid.step);
    expect(restored.logits).toEqual(mid.logits);
  });
});

describe("presets", () => {
  it("every preset builds a valid initial state", () => {
    for (const preset of PRESETS) {
      const state = initialState(preset.id);
      expect(state.logits).toHaveLength(ANSWER_KINDS.length);
      const total = policyProbs(state).reduce((sum, p) => sum + p, 0);
      expect(total).toBeCloseTo(1, 6);
    }
  });
});
