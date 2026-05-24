import { describe, expect, it } from "vitest";

import {
  getPreset,
  implicitReward,
  initialState,
  policyKl,
  PROMPTS,
  softmax,
  step,
  type SimState,
} from "./direct-preference-optimization-model";

function run(state: SimState, steps: number): SimState {
  let current = state;
  for (let i = 0; i < steps; i++) current = step(current, { kind: "tick" });
  return current;
}

function preferredProb(
  state: SimState,
  promptId: string,
  index: number,
): number {
  const logits = state.logits[promptId] ?? [];
  return softmax(logits)[index] ?? 0;
}

describe("DPO loss and implicit reward", () => {
  it("starts at log(2) when the policy equals the reference", () => {
    // With pi_theta = pi_ref every log-ratio is zero, the margin is zero, and
    // -log sigma(0) = log 2 for every pair.
    const state = initialState("align");
    expect(state.loss).toBeCloseTo(Math.log(2), 6);
  });

  it("gives every completion zero implicit reward at the start", () => {
    const state = initialState("align");
    for (const prompt of PROMPTS) {
      for (let i = 0; i < prompt.completions.length; i++) {
        expect(implicitReward(state, prompt.id, i)).toBeCloseTo(0, 6);
      }
    }
  });

  it("is deterministic given the same preset", () => {
    const a = run(initialState("align"), 30);
    const b = run(initialState("align"), 30);
    expect(b.logits).toEqual(a.logits);
    expect(b.loss).toBeCloseTo(a.loss, 12);
  });

  it("keeps the loss non-negative", () => {
    const state = run(initialState("align"), 40);
    expect(state.loss).toBeGreaterThanOrEqual(0);
  });
});

describe("the DPO gradient step", () => {
  it("raises the preferred completion and lowers the dispreferred one", () => {
    const before = initialState("align");
    const after = run(before, 25);
    const review = "review";
    expect(preferredProb(after, review, 0)).toBeGreaterThan(
      preferredProb(before, review, 0),
    );
    expect(preferredProb(after, review, 3)).toBeLessThan(
      preferredProb(before, review, 3),
    );
  });

  it("drives the loss down on the honest dataset", () => {
    const before = initialState("align");
    const after = run(before, 40);
    expect(after.loss).toBeLessThan(before.loss);
  });

  it("makes the preferred completion the implicit reward winner", () => {
    const after = run(initialState("align"), 40);
    const rewardBest = implicitReward(after, "review", 0);
    const rewardWorst = implicitReward(after, "review", 3);
    expect(rewardBest).toBeGreaterThan(rewardWorst);
  });
});

describe("the sigma weight prevents degeneration", () => {
  it("lets the policy drift farther from the reference when the weight is off", () => {
    const withWeight = run(initialState("align"), 60);
    const withoutWeight = run(initialState("degenerate"), 60);
    expect(policyKl(withoutWeight)).toBeGreaterThan(policyKl(withWeight));
  });

  it("degenerates the preferred probability toward one without the weight", () => {
    const adaptive = run(initialState("align"), 120);
    const naive = run(initialState("degenerate"), 120);
    const adaptiveTop = softmax(adaptive.logits.review ?? [])[0] ?? 0;
    const naiveTop = softmax(naive.logits.review ?? [])[0] ?? 0;
    // The naive objective shoves the winner toward probability one; the adaptive
    // weight cools off once the pair is ordered correctly, so it stops short.
    expect(naiveTop).toBeGreaterThan(0.95);
    expect(adaptiveTop).toBeLessThan(naiveTop);
  });
});

describe("beta controls the KL constraint", () => {
  it("holds the policy closer to the reference at a larger beta", () => {
    const loose = run(initialState("align"), 50);
    const tight = run(initialState("tight"), 50);
    expect(policyKl(tight)).toBeLessThan(policyKl(loose));
  });
});

describe("a mislabeled pair changes the outcome", () => {
  it("chases the dispreferred completion up when the label is flipped", () => {
    const honest = run(initialState("align"), 40);
    const flipped = run(initialState("mislabel"), 40);
    // In the honest run the rude pan (index 3) falls; in the flipped run the
    // labeler prefers it, so DPO raises it instead.
    expect(preferredProb(flipped, "review", 3)).toBeGreaterThan(
      preferredProb(honest, "review", 3),
    );
    expect(preferredProb(flipped, "review", 0)).toBeLessThan(
      preferredProb(honest, "review", 0),
    );
  });

  it("flips a pair mid-run through an intervention and reverses the pressure", () => {
    let state = run(initialState("align"), 20);
    const beforeWorst = preferredProb(state, "review", 3);
    state = step(state, {
      kind: "intervene",
      action: { type: "flipPair", index: 0 },
    });
    state = run(state, 30);
    expect(preferredProb(state, "review", 3)).toBeGreaterThan(beforeWorst);
  });
});

describe("interventions and presets", () => {
  it("loads a preset by id", () => {
    const state = step(initialState("align"), {
      kind: "intervene",
      action: { type: "loadPreset", id: "degenerate" },
    });
    expect(state.params.useAdaptiveWeight).toBe(false);
    expect(state.params).toEqual(getPreset("degenerate").params);
  });

  it("removing every pair leaves a zero loss and a still policy", () => {
    let state = initialState("align");
    state = step(state, {
      kind: "intervene",
      action: { type: "removePair", index: 1 },
    });
    state = step(state, {
      kind: "intervene",
      action: { type: "removePair", index: 0 },
    });
    expect(state.dataset.length).toBe(0);
    const before = state.logits;
    state = run(state, 10);
    expect(state.logits).toEqual(before);
    expect(state.loss).toBe(0);
  });

  it("records each step in the event log with a restorable snapshot", () => {
    const state = run(initialState("align"), 5);
    expect(state.log.events.length).toBeGreaterThan(0);
    const last = state.log.events.at(-1);
    expect(last?.snapshot.logits).toBeDefined();
  });

  it("rewinds to an earlier snapshot through restore", () => {
    const early = run(initialState("align"), 5);
    const snap = early.log.events.at(-1)?.snapshot;
    expect(snap).toBeDefined();
    const later = run(early, 30);
    if (!snap) throw new Error("missing snapshot");
    const restored = step(later, {
      kind: "intervene",
      action: { type: "restore", snapshot: snap, label: "rewind" },
    });
    expect(restored.logits).toEqual(snap.logits);
  });
});

describe("softmax", () => {
  it("returns a probability distribution", () => {
    const p = softmax([1, 2, 3, 0]);
    const total = p.reduce((sum, value) => sum + value, 0);
    expect(total).toBeCloseTo(1, 6);
    expect(p.every((value) => value >= 0)).toBe(true);
  });
});
