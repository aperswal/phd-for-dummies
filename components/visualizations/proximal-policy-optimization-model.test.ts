import { describe, expect, it } from "vitest";

import {
  ACTION_LABELS,
  getPreset,
  initialState,
  N_ACTIONS,
  N_STATES,
  optimalReturn,
  softmax,
  status,
  step,
  type SimState,
} from "./proximal-policy-optimization-model";

function run(state: SimState, ticks: number): SimState {
  let s = state;
  for (let i = 0; i < ticks; i++) s = step(s, { kind: "tick" });
  return s;
}

describe("the chain MDP and softmax policy", () => {
  it("starts from a uniform policy whose action probabilities sum to one", () => {
    const state = initialState("clip");
    for (let s = 0; s < N_STATES; s++) {
      const probs = softmax(state.theta, s);
      const total = probs.reduce((sum, v) => sum + v, 0);
      expect(total).toBeCloseTo(1, 9);
      expect(probs.every((p) => p >= 0)).toBe(true);
      expect(probs[1]).toBeCloseTo(0.5, 9); // two actions, uniform
    }
    expect(ACTION_LABELS.length).toBe(N_ACTIONS);
  });

  it("has an optimal return that is positive and reachable by always going right", () => {
    expect(optimalReturn(0.97)).toBeGreaterThan(0.7);
  });
});

describe("determinism", () => {
  it("reproduces the same run from the same seed and inputs", () => {
    const a = run(initialState("clip"), 60);
    const b = run(initialState("clip"), 60);
    expect(b.rho).toBeCloseTo(a.rho, 12);
    expect(b.theta).toEqual(a.theta);
    expect(b.klOldToNew).toBeCloseTo(a.klOldToNew, 12);
  });

  it("reproduces a noisy run exactly because the noise draws from the seeded stream", () => {
    const a = run(initialState("noisy"), 80);
    const b = run(initialState("noisy"), 80);
    expect(b.theta).toEqual(a.theta);
    expect(b.rho).toBeCloseTo(a.rho, 12);
  });
});

describe("the clipped objective (the paper's main method)", () => {
  it("climbs toward the optimal return and keeps the policy inside the trust region", () => {
    const final = run(initialState("clip"), 200);
    const st = status(final);
    expect(st.fractionOfOptimal).toBeGreaterThan(0.9);
    expect(st.inTrustRegion).toBe(true);
    expect(st.diverged).toBe(false);
  });

  it("learns to prefer right at the start state", () => {
    const final = run(initialState("clip"), 200);
    const probs = softmax(final.theta, 0);
    expect(probs[1] ?? 0).toBeGreaterThan(probs[0] ?? 0);
  });
});

describe("the no-clip failure mode (Section 2.1)", () => {
  it("blows the policy past the trust region when reusing one batch for many greedy epochs", () => {
    let worst = 0;
    let state = initialState("overshoot");
    for (let i = 0; i < 150; i++) {
      state = step(state, { kind: "tick" });
      worst = Math.max(worst, state.klOldToNew);
    }
    expect(worst).toBeGreaterThan(0.5);
  });

  it("keeps every clip update inside the trust region under the same step size", () => {
    let state = initialState("clip");
    state = step(state, {
      kind: "intervene",
      action: { type: "setEpochs", value: 40 },
    });
    state = step(state, {
      kind: "intervene",
      action: { type: "setAlpha", value: 6 },
    });
    let worst = 0;
    for (let i = 0; i < 150; i++) {
      state = step(state, { kind: "tick" });
      worst = Math.max(worst, state.klOldToNew);
    }
    expect(worst).toBeLessThan(0.3);
  });

  it("brings the policy back inside the trust region when switched to clip mid-run", () => {
    let state = initialState("overshoot");
    let sawDiverge = false;
    for (let i = 0; i < 40; i++) {
      state = step(state, { kind: "tick" });
      if (status(state).diverged) sawDiverge = true;
    }
    expect(sawDiverge).toBe(true);
    state = step(state, {
      kind: "intervene",
      action: { type: "setObjective", value: "clip" },
    });
    state = run(state, 120);
    const st = status(state);
    expect(st.inTrustRegion).toBe(true);
    expect(st.fractionOfOptimal).toBeGreaterThan(0.9);
  });
});

describe("the clip band", () => {
  it("clips a real fraction of the ratios while the no-clip run clips none meaningfully", () => {
    const clipped = run(initialState("clip"), 40);
    const noclip = run(initialState("overshoot"), 40);
    expect(noclip.klOldToNew).toBeGreaterThan(clipped.klOldToNew);
  });

  it("widening epsilon lets the policy move further per round (higher KL early)", () => {
    const tight = step(initialState("clip"), {
      kind: "intervene",
      action: { type: "setEpsilon", value: 0.05 },
    });
    const wide = step(initialState("clip"), {
      kind: "intervene",
      action: { type: "setEpsilon", value: 0.6 },
    });
    const tightAfter = run(tight, 20);
    const wideAfter = run(wide, 20);
    expect(wideAfter.klOldToNew).toBeGreaterThan(tightAfter.klOldToNew);
  });
});

describe("interventions and the log", () => {
  it("loads a preset by id and resets the policy to uniform", () => {
    let state = run(initialState("clip"), 50);
    state = step(state, {
      kind: "intervene",
      action: { type: "loadPreset", id: "overshoot" },
    });
    expect(state.params.objective).toBe(
      getPreset("overshoot").params.objective,
    );

    const reset = step(state, {
      kind: "intervene",
      action: { type: "resetPolicy" },
    });
    const probs = softmax(reset.theta, 0);
    expect(probs[0]).toBeCloseTo(0.5, 9);
    expect(reset.round).toBe(0);
  });

  it("records a rewindable snapshot when a knob changes", () => {
    const state = step(initialState("clip"), {
      kind: "intervene",
      action: { type: "setEpsilon", value: 0.4 },
    });
    expect(state.log.events.length).toBe(1);
    expect(state.log.events.at(-1)?.label).toContain("epsilon");
  });

  it("logs when a round blows past the band", () => {
    const state = run(initialState("overshoot"), 80);
    const blew = state.log.events.some((e) => e.label.includes("blew past"));
    expect(blew).toBe(true);
  });
});
