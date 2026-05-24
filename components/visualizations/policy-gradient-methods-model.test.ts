import { describe, expect, it } from "vitest";

import {
  initialState,
  isTerminal,
  N_ACTIONS,
  N_STATES,
  status,
  step,
  type Intervention,
  type SimState,
} from "@/components/visualizations/policy-gradient-methods-model";

function run(state: SimState, ticks: number): SimState {
  let next = state;
  for (let i = 0; i < ticks; i++) next = step(next, { kind: "tick" });
  return next;
}

function intervene(state: SimState, action: Intervention): SimState {
  return step(state, { kind: "intervene", action });
}

describe("policy gradient theorem on a gridworld", () => {
  it("reproduces a run exactly from the same seed and inputs", () => {
    const a = run(initialState("reinforce"), 40);
    const b = run(initialState("reinforce"), 40);
    expect(b.theta).toEqual(a.theta);
    expect(b.rng).toBe(a.rng);
    expect(b.rho).toBe(a.rho);
  });

  it("never puts gradient on terminal states (no left-over policy there)", () => {
    const state = run(initialState("exact"), 50);
    for (let s = 0; s < N_STATES; s++) {
      if (!isTerminal(state.mdp, s)) continue;
      for (let a = 0; a < N_ACTIONS; a++) {
        expect(state.exactGrad[s * N_ACTIONS + a]).toBe(0);
      }
    }
  });

  it("follows the exact gradient with cosine 1 (Theorem 1) and climbs to optimal", () => {
    const start = initialState("exact");
    const mid = run(start, 5);
    expect(mid.appliedCos).toBeGreaterThan(0.999);

    const end = run(start, 500);
    const score = status(end);
    expect(score.fractionOfOptimal).toBeGreaterThan(0.9);
    expect(score.nearOptimal).toBe(true);
  });

  it("compatible critic reproduces the exact gradient with zero bias (Theorem 2)", () => {
    const exactEnd = run(initialState("exact"), 300);
    const compatibleEnd = run(initialState("compatible"), 300);
    // Identical step for step: the compatible critic's fit equals A^pi exactly.
    expect(compatibleEnd.theta).toEqual(exactEnd.theta);
    expect(status(compatibleEnd).cos).toBeGreaterThan(0.999);
  });

  it("incompatible critic biases the gradient and stalls well short of optimal", () => {
    const end = run(initialState("incompatible"), 800);
    const score = status(end);
    expect(score.cos).toBeLessThan(0.9);
    expect(score.fractionOfOptimal).toBeLessThan(0.6);
  });

  it("recovers when the critic is switched from incompatible to compatible mid-run", () => {
    const stalled = run(initialState("incompatible"), 400);
    expect(status(stalled).fractionOfOptimal).toBeLessThan(0.6);

    const switched = intervene(stalled, {
      type: "setCritic",
      value: "compatible",
    });
    const recovered = run(switched, 600);
    const score = status(recovered);
    expect(score.cos).toBeGreaterThan(0.999);
    expect(score.fractionOfOptimal).toBeGreaterThan(0.9);
  });

  it("sampled returns are an unbiased estimate that points uphill on average", () => {
    // Average the applied cosine across a window of REINFORCE steps; the noisy
    // estimate should align positively with the true gradient.
    const mid = run(initialState("reinforce"), 80);
    const window = mid.cosHistory.slice(-40);
    const mean = window.reduce((sum, value) => sum + value, 0) / window.length;
    expect(mean).toBeGreaterThan(0);
  });

  it("logs the moment the exact run reaches near-optimal return", () => {
    const end = run(initialState("exact"), 500);
    expect(
      end.log.events.some((event) => event.label.includes("near-optimal")),
    ).toBe(true);
  });

  it("resets the policy to uniform on demand", () => {
    const learned = run(initialState("exact"), 200);
    const reset = intervene(learned, { type: "resetPolicy" });
    expect(reset.theta.every((value) => value === 0)).toBe(true);
    // Uniform softmax means every action equally likely.
    expect(reset.pi[reset.mdp.start]).toEqual([0.25, 0.25, 0.25, 0.25]);
  });
});
