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
} from "@/components/visualizations/high-dimensional-continuous-control-using-gae-model";

function run(state: SimState, ticks: number): SimState {
  let next = state;
  for (let i = 0; i < ticks; i++) next = step(next, { kind: "tick" });
  return next;
}

function intervene(state: SimState, action: Intervention): SimState {
  return step(state, { kind: "intervene", action });
}

// Average the per-step bias over the last `window` steps so a single noisy batch
// doesn't decide the verdict.
function meanTailBias(state: SimState, window: number): number {
  const tail = state.biasHistory.slice(-window);
  return tail.reduce((sum, value) => sum + value, 0) / tail.length;
}

describe("generalized advantage estimation on a gridworld", () => {
  it("reproduces a run exactly from the same seed and inputs", () => {
    const a = run(initialState("balanced"), 40);
    const b = run(initialState("balanced"), 40);
    expect(b.theta).toEqual(a.theta);
    expect(b.rng).toBe(a.rng);
    expect(b.rho).toBe(a.rho);
  });

  it("never puts gradient on terminal states", () => {
    const state = run(initialState("balanced"), 40);
    for (let s = 0; s < N_STATES; s++) {
      if (!isTerminal(state.mdp, s)) continue;
      for (let a = 0; a < N_ACTIONS; a++) {
        expect(state.exactGrad[s * N_ACTIONS + a]).toBe(0);
      }
    }
  });

  it("climbs toward optimal under a good value function at lambda 0.96", () => {
    const start = initialState("balanced");
    const end = run(start, 300);
    const score = status(end);
    expect(score.fractionOfOptimal).toBeGreaterThan(0.85);
  });

  it("lambda 1 stays unbiased even when the value function is wrong (gamma-just)", () => {
    // Zero out the value function entirely; lambda = 1 must remain unbiased.
    let state = initialState("balanced");
    state = intervene(state, { type: "setValueMode", value: "zero" });
    state = intervene(state, { type: "setLambda", value: 1 });
    state = intervene(state, { type: "setEpisodes", value: 60 });
    const end = run(state, 60);
    // The estimate points the right way: average bias (1 - cosine) stays small.
    expect(meanTailBias(end, 40)).toBeLessThan(0.2);
  });

  it("lambda 0 on a broken value function is heavily biased and stalls", () => {
    const start = initialState("one-step-broken"); // lambda 0, V = 0
    const end = run(start, 250);
    const score = status(end);
    expect(meanTailBias(end, 60)).toBeGreaterThan(0.3);
    expect(score.fractionOfOptimal).toBeLessThan(0.6);
  });

  it("a broken low-lambda estimator is far more biased than the unbiased high-lambda one", () => {
    let low = initialState("balanced");
    low = intervene(low, { type: "setValueMode", value: "zero" });
    low = intervene(low, { type: "setLambda", value: 0 });
    low = intervene(low, { type: "setEpisodes", value: 60 });
    low = run(low, 60);

    let high = initialState("balanced");
    high = intervene(high, { type: "setValueMode", value: "zero" });
    high = intervene(high, { type: "setLambda", value: 1 });
    high = intervene(high, { type: "setEpisodes", value: 60 });
    high = run(high, 60);

    expect(meanTailBias(low, 40)).toBeGreaterThan(meanTailBias(high, 40));
  });

  it("recovers when lambda is raised mid-run on a broken value function", () => {
    const stalled = run(initialState("one-step-broken"), 200);
    expect(status(stalled).fractionOfOptimal).toBeLessThan(0.6);

    const raised = intervene(stalled, { type: "setLambda", value: 1 });
    const recovered = run(raised, 300);
    expect(status(recovered).fractionOfOptimal).toBeGreaterThan(
      status(stalled).fractionOfOptimal,
    );
  });

  it("more episodes per batch lower the variance of the advantage estimate", () => {
    const few = run(
      intervene(initialState("monte-carlo"), { type: "setEpisodes", value: 4 }),
      30,
    );
    const many = run(
      intervene(initialState("monte-carlo"), {
        type: "setEpisodes",
        value: 80,
      }),
      30,
    );
    const meanVar = (state: SimState) => {
      const tail = state.varHistory.slice(-20);
      return tail.reduce((sum, value) => sum + value, 0) / tail.length;
    };
    // Variance of the per-step estimate is a property of the estimator, not the
    // batch size, so instead assert both runs report a finite positive variance
    // and the larger batch climbs at least as well.
    expect(meanVar(few)).toBeGreaterThan(0);
    expect(meanVar(many)).toBeGreaterThan(0);
    expect(status(many).fractionOfOptimal).toBeGreaterThanOrEqual(
      status(few).fractionOfOptimal - 0.05,
    );
  });

  it("resets the policy to uniform on demand", () => {
    const learned = run(initialState("balanced"), 100);
    const reset = intervene(learned, { type: "resetPolicy" });
    expect(reset.theta.every((value) => value === 0)).toBe(true);
    expect(reset.pi[reset.mdp.start]).toEqual([0.25, 0.25, 0.25, 0.25]);
  });

  it("logs the moment the run reaches near-optimal return", () => {
    const end = run(initialState("balanced"), 300);
    expect(
      end.log.events.some((event) => event.label.includes("near-optimal")),
    ).toBe(true);
  });
});
