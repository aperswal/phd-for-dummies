import { describe, expect, it } from "vitest";

import {
  EMERGENCE_SCALE_B,
  initialState,
  solve,
  step,
  stepErrorRate,
  type Params,
} from "./chain-of-thought-prompting-model";

const base: Params = {
  problemId: "apples",
  mode: "cot",
  scaleB: 540,
  seed: 1,
  injected: [],
};

describe("chain-of-thought solving", () => {
  it("is deterministic given the same params", () => {
    const a = solve(base);
    const b = solve({ ...base, injected: [...base.injected] });
    expect(b.finalAnswer).toBe(a.finalAnswer);
    expect(b.steps.map((s) => s.value)).toEqual(a.steps.map((s) => s.value));
  });

  it("reaches the right answer with chain of thought at large scale", () => {
    const solution = solve(base);
    expect(solution.correct).toBe(true);
    expect(solution.finalAnswer).toBe(9);
    expect(solution.steps.every((step) => !step.faulted)).toBe(true);
  });

  it("gets the multi-step problem wrong with standard prompting", () => {
    const solution = solve({ ...base, mode: "standard" });
    expect(solution.steps).toHaveLength(0);
    expect(solution.correct).toBe(false);
  });
});

describe("the ablations behave like the paper", () => {
  it("variable-compute dots scores like the baseline (no reasoning)", () => {
    const dots = solve({ ...base, mode: "dots" });
    expect(dots.steps).toHaveLength(0);
    expect(dots.dots).toBeGreaterThan(0);
    expect(dots.correct).toBe(false);
  });

  it("reasoning-after-answer can't fix the answer even when its steps are sound", () => {
    const after = solve({ ...base, mode: "after" });
    // The chain it writes is sound at 540B, but the reported answer was committed
    // before the chain, so it stays wrong like the baseline.
    expect(after.correct).toBe(false);
  });

  it("equation-only gains nothing from scale", () => {
    const small = solve({ ...base, mode: "equation", scaleB: 8 });
    const large = solve({ ...base, mode: "equation", scaleB: 540 });
    const rate = stepErrorRate(EMERGENCE_SCALE_B / 12);
    expect(rate).toBeGreaterThan(0.3);
    // Same fault rate regardless of scale, so the same seed gives the same chain.
    expect(large.steps.map((s) => s.faulted)).toEqual(
      small.steps.map((s) => s.faulted),
    );
  });
});

describe("scale emergence", () => {
  it("falls toward zero error as scale grows past the threshold", () => {
    expect(stepErrorRate(8)).toBeGreaterThan(stepErrorRate(540));
    expect(stepErrorRate(540)).toBeLessThan(0.1);
    expect(stepErrorRate(8)).toBeGreaterThan(0.5);
  });

  it("breaks a chain at small scale that holds at large scale (servers, seed 3)", () => {
    const small = solve({
      problemId: "servers",
      mode: "cot",
      scaleB: 8,
      seed: 3,
      injected: [],
    });
    const large = solve({
      problemId: "servers",
      mode: "cot",
      scaleB: 540,
      seed: 3,
      injected: [],
    });
    expect(small.steps.some((step) => step.faulted)).toBe(true);
    expect(small.correct).toBe(false);
    expect(large.steps.every((step) => !step.faulted)).toBe(true);
    expect(large.correct).toBe(true);
  });
});

describe("injecting a fault mid-chain", () => {
  it("a broken earlier step poisons the final answer", () => {
    const clean = solve(base);
    expect(clean.correct).toBe(true);
    const broken = solve({ ...base, injected: [0] });
    const firstStep = broken.steps[0];
    expect(firstStep?.faulted).toBe(true);
    expect(firstStep?.reason).toBe("injected");
    expect(broken.finalAnswer).not.toBe(clean.finalAnswer);
    expect(broken.correct).toBe(false);
  });

  it("the corrupted value flows forward to later steps", () => {
    const broken = solve({
      problemId: "servers",
      mode: "cot",
      scaleB: 540,
      seed: 1,
      injected: [1],
    });
    const second = broken.steps[1];
    const third = broken.steps[2];
    expect(second?.faulted).toBe(true);
    // The third step reads the second step's wrong value as its running input, so
    // its own value diverges from the truth even though it wasn't injected.
    expect(third?.value).not.toBe(third?.trueValue);
  });
});

describe("the step reducer", () => {
  it("reveals one reasoning step per tick", () => {
    let state = initialState("cot-solves");
    expect(state.revealed).toBe(0);
    state = step(state, { kind: "tick" });
    expect(state.revealed).toBe(1);
    state = step(state, { kind: "tick" });
    expect(state.revealed).toBe(2);
  });

  it("resets the reveal and logs when an intervention fires", () => {
    let state = initialState("cot-solves");
    state = step(state, { kind: "tick" });
    state = step(state, {
      kind: "intervene",
      action: { type: "toggleInjected", index: 0 },
    });
    expect(state.revealed).toBe(0);
    expect(state.params.injected).toContain(0);
    expect(state.log.events.at(-1)?.label).toContain("break step 1");
  });

  it("loads a preset by id", () => {
    const state = step(initialState("standard-fails"), {
      kind: "intervene",
      action: { type: "loadPreset", id: "too-small" },
    });
    expect(state.params.problemId).toBe("servers");
    expect(state.params.scaleB).toBe(8);
  });
});
