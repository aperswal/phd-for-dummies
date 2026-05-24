import { describe, expect, it } from "vitest";

import {
  CONTEXT_BUDGET,
  getPreset,
  getTask,
  initialState,
  predictSuccess,
  runningStats,
  step,
  tokensUsed,
  type PromptConfig,
  type SimState,
} from "./language-models-are-few-shot-learners-model";

const base: PromptConfig = {
  taskId: "arithmetic",
  modelId: "175b",
  exampleCount: 0,
  useDescription: true,
  corruptedExamples: 0,
  seed: 7,
};

function runTicks(state: SimState, count: number): SimState {
  let next = state;
  for (let i = 0; i < count; i++) {
    next = step(next, { kind: "tick" });
  }
  return next;
}

describe("predicted success probability", () => {
  it("always stays a valid probability", () => {
    for (const task of ["arithmetic", "unscramble", "reverse"]) {
      for (const model of ["125m", "1.3b", "13b", "175b"]) {
        for (let examples = 0; examples <= 6; examples++) {
          const p = predictSuccess({
            ...base,
            taskId: task,
            modelId: model,
            exampleCount: examples,
          });
          expect(p).toBeGreaterThanOrEqual(0);
          expect(p).toBeLessThanOrEqual(1);
        }
      }
    }
  });

  it("is deterministic given the same prompt", () => {
    expect(predictSuccess(base)).toBe(predictSuccess({ ...base }));
  });
});

describe("the in-context learning curve", () => {
  it("rises as examples are added to a 175B prompt", () => {
    const zero = predictSuccess({ ...base, exampleCount: 0 });
    const one = predictSuccess({ ...base, exampleCount: 1 });
    const few = predictSuccess({ ...base, exampleCount: 5 });
    expect(one).toBeGreaterThan(zero);
    expect(few).toBeGreaterThan(one);
    expect(few).toBeGreaterThan(0.95);
  });

  it("barely lifts for a small model no matter how many examples it sees", () => {
    const small0 = predictSuccess({
      ...base,
      modelId: "125m",
      exampleCount: 0,
    });
    const small5 = predictSuccess({
      ...base,
      modelId: "125m",
      exampleCount: 5,
    });
    const big5 = predictSuccess({ ...base, modelId: "175b", exampleCount: 5 });
    expect(small5 - small0).toBeLessThan(0.15);
    expect(big5 - small5).toBeGreaterThan(0.5);
  });

  it("gives a steeper few-shot slope to bigger models", () => {
    const slope = (model: string): number =>
      predictSuccess({ ...base, modelId: model, exampleCount: 5 }) -
      predictSuccess({ ...base, modelId: model, exampleCount: 0 });
    expect(slope("175b")).toBeGreaterThan(slope("13b"));
    expect(slope("13b")).toBeGreaterThan(slope("125m"));
  });
});

describe("the reversed-word wall", () => {
  it("stays near zero even at full scale with a full prompt", () => {
    const wall = predictSuccess({
      ...base,
      taskId: "reverse",
      modelId: "175b",
      exampleCount: 6,
    });
    expect(wall).toBeLessThan(0.02);
  });
});

describe("mislabeled examples", () => {
  it("make the model do worse than the same prompt with clean examples", () => {
    const clean = predictSuccess({
      ...base,
      exampleCount: 5,
      corruptedExamples: 0,
    });
    const dirty = predictSuccess({
      ...base,
      exampleCount: 5,
      corruptedExamples: 3,
    });
    expect(dirty).toBeLessThan(clean);
  });
});

describe("the context budget", () => {
  it("reports going over budget once the prompt holds too many examples", () => {
    const task = getTask("unscramble");
    const small = tokensUsed(task, 1, true);
    const big = tokensUsed(task, 6, true);
    expect(big).toBeGreaterThan(small);
    expect(big).toBeGreaterThan(CONTEXT_BUDGET);
  });
});

describe("the trial process", () => {
  it("is reproducible from the same seed and prompt", () => {
    const a = runTicks(initialState("few-shot-175b"), 20);
    const b = runTicks(initialState("few-shot-175b"), 20);
    expect(b.trials).toEqual(a.trials);
    expect(b.correctCount).toBe(a.correctCount);
  });

  it("converges its running accuracy toward the predicted probability", () => {
    const state = runTicks({ ...initialState("few-shot-175b") }, 400);
    const stats = runningStats(state);
    expect(Math.abs(stats.accuracy - stats.predicted)).toBeLessThan(0.08);
  });

  it("scores far worse on a small model than on GPT-3 over many trials", () => {
    const big = runningStats(runTicks(initialState("few-shot-175b"), 300));
    const small = runningStats(runTicks(initialState("scale-wall"), 300));
    expect(big.accuracy).toBeGreaterThan(small.accuracy + 0.4);
  });
});

describe("the step reducer", () => {
  it("resets the tally when the prompt changes", () => {
    let state = runTicks(initialState("few-shot-175b"), 30);
    expect(state.attempted).toBe(30);
    state = step(state, { kind: "intervene", action: { type: "addExample" } });
    expect(state.attempted).toBe(0);
    expect(state.correctCount).toBe(0);
  });

  it("adding examples lifts the prediction and then a small model flattens it", () => {
    let state = step(initialState("zero-shot"), {
      kind: "intervene",
      action: { type: "addExample" },
    });
    const liftedBig = runningStats(state).predicted;
    state = step(state, {
      kind: "intervene",
      action: { type: "setModel", id: "125m" },
    });
    expect(runningStats(state).predicted).toBeLessThan(liftedBig);
  });

  it("corrupting an example lowers the prediction", () => {
    let state = initialState("few-shot-175b");
    const before = runningStats(state).predicted;
    state = step(state, {
      kind: "intervene",
      action: { type: "corruptExample" },
    });
    expect(runningStats(state).predicted).toBeLessThan(before);
  });

  it("records each change so the reader can rewind to it", () => {
    const state = step(initialState("few-shot-175b"), {
      kind: "intervene",
      action: { type: "setModel", id: "13b" },
    });
    const last = state.log.events.at(-1);
    expect(last?.label).toContain("model");
    expect(last?.snapshot.modelId).toBe("13b");
  });

  it("loads a preset by id", () => {
    const state = step(initialState("few-shot-175b"), {
      kind: "intervene",
      action: { type: "loadPreset", id: "the-wall" },
    });
    expect(state.config).toEqual(getPreset("the-wall").config);
  });
});
