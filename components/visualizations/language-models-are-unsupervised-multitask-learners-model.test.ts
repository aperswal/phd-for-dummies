import { describe, expect, it } from "vitest";

import {
  GPT2_SIZE_INDEX,
  MODEL_SIZES,
  getPreset,
  getTask,
  goldMatch,
  goldProbability,
  initialState,
  scoreCandidates,
  step,
  type GenParams,
  type SimState,
} from "./language-models-are-unsupervised-multitask-learners-model";

const qaBig: GenParams = {
  taskId: "qa",
  sizeIndex: GPT2_SIZE_INDEX,
  temperature: 0.7,
  topK: 3,
  seed: 7,
  hintPresent: true,
};

function runToEnd(start: SimState): SimState {
  let state = start;
  for (let i = 0; i < 20 && !state.finished; i++) {
    state = step(state, { kind: "tick" });
  }
  return state;
}

describe("the next-token distribution", () => {
  it("is a valid probability distribution over the kept tokens", () => {
    const candidates = scoreCandidates(qaBig, 0);
    const live = candidates.filter((c) => !c.masked);
    const total = live.reduce((sum, c) => sum + c.prob, 0);
    expect(total).toBeCloseTo(1, 6);
    expect(live.every((c) => c.prob >= 0)).toBe(true);
  });

  it("keeps exactly top-k tokens unmasked", () => {
    const candidates = scoreCandidates({ ...qaBig, topK: 2 }, 0);
    expect(candidates.filter((c) => !c.masked).length).toBe(2);
  });

  it("is deterministic given the same parameters", () => {
    const a = scoreCandidates(qaBig, 0);
    const b = scoreCandidates({ ...qaBig }, 0);
    expect(b.map((c) => c.prob)).toEqual(a.map((c) => c.prob));
  });
});

describe("capacity drives the log-linear lift", () => {
  it("assigns the correct token more probability as the model grows", () => {
    const probs = MODEL_SIZES.map((_, index) =>
      goldProbability({ ...qaBig, sizeIndex: index }, 0),
    );
    for (let i = 1; i < probs.length; i++) {
      expect(probs[i] ?? 0).toBeGreaterThan(probs[i - 1] ?? 0);
    }
  });

  it("lets the big model emit the gold answer and starves the tiny one of it", () => {
    const big = runToEnd(initialState("qa-happy"));
    expect(big.generated.slice(0, 2)).toEqual(["Charles", "Darwin"]);
    expect(goldMatch(big)).toBe(1);
    const bigProb = goldProbability(qaBig, 0);
    const smallProb = goldProbability({ ...qaBig, sizeIndex: 0 }, 0);
    expect(bigProb).toBeGreaterThan(smallProb);
    expect(smallProb).toBeLessThan(0.5);
  });
});

describe("the prompt selects the task", () => {
  it("runs a different conditional for each task id", () => {
    const tasks = ["qa", "summarize", "translate", "reading"];
    for (const id of tasks) {
      const state = runToEnd(
        step(initialState("qa-happy"), {
          kind: "intervene",
          action: { type: "setTask", id },
        }),
      );
      expect(state.generated[0]).toBe(getTask(id).gold[0]);
    }
  });
});

describe("the hint ablation", () => {
  it("collapses the gold probability when the hint is removed", () => {
    const withHint = goldProbability(
      { ...qaBig, taskId: "summarize", hintPresent: true },
      0,
    );
    const withoutHint = goldProbability(
      { ...qaBig, taskId: "summarize", hintPresent: false },
      0,
    );
    expect(withHint).toBeGreaterThan(withoutHint);
  });

  it("drifts the output off the gold summary once the hint is toggled off", () => {
    const start = initialState("summarize-hint");
    const ablated = step(start, {
      kind: "intervene",
      action: { type: "toggleHint" },
    });
    expect(ablated.params.hintPresent).toBe(false);
    const withHint = goldMatch(runToEnd(start));
    const withoutHint = goldMatch(runToEnd(ablated));
    expect(withHint).toBeGreaterThan(withoutHint);
  });
});

describe("the step reducer", () => {
  it("appends one token per tick and stops at the cap", () => {
    let state = initialState("qa-happy");
    const first = step(state, { kind: "tick" });
    expect(first.generated.length).toBe(1);
    state = runToEnd(state);
    expect(state.finished).toBe(true);
    expect(state.generated.length).toBeGreaterThan(0);
  });

  it("reproduces the same run from the same seed", () => {
    const a = runToEnd(initialState("qa-happy"));
    const b = runToEnd(initialState("qa-happy"));
    expect(b.generated).toEqual(a.generated);
  });

  it("reshuffle advances the seed and changes the run on a flat distribution", () => {
    // A tiny model at high temperature has a near-uniform next-token
    // distribution, so the seed genuinely steers which token gets sampled.
    // Reshuffle replaces the setSeed slider: it advances the seed by a large
    // prime so the reader gets a different sample without a meaningless dial.
    const flat: GenParams = {
      taskId: "qa",
      sizeIndex: 0,
      temperature: 1.8,
      topK: 5,
      seed: 2,
      hintPresent: true,
    };
    const baseRun = runToEnd({
      ...initialState("small-model"),
      params: flat,
      rngState: flat.seed >>> 0,
    });
    const reshuffled = step(
      { ...initialState("small-model"), params: flat, rngState: flat.seed >>> 0 },
      { kind: "intervene", action: { type: "reshuffle" } },
    );
    expect(reshuffled.params.seed).not.toBe(flat.seed);
    const reshuffledRun = runToEnd(reshuffled);
    expect(reshuffledRun.generated.length).toBeGreaterThan(0);
    expect(reshuffledRun.generated).not.toEqual(baseRun.generated);
  });

  it("loads a preset by id", () => {
    const state = step(initialState("qa-happy"), {
      kind: "intervene",
      action: { type: "loadPreset", id: "translate" },
    });
    expect(state.params.taskId).toBe(getPreset("translate").params.taskId);
    expect(state.generated.length).toBe(0);
  });
});
