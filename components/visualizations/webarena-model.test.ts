import { describe, expect, it } from "vitest";

import {
  evaluateRun,
  getPreset,
  getResolvedPlan,
  getTask,
  initialState,
  reduce,
  type SimParams,
  type SimState,
} from "./webarena-model";

const base: SimParams = {
  taskId: "itinerary",
  plan: "reference",
  uaHint: false,
  metric: "functional",
  seed: 1,
};

// Run a plan to completion by ticking until the model reports finished.
function runToEnd(params: SimParams): SimState {
  let state = initialState(params);
  for (let i = 0; i < 20 && !state.finished; i++) {
    state = reduce(state, { kind: "tick" });
  }
  return state;
}

function scores(params: SimParams) {
  const state = runToEnd(params);
  const task = getTask(params.taskId);
  return evaluateRun(task, state.actions, state.trajectory);
}

describe("the reference path", () => {
  it("passes both functional and surface-form scoring", () => {
    const result = scores(base);
    expect(result.functional).toBe(1);
    expect(result.surface).toBe(1);
  });

  it("is deterministic: the same params reproduce the same trajectory", () => {
    const a = runToEnd(base);
    const b = runToEnd({ ...base });
    expect(b.actions.map((x) => x.kind)).toEqual(a.actions.map((x) => x.kind));
    expect(b.trajectory.at(-1)).toEqual(a.trajectory.at(-1));
  });
});

describe("the alternative valid path (the paper's central point)", () => {
  it("passes the functional reward but fails surface-form matching", () => {
    const result = scores({ ...base, plan: "alternative" });
    expect(result.functional).toBe(1);
    expect(result.surface).toBe(0);
  });

  it("reaches the same end state as the reference path", () => {
    const ref = runToEnd(base).trajectory.at(-1);
    const alt = runToEnd({ ...base, plan: "alternative" }).trajectory.at(-1);
    expect(alt?.repoFiles).toEqual(ref?.repoFiles);
  });
});

describe("an off-target run", () => {
  it("finishes without error yet fails the functional reward", () => {
    const state = runToEnd({ ...base, plan: "offTarget" });
    expect(state.finished).toBe(true);
    const result = evaluateRun(
      getTask("itinerary"),
      state.actions,
      state.trajectory,
    );
    expect(result.functional).toBe(0);
  });

  it("fails because a required state check does not hold", () => {
    const result = scores({ ...base, plan: "offTarget" });
    expect(result.checks.some((check) => !check.pass)).toBe(true);
  });
});

describe("unachievable tasks and the UA hint", () => {
  it("answers N/A correctly when the hint is on", () => {
    const params: SimParams = { ...base, taskId: "unachievable", uaHint: true };
    const { plan } = getResolvedPlan(params);
    expect(plan).toBe("reference");
    const result = scores(params);
    expect(result.functional).toBe(1);
  });

  it("hallucinates an answer and fails when the hint is off", () => {
    const params: SimParams = {
      ...base,
      taskId: "unachievable",
      uaHint: false,
    };
    const { plan } = getResolvedPlan(params);
    expect(plan).toBe("offTarget");
    const result = scores(params);
    expect(result.functional).toBe(0);
  });
});

describe("the UA hint backfiring on a feasible task", () => {
  it("can flip a good plan into an early give-up for some seeds", () => {
    const gaveUpSeeds: number[] = [];
    const finishedSeeds: number[] = [];
    for (let seed = 0; seed < 12; seed++) {
      const { plan } = getResolvedPlan({
        ...base,
        taskId: "order",
        uaHint: true,
        seed,
      });
      if (plan === "giveUp") gaveUpSeeds.push(seed);
      else finishedSeeds.push(seed);
    }
    expect(gaveUpSeeds.length).toBeGreaterThan(0);
    expect(finishedSeeds.length).toBeGreaterThan(0);
  });

  it("never trips the false give-up when the hint is off", () => {
    for (let seed = 0; seed < 12; seed++) {
      const { plan } = getResolvedPlan({
        ...base,
        taskId: "order",
        uaHint: false,
        seed,
      });
      expect(plan).toBe("reference");
    }
  });
});

describe("the step reducer", () => {
  it("advances one action per tick and grows the trajectory", () => {
    let state = initialState(base);
    const startLen = state.trajectory.length;
    state = reduce(state, { kind: "tick" });
    expect(state.cursor).toBe(1);
    expect(state.trajectory.length).toBe(startLen + 1);
  });

  it("switching the metric keeps the run instead of resetting it", () => {
    let state = runToEnd(base);
    const before = state.cursor;
    state = reduce(state, {
      kind: "intervene",
      action: { type: "setMetric", metric: "surface" },
    });
    expect(state.params.metric).toBe("surface");
    expect(state.cursor).toBe(before);
  });

  it("changing the plan resets the run from a fresh world", () => {
    let state = runToEnd(base);
    state = reduce(state, {
      kind: "intervene",
      action: { type: "setPlan", plan: "offTarget" },
    });
    expect(state.cursor).toBe(0);
    expect(state.finished).toBe(false);
    expect(state.trajectory.length).toBe(1);
  });

  it("loads a preset by id", () => {
    const state = reduce(initialState(base), {
      kind: "intervene",
      action: { type: "loadPreset", id: "unachievable" },
    });
    expect(state.params).toEqual(getPreset("unachievable").params);
  });
});
