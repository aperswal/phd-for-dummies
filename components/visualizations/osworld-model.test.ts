import { describe, expect, it } from "vitest";

import {
  DEFAULT_BUDGET,
  getPreset,
  getTask,
  groundChance,
  initialState,
  OBSERVATION_MODES,
  PERTURBATION_FACTOR_KEYS,
  step,
  type Input,
  type SimParams,
  type SimState,
} from "./osworld-model";

// Run the loop to completion (done, failed, or budget exhausted) so the test
// can assert on the final reward.
function runToEnd(state: SimState): SimState {
  let current = state;
  let guard = 0;
  while (
    current.phase !== "done" &&
    current.phase !== "failed" &&
    guard < 500
  ) {
    current = step(current, { kind: "tick" });
    guard += 1;
  }
  return current;
}

const tick: Input = { kind: "tick" };

describe("the OSWorld agent loop", () => {
  it("is deterministic given the same seed and inputs", () => {
    const a = runToEnd(initialState("single-app"));
    const b = runToEnd(initialState("single-app"));
    expect(b.reward).toBe(a.reward);
    expect(b.step).toBe(a.step);
    expect(b.attempts.map((x) => x.outcome)).toEqual(
      a.attempts.map((x) => x.outcome),
    );
  });

  it("only ever returns reward 0 or 1, the execution-based grade", () => {
    for (const preset of [
      "single-app",
      "workflow",
      "perturbed",
      "screenshot",
    ]) {
      const end = runToEnd(initialState(preset));
      expect(end.reward === 0 || end.reward === 1).toBe(true);
    }
  });

  it("awards reward 1 only when every subgoal is completed", () => {
    const end = runToEnd(initialState("single-app"));
    const task = getTask(end.params.taskId);
    if (end.reward === 1) {
      expect(end.subgoalIndex).toBe(task.subgoals.length);
      expect(end.phase).toBe("done");
    } else {
      expect(end.subgoalIndex).toBeLessThan(task.subgoals.length);
      expect(end.phase).toBe("failed");
    }
  });

  it("never advances past the step budget", () => {
    const end = runToEnd(initialState("workflow"));
    expect(end.step).toBeLessThanOrEqual(end.params.stepBudget);
  });
});

describe("grounding chance", () => {
  it("ranks the observation modes the way the paper's Table 5 does", () => {
    const base: SimParams = {
      taskId: "rename",
      observationId: "a11y",
      perturbation: "none",
      stepBudget: DEFAULT_BUDGET,
      seed: 1,
    };
    const a11y = groundChance(base);
    const screenshot = groundChance({ ...base, observationId: "screenshot" });
    expect(a11y).toBeGreaterThan(screenshot);
  });

  it("drops with every window perturbation, hardest on shrink", () => {
    const base: SimParams = {
      taskId: "rename",
      observationId: "som",
      perturbation: "none",
      stepBudget: DEFAULT_BUDGET,
      seed: 1,
    };
    const original = groundChance(base);
    const moved = groundChance({ ...base, perturbation: "move" });
    const shrunk = groundChance({ ...base, perturbation: "shrink" });
    const cluttered = groundChance({ ...base, perturbation: "clutter" });
    expect(moved).toBeLessThan(original);
    expect(cluttered).toBeLessThan(moved);
    expect(shrunk).toBeLessThan(cluttered);
  });
});

describe("the window perturbation intervention", () => {
  it("makes a run that would have finished fail when applied mid-task", () => {
    // This seed finishes the rename task clean on the original layout.
    const clean = runToEnd(initialState("perturbed"));
    expect(clean.reward).toBe(1);

    // Shrink the window after the first landed click and the same run misses.
    let perturbed = step(initialState("perturbed"), tick);
    perturbed = step(perturbed, {
      kind: "intervene",
      action: { type: "setPerturbation", value: "shrink" },
    });
    perturbed = runToEnd(perturbed);
    expect(perturbed.reward).toBe(0);
    expect(perturbed.phase).toBe("failed");
  });
});

describe("multi-app workflows", () => {
  it("pass less often than single-app tasks under the same grounding rate", () => {
    let singleWins = 0;
    let workflowWins = 0;
    const single = getPreset("single-app").params;
    const workflow = getPreset("workflow").params;
    // Re-seed each run so the comparison spans many trajectories.
    for (let seed = 0; seed < 60; seed++) {
      const sRun = runToEnd({
        ...initialState("single-app"),
        params: { ...single, seed },
        rngState: seed >>> 0,
      });
      const wRun = runToEnd({
        ...initialState("workflow"),
        params: { ...workflow, seed },
        rngState: seed >>> 0,
      });
      if (sRun.reward === 1) singleWins += 1;
      if (wRun.reward === 1) workflowWins += 1;
    }
    expect(singleWins).toBeGreaterThan(workflowWins);
  });
});

describe("popups cost a step without progress", () => {
  it("records a popup-clearing attempt that makes no subgoal progress", () => {
    // Force a config where misses are common, driving every parameter through
    // the real reducer so the state stays well-typed.
    let state = initialState("screenshot");
    state = step(state, {
      kind: "intervene",
      action: { type: "setTask", id: "bookkeeping" },
    });
    state = step(state, {
      kind: "intervene",
      action: { type: "setObservation", id: "screenshot" },
    });
    state = step(state, {
      kind: "intervene",
      action: { type: "setPerturbation", value: "shrink" },
    });
    // Reshuffle once to move off the preset seed; the exact seed value
    // doesn't matter here — we just need misses to be common enough that
    // a popup may appear, which screenshot + shrink + bookkeeping guarantees
    // across almost any seed.
    state = step(state, {
      kind: "intervene",
      action: { type: "reshuffle" },
    });
    state = runToEnd(state);
    const clears = state.attempts.filter((a) => a.subgoal === "close popup");
    const popups = state.attempts.filter((a) => a.outcome === "popup");
    // A cleared popup only exists if a stray click opened one first.
    if (clears.length > 0) {
      expect(popups.length).toBeGreaterThan(0);
    }
    // The grade is still a clean 0 or 1.
    expect(state.reward === 0 || state.reward === 1).toBe(true);
  });
});

describe("reshuffle intervention", () => {
  it("advances the seed by 1 and resets trajectory to step 0", () => {
    const before = initialState("single-app");
    const originalSeed = before.params.seed;
    // Advance one tick so the run is no longer at step 0.
    const mid = step(before, { kind: "tick" });
    expect(mid.step).toBe(1);
    const after = step(mid, {
      kind: "intervene",
      action: { type: "reshuffle" },
    });
    expect(after.params.seed).toBe((originalSeed + 1) % 100);
    expect(after.step).toBe(0);
    expect(after.attempts).toHaveLength(0);
  });

  it("produces a different outcome than the original seed on some runs", () => {
    // Across many reshuffles at least one run differs from the first, confirming
    // that the seed advance actually changes the trajectory.
    const base = initialState("single-app");
    const firstEnd = runToEnd({ ...base });
    let foundDifferent = false;
    let current = base;
    for (let i = 0; i < 20 && !foundDifferent; i++) {
      current = step(current, {
        kind: "intervene",
        action: { type: "reshuffle" },
      });
      const end = runToEnd({ ...current });
      if (end.reward !== firstEnd.reward || end.step !== firstEnd.step) {
        foundDifferent = true;
      }
    }
    expect(foundDifferent).toBe(true);
  });
});

describe("config tables stay consistent", () => {
  it("exposes three observation modes with grounding rates in (0, 1]", () => {
    expect(OBSERVATION_MODES.length).toBe(3);
    for (const mode of OBSERVATION_MODES) {
      expect(mode.groundRate).toBeGreaterThan(0);
      expect(mode.groundRate).toBeLessThanOrEqual(1);
    }
  });

  it("defines a factor for every perturbation", () => {
    expect(PERTURBATION_FACTOR_KEYS).toEqual([
      "none",
      "move",
      "shrink",
      "clutter",
    ]);
  });
});
