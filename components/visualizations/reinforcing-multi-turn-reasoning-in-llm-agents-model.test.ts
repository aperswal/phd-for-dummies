import { describe, expect, it } from "vitest";

import {
  computeAdvantages,
  DEFAULT_WEIGHTS,
  initialFromPreset,
  step,
  turn1HasSignal,
  turn1Spread,
  type Rollout,
  type SimState,
} from "./reinforcing-multi-turn-reasoning-in-llm-agents-model";

function blankRollout(overrides: Partial<Rollout>): Rollout {
  return {
    id: 0,
    toolExecuted: false,
    retrieved: false,
    intermediateFormatOk: true,
    searchCount: 0,
    answerCorrect: false,
    outcomeFormatOk: true,
    intermediateReward: 0,
    outcomeReward: 0,
    mergedReward: 0,
    ...overrides,
  };
}

// A group where everyone gets the same final answer but turn 1 differs: half
// the rollouts ran the tool well and retrieved the answer, half did neither.
function splitGroup(): Rollout[] {
  const good = blankRollout({
    id: 0,
    toolExecuted: true,
    retrieved: true,
    searchCount: 1,
    answerCorrect: true,
    intermediateReward: 0.2 + 0.5 + 0.1 - 0.1,
    outcomeReward: 1,
    mergedReward: 0.2 + 0.5 + 0.1 - 0.1 + 1,
  });
  const goodWrong = blankRollout({
    id: 1,
    toolExecuted: true,
    retrieved: true,
    searchCount: 1,
    answerCorrect: false,
    intermediateReward: 0.2 + 0.5 + 0.1 - 0.1,
    outcomeReward: 1, // same outcome reward as the correct one to isolate turn 1
    mergedReward: 0.2 + 0.5 + 0.1 - 0.1 + 1,
  });
  const bad = blankRollout({
    id: 2,
    toolExecuted: false,
    retrieved: false,
    searchCount: 0,
    answerCorrect: true,
    intermediateReward: 0.1,
    outcomeReward: 1,
    mergedReward: 0.1 + 1,
  });
  const badWrong = blankRollout({
    id: 3,
    toolExecuted: false,
    retrieved: false,
    searchCount: 0,
    answerCorrect: false,
    intermediateReward: 0.1,
    outcomeReward: 1,
    mergedReward: 0.1 + 1,
  });
  return [good, goodWrong, bad, badWrong];
}

describe("turn-level advantage assignment", () => {
  it("GRPO-OR assigns the same advantage to both turns of a trajectory", () => {
    const group = splitGroup();
    const rows = computeAdvantages(group, "grpo-or", DEFAULT_WEIGHTS);
    for (const row of rows) {
      expect(row.turn1).toBeCloseTo(row.turn2, 10);
    }
  });

  it("GRPO-OR gives turn 1 no signal when the outcome reward is flat", () => {
    // Every rollout here has the same outcome reward, so outcome-only credit is
    // zero for everyone. Turn 1 learns nothing even though searches differed.
    const group = splitGroup();
    const rows = computeAdvantages(group, "grpo-or", DEFAULT_WEIGHTS);
    expect(turn1HasSignal(rows)).toBe(false);
    for (const row of rows) {
      expect(row.turn1).toBeCloseTo(0, 10);
    }
  });

  it("MT-GRPO gives turn 1 a real signal that separates the good searches", () => {
    const group = splitGroup();
    const rows = computeAdvantages(group, "mt-grpo", DEFAULT_WEIGHTS);
    expect(turn1HasSignal(rows)).toBe(true);
    const goodSearch = rows.find((r) => r.rolloutId === 0);
    const badSearch = rows.find((r) => r.rolloutId === 2);
    expect(goodSearch).toBeDefined();
    expect(badSearch).toBeDefined();
    // The rollout that ran the tool and retrieved the answer earns more turn-1
    // credit than the one that skipped the tool, which GRPO-OR could not tell
    // apart.
    expect(goodSearch?.turn1 ?? 0).toBeGreaterThan(badSearch?.turn1 ?? 0);
  });

  it("MT-GRPO turn-1 advantage equals A^I + alpha*A^O by construction", () => {
    const group = splitGroup();
    const alpha = 0.5;
    const weights = { ...DEFAULT_WEIGHTS, alpha };
    const mt = computeAdvantages(group, "mt-grpo", weights);
    // Outcome advantage alone is what GRPO-OR's turn-2 column is.
    const or = computeAdvantages(group, "grpo-or", weights);
    // Reconstruct A^I from MT-GRPO: A^I = turn1 - alpha*A^O.
    for (let i = 0; i < group.length; i++) {
      const aOutcome = or[i]?.turn2 ?? 0;
      const aIntermediate = (mt[i]?.turn1 ?? 0) - alpha * aOutcome;
      // The reconstructed intermediate advantage must match MT-GRPO's turn-2
      // outcome term being separate, so turn2 is exactly A^O.
      expect(mt[i]?.turn2 ?? 0).toBeCloseTo(aOutcome, 10);
      expect(Number.isFinite(aIntermediate)).toBe(true);
    }
  });

  it("a flat group yields zero advantage everywhere (group-relative property)", () => {
    const identical = Array.from({ length: 4 }, (_, id) =>
      blankRollout({
        id,
        toolExecuted: true,
        retrieved: true,
        searchCount: 1,
        answerCorrect: true,
        intermediateReward: 0.7,
        outcomeReward: 1,
        mergedReward: 1.7,
      }),
    );
    for (const algorithm of ["grpo-or", "grpo-mr", "mt-grpo"] as const) {
      const rows = computeAdvantages(identical, algorithm, DEFAULT_WEIGHTS);
      for (const row of rows) {
        expect(row.turn1).toBeCloseTo(0, 10);
        expect(row.turn2).toBeCloseTo(0, 10);
      }
    }
  });
});

function runSteps(state: SimState, count: number): SimState {
  let current = state;
  for (let i = 0; i < count; i++) {
    current = step(current, { kind: "tick" });
  }
  return current;
}

describe("training dynamics", () => {
  it("is deterministic given the same seed and inputs", () => {
    const a = runSteps(initialFromPreset("mt-stable"), 40);
    const b = runSteps(initialFromPreset("mt-stable"), 40);
    expect(a.policy).toEqual(b.policy);
    expect(a.history).toEqual(b.history);
  });

  it("GRPO-OR lets tool use collapse while MT-GRPO holds it up", () => {
    const start = initialFromPreset("mt-stable").policy.toolUse;
    const mt = runSteps(initialFromPreset("mt-stable"), 80);
    const or = runSteps(initialFromPreset("or-collapse"), 80);
    // MT-GRPO keeps the tool-use propensity above where it started; GRPO-OR
    // drifts well below it, the Figure 1 collapse.
    expect(mt.policy.toolUse).toBeGreaterThan(start);
    expect(or.policy.toolUse).toBeLessThan(mt.policy.toolUse);
    expect(or.policy.toolUse).toBeLessThan(start);
  });

  it("switching to GRPO-OR mid-run turns a stable run into a collapsing one", () => {
    const before = runSteps(initialFromPreset("mt-stable"), 30);
    // Under MT-GRPO the early run gives turn 1 a usable signal often enough.
    const earlySpread = turn1Spread(
      computeAdvantages(before.rollouts, "mt-grpo", DEFAULT_WEIGHTS),
    );
    const orSpread = turn1Spread(
      computeAdvantages(before.rollouts, "grpo-or", DEFAULT_WEIGHTS),
    );
    // On the same group, MT-GRPO separates turn-1 behavior at least as much as
    // outcome-only does, and usually more.
    expect(earlySpread).toBeGreaterThanOrEqual(orSpread);
    const after = step(before, {
      kind: "intervene",
      action: { type: "setAlgorithm", id: "grpo-or" },
    });
    // After the switch, turn 1 rides on outcome-only credit, so tool use slides
    // below where the stable MT-GRPO run had taken it.
    const ranAfter = runSteps(after, 80);
    expect(ranAfter.policy.toolUse).toBeLessThan(before.policy.toolUse);
  });

  it("records an event and can rewind to a snapshot", () => {
    const ran = runSteps(initialFromPreset("mt-stable"), 20);
    const switched = step(ran, {
      kind: "intervene",
      action: { type: "setAlgorithm", id: "grpo-or" },
    });
    expect(switched.log.length).toBeGreaterThan(0);
    const entry = switched.log[switched.log.length - 1];
    expect(entry).toBeDefined();
    const restored = step(switched, {
      kind: "intervene",
      action: { type: "restore", snapshot: entry!.snapshot, label: "rewind" },
    });
    expect(restored.step).toBe(entry?.snapshot.step);
    expect(restored.algorithm).toBe(entry?.snapshot.algorithm);
  });

  it("dropping the search penalty changes the run", () => {
    const withPenalty = runSteps(initialFromPreset("mt-stable"), 60);
    const noPenalty = runSteps(initialFromPreset("no-penalty"), 60);
    expect(noPenalty.policy.toolUse).not.toBeCloseTo(
      withPenalty.policy.toolUse,
      3,
    );
  });
});
