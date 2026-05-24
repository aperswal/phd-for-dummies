import { describe, expect, it } from "vitest";

import {
  initialState,
  isTerminal,
  safetyHolds,
  step,
  type Input,
  type Intervention,
  type SimState,
} from "@/components/visualizations/reflexion-language-agents-with-verbal-reinforcement-learning-model";

function tick(state: SimState): SimState {
  return step(state, { kind: "tick" });
}

function act(state: SimState, action: Intervention): SimState {
  return step(state, { kind: "intervene", action });
}

// Run plain ticks until the loop reaches a terminal phase or a step cap, so a
// non-converging run can't hang the test.
function runToEnd(start: SimState, cap = 2000): SimState {
  let state = start;
  let steps = 0;
  while (!isTerminal(state) && steps < cap) {
    state = tick(state);
    steps += 1;
  }
  return state;
}

describe("reflexion model", () => {
  it("is deterministic for the same seed and inputs", () => {
    const a = runToEnd(initialState("reflexion"));
    const b = runToEnd(initialState("reflexion"));
    expect(a.phase).toBe(b.phase);
    expect(a.trial).toBe(b.trial);
    expect(a.outcomes.length).toBe(b.outcomes.length);
  });

  it("happy path converges to a truthful pass", () => {
    const end = runToEnd(initialState("reflexion"));
    expect(end.phase).toBe("passed");
    expect(end.truthfulPass).toBe(true);
    expect(safetyHolds(end)).toBe(true);
  });

  it("the first trial fails before any lesson exists", () => {
    let state = initialState("reflexion");
    // Walk the first trajectory and its evaluation.
    while (state.trial === 0 && state.reward === null) {
      state = tick(state);
    }
    expect(state.reward).toBe(0);
    expect(state.memory.length).toBe(0);
  });

  it("memory grows by one correct lesson per failed trial on the happy path", () => {
    let state = initialState("reflexion");
    // Advance through the first failure and its reflection.
    while (state.memory.length === 0 && !isTerminal(state)) {
      state = tick(state);
    }
    expect(state.memory.length).toBe(1);
    expect(state.memory[0]!.correct).toBe(true);
  });

  it("turning reflection off flatlines the agent: it never passes", () => {
    const end = runToEnd(initialState("react-only"));
    expect(end.phase).toBe("exhausted");
    expect(end.outcomes.every((outcome) => !outcome.passed)).toBe(true);
    expect(end.memory.length).toBe(0);
  });

  it("a low-accuracy self-reflection stays stuck in a local minimum", () => {
    const end = runToEnd(initialState("local-minimum"));
    expect(end.phase).toBe("exhausted");
  });

  it("a flaky evaluator breaks the safety invariant by passing a wrong answer", () => {
    const end = runToEnd(initialState("flaky-evaluator"));
    expect(end.phase).toBe("passed");
    // It reported a pass, but on a trajectory that was not truthfully correct.
    expect(end.truthfulPass).toBe(false);
    expect(safetyHolds(end)).toBe(false);
  });

  it("corrupting a correct lesson makes the next trial repeat the mistake", () => {
    // Get to the point where memory holds exactly one correct lesson.
    let state = initialState("reflexion");
    while (state.memory.length === 0 && !isTerminal(state)) {
      state = tick(state);
    }
    expect(state.memory[0]!.correct).toBe(true);
    const correctedPoint = state.memory[0]!.pointId;

    const corrupted = act(state, { type: "corruptLastReflection" });
    expect(corrupted.memory[0]!.correct).toBe(false);

    // Walk the next trajectory and check the corrected point now fails again.
    let next = corrupted;
    const targetTrial = next.trial;
    while (next.trial === targetTrial && next.cursor <= correctedPoint) {
      next = tick(next);
    }
    const chosen = next.trajectory[correctedPoint];
    const correctAction = next.task[correctedPoint]!.correctAction;
    expect(chosen).not.toBe(correctAction);
  });

  it("shrinking the memory window evicts the oldest lessons", () => {
    let state = initialState("reflexion");
    // Build up at least two lessons.
    while (state.memory.length < 2 && !isTerminal(state)) {
      state = tick(state);
    }
    expect(state.memory.length).toBeGreaterThanOrEqual(2);
    const shrunk = act(state, { type: "setMemoryCapacity", value: 1 });
    expect(shrunk.memory.length).toBe(1);
    expect(shrunk.params.memoryCapacity).toBe(1);
  });

  it("ticking a terminal state is a no-op", () => {
    const end = runToEnd(initialState("reflexion"));
    const again = tick(end);
    expect(again.phase).toBe(end.phase);
    expect(again.trial).toBe(end.trial);
    expect(again.outcomes.length).toBe(end.outcomes.length);
  });

  it("restore rewinds to a recorded snapshot", () => {
    let state = initialState("reflexion");
    for (let i = 0; i < 6; i++) state = tick(state);
    const event = state.log[2];
    expect(event).toBeDefined();
    const restored = act(state, {
      type: "restore",
      snapshot: event!.snapshot,
      label: "rewind",
    });
    expect(restored.trial).toBe(event!.snapshot.trial);
    expect(restored.cursor).toBe(event!.snapshot.cursor);
  });
});

// Type-only guard: the reducer never widens Input.
const _inputs: Input[] = [
  { kind: "tick" },
  { kind: "intervene", action: { type: "addTrap" } },
];
void _inputs;
