import { describe, expect, it } from "vitest";

import {
  AGENTS,
  getTask,
  initialState,
  MAX_TURNS,
  PRESETS,
  step,
  totalSteps,
  type CrmState,
  type Input,
} from "@/components/visualizations/crmarena-model";

function run(state: CrmState, inputs: Input[]): CrmState {
  return inputs.reduce((current, input) => step(current, input), state);
}

function runToEnd(state: CrmState): CrmState {
  let current = state;
  let guard = 0;
  while (current.status === "running" && guard < MAX_TURNS + 4) {
    current = step(current, { kind: "tick" });
    guard += 1;
  }
  return current;
}

describe("crmarena model", () => {
  it("is deterministic: same preset and ticks reproduce the run", () => {
    const a = runToEnd(initialState("reasoner-wins"));
    const b = runToEnd(initialState("reasoner-wins"));
    expect(a.status).toBe(b.status);
    expect(a.turn).toBe(b.turn);
    expect(a.turns.map((t) => t.kind)).toEqual(b.turns.map((t) => t.kind));
  });

  it("a run always ends solved or failed, never stuck", () => {
    for (const preset of PRESETS) {
      const ended = runToEnd(initialState(preset.id));
      expect(["solved", "failed"]).toContain(ended.status);
    }
  });

  it("never exceeds the turn budget", () => {
    for (const preset of PRESETS) {
      const ended = runToEnd(initialState(preset.id));
      expect(ended.turn).toBeLessThanOrEqual(MAX_TURNS + 1);
    }
  });

  it("the strong reasoner preset clears the chain and solves", () => {
    const ended = runToEnd(initialState("reasoner-wins"));
    expect(ended.status).toBe("solved");
    expect(ended.turns.some((t) => t.kind === "submit-correct")).toBe(true);
  });

  it("strong models solve far more often than the weak model across seeds", () => {
    let strong = 0;
    let weak = 0;
    for (let seed = 0; seed < 40; seed++) {
      const strongRun = runToEnd(
        step(initialState("reasoner-wins"), {
          kind: "intervene",
          action: { type: "setSeed", value: seed },
        }),
      );
      const weakRun = runToEnd(
        step(initialState("weak-stalls"), {
          kind: "intervene",
          action: { type: "setSeed", value: seed },
        }),
      );
      if (strongRun.status === "solved") strong += 1;
      if (weakRun.status === "solved") weak += 1;
    }
    expect(strong).toBeGreaterThan(weak + 10);
  });

  it("forcing a guess fails an unanswerable task immediately", () => {
    const start = initialState("false-premise");
    const guessed = step(start, {
      kind: "intervene",
      action: { type: "forceGuess" },
    });
    const after = step(guessed, { kind: "tick" });
    expect(after.status).toBe("failed");
    expect(after.turns.at(-1)?.kind).toBe("submit-wrong");
  });

  it("dropping a tool result costs a turn and can knock the agent back", () => {
    const start = initialState("reasoner-wins");
    const before = step(start, { kind: "tick" });
    const dropped = step(before, {
      kind: "intervene",
      action: { type: "dropResult" },
    });
    expect(dropped.turn).toBe(before.turn + 1);
    expect(dropped.turns.some((t) => t.kind === "dropped")).toBe(true);
  });

  it("removing the needed function on a specific toolset lowers the weak model's solve rate", () => {
    // Compare the weak model on a specific toolset with and without pulling the
    // function the second step depends on, across seeds.
    let withFn = 0;
    let withoutFn = 0;
    for (let seed = 0; seed < 50; seed++) {
      const base = step(initialState("tools-backfire"), {
        kind: "intervene",
        action: { type: "setSeed", value: seed },
      });
      withFn += runToEnd(base).status === "solved" ? 1 : 0;

      // Advance one step so the function-bearing step is current, then remove it.
      const stepped = step(base, { kind: "tick" });
      const removed =
        stepped.status === "running"
          ? step(stepped, {
              kind: "intervene",
              action: { type: "removeFunction" },
            })
          : stepped;
      withoutFn += runToEnd(removed).status === "solved" ? 1 : 0;
    }
    expect(withoutFn).toBeLessThanOrEqual(withFn);
  });

  it("every task step count matches its declared chain", () => {
    for (const preset of PRESETS) {
      const params = preset.params;
      expect(totalSteps(params)).toBe(getTask(params.taskId).steps.length);
    }
  });

  it("an intervention resets the run to a clean start", () => {
    const advanced = run(initialState("reasoner-wins"), [
      { kind: "tick" },
      { kind: "tick" },
    ]);
    const switched = step(advanced, {
      kind: "intervene",
      action: { type: "setAgent", id: "gpt4o" },
    });
    expect(switched.turn).toBe(0);
    expect(switched.step).toBe(0);
    expect(switched.status).toBe("running");
    expect(switched.params.agentId).toBe("gpt4o");
  });

  it("exposes the four headline agent profiles", () => {
    expect(AGENTS.map((a) => a.id)).toEqual([
      "o1",
      "gpt4o",
      "claude35",
      "llama8b",
    ]);
  });
});
