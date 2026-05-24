import { describe, expect, it } from "vitest";

import {
  getPreset,
  getTask,
  grounding,
  initialState,
  step,
  type Intervention,
  type SimState,
} from "./react-synergizing-reasoning-and-acting-model";

// Run the agent to a terminal status (solved / wrong / stuck) or until the step
// budget runs out, applying an optional intervention at a chosen tick.
function run(
  presetId: string,
  options: { at?: number; intervene?: Intervention; maxTicks?: number } = {},
): SimState {
  let state = initialState(presetId);
  const maxTicks = options.maxTicks ?? 200;
  for (let t = 0; t < maxTicks; t++) {
    if (options.intervene && t === (options.at ?? 0)) {
      state = step(state, { kind: "intervene", action: options.intervene });
    }
    if (state.status !== "running") break;
    state = step(state, { kind: "tick" });
  }
  return state;
}

describe("the ReAct loop", () => {
  it("grounds its way to the paper's answer on the Apple Remote question", () => {
    const state = run("react-grounded");
    expect(state.status).toBe("solved");
    expect(state.finalAnswer).toBe("keyboard function keys");
    expect(state.groundedFacts).toBeGreaterThan(0);
  });

  it("interleaves thought, action, and observation in order", () => {
    let state = initialState("react-grounded");
    state = step(state, { kind: "tick" }); // thought 1
    state = step(state, { kind: "tick" }); // action 1
    state = step(state, { kind: "tick" }); // observation 1
    const kinds = state.steps.slice(1).map((s) => s.kind);
    expect(kinds).toEqual(["thought", "action", "observation"]);
  });

  it("is deterministic given the same seed and inputs", () => {
    const a = run("react-grounded");
    const b = run("react-grounded");
    expect(b.finalAnswer).toBe(a.finalAnswer);
    expect(b.steps.length).toBe(a.steps.length);
    expect(b.tick).toBe(a.tick);
  });
});

describe("the CoT baseline hallucinates", () => {
  it("answers from memory and lands on the wrong device", () => {
    const state = run("cot-hallucinates");
    expect(state.status).toBe("wrong");
    expect(state.finalAnswer).toBe(getTask("apple").hallucinatedAnswer);
  });

  it("grounds nothing, so the grounding ratio is zero", () => {
    const state = run("cot-hallucinates");
    const g = grounding(state);
    expect(g.grounded).toBe(0);
    expect(g.memory).toBeGreaterThan(0);
  });

  it("never emits an action other than the final answer", () => {
    const state = run("cot-hallucinates");
    const searches = state.steps.filter((s) => s.actionType === "search");
    expect(searches.length).toBe(0);
  });
});

describe("the Act baseline", () => {
  it("emits no thoughts", () => {
    const state = run("act-loops", { maxTicks: 60 });
    const thoughts = state.steps.filter((s) => s.kind === "thought");
    expect(thoughts.length).toBe(0);
  });
});

describe("interventions change the outcome under the rules", () => {
  it("a dead search derails ReAct so it never reaches the answer", () => {
    const state = run("dead-search");
    expect(state.status).toBe("stuck");
    expect(state.finalAnswer).toBeNull();
  });

  it("the same question without the dead search solves cleanly", () => {
    // Same task and mode as the dead-search preset, but no injected fault.
    let state = initialState("react-grounded");
    state = step(state, {
      kind: "intervene",
      action: { type: "setTask", id: "colorado" },
    });
    for (let t = 0; t < 200 && state.status === "running"; t++) {
      state = step(state, { kind: "tick" });
    }
    expect(state.status).toBe("solved");
    expect(state.finalAnswer).toBe("1,800 to 7,000 ft");
  });

  it("injecting a dead search mid-run flips a solving ReAct run to stuck", () => {
    const clean = run("react-grounded");
    expect(clean.status).toBe("solved");

    let state = initialState("react-grounded");
    state = step(state, { kind: "tick" }); // thought 1
    state = step(state, {
      kind: "intervene",
      action: { type: "injectDeadSearch" },
    });
    for (let t = 0; t < 200 && state.status === "running"; t++) {
      state = step(state, { kind: "tick" });
    }
    expect(state.status).not.toBe("solved");
  });

  it("corrupting memory strips grounding from every stated belief", () => {
    let state = initialState("react-grounded");
    // Advance through one full beat so a fact is grounded.
    state = step(state, { kind: "tick" }); // thought
    state = step(state, { kind: "tick" }); // action
    state = step(state, { kind: "tick" }); // observation -> grounds a fact
    expect(state.groundedFacts).toBeGreaterThan(0);
    state = step(state, {
      kind: "intervene",
      action: { type: "corruptMemory" },
    });
    expect(state.groundedFacts).toBe(0);
    expect(grounding(state).ratio).toBeLessThan(1);
  });

  it("editing a CoT guess into the truth removes the hallucination", () => {
    let state = initialState("cot-hallucinates");
    state = step(state, {
      kind: "intervene",
      action: {
        type: "editThought",
        text: "Apple Remote controls Front Row, controlled by keyboard function keys.",
      },
    });
    for (let t = 0; t < 60 && state.status === "running"; t++) {
      state = step(state, { kind: "tick" });
    }
    // The first guess is now grounded truth, so the run no longer hallucinates.
    const firstThought = state.steps.find((s) => s.kind === "thought");
    expect(firstThought?.hallucinated).toBe(false);
  });

  it("forcing a loop stops the run as stuck", () => {
    let state = initialState("react-grounded");
    state = step(state, { kind: "tick" });
    state = step(state, { kind: "intervene", action: { type: "forceLoop" } });
    expect(state.status).toBe("stuck");
  });
});

describe("the Fever claim", () => {
  it("ReAct grounds a SUPPORTS verdict", () => {
    let state = initialState("react-grounded");
    state = step(state, {
      kind: "intervene",
      action: { type: "setTask", id: "fever" },
    });
    for (let t = 0; t < 60 && state.status === "running"; t++) {
      state = step(state, { kind: "tick" });
    }
    expect(state.status).toBe("solved");
    expect(state.finalAnswer).toBe("SUPPORTS");
  });
});

describe("presets", () => {
  it("loads each preset's task and mode", () => {
    for (const preset of [
      "react-grounded",
      "cot-hallucinates",
      "act-loops",
      "dead-search",
    ]) {
      const state = initialState(preset);
      const p = getPreset(preset);
      expect(state.taskId).toBe(p.taskId);
      expect(state.mode).toBe(p.mode);
    }
  });
});
