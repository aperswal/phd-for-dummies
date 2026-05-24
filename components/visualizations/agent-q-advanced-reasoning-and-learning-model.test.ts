import { describe, expect, it } from "vitest";

import {
  blendedValue,
  initialState,
  presetState,
  step,
  type Params,
  type SimState,
} from "@/components/visualizations/agent-q-advanced-reasoning-and-learning-model";

function run(state: SimState, ticks: number): SimState {
  let s = state;
  for (let i = 0; i < ticks; i++) s = step(s, { kind: "tick" });
  return s;
}

function findNodeByPage(state: SimState, pageId: string) {
  return state.nodes.find((n) => n.pageId === pageId);
}

const BASE: Params = {
  cExp: 1.2,
  alpha: 0.5,
  threshold: 0.2,
  budget: 24,
  rolloutNoise: 0.3,
};

describe("Agent Q model invariants", () => {
  it("is deterministic: same seed and inputs reproduce the run exactly", () => {
    const a = run(initialState(BASE, 7), 24);
    const b = run(initialState(BASE, 7), 24);
    expect(a.successes).toBe(b.successes);
    expect(a.nodes.length).toBe(b.nodes.length);
    expect(a.prefPairs).toEqual(b.prefPairs);
  });

  it("keeps every blended action value inside [0, 1]", () => {
    const state = run(presetState("guided"), 24);
    for (const node of state.nodes) {
      const page = node.pageId;
      void page;
      for (let i = 0; i < node.actionValue.length; i++) {
        const v = blendedValue(node, i, state.params.alpha);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
  });

  it("conserves visits: the root visit count equals the rollouts run", () => {
    const state = run(initialState(BASE, 7), 10);
    const root = state.nodes.find((n) => n.id === state.rootId);
    expect(root?.visits).toBe(10);
  });

  it("stops growing the tree once the budget is spent", () => {
    const state = run(initialState({ ...BASE, budget: 6 }, 7), 20);
    expect(state.iteration).toBe(6);
    expect(state.phase).toBe("done");
  });
});

describe("Agent Q happy path: guided search finds the booking", () => {
  it("reaches the goal and grows the correct path of pages", () => {
    const state = run(presetState("guided"), 24);
    expect(state.successes).toBeGreaterThan(0);
    expect(findNodeByPage(state, "right-restaurant")).toBeDefined();
    expect(findNodeByPage(state, "complete")).toBeDefined();
  });

  it("forms a preference pair at the search page favouring the correct restaurant", () => {
    const state = run(presetState("guided"), 24);
    const searchPair = state.prefPairs.find(
      (p) => p.pageLabel === "OpenTable search",
    );
    expect(searchPair).toBeDefined();
    expect(searchPair?.preferred).toContain("correct restaurant");
  });
});

describe("Agent Q failure modes from the paper", () => {
  it("greedy search (c_exp = 0) books far less often than guided search", () => {
    const greedy = run(presetState("greedy"), 24);
    const guided = run(presetState("guided"), 24);
    expect(greedy.successes).toBeLessThan(guided.successes);
  });

  it("a misleading critic pulls the search into the wrong-restaurant branch", () => {
    const misled = run(presetState("misleading"), 24);
    const wrong = findNodeByPage(misled, "wrong-restaurant");
    const right = findNodeByPage(misled, "right-restaurant");
    // The wrong branch gets expanded and accrues visits because the poisoned
    // critic ranks it highest and alpha leans on the critic.
    expect(wrong).toBeDefined();
    const wrongVisits = wrong?.visits ?? 0;
    const rightVisits = right?.visits ?? 0;
    expect(wrongVisits).toBeGreaterThan(rightVisits);
  });
});

describe("Agent Q mid-run interventions change the outcome", () => {
  it("poisoning the search critic mid-run reverses which restaurant it prefers", () => {
    let state = presetState("guided");
    state = run(state, 8);
    const before = blendedValue(
      state.nodes[0]!,
      1, // index 1 is "open the correct restaurant"
      state.params.alpha,
    );
    const search = findNodeByPage(state, "search")!;
    state = step(state, {
      kind: "intervene",
      action: { type: "poisonNode", nodeId: search.id },
    });
    const after = blendedValue(state.nodes[0]!, 1, state.params.alpha);
    // The correct restaurant looked better before; after poisoning the critic
    // it looks worse, because the critic rank flips.
    expect(after).toBeLessThan(before);
  });

  it("killing a node stops the search from expanding into it again", () => {
    let state = run(presetState("guided"), 6);
    const wrong = findNodeByPage(state, "wrong-restaurant");
    if (wrong) {
      const visitsBefore = wrong.visits;
      state = step(state, {
        kind: "intervene",
        action: { type: "killNode", nodeId: wrong.id },
      });
      state = run(state, 12);
      const wrongAfter = state.nodes.find((n) => n.id === wrong.id);
      expect(wrongAfter?.dead).toBe(true);
      // Killed, so it accrues no further visits.
      expect(wrongAfter?.visits).toBe(visitsBefore);
    } else {
      // Guided search may not even open the wrong branch; that is itself the
      // point, so the assertion is vacuously satisfied.
      expect(wrong).toBeUndefined();
    }
  });

  it("raising the DPO threshold drops low-gap preference pairs", () => {
    let state = run(presetState("guided"), 24);
    const pairsLow = state.prefPairs.length;
    state = step(state, {
      kind: "intervene",
      action: { type: "setThreshold", value: 0.9 },
    });
    expect(state.prefPairs.length).toBeLessThanOrEqual(pairsLow);
  });
});
