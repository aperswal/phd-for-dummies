import { describe, expect, it } from "vitest";

import {
  defaultParams,
  getScenario,
  initialState,
  playsBestMove,
  puctScore,
  rootMoveStats,
  selectedMove,
  step,
  type Input,
  type SimState,
} from "./mastering-chess-and-shogi-by-self-play-model";

function run(state: SimState, sims: number): SimState {
  let next = state;
  for (let i = 0; i < sims; i++) {
    next = step(next, { kind: "tick" });
  }
  return next;
}

function intervene(state: SimState, action: Input): SimState {
  return step(state, action);
}

describe("PUCT selection rule", () => {
  it("rewards a high prior on an untried move", () => {
    const tried = puctScore(0, 0.5, 9, 9, 1.5);
    const untried = puctScore(0, 0.5, 9, 0, 1.5);
    expect(untried).toBeGreaterThan(tried);
  });

  it("rewards a higher mean value at equal visits", () => {
    const low = puctScore(-0.2, 0.4, 9, 3, 1.5);
    const high = puctScore(0.6, 0.4, 9, 3, 1.5);
    expect(high).toBeGreaterThan(low);
  });

  it("widens the exploration bonus as c_puct grows", () => {
    const calm = puctScore(0, 0.5, 16, 1, 0.5);
    const eager = puctScore(0, 0.5, 16, 1, 3);
    expect(eager - calm).toBeGreaterThan(0);
  });
});

describe("determinism", () => {
  it("reproduces the exact same run for the same seed and inputs", () => {
    const a = run(initialState("wrong-net"), 40);
    const b = run(initialState("wrong-net"), 40);
    expect(rootMoveStats(a)).toEqual(rootMoveStats(b));
  });

  it("reproduces the same run with root noise on a fixed seed", () => {
    const params = { ...defaultParams(), rootNoise: 0.4, seed: 3 };
    const a = run(initialState("wrong-net", params), 30);
    const b = run(initialState("wrong-net", params), 30);
    expect(rootMoveStats(a)).toEqual(rootMoveStats(b));
  });
});

describe("invariants", () => {
  it("conserves visits: every simulation adds exactly one root visit", () => {
    const state = run(initialState("wrong-net"), 25);
    const rootVisits = rootMoveStats(state).reduce(
      (sum, move) => sum + move.visits,
      0,
    );
    expect(rootVisits).toBe(25);
  });

  it("never exceeds the simulation budget", () => {
    const params = { ...defaultParams(), simBudget: 12 };
    const state = run(initialState("wrong-net", params), 100);
    expect(state.completed).toBe(12);
  });

  it("a tick past the budget is a no-op", () => {
    const params = { ...defaultParams(), simBudget: 5 };
    const filled = run(initialState("wrong-net", params), 5);
    const after = step(filled, { kind: "tick" });
    expect(after.completed).toBe(5);
    expect(rootMoveStats(after)).toEqual(rootMoveStats(filled));
  });
});

describe("policy improvement: search corrects a wrong prior", () => {
  it("echoes the wrong prior when starved of simulations", () => {
    const params = { ...defaultParams(), simBudget: 2, cPuct: 1.5 };
    const state = run(initialState("wrong-net", params), 2);
    const chosen = selectedMove(state);
    // With only two simulations the search just visits the top-prior move A.
    expect(chosen?.label).toBe("A");
    expect(playsBestMove(state)).toBe(false);
  });

  it("drives visits onto the true best move with enough simulations", () => {
    const state = run(initialState("wrong-net"), 60);
    const chosen = selectedMove(state);
    expect(chosen?.label).toBe("C");
    expect(playsBestMove(state)).toBe(true);
  });
});

describe("scenarios", () => {
  it("confirms an already-good prior on the aligned net", () => {
    const state = run(initialState("aligned-net"), 40);
    expect(playsBestMove(state)).toBe(true);
    expect(selectedMove(state)?.label).toBe(
      getScenario("aligned-net").tree.nodes[getScenario("aligned-net").bestMove]
        ?.label,
    );
  });

  it("uncovers the hidden trap and avoids the surface-good move", () => {
    const state = run(initialState("trap"), 60);
    const chosen = selectedMove(state);
    expect(chosen?.label).not.toBe("A");
    expect(playsBestMove(state)).toBe(true);
  });
});

describe("interventions change the outcome", () => {
  it("raising c_puct mid-run lets a starved search escape the wrong prior", () => {
    // The same simulation budget, two exploration settings. Low exploration
    // exploits the values it has already seen and never finds the real win,
    // settling on the safe draw B. High exploration keeps probing and uncovers
    // move C, the true best, so the chosen move changes under the paper's rule.
    const timid = run(
      initialState("wrong-net", {
        ...defaultParams(),
        simBudget: 24,
        cPuct: 0.2,
      }),
      24,
    );
    const bold = run(
      initialState("wrong-net", {
        ...defaultParams(),
        simBudget: 24,
        cPuct: 4,
      }),
      24,
    );

    expect(playsBestMove(timid)).toBe(false);
    expect(selectedMove(bold)?.label).toBe("C");
    expect(playsBestMove(bold)).toBe(true);
    expect(selectedMove(bold)?.label).not.toBe(selectedMove(timid)?.label);
  });

  it("rewind restores an earlier moment exactly", () => {
    let state = run(initialState("wrong-net"), 10);
    const checkpoint = state.events[state.events.length - 1];
    expect(checkpoint).toBeDefined();
    const snapshot = checkpoint?.snapshot;
    if (!snapshot) throw new Error("no checkpoint");

    state = run(state, 20);
    const restored = intervene(state, {
      kind: "intervene",
      action: { type: "restore", snapshot, label: "rewind" },
    });
    expect(restored.completed).toBe(snapshot.completed);
    expect(restored.stats).toEqual(snapshot.stats);
  });

  it("loading a scenario resets the search", () => {
    const played = run(initialState("wrong-net"), 30);
    const switched = intervene(played, {
      kind: "intervene",
      action: { type: "loadScenario", id: "aligned-net" },
    });
    expect(switched.completed).toBe(0);
    expect(switched.scenarioId).toBe("aligned-net");
  });
});
