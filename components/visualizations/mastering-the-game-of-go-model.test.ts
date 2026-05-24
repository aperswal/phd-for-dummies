import { describe, expect, it } from "vitest";

import {
  chosenMove,
  initialState,
  isCorrect,
  minimaxValue,
  MOVE_NAMES,
  step,
  totalRootVisits,
  trueBestMove,
  type Intervention,
  type SimState,
} from "@/components/visualizations/mastering-the-game-of-go-model";

function run(state: SimState, ticks: number): SimState {
  let next = state;
  for (let i = 0; i < ticks; i++) {
    next = step(next, { kind: "tick" });
  }
  return next;
}

function intervene(state: SimState, action: Intervention): SimState {
  return step(state, { kind: "intervene", action });
}

function moveSlot(state: SimState, nodeId: number | null): number | null {
  if (nodeId === null) return null;
  return state.nodes[nodeId]?.childSlot ?? null;
}

describe("AlphaGo MCTS model", () => {
  it("knows the true-best root move is 'center' (slot 0) under minimax", () => {
    const state = initialState("honest");
    // center = 0.55, contact = -0.6 (trap), corner = 0.3. center wins.
    expect(moveSlot(state, trueBestMove(state))).toBe(0);
    expect(MOVE_NAMES[0]).toBe("center");
  });

  it("starts with the root expanded into three moves and zero visits", () => {
    const state = initialState("honest");
    const root = state.nodes[state.root]!;
    expect(root.expanded).toBe(true);
    expect(root.children).toHaveLength(3);
    expect(totalRootVisits(state)).toBe(0);
  });

  it("conserves visit counts: every simulation adds exactly one root visit", () => {
    let state = initialState("honest");
    for (let i = 1; i <= 40; i++) {
      state = step(state, { kind: "tick" });
      expect(state.simulations).toBe(i);
      // Each simulation backs up through exactly one root child, so the sum of
      // root-child visits equals the simulation count.
      expect(totalRootVisits(state)).toBe(i);
    }
  });

  it("happy path: honest search converges to the true-best move", () => {
    const state = run(initialState("honest"), 220);
    expect(isCorrect(state)).toBe(true);
    expect(moveSlot(state, chosenMove(state))).toBe(0);
  });

  it("is deterministic: same seed and ticks reproduce the run exactly", () => {
    const a = run(initialState("honest", 11), 120);
    const b = run(initialState("honest", 11), 120);
    expect(chosenMove(a)).toBe(chosenMove(b));
    expect(totalRootVisits(a)).toBe(totalRootVisits(b));
    expect(a.rng).toBe(b.rng);
  });

  it("value-net lie at lambda 0 makes search pick the losing move", () => {
    // The preset pins a lie on the trap move and sets lambda 0 (value only).
    const state = run(initialState("value-lies"), 200);
    expect(moveSlot(state, chosenMove(state))).toBe(1); // contact, the trap
    expect(isCorrect(state)).toBe(false);
  });

  it("raising lambda lets rollouts overrule the lying value net", () => {
    let state = initialState("value-lies");
    state = intervene(state, { type: "setLambda", value: 0.5 });
    state = run(state, 240);
    // Rollouts see the trap move's true value (-0.6) and pull search off it.
    expect(moveSlot(state, chosenMove(state))).toBe(0);
    expect(isCorrect(state)).toBe(true);
  });

  it("greedy trap (low c_puct, high bad prior) tunnels into the wrong move", () => {
    const state = run(initialState("trap"), 200);
    expect(moveSlot(state, chosenMove(state))).toBe(1);
    expect(isCorrect(state)).toBe(false);
  });

  it("more exploration escapes the greedy trap", () => {
    let state = initialState("trap");
    state = intervene(state, { type: "setCPuct", value: 2.5 });
    state = run(state, 320);
    expect(moveSlot(state, chosenMove(state))).toBe(0);
    expect(isCorrect(state)).toBe(true);
  });

  it("pruning the true-best move forces search onto the next-best legal move", () => {
    let state = initialState("honest");
    const center = state.nodes.find(
      (n) => n.parent === state.root && n.childSlot === 0,
    )!;
    state = intervene(state, { type: "prune", node: center.id });
    state = run(state, 200);
    // center is gone from selection, so the most-visited move is no longer 0.
    expect(moveSlot(state, chosenMove(state))).not.toBe(0);
    // corner (0.3) beats the trap (-0.6), so search lands on corner.
    expect(moveSlot(state, chosenMove(state))).toBe(2);
  });

  it("the search tree grows: nodes get expanded as simulations descend", () => {
    const before = initialState("honest").nodes.length;
    const after = run(initialState("honest"), 60).nodes.length;
    expect(after).toBeGreaterThan(before);
  });

  it("restore rewinds to a recorded snapshot", () => {
    let state = run(initialState("honest"), 30);
    const snapshot = {
      nodes: state.nodes.map((n) => ({ ...n, children: [...n.children] })),
      rng: state.rng,
      simulations: state.simulations,
      params: { ...state.params },
    };
    const simsAtSnapshot = state.simulations;
    state = run(state, 50);
    expect(state.simulations).toBe(simsAtSnapshot + 50);
    state = intervene(state, { type: "restore", snapshot, label: "rewind" });
    expect(state.simulations).toBe(simsAtSnapshot);
  });

  it("minimax of the trap move is its worst reply (-0.6)", () => {
    const state = initialState("honest");
    const trap = state.nodes.find(
      (n) => n.parent === state.root && n.childSlot === 1,
    )!;
    expect(minimaxValue(state, trap.id)).toBeCloseTo(-0.6, 5);
  });
});
