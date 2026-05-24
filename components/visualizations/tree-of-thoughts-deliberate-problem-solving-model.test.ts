import { describe, expect, it } from "vitest";

import {
  canReach24,
  getPreset,
  initialState,
  runToEnd,
  step,
  type Params,
} from "./tree-of-thoughts-deliberate-problem-solving-model";

const base: Params = {
  algorithm: "bfs",
  beam: 5,
  threshold: 0.3,
  honest: true,
  evalNoise: 0.3,
  proposals: 6,
  puzzleId: "4-9-10-13",
};

describe("Game of 24 ground truth", () => {
  it("knows the paper's example reaches 24", () => {
    expect(canReach24([4, 9, 10, 13])).toBe(true);
  });

  it("knows 1 1 1 1 cannot reach 24", () => {
    expect(canReach24([1, 1, 1, 1])).toBe(false);
  });

  it("calls a single 24 a solution and a single 23 not", () => {
    expect(canReach24([24])).toBe(true);
    expect(canReach24([23])).toBe(false);
  });
});

describe("the search is deterministic", () => {
  it("reproduces the same run from the same seed and inputs", () => {
    const a = runToEnd(base, 7);
    const b = runToEnd(base, 7);
    expect(b.outcome).toBe(a.outcome);
    expect(b.nodes.length).toBe(a.nodes.length);
    expect(b.solutionId).toBe(a.solutionId);
  });
});

describe("safety: a found solution really reaches 24", () => {
  it("never reports solved on a state that does not equal 24", () => {
    const solved = runToEnd(base, 7);
    expect(solved.outcome).toBe("solved");
    const solution = solved.nodes.find((n) => n.id === solved.solutionId);
    expect(solution?.numbers).toEqual([24]);
  });

  it("fails honestly on an unsolvable puzzle instead of inventing an answer", () => {
    const failed = runToEnd({ ...base, puzzleId: "1-1-1-1" }, 7);
    expect(failed.outcome).toBe("failed");
    expect(failed.solutionId).toBeNull();
  });
});

describe("an honest evaluator with an adequate beam always solves a solvable puzzle", () => {
  it("solves the paper's example with BFS, with DFS, and even with beam 1", () => {
    expect(runToEnd(base, 7).outcome).toBe("solved");
    expect(runToEnd({ ...base, beam: 1 }, 7).outcome).toBe("solved");
    expect(runToEnd({ ...base, algorithm: "dfs" }, 7).outcome).toBe("solved");
  });
});

describe("beam width changes the outcome under a noisy evaluator (the b=1 vs b=5 result)", () => {
  it("a wider beam recovers a run a narrow beam fails", () => {
    const noisy = { ...base, honest: false, evalNoise: 0.5 };
    const narrow = runToEnd({ ...noisy, beam: 1 }, 7);
    const wide = runToEnd({ ...noisy, beam: 5 }, 7);
    expect(narrow.outcome).toBe("failed");
    expect(wide.outcome).toBe("solved");
  });
});

describe("the evaluator's honesty changes the outcome (the crosswords pruning lesson)", () => {
  it("a noisy evaluator prunes the only path and fails where an honest one solves", () => {
    const noisy = runToEnd(
      { ...base, beam: 2, honest: false, evalNoise: 0.55 },
      3,
    );
    const honest = runToEnd({ ...base, beam: 2, honest: true }, 3);
    expect(noisy.outcome).toBe("failed");
    expect(honest.outcome).toBe("solved");
    // the noisy evaluator marked at least one truly-reachable state impossible.
    expect(noisy.nodes.some((n) => n.misjudged)).toBe(true);
  });
});

describe("DFS explores, prunes, and backtracks", () => {
  it("reaches a terminal outcome and visits more than the root", () => {
    const dfs = runToEnd({ ...base, algorithm: "dfs" }, 7);
    expect(["solved", "failed"]).toContain(dfs.outcome);
    expect(dfs.visited).toBeGreaterThan(0);
  });

  it("a higher threshold prunes harder, visiting no more nodes", () => {
    const lenient = runToEnd({ ...base, algorithm: "dfs", threshold: 0.1 }, 7);
    const strict = runToEnd({ ...base, algorithm: "dfs", threshold: 0.9 }, 7);
    expect(strict.visited).toBeLessThanOrEqual(lenient.visited);
  });
});

describe("mid-run interventions", () => {
  it("force-pruning the kept frontier can stall a search that would have solved", () => {
    let state = initialState({ ...base, beam: 1 }, 7);
    // first BFS layer keeps one node; prune it and the frontier empties.
    state = step(state, { kind: "tick" });
    const kept = state.frontier;
    for (const id of kept) {
      state = step(state, {
        kind: "intervene",
        action: { type: "forcePrune", nodeId: id },
      });
    }
    state = step(state, { kind: "tick" });
    expect(state.outcome).toBe("failed");
  });

  it("flipping a node's verdict to impossible marks it misjudged when it was reachable", () => {
    let state = initialState(base, 7);
    state = step(state, { kind: "tick" });
    const reachable = state.nodes.find(
      (n) => n.reachable && n.status === "kept",
    );
    expect(reachable).toBeDefined();
    if (!reachable) return;
    // cycle the verdict until it lands on impossible.
    let target = reachable;
    for (let i = 0; i < 3 && target.verdict !== "impossible"; i++) {
      state = step(state, {
        kind: "intervene",
        action: { type: "flipVerdict", nodeId: reachable.id },
      });
      target = state.nodes.find((n) => n.id === reachable.id) ?? target;
    }
    expect(target.verdict).toBe("impossible");
    expect(target.misjudged).toBe(true);
  });

  it("loads each preset by id", () => {
    let state = initialState(base, 7);
    for (const preset of ["bfs-wide", "bfs-narrow", "noisy-eval", "dfs"]) {
      state = step(state, {
        kind: "intervene",
        action: { type: "loadPreset", id: preset },
      });
      expect(state.params).toEqual(getPreset(preset).params);
    }
  });
});
