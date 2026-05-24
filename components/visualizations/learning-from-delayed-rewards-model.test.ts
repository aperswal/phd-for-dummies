import { describe, expect, it } from "vitest";

import {
  ACTIONS,
  buildWorld,
  convergence,
  getPreset,
  greedyPolicyAction,
  initialState,
  optimalValues,
  RIGHT,
  step,
  transitionOutcomes,
  valueOf,
  type Input,
  type Params,
  type SimState,
} from "./learning-from-delayed-rewards-model";

function run(state: SimState, ticks: number): SimState {
  const tick: Input = { kind: "tick" };
  let next = state;
  for (let i = 0; i < ticks; i++) next = step(next, tick);
  return next;
}

const CORRIDOR: Params = {
  alpha: 0.5,
  gamma: 0.9,
  epsilon: 0,
  slip: 0,
  stepCost: -0.04,
  goalReward: 1,
  penaltyReward: -1,
  behavior: "greedy",
};

// A three-cell corridor S . G lets us check the update rule by hand: the agent
// at the left cell, forced right, lands on the empty middle cell.
function corridorState(): SimState {
  const world = buildWorld(["S.G"]);
  return {
    world,
    params: { ...CORRIDOR },
    q: [
      [-1, 0.5, -1, -1], // cell 0: greedy picks RIGHT with no tie
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ],
    visits: [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ],
    agent: 0,
    rng: 1,
    episodes: 0,
    steps: 0,
    stepsThisEpisode: 0,
    episodeLengths: [],
    last: null,
    log: { events: [], nextEventId: 1 },
  };
}

describe("the one-step Q-learning update", () => {
  it("nudges Q(x,a) toward r + gamma*max_b Q(y,b) by exactly alpha", () => {
    const state = corridorState();
    const next = step(state, { kind: "tick" });
    // RIGHT from cell 0 -> cell 1 (empty), reward = stepCost, max Q(cell1) = 0.
    const target = CORRIDOR.stepCost + CORRIDOR.gamma * 0;
    const expected = (1 - CORRIDOR.alpha) * 0.5 + CORRIDOR.alpha * target;
    expect(next.q[0]?.[RIGHT]).toBeCloseTo(expected, 10);
    expect(next.last?.x).toBe(0);
    expect(next.last?.y).toBe(1);
  });

  it("with alpha = 1 copies the target outright, keeping nothing of the old value", () => {
    const state = corridorState();
    state.params.alpha = 1;
    const next = step(state, { kind: "tick" });
    const target = CORRIDOR.stepCost + CORRIDOR.gamma * 0;
    expect(next.q[0]?.[RIGHT]).toBeCloseTo(target, 10);
  });

  it("uses a zero bootstrap when the move reaches the terminal goal", () => {
    const state = corridorState();
    state.agent = 1;
    state.q[1] = [-1, 0.5, -1, -1]; // greedy RIGHT into the goal
    const next = step(state, { kind: "tick" });
    const target = CORRIDOR.goalReward + CORRIDOR.gamma * 0;
    const expected = (1 - CORRIDOR.alpha) * 0.5 + CORRIDOR.alpha * target;
    expect(next.q[1]?.[RIGHT]).toBeCloseTo(expected, 10);
    expect(next.last?.terminal).toBe(true);
    expect(next.episodes).toBe(1);
    expect(next.agent).toBe(state.world.start);
  });
});

describe("the world dynamics", () => {
  it("keeps the agent in place when a move hits a wall or edge", () => {
    const world = buildWorld(["S#", "..G"]);
    // From cell 0, RIGHT is a wall and UP is off-grid, both stay put.
    expect(transitionOutcomes(world, CORRIDOR, 0, RIGHT)).toEqual([
      { to: 0, prob: 1 },
    ]);
  });

  it("splits a slippery move over the intended and perpendicular cells, summing to one", () => {
    const world = buildWorld(["...", "...", "..G"]);
    const outcomes = transitionOutcomes(
      world,
      { ...CORRIDOR, slip: 0.3 },
      4,
      RIGHT,
    );
    const total = outcomes.reduce((sum, o) => sum + o.prob, 0);
    expect(total).toBeCloseTo(1, 10);
    expect(outcomes.length).toBeGreaterThan(1);
  });
});

describe("the value-iteration oracle", () => {
  it("gives higher optimal value to cells nearer the goal", () => {
    const world = buildWorld(["S...G"]);
    const star = optimalValues(world, CORRIDOR);
    expect(star[3]).toBeGreaterThan(star[0] ?? 0); // one step from goal beats four
  });
});

describe("learning converges on the happy path", () => {
  it("drives the greedy policy to optimal in the maze preset", () => {
    const state = run(initialState("backup"), 40000);
    const result = convergence(state);
    expect(result.valueError).toBeLessThan(0.05);
    expect(result.greedyOptimal).toBe(true);
  });
});

describe("off-policy learning", () => {
  it("learns the optimal policy even while behaving completely at random", () => {
    const state = run(initialState("offpolicy"), 40000);
    const result = convergence(state);
    expect(state.params.behavior).toBe("random");
    expect(result.greedyOptimal).toBe(true);
    expect(result.coverage).toBeGreaterThan(0.99);
  });
});

describe("exploration is required", () => {
  it("leaves a greedy, no-cost agent with worse coverage than an exploring one", () => {
    const greedy = run(initialState("noexplore"), 8000);
    const turnedOn = step(
      step(initialState("noexplore"), {
        kind: "intervene",
        action: { type: "setBehavior", value: "epsilon-greedy" },
      }),
      { kind: "intervene", action: { type: "setEpsilon", value: 0.3 } },
    );
    const exploring = run(turnedOn, 8000);
    expect(convergence(greedy).coverage).toBeLessThan(1);
    expect(convergence(exploring).coverage).toBeGreaterThan(
      convergence(greedy).coverage,
    );
  });
});

describe("interventions", () => {
  it("toggling a wall changes the world and logs the moment to rewind to", () => {
    const before = initialState("backup");
    const walled = step(before, {
      kind: "intervene",
      action: { type: "toggleWall", cell: 2 },
    });
    expect(walled.world.walls[2]).toBe(true);
    const event = walled.log.events.at(-1);
    expect(event?.snapshot.world.walls[2]).toBe(false); // snapshot is pre-change
  });

  it("refuses to wall in the start or the goal", () => {
    const state = initialState("backup");
    const tryGoal = step(state, {
      kind: "intervene",
      action: { type: "toggleWall", cell: state.world.goal },
    });
    expect(tryGoal.world.walls[state.world.goal]).toBe(false);
  });

  it("restores a snapshot exactly", () => {
    const state = run(initialState("backup"), 200);
    const moved = run(state, 50);
    const restored = step(moved, {
      kind: "intervene",
      action: { type: "restore", snapshot: state, label: "rewind" },
    });
    expect(restored.steps).toBe(state.steps);
    expect(restored.agent).toBe(state.agent);
    expect(valueOf(restored.q, 0)).toBeCloseTo(valueOf(state.q, 0), 10);
  });
});

describe("determinism", () => {
  it("reproduces a run from the same seed and inputs", () => {
    const a = run(initialState("backup", 7), 1500);
    const b = run(initialState("backup", 7), 1500);
    expect(b.q).toEqual(a.q);
    expect(b.agent).toBe(a.agent);
    expect(b.episodes).toBe(a.episodes);
  });

  it("greedy policy reads the argmax action", () => {
    const q = [[0.1, 0.9, 0.2, 0.3]];
    expect(greedyPolicyAction(q, 0)).toBe(RIGHT);
    expect(ACTIONS.length).toBe(4);
  });
});

describe("presets", () => {
  it("every preset builds a world with a goal and a start", () => {
    for (const preset of [getPreset("backup"), getPreset("offpolicy")]) {
      const world = buildWorld(preset.rows);
      expect(world.goal).toBeGreaterThanOrEqual(0);
      expect(world.start).toBeGreaterThanOrEqual(0);
    }
  });
});
