// A pure, deterministic model of one-step Q-learning from Watkins' 1989 thesis
// "Learning from Delayed Rewards". The agent has no model of the world. It keeps
// a table Q(x, a) of action-values and, on every observed step [x, a, r, y],
// nudges one entry toward the paper's target:
//
//   Q(x, a) <- (1 - alpha) Q(x, a) + alpha ( r + gamma * max_b Q(y, b) )
//
// which is exactly the rule on page 96 of the thesis. The greedy policy reads
// off argmax_a Q(x, a). The agent is "free to take whatever actions it likes",
// so the behaviour policy (how it moves) is separate from the greedy policy it
// learns. That split is what makes Q-learning off-policy. All randomness is
// drawn from a seeded stream threaded through the state, so a seed plus a list
// of inputs reproduces a run exactly.

import { at } from "@/lib/simulation/array";
import { emptyLog, record, type EventLog } from "@/lib/simulation/history";
import { nextRandom } from "@/lib/simulation/rng";

// Actions are the four grid moves, indexed so that (dir + 2) mod 4 reverses a
// direction and the perpendicular pair is { dir + 1, dir + 3 } (mod 4).
export const UP = 0;
export const RIGHT = 1;
export const DOWN = 2;
export const LEFT = 3;
export const ACTIONS = [UP, RIGHT, DOWN, LEFT] as const;

// Indexed by action, so DELTA[UP] is the row/col step for moving up.
const DELTA = [
  { dr: -1, dc: 0 },
  { dr: 0, dc: 1 },
  { dr: 1, dc: 0 },
  { dr: 0, dc: -1 },
] as const;

export type Behavior = "epsilon-greedy" | "greedy" | "random";

// The world is the Markov decision process: a grid with walls, one terminal
// goal, an optional non-terminal penalty cell, and a fixed start. Interventions
// (toggling a wall, moving the goal) mutate the world, which is the whole point
// of letting the reader perturb a process that is already running.
export interface World {
  width: number;
  height: number;
  walls: boolean[];
  penalties: boolean[]; // non-terminal hazard cells that pay penaltyReward
  goal: number;
  start: number;
}

// The knobs the reader turns live. alpha is the learning factor, gamma the
// discount, epsilon the exploration rate of the behaviour policy, slip the
// probability a move is deflected to a perpendicular direction, stepCost the
// per-move reward, and goal/penalty the special rewards.
export interface Params {
  alpha: number;
  gamma: number;
  epsilon: number;
  slip: number;
  stepCost: number;
  goalReward: number;
  penaltyReward: number;
  behavior: Behavior;
}

export interface Transition {
  x: number;
  a: number;
  r: number;
  y: number;
  tdError: number;
  terminal: boolean;
}

export interface SimState {
  world: World;
  params: Params;
  q: number[][]; // q[cell][action]
  visits: number[][]; // how often each (cell, action) has been tried
  agent: number;
  rng: number;
  episodes: number;
  steps: number;
  stepsThisEpisode: number;
  episodeLengths: number[]; // recent episode lengths, the learning curve
  last: Transition | null;
  log: EventLog<Snapshot>;
}

export type Snapshot = Omit<SimState, "log">;

export function cellOf(world: World, row: number, col: number): number {
  return row * world.width + col;
}

export function rowCol(
  world: World,
  cell: number,
): { row: number; col: number } {
  return { row: Math.floor(cell / world.width), col: cell % world.width };
}

export function isGoal(world: World, cell: number): boolean {
  return cell === world.goal;
}

export function isWall(world: World, cell: number): boolean {
  return at(world.walls, cell);
}

export function isPenalty(world: World, cell: number): boolean {
  return at(world.penalties, cell);
}

// The cell a move lands in before any slip. A move into a wall or off the grid
// keeps the agent where it is, matching the thesis demo where an attempt to
// leave the square is curtailed.
function intendedNext(world: World, cell: number, action: number): number {
  const { row, col } = rowCol(world, cell);
  const delta = at(DELTA, action);
  const nextRow = row + delta.dr;
  const nextCol = col + delta.dc;
  if (
    nextRow < 0 ||
    nextRow >= world.height ||
    nextCol < 0 ||
    nextCol >= world.width
  ) {
    return cell;
  }
  const target = cellOf(world, nextRow, nextCol);
  return isWall(world, target) ? cell : target;
}

function perpendiculars(action: number): [number, number] {
  return [(action + 1) % 4, (action + 3) % 4];
}

// The full transition distribution for taking `action` in `cell`: probability
// 1 - slip of the intended move, and slip split evenly over the two
// perpendicular moves. The same distribution drives both the sampled
// environment and the value-iteration oracle, so the two never disagree.
export function transitionOutcomes(
  world: World,
  params: Params,
  cell: number,
  action: number,
): { to: number; prob: number }[] {
  const [perpA, perpB] = perpendiculars(action);
  const raw = [
    { to: intendedNext(world, cell, action), prob: 1 - params.slip },
    { to: intendedNext(world, cell, perpA), prob: params.slip / 2 },
    { to: intendedNext(world, cell, perpB), prob: params.slip / 2 },
  ];
  const merged = new Map<number, number>();
  for (const { to, prob } of raw) {
    merged.set(to, (merged.get(to) ?? 0) + prob);
  }
  return [...merged]
    .filter(([, prob]) => prob > 0)
    .map(([to, prob]) => ({ to, prob }));
}

export function rewardFor(world: World, params: Params, to: number): number {
  if (to === world.goal) return params.goalReward;
  if (isPenalty(world, to)) return params.penaltyReward;
  return params.stepCost;
}

function maxQ(q: number[][], cell: number): number {
  return at(q, cell).reduce((best, value) => Math.max(best, value), -Infinity);
}

export function valueOf(q: number[][], cell: number): number {
  return maxQ(q, cell);
}

export function greedyPolicyAction(q: number[][], cell: number): number {
  const values = at(q, cell);
  let best = -Infinity;
  let bestAction = UP;
  for (const action of ACTIONS) {
    const value = values[action] ?? -Infinity;
    if (value > best) {
      best = value;
      bestAction = action;
    }
  }
  return bestAction;
}

// argmax over actions with ties broken from the seeded stream, so a converged Q
// whose moves tie still picks deterministically for a given seed.
function greedyAction(
  q: number[][],
  cell: number,
  rng: number,
): { action: number; rng: number } {
  const values = at(q, cell);
  let best = -Infinity;
  for (const value of values) best = Math.max(best, value);
  const tied = ACTIONS.filter((a) => (values[a] ?? -Infinity) >= best - 1e-12);
  if (tied.length === 1) return { action: at(tied, 0), rng };
  const draw = nextRandom(rng);
  return {
    action: at(tied, Math.floor(draw.value * tied.length)),
    rng: draw.state,
  };
}

function randomAction(rng: number): { action: number; rng: number } {
  const draw = nextRandom(rng);
  return { action: at(ACTIONS, Math.floor(draw.value * 4)), rng: draw.state };
}

function chooseAction(
  state: SimState,
  cell: number,
): { action: number; rng: number } {
  const { params, q } = state;
  if (params.behavior === "greedy") return greedyAction(q, cell, state.rng);
  if (params.behavior === "random") return randomAction(state.rng);
  const roll = nextRandom(state.rng);
  if (roll.value < params.epsilon) return randomAction(roll.state);
  return greedyAction(q, cell, roll.state);
}

// Where the move actually lands, deflecting to a perpendicular with probability
// slip.
function sampleNext(
  world: World,
  params: Params,
  rng: number,
  cell: number,
  action: number,
): { to: number; rng: number } {
  if (params.slip <= 0) return { to: intendedNext(world, cell, action), rng };
  const roll = nextRandom(rng);
  if (roll.value >= params.slip) {
    return { to: intendedNext(world, cell, action), rng: roll.state };
  }
  const [perpA, perpB] = perpendiculars(action);
  const pick = roll.value < params.slip / 2 ? perpA : perpB;
  return { to: intendedNext(world, cell, pick), rng: roll.state };
}

function clone2d(grid: number[][]): number[][] {
  return grid.map((row) => [...row]);
}

// Everything that defines a moment except the log itself, so a log entry can
// carry a snapshot to rewind to without nesting a log inside a log. Listing the
// fields keeps it type-checked: drop one and the compiler complains.
function snapshotOf(state: SimState): Snapshot {
  return {
    world: state.world,
    params: state.params,
    q: state.q,
    visits: state.visits,
    agent: state.agent,
    rng: state.rng,
    episodes: state.episodes,
    steps: state.steps,
    stepsThisEpisode: state.stepsThisEpisode,
    episodeLengths: state.episodeLengths,
    last: state.last,
  };
}

function logEvent(
  state: SimState,
  snapshot: Snapshot,
  label: string,
): SimState {
  return { ...state, log: record(state.log, state.steps, label, snapshot) };
}

// One agent step: choose an action, see where it lands and what it pays, apply
// the one-step Q-learning update, and move. Reaching the goal ends the episode
// and resets the agent to the start. This is the only place Q changes during
// normal running.
function advance(state: SimState): SimState {
  const x = state.agent;
  const chosen = chooseAction(state, x);
  const sampled = sampleNext(
    state.world,
    state.params,
    chosen.rng,
    x,
    chosen.action,
  );
  const y = sampled.to;
  const a = chosen.action;
  const r = rewardFor(state.world, state.params, y);
  const terminal = isGoal(state.world, y);

  const q = clone2d(state.q);
  const old = at(at(q, x), a);
  const bootstrap = terminal ? 0 : maxQ(q, y);
  const target = r + state.params.gamma * bootstrap;
  const tdError = target - old;
  at(q, x)[a] = old + state.params.alpha * tdError;

  const visits = clone2d(state.visits);
  at(visits, x)[a] = at(at(visits, x), a) + 1;

  const last: Transition = { x, a, r, y, tdError, terminal };
  const steps = state.steps + 1;
  const stepsThisEpisode = state.stepsThisEpisode + 1;

  if (!terminal) {
    return {
      ...state,
      q,
      visits,
      rng: sampled.rng,
      agent: y,
      steps,
      stepsThisEpisode,
      last,
    };
  }

  const episodeLengths = [...state.episodeLengths, stepsThisEpisode].slice(-40);
  const next: SimState = {
    ...state,
    q,
    visits,
    rng: sampled.rng,
    agent: state.world.start,
    episodes: state.episodes + 1,
    steps,
    stepsThisEpisode: 0,
    episodeLengths,
    last,
  };
  return logEvent(
    next,
    snapshotOf(next),
    `episode ${next.episodes}: reached goal in ${stepsThisEpisode} steps`,
  );
}

// Value iteration on the true model, used only to score how close the learned Q
// is to optimal. The agent never sees this; it is the oracle behind the
// "value error" readout and the "greedy policy optimal" light.
export function optimalValues(world: World, params: Params): number[] {
  const n = world.width * world.height;
  const values = new Array<number>(n).fill(0);
  for (let sweep = 0; sweep < 400; sweep++) {
    let delta = 0;
    for (let cell = 0; cell < n; cell++) {
      if (isWall(world, cell) || isGoal(world, cell)) continue;
      let best = -Infinity;
      for (const action of ACTIONS) {
        let value = 0;
        for (const { to, prob } of transitionOutcomes(
          world,
          params,
          cell,
          action,
        )) {
          const cont = isGoal(world, to) ? 0 : at(values, to);
          value += prob * (rewardFor(world, params, to) + params.gamma * cont);
        }
        best = Math.max(best, value);
      }
      delta = Math.max(delta, Math.abs(best - at(values, cell)));
      values[cell] = best;
    }
    if (delta < 1e-9) break;
  }
  return values;
}

// Expected discounted return of following the agent's current greedy policy
// from the start, evaluated on the true model. Compared against the optimal
// start value to decide whether the learned policy is already optimal.
export function greedyPolicyValue(
  world: World,
  params: Params,
  q: number[][],
): number {
  const n = world.width * world.height;
  const values = new Array<number>(n).fill(0);
  for (let sweep = 0; sweep < 400; sweep++) {
    let delta = 0;
    for (let cell = 0; cell < n; cell++) {
      if (isWall(world, cell) || isGoal(world, cell)) continue;
      const action = greedyPolicyAction(q, cell);
      let value = 0;
      for (const { to, prob } of transitionOutcomes(
        world,
        params,
        cell,
        action,
      )) {
        const cont = isGoal(world, to) ? 0 : at(values, to);
        value += prob * (rewardFor(world, params, to) + params.gamma * cont);
      }
      delta = Math.max(delta, Math.abs(value - at(values, cell)));
      values[cell] = value;
    }
    if (delta < 1e-9) break;
  }
  return at(values, world.start);
}

export interface Convergence {
  valueError: number; // max |maxQ(x) - V*(x)| over reachable cells
  coverage: number; // fraction of (cell, action) pairs tried at least once
  greedyOptimal: boolean; // greedy policy value within tolerance of optimal
}

export function convergence(state: SimState): Convergence {
  const { world, params, q, visits } = state;
  const star = optimalValues(world, params);
  const n = world.width * world.height;

  let valueError = 0;
  let tried = 0;
  let total = 0;
  for (let cell = 0; cell < n; cell++) {
    if (isWall(world, cell) || isGoal(world, cell)) continue;
    valueError = Math.max(
      valueError,
      Math.abs(valueOf(q, cell) - at(star, cell)),
    );
    for (const action of ACTIONS) {
      total += 1;
      if (at(at(visits, cell), action) > 0) tried += 1;
    }
  }
  const greedyValue = greedyPolicyValue(world, params, q);
  const tol = Math.max(0.02, Math.abs(at(star, world.start)) * 0.02);
  return {
    valueError,
    coverage: total === 0 ? 1 : tried / total,
    greedyOptimal: Math.abs(greedyValue - at(star, world.start)) <= tol,
  };
}

export type Intervention =
  | { type: "setAlpha"; value: number }
  | { type: "setGamma"; value: number }
  | { type: "setEpsilon"; value: number }
  | { type: "setSlip"; value: number }
  | { type: "setStepCost"; value: number }
  | { type: "setBehavior"; value: Behavior }
  | { type: "toggleWall"; cell: number }
  | { type: "setGoal"; cell: number }
  | { type: "teleport"; cell: number }
  | { type: "pokeValue"; cell: number; action: number; value: number }
  | { type: "resetLearning" }
  | { type: "setSeed"; value: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: Snapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

// ---- Worlds and presets -------------------------------------------------

// A world from an ASCII map: '#' wall, 'G' goal, 'S' start, 'P' penalty, '.'
// empty. Rows must be equal length. Keeping layouts as text makes the presets
// readable and easy to add.
export function buildWorld(rows: string[]): World {
  const height = rows.length;
  const width = at(rows, 0).length;
  const walls = new Array<boolean>(width * height).fill(false);
  const penalties = new Array<boolean>(width * height).fill(false);
  let goal = -1;
  let start = 0;
  for (let row = 0; row < height; row++) {
    const line = at(rows, row);
    for (let col = 0; col < width; col++) {
      const cell = row * width + col;
      const ch = line[col];
      if (ch === "#") walls[cell] = true;
      else if (ch === "G") goal = cell;
      else if (ch === "S") start = cell;
      else if (ch === "P") penalties[cell] = true;
    }
  }
  return { width, height, walls, penalties, goal, start };
}

// A serpentine maze: the only route to the goal threads the whole grid, so the
// reward at the bottom-right has to back up across roughly eighteen cells. That
// long climb is exactly the delayed-reward credit assignment the thesis is
// about, and it is dramatic to watch fill in.
const MAZE = ["S......", "######.", ".......", ".######", "......G"];

const OPEN = ["S.....", "......", "..##..", "..##..", "......", ".....G"];

// A hazard band the agent must route around: the short way crosses the cliff,
// the optimal way goes around it.
const CLIFF = ["S.....", "......", ".PPPP.", "......", "G....."];

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  rows: string[];
  params: Params;
}

const BASE: Params = {
  alpha: 0.5,
  gamma: 0.95,
  epsilon: 0.2,
  slip: 0,
  stepCost: -0.04,
  goalReward: 1,
  penaltyReward: -1,
  behavior: "epsilon-greedy",
};

export const PRESETS: Preset[] = [
  {
    id: "backup",
    label: "Value backs up",
    blurb:
      "An e-greedy agent in a maze. Run it and watch the goal's reward seep backward cell by cell until every cell's arrow points along the shortest path. The value error falls and the optimal light turns on.",
    rows: MAZE,
    params: { ...BASE },
  },
  {
    id: "offpolicy",
    label: "Off-policy: pure random",
    blurb:
      "The agent moves completely at random, never using what it learns to decide where to go. The greedy arrows still line up into the optimal policy, because Q-learning learns the best route no matter how you wander, as long as every move gets tried.",
    rows: OPEN,
    params: { ...BASE, behavior: "random" },
  },
  {
    id: "cliff",
    label: "Rewards and penalties",
    blurb:
      "A band of penalty cells sits between the start and the goal. The agent learns to weigh the cost of crossing the hazard against the longer safe route, and the greedy arrows curve around the cliff.",
    rows: CLIFF,
    params: { ...BASE, penaltyReward: -1, epsilon: 0.25 },
  },
  {
    id: "noexplore",
    label: "No exploration stalls",
    blurb:
      "Pure greedy behaviour with no step cost. The agent commits to the first path it stumbles onto and stops trying anything else, so most cells never get visited and the policy stays stuck. Push epsilon up and watch coverage climb and the light flip.",
    rows: OPEN,
    params: { ...BASE, behavior: "greedy", epsilon: 0, stepCost: 0 },
  },
  {
    id: "slippery",
    label: "Big alpha + slippery floor",
    blurb:
      "A slippery floor makes moves random a quarter of the time, and alpha is cranked to 1 so each update throws away everything learned before and copies the latest noisy sample. The value error never settles. Lower alpha and it calms down.",
    rows: OPEN,
    params: { ...BASE, alpha: 1, slip: 0.25 },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

function freshQ(world: World): number[][] {
  return Array.from({ length: world.width * world.height }, () => [0, 0, 0, 0]);
}

export function initialState(presetId = "backup", seed = 1): SimState {
  const preset = getPreset(presetId);
  const world = buildWorld(preset.rows);
  return {
    world,
    params: { ...preset.params },
    q: freshQ(world),
    visits: freshQ(world).map((row) => row.map(() => 0)),
    agent: world.start,
    rng: seed >>> 0,
    episodes: 0,
    steps: 0,
    stepsThisEpisode: 0,
    episodeLengths: [],
    last: null,
    log: emptyLog<Snapshot>(),
  };
}

function resetLearningState(state: SimState): SimState {
  return {
    ...state,
    q: freshQ(state.world),
    visits: freshQ(state.world).map((row) => row.map(() => 0)),
    agent: state.world.start,
    episodes: 0,
    steps: state.steps,
    stepsThisEpisode: 0,
    episodeLengths: [],
    last: null,
  };
}

function applyIntervention(
  state: SimState,
  action: Intervention,
): { state: SimState; label: string } {
  switch (action.type) {
    case "setAlpha":
      return {
        state: { ...state, params: { ...state.params, alpha: action.value } },
        label: `alpha -> ${action.value.toFixed(2)}`,
      };
    case "setGamma":
      return {
        state: { ...state, params: { ...state.params, gamma: action.value } },
        label: `gamma -> ${action.value.toFixed(2)}`,
      };
    case "setEpsilon":
      return {
        state: { ...state, params: { ...state.params, epsilon: action.value } },
        label: `epsilon -> ${action.value.toFixed(2)}`,
      };
    case "setSlip":
      return {
        state: { ...state, params: { ...state.params, slip: action.value } },
        label: `slip -> ${action.value.toFixed(2)}`,
      };
    case "setStepCost":
      return {
        state: {
          ...state,
          params: { ...state.params, stepCost: action.value },
        },
        label: `step cost -> ${action.value.toFixed(2)}`,
      };
    case "setBehavior":
      return {
        state: {
          ...state,
          params: { ...state.params, behavior: action.value },
        },
        label: `behaviour -> ${action.value}`,
      };
    case "toggleWall": {
      if (
        action.cell === state.world.goal ||
        action.cell === state.world.start
      ) {
        return { state, label: "wall blocked (start/goal)" };
      }
      const walls = [...state.world.walls];
      walls[action.cell] = !walls[action.cell];
      return {
        state: { ...state, world: { ...state.world, walls } },
        label: `${walls[action.cell] ? "wall on" : "wall off"} cell ${action.cell}`,
      };
    }
    case "setGoal": {
      if (isWall(state.world, action.cell))
        return { state, label: "goal blocked (wall)" };
      return {
        state: { ...state, world: { ...state.world, goal: action.cell } },
        label: `goal -> cell ${action.cell}`,
      };
    }
    case "teleport":
      if (isWall(state.world, action.cell))
        return { state, label: "teleport blocked (wall)" };
      return {
        state: { ...state, agent: action.cell },
        label: `agent -> cell ${action.cell}`,
      };
    case "pokeValue": {
      const q = clone2d(state.q);
      at(q, action.cell)[action.action] = action.value;
      return {
        state: { ...state, q },
        label: `Q poked at cell ${action.cell}`,
      };
    }
    case "resetLearning":
      return { state: resetLearningState(state), label: "Q reset to zero" };
    case "setSeed":
      return {
        state: { ...resetLearningState(state), rng: action.value >>> 0 },
        label: `seed -> ${action.value}, Q reset`,
      };
    case "loadPreset": {
      const preset = getPreset(action.id);
      return {
        state: { ...initialState(action.id), log: state.log },
        label: `preset -> ${preset.label}`,
      };
    }
    case "restore":
      return {
        state: { ...action.snapshot, log: state.log },
        label: action.label,
      };
  }
}

// The pure reducer. A tick runs one learning step; an intervention changes a
// parameter or the world and logs the moment before it took effect so the event
// log can rewind to just before the change.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return advance(state);
  const before = snapshotOf(state);
  const { state: changed, label } = applyIntervention(state, input.action);
  return logEvent(changed, before, label);
}
