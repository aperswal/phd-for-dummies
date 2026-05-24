// A pure, deterministic model of Deep Q-learning with experience replay from
// "Playing Atari with Deep Reinforcement Learning". The training loop is exactly
// Algorithm 1: behave epsilon-greedily, store each transition in a replay memory,
// sample a minibatch, regress the network toward the bootstrapped target
// y = r + gamma * max_a' Q(s', a'), and take a gradient step on (y - Q)^2.
//
// Two honest simplifications stand in for the paper's scale. The game is a tiny
// deterministic catch grid instead of an Atari screen, and the Q-network is a
// linear approximator over coarse, deliberately overlapping tile features
// instead of a convolutional net. The overlap matters: it recreates the exact
// instability the paper warns about (function approximation + bootstrapping +
// off-policy learning, the "deadly triad" the paper cites as a divergence risk).
// With replay off, consecutive correlated samples push the shared weights into a
// feedback loop and the value estimate blows up. With replay on, random
// minibatches average over many past behaviours and the estimate settles. That
// is the paper's central stability claim, reproduced where a reader can poke it.
//
// No Math.random and no wall clock reach the model. A seeded mulberry32 cursor
// lives inside the state, so the same seed plus the same inputs reproduce the
// exact same run, which makes scrubbing and "what just happened" honest.

import { at } from "@/lib/simulation/array";
import {
  emptyLog,
  record,
  type EventLog,
  type SimEvent,
} from "@/lib/simulation/history";

// ---------------------------------------------------------------------------
// Seeded randomness, threaded through immutable state.
// ---------------------------------------------------------------------------

// One step of mulberry32 as a pure function: given the cursor, return the next
// value in [0, 1) and the next cursor. Threading the cursor through state keeps
// the whole reducer deterministic.
function nextRandom(cursor: number): { value: number; cursor: number } {
  const a = (cursor + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, cursor: a };
}

function randInt(cursor: number, n: number): { value: number; cursor: number } {
  const draw = nextRandom(cursor);
  return { value: Math.floor(draw.value * n), cursor: draw.cursor };
}

// ---------------------------------------------------------------------------
// The environment: a tiny deterministic "catch" game.
//
// A ball sits in one of GRID_W columns at the top. A paddle sits at the bottom
// and moves left, stays, or moves right. Each step the ball drops one row. When
// it reaches the bottom row, the paddle catches it (reward +1) if their columns
// match, else misses it (reward -1). Then a new ball spawns. The state is the
// (ballColumn, paddleColumn) pair, which the agent walks through in correlated
// runs exactly the way an Atari agent sees correlated consecutive frames.
// ---------------------------------------------------------------------------

export const GRID_W = 5;
export const GRID_H = 5;
export const ACTIONS = ["left", "stay", "right"] as const;
export type ActionName = (typeof ACTIONS)[number];
export const ACTION_COUNT = ACTIONS.length;

export interface GameState {
  ballCol: number;
  ballRow: number;
  paddleCol: number;
}

// The features the linear Q-network sees. They are deliberately coarse and
// overlapping, not a clean one-hot per state. The paddle column is binned into
// three wide regions (left / middle / right) and the ball into three regions
// too, so many distinct game states collapse onto the same feature vector. This
// aliasing is the crux: a gradient step taken at one state spills onto every
// other state that shares its bin, which is exactly the harmful generalization a
// neural network does and a tabular lookup does not. Combined with bootstrapping
// and off-policy behaviour it is the "deadly triad" the paper cites as a
// divergence risk and tames with experience replay.
// The features the linear Q-network sees. The paddle and ball columns are
// one-hot so the task is genuinely learnable, but a single shared "relation"
// tile (ball left of, on, or right of the paddle) is added on top and weighted
// heavily. That shared tile is the harmful generalization: nudging it at one
// state nudges the value of every state in the same relation, and because the
// bootstrap target reads that same inflated tile the update can feed back on
// itself. With near-greedy off-policy behaviour and bootstrapping that is the
// deadly triad the paper cites as a divergence risk. Experience replay tames it
// by averaging the shared tile's update over transitions from many relations.
const REL_BINS = 3;
const REL_WEIGHT = 2;
export const FEATURE_DIM = GRID_W + GRID_W + GRID_H + REL_BINS + 1;

function relationBin(game: GameState): number {
  const rel = game.ballCol - game.paddleCol;
  return rel < 0 ? 0 : rel > 0 ? 2 : 1;
}

export function featurize(game: GameState): number[] {
  const f = new Array<number>(FEATURE_DIM).fill(0);
  let base = 0;
  f[base + game.paddleCol] = 1;
  base += GRID_W;
  f[base + game.ballCol] = 1;
  base += GRID_W;
  f[base + game.ballRow] = 1;
  base += GRID_H;
  f[base + relationBin(game)] = REL_WEIGHT;
  base += REL_BINS;
  f[base] = 1; // bias
  return f;
}

function spawnBall(cursor: number): { ballCol: number; cursor: number } {
  const draw = randInt(cursor, GRID_W);
  return { ballCol: draw.value, cursor: draw.cursor };
}

function stepGame(
  game: GameState,
  action: number,
  cursor: number,
): { next: GameState; reward: number; done: boolean; cursor: number } {
  const move = action - 1; // left = -1, stay = 0, right = +1
  const paddleCol = Math.min(Math.max(game.paddleCol + move, 0), GRID_W - 1);
  const ballRow = game.ballRow + 1;

  if (ballRow >= GRID_H - 1) {
    const reward = game.ballCol === paddleCol ? 1 : -1;
    const spawn = spawnBall(cursor);
    const next: GameState = {
      ballCol: spawn.ballCol,
      ballRow: 0,
      paddleCol,
    };
    return { next, reward, done: true, cursor: spawn.cursor };
  }

  return {
    next: { ballCol: game.ballCol, ballRow, paddleCol },
    reward: 0,
    done: false,
    cursor,
  };
}

// ---------------------------------------------------------------------------
// The linear Q-network. weights[action] is the weight vector for one action's
// output unit, exactly the paper's choice of one output per action so a single
// forward pass scores every action. Q(s, a) = w_a . phi(s).
// ---------------------------------------------------------------------------

type Weights = number[][]; // [action][feature]

function zeroWeights(): Weights {
  return Array.from({ length: ACTION_COUNT }, () =>
    new Array<number>(FEATURE_DIM).fill(0),
  );
}

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += at(a, i) * at(b, i);
  return sum;
}

export function qValues(weights: Weights, features: number[]): number[] {
  return weights.map((w) => dot(w, features));
}

function greedyAction(weights: Weights, features: number[]): number {
  const q = qValues(weights, features);
  let best = 0;
  for (let a = 1; a < q.length; a++) if (at(q, a) > at(q, best)) best = a;
  return best;
}

// ---------------------------------------------------------------------------
// The replay memory: a fixed-capacity ring buffer of transitions, overwriting
// oldest first exactly like the paper's last-N memory.
// ---------------------------------------------------------------------------

export interface Transition {
  features: number[];
  action: number;
  reward: number;
  nextFeatures: number[];
  done: boolean;
}

// ---------------------------------------------------------------------------
// Tunable parameters the reader controls live.
// ---------------------------------------------------------------------------

export interface Params {
  replayOn: boolean;
  bufferCapacity: number;
  batchSize: number;
  epsilon: number;
  gamma: number;
  learningRate: number;
  forceColumn: number; // -1 = off; otherwise pin the ball to this column
}

// The learning rate sits in the regime the paper's stability argument lives in:
// high enough that correlated online updates run away, but replay over a diverse
// buffer keeps the same rate bounded. Drop it toward 0.05 and both paths stay
// calm, which is its own lesson.
export const DEFAULT_PARAMS: Params = {
  replayOn: true,
  bufferCapacity: 400,
  batchSize: 16,
  epsilon: 0.1,
  gamma: 0.95,
  learningRate: 0.3,
  forceColumn: -1,
};

// ---------------------------------------------------------------------------
// The full simulation state.
// ---------------------------------------------------------------------------

export interface Metrics {
  step: number;
  loss: number;
  maxAbsQ: number;
  diverged: boolean;
  returnsWindow: number[]; // recent episode returns, newest last
  qHistory: number[]; // recent maxAbsQ samples for the plot
  lossHistory: number[];
}

export interface Snapshot {
  params: Params;
  step: number;
}

export interface SimState {
  params: Params;
  weights: Weights;
  game: GameState;
  buffer: Transition[];
  cursor: number;
  episodeReturn: number;
  metrics: Metrics;
  log: EventLog<Snapshot>;
}

const DIVERGENCE_LIMIT = 1e4;
const PLOT_WINDOW = 120;
const RETURN_WINDOW = 12;

export type SimEventEntry = SimEvent<Snapshot>;

function initialGame(cursor: number): { game: GameState; cursor: number } {
  const spawn = spawnBall(cursor);
  return {
    game: { ballCol: spawn.ballCol, ballRow: 0, paddleCol: 2 },
    cursor: spawn.cursor,
  };
}

export function initialState(presetId = "stable"): SimState {
  const preset = getPreset(presetId);
  const seedCursor = 0x9e3779b9 ^ presetId.length;
  const start = initialGame(seedCursor);
  return {
    params: { ...preset.params },
    weights: zeroWeights(),
    game: start.game,
    buffer: [],
    cursor: start.cursor,
    episodeReturn: 0,
    metrics: {
      step: 0,
      loss: 0,
      maxAbsQ: 0,
      diverged: false,
      returnsWindow: [],
      qHistory: [],
      lossHistory: [],
    },
    log: emptyLog(),
  };
}

// One SGD update on a batch of transitions, computed exactly like minibatch
// gradient descent. The gradient of (y - Q(s,a))^2 with respect to the taken
// action's weights is -2 (y - Q) phi(s); the batch averages those per-sample
// gradients into one update. A single transition (online Q-learning, replay off)
// is just a batch of size one. So both paths take one update per environment
// step, and the only difference is decorrelation: the batch averages over many
// past behaviours, while the single sample is whatever the agent just lived
// through, correlated with the last. The targets are computed against the
// weights at the start of the batch, matching the paper's held-fixed targets.
function trainOnBatch(
  weights: Weights,
  batch: Transition[],
  params: Params,
): { weights: number[][]; loss: number; diverged: boolean } {
  const gradient = zeroWeights();
  let lossSum = 0;
  for (const t of batch) {
    const predicted = dot(at(weights, t.action), t.features);
    const nextQ = t.done ? 0 : Math.max(...qValues(weights, t.nextFeatures));
    const target = t.reward + (t.done ? 0 : params.gamma * nextQ);
    const tdError = target - predicted;
    lossSum += tdError * tdError;
    const row = at(gradient, t.action);
    for (let i = 0; i < FEATURE_DIM; i++) {
      row[i] = at(row, i) + tdError * at(t.features, i);
    }
  }
  const scale = params.learningRate / Math.max(batch.length, 1);
  const next = weights.map((w, a) =>
    w.map((value, i) => value + scale * at(at(gradient, a), i)),
  );
  const maxAbs = next.reduce(
    (m, row) => Math.max(m, ...row.map((v) => Math.abs(v))),
    0,
  );
  return {
    weights: next,
    loss: lossSum / Math.max(batch.length, 1),
    diverged: !Number.isFinite(maxAbs) || maxAbs > DIVERGENCE_LIMIT,
  };
}

function sampleBatch(
  buffer: Transition[],
  size: number,
  cursor: number,
): { batch: Transition[]; cursor: number } {
  const batch: Transition[] = [];
  let c = cursor;
  for (let i = 0; i < size; i++) {
    const draw = randInt(c, buffer.length);
    c = draw.cursor;
    batch.push(at(buffer, draw.value));
  }
  return { batch, cursor: c };
}

function pushTransition(
  buffer: Transition[],
  transition: Transition,
  capacity: number,
): Transition[] {
  const next = [...buffer, transition];
  return next.length > capacity ? next.slice(next.length - capacity) : next;
}

function maxAbsWeight(weights: Weights): number {
  return weights.reduce(
    (m, row) => Math.max(m, ...row.map((v) => Math.abs(v))),
    0,
  );
}

// One environment step plus one learning update. This is the inner loop of
// Algorithm 1 run once.
function advance(state: SimState): SimState {
  if (state.metrics.diverged) return state;

  let cursor = state.cursor;
  const params = state.params;

  // Optionally pin the ball to one column. This is the reader's "force the
  // behaviour into one region" intervention, which starves replay's diversity
  // and is the data-distribution shift the paper says online learning chases.
  let game = state.game;
  if (params.forceColumn >= 0 && game.ballRow === 0) {
    game = { ...game, ballCol: params.forceColumn };
  }

  const features = featurize(game);

  // Epsilon-greedy behaviour.
  const explore = nextRandom(cursor);
  cursor = explore.cursor;
  let action: number;
  if (explore.value < params.epsilon) {
    const pick = randInt(cursor, ACTION_COUNT);
    cursor = pick.cursor;
    action = pick.value;
  } else {
    action = greedyAction(state.weights, features);
  }

  const outcome = stepGame(game, action, cursor);
  cursor = outcome.cursor;
  if (params.forceColumn >= 0 && outcome.done && outcome.next.ballRow === 0) {
    outcome.next.ballCol = params.forceColumn;
  }
  const nextFeatures = featurize(outcome.next);

  const transition: Transition = {
    features,
    action,
    reward: outcome.reward,
    nextFeatures,
    done: outcome.done,
  };
  const buffer = pushTransition(
    state.buffer,
    transition,
    params.bufferCapacity,
  );

  // Replay on: sample a random minibatch from the whole buffer and take one
  // averaged gradient step. Replay off: take that step on the single transition
  // just seen, which is online Q-learning on correlated samples, the unstable
  // case. Both take exactly one update per step; only the sample diversity
  // differs.
  let batch: Transition[];
  if (params.replayOn) {
    const sampled = sampleBatch(buffer, params.batchSize, cursor);
    cursor = sampled.cursor;
    batch = sampled.batch;
  } else {
    batch = [transition];
  }
  const trained = trainOnBatch(state.weights, batch, params);
  const weights = trained.weights;
  const lossSum = trained.loss;
  const diverged = trained.diverged;

  const episodeReturn = state.episodeReturn + outcome.reward;
  const returnsWindow = outcome.done
    ? [...state.metrics.returnsWindow, episodeReturn].slice(-RETURN_WINDOW)
    : state.metrics.returnsWindow;

  const maxAbsQ = maxAbsWeight(weights);
  const stepCount = state.metrics.step + 1;

  return {
    params,
    weights,
    game: outcome.next,
    buffer,
    cursor,
    episodeReturn: outcome.done ? 0 : episodeReturn,
    metrics: {
      step: stepCount,
      loss: lossSum,
      maxAbsQ,
      diverged: diverged || state.metrics.diverged,
      returnsWindow,
      qHistory: [...state.metrics.qHistory, maxAbsQ].slice(-PLOT_WINDOW),
      lossHistory: [
        ...state.metrics.lossHistory,
        Math.min(lossSum, DIVERGENCE_LIMIT),
      ].slice(-PLOT_WINDOW),
    },
    log: state.log,
  };
}

// ---------------------------------------------------------------------------
// Interventions and the reducer.
// ---------------------------------------------------------------------------

export type Intervention =
  | { type: "toggleReplay" }
  | { type: "setBufferCapacity"; value: number }
  | { type: "setBatchSize"; value: number }
  | { type: "setEpsilon"; value: number }
  | { type: "setGamma"; value: number }
  | { type: "setLearningRate"; value: number }
  | { type: "setForceColumn"; value: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: Snapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

function logged(state: SimState, label: string): SimState {
  return {
    ...state,
    log: record(state.log, state.metrics.step, label, {
      params: { ...state.params },
      step: state.metrics.step,
    }),
  };
}

function applyParams(state: SimState, params: Params, label: string): SimState {
  return logged({ ...state, params }, label);
}

// Restoring a snapshot replays the run from a fresh start under the snapshot's
// parameters. Because the model is deterministic, fast-forwarding to the
// snapshot's step reproduces that exact moment.
function restore(state: SimState, snapshot: Snapshot, label: string): SimState {
  let rebuilt = initialState();
  rebuilt = { ...rebuilt, params: { ...snapshot.params } };
  for (let i = 0; i < snapshot.step; i++) rebuilt = advance(rebuilt);
  return logged({ ...rebuilt, log: state.log }, label);
}

function describeForce(value: number): string {
  return value < 0 ? "off" : `column ${value}`;
}

function applyIntervention(state: SimState, action: Intervention): SimState {
  switch (action.type) {
    case "toggleReplay": {
      const replayOn = !state.params.replayOn;
      return applyParams(
        state,
        { ...state.params, replayOn },
        `experience replay ${replayOn ? "on" : "off"}`,
      );
    }
    case "setBufferCapacity":
      return applyParams(
        state,
        { ...state.params, bufferCapacity: action.value },
        `buffer capacity -> ${action.value}`,
      );
    case "setBatchSize":
      return applyParams(
        state,
        { ...state.params, batchSize: action.value },
        `minibatch -> ${action.value}`,
      );
    case "setEpsilon":
      return applyParams(
        state,
        { ...state.params, epsilon: action.value },
        `epsilon -> ${action.value.toFixed(2)}`,
      );
    case "setGamma":
      return applyParams(
        state,
        { ...state.params, gamma: action.value },
        `gamma -> ${action.value.toFixed(2)}`,
      );
    case "setLearningRate":
      return applyParams(
        state,
        { ...state.params, learningRate: action.value },
        `learning rate -> ${action.value.toFixed(3)}`,
      );
    case "setForceColumn":
      return applyParams(
        state,
        { ...state.params, forceColumn: action.value },
        `force ball -> ${describeForce(action.value)}`,
      );
    case "loadPreset": {
      const preset = getPreset(action.id);
      const fresh = initialState(action.id);
      return logged(fresh, `preset -> ${preset.label}`);
    }
    case "restore":
      return restore(state, action.snapshot, action.label);
  }
}

// The pure reducer. A tick runs one inner-loop step of Algorithm 1; an
// intervention changes a parameter and records a rewindable snapshot.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return advance(state);
  return applyIntervention(state, input.action);
}

// ---------------------------------------------------------------------------
// Presets: the happy path and the failure cases the paper warns about.
// ---------------------------------------------------------------------------

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: Params;
}

export const PRESETS: Preset[] = [
  {
    id: "stable",
    label: "Stable (replay on)",
    blurb:
      "Algorithm 1 as published. Random minibatches from the replay memory keep the value estimate bounded while the paddle learns to catch.",
    params: { ...DEFAULT_PARAMS },
  },
  {
    id: "diverge",
    label: "Divergence (replay off)",
    blurb:
      "Online Q-learning on consecutive correlated samples at the same learning rate. The shared relation tile feeds the bootstrap back on itself and the value estimate runs away, the deadly triad the paper cites.",
    params: { ...DEFAULT_PARAMS, replayOn: false },
  },
  {
    id: "tiny-buffer",
    label: "Buffer too small",
    blurb:
      "Replay on, but the memory holds only the last few transitions, so a sampled batch is still nearly all correlated. The averaging that keeps the value bounded has almost nothing to average.",
    params: { ...DEFAULT_PARAMS, bufferCapacity: 2, batchSize: 2 },
  },
  {
    id: "shift",
    label: "Distribution shift",
    blurb:
      "The ball is pinned to one column, so the agent only ever sees one slice of the game. Replay can only shuffle what it has stored.",
    params: { ...DEFAULT_PARAMS, forceColumn: 0 },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function meanReturn(window: number[]): number {
  if (window.length === 0) return 0;
  return window.reduce((s, v) => s + v, 0) / window.length;
}
