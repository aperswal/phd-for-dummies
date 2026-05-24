// A pure, deterministic model of "Deep Reinforcement Learning from Human
// Preferences" (Christiano et al., 2017), shrunk to a small gridworld so the
// whole feedback loop fits on one screen and a reader can poke it.
//
// The paper runs three processes at once. (1) A policy acts in the world and is
// trained by RL on a predicted reward r-hat. (2) Pairs of short trajectory
// segments (clips) are shown to a human, who marks which one is better. (3) The
// reward predictor r-hat is fit by supervised learning to those comparisons,
// using the Bradley-Terry model: the chance a person prefers clip 1 over clip 2
// rises with the difference of their summed predicted reward.
//
// We keep the paper's real rules. r-hat is a per-cell reward (the analogue of a
// reward network over observations). It is fit online by gradient descent on the
// exact cross-entropy / Bradley-Terry loss the paper minimizes. An ensemble of
// predictors estimates disagreement, which drives the query selection the paper
// uses. The policy is the soft-greedy policy of value iteration run on r-hat.
// The true reward is never shown to the learner; it only labels clips when the
// labeler is the synthetic oracle, and it only scores the final alignment.
//
// Everything is deterministic given the seed and the sequence of inputs. No
// Math.random, no Date.now. Same seed plus same clicks reproduce the run.

import { at } from "@/lib/simulation/array";
import {
  emptyLog,
  record,
  type EventLog,
  type SimEvent,
} from "@/lib/simulation/history";
import { nextRandom } from "@/lib/simulation/rng";

export const GRID = 5;
export const CELLS = GRID * GRID;
export const GOAL = 24; // bottom-right corner
export const CLIP_LEN = 3; // a clip is CLIP_LEN consecutive cells the agent visited
export const ENSEMBLE = 3; // number of reward predictors, for disagreement
const LEARN_RATE = 0.6;
const TRUE_GOAL_REWARD = 1;
const TRUE_STEP_COST = -0.04;

export type Labeler = "oracle" | "you";
export type QueryRule = "disagreement" | "random";

// The hidden true reward the learner never sees: the goal cell pays, every other
// step costs a little. Used only by the oracle to label clips and to score how
// well r-hat has lined up with the truth.
export function trueReward(cell: number): number {
  return cell === GOAL ? TRUE_GOAL_REWARD : TRUE_STEP_COST;
}

function rowCol(cell: number): { row: number; col: number } {
  return { row: Math.floor(cell / GRID), col: cell % GRID };
}

function clampCell(row: number, col: number): number {
  const r = Math.min(Math.max(row, 0), GRID - 1);
  const c = Math.min(Math.max(col, 0), GRID - 1);
  return r * GRID + c;
}

export interface Clip {
  id: number;
  cells: number[]; // the cells the agent occupied during this segment
}

// A pending comparison the human (or oracle) has not yet labeled.
export interface Query {
  id: number;
  a: Clip;
  b: Clip;
  disagreement: number; // ensemble variance on this pair, for query selection
}

// One labeled comparison in the database D. mu is the paper's distribution over
// {1, 2}: all mass on the preferred clip, or split 0.5/0.5 for a tie.
export interface Comparison {
  query: Query;
  mu: [number, number];
}

// The reward predictor is a per-cell scalar table. The ensemble holds ENSEMBLE
// independently initialized tables, normalized and averaged for the estimate,
// exactly the ensembling the paper describes.
export interface Predictor {
  cell: number[];
}

export interface Params {
  labeler: Labeler;
  queryRule: QueryRule;
  mislabelRate: number; // chance the labeler flips its preference (human error)
  queryEvery: number; // virtual steps between new queries (lower = more feedback)
  onlineQueries: boolean; // false freezes querying after the offline batch
}

export interface SimState {
  params: Params;
  rngState: number;
  step: number;
  predictors: Predictor[]; // the ensemble
  rhat: number[]; // averaged, normalized estimate over cells
  agentCell: number; // where the policy currently sits, for the live trajectory
  policy: number[]; // greedy action per cell under rhat, for rendering
  database: Comparison[]; // labeled comparisons gathered so far
  pending: Query | null; // a query awaiting the reader's manual label
  labelCount: number;
  loss: number; // current Bradley-Terry loss on the database
  alignment: number; // correlation of rhat with true reward, in [-1, 1]
  optimalShare: number; // fraction of cells whose greedy action is optimal
  reachedGoal: boolean; // did the live trajectory reach the goal this rollout
  log: EventLog<Snapshot>;
}

export interface Snapshot {
  params: Params;
  rngState: number;
  step: number;
  predictors: Predictor[];
  agentCell: number;
  database: Comparison[];
  labelCount: number;
}

// --- randomness, threaded through state so the run is replayable ---

function draw(state: { rngState: number }): number {
  const next = nextRandom(state.rngState);
  state.rngState = next.state;
  return next.value;
}

function randInt(state: { rngState: number }, n: number): number {
  return Math.floor(draw(state) * n);
}

// --- predictors ---

function makePredictor(seed: number): Predictor {
  let s = seed >>> 0;
  const cell = new Array<number>(CELLS);
  for (let i = 0; i < CELLS; i++) {
    const next = nextRandom(s);
    s = next.state;
    cell[i] = (next.value - 0.5) * 0.2; // small spread so the ensemble disagrees
  }
  return { cell };
}

function makeEnsemble(seed: number): Predictor[] {
  return Array.from({ length: ENSEMBLE }, (_, k) =>
    makePredictor(seed * 9176 + k * 2531 + 17),
  );
}

// Normalize a predictor's cells to zero mean and unit-ish scale, then average
// across the ensemble. The paper normalizes each predictor independently before
// averaging; the same idea here keeps one predictor's offset from dominating.
function averagePredictors(predictors: Predictor[]): number[] {
  const normalized = predictors.map((p) => {
    const mean = p.cell.reduce((s, v) => s + v, 0) / CELLS;
    const variance =
      p.cell.reduce((s, v) => s + (v - mean) * (v - mean), 0) / CELLS;
    const sd = Math.sqrt(variance) || 1;
    return p.cell.map((v) => (v - mean) / sd);
  });
  return Array.from({ length: CELLS }, (_, c) => {
    let sum = 0;
    for (const n of normalized) sum += at(n, c);
    return sum / ENSEMBLE;
  });
}

function clipScore(predictor: Predictor, clip: Clip): number {
  let sum = 0;
  for (const cell of clip.cells) sum += at(predictor.cell, cell);
  return sum;
}

function sigmoid(x: number): number {
  if (x < -40) return 0;
  if (x > 40) return 1;
  return 1 / (1 + Math.exp(-x));
}

// Equation 1 of the paper: the predicted probability that clip a is preferred to
// clip b is the softmax (here, sigmoid) of the difference of their summed
// predicted reward.
export function preferProbability(
  predictor: Predictor,
  a: Clip,
  b: Clip,
): number {
  return sigmoid(clipScore(predictor, a) - clipScore(predictor, b));
}

// One gradient step of the Bradley-Terry cross-entropy loss for one comparison,
// applied to one predictor. d loss / d score = (predicted - target), and each
// cell in clip a gets +(p - mu1), each cell in clip b gets the opposite sign,
// because the score is a plain sum over the clip's cells.
function trainStep(predictor: Predictor, comp: Comparison): void {
  const p = preferProbability(predictor, comp.query.a, comp.query.b);
  const gradient = p - comp.mu[0]; // positive means we over-predicted "a wins"
  for (const cell of comp.query.a.cells) {
    predictor.cell[cell] = at(predictor.cell, cell) - LEARN_RATE * gradient;
  }
  for (const cell of comp.query.b.cells) {
    predictor.cell[cell] = at(predictor.cell, cell) + LEARN_RATE * gradient;
  }
}

function databaseLoss(predictors: Predictor[], db: Comparison[]): number {
  if (db.length === 0) return 0;
  const rhat = averagePredictors(predictors);
  const board: Predictor = { cell: rhat };
  let total = 0;
  for (const comp of db) {
    const p = Math.min(
      Math.max(preferProbability(board, comp.query.a, comp.query.b), 1e-6),
      1 - 1e-6,
    );
    total += -(comp.mu[0] * Math.log(p) + comp.mu[1] * Math.log(1 - p));
  }
  return total / db.length;
}

// --- clips, queries, labels ---

// Build a plausible clip: start somewhere, then walk CLIP_LEN cells either
// toward or away from the goal, so some clips genuinely earn more true reward
// than others and the comparison carries signal.
function sampleClip(state: { rngState: number }, id: number): Clip {
  const start = randInt(state, CELLS);
  const towardGoal = draw(state) < 0.5;
  const cells = [start];
  let current = start;
  for (let i = 1; i < CLIP_LEN; i++) {
    const { row, col } = rowCol(current);
    const goal = rowCol(GOAL);
    const stepRow = towardGoal
      ? Math.sign(goal.row - row)
      : -Math.sign(goal.row - row);
    const stepCol = towardGoal
      ? Math.sign(goal.col - col)
      : -Math.sign(goal.col - col);
    // move one axis at a time, picking the row or column step at random
    const moveRow = draw(state) < 0.5 && stepRow !== 0;
    current = moveRow
      ? clampCell(row + stepRow, col)
      : clampCell(row, col + (stepCol !== 0 ? stepCol : stepRow));
    cells.push(current);
  }
  return { id, cells };
}

function clipTrueReturn(clip: Clip): number {
  let sum = 0;
  for (const cell of clip.cells) sum += trueReward(cell);
  return sum;
}

// Ensemble disagreement on a pair: variance of each predictor's preference
// probability. The paper queries the pairs with the highest variance across the
// ensemble.
function ensembleDisagreement(
  predictors: Predictor[],
  a: Clip,
  b: Clip,
): number {
  const ps = predictors.map((p) => preferProbability(p, a, b));
  const mean = ps.reduce((s, v) => s + v, 0) / ps.length;
  return ps.reduce((s, v) => s + (v - mean) * (v - mean), 0) / ps.length;
}

// Build one query. With the disagreement rule, sample several candidate pairs
// and keep the one the ensemble splits on most; with random, take the first.
function buildQuery(state: SimState, id: number): Query {
  const candidates = state.params.queryRule === "disagreement" ? 6 : 1;
  let best: Query | null = null;
  for (let i = 0; i < candidates; i++) {
    const a = sampleClip(state, id * 100 + i * 2);
    const b = sampleClip(state, id * 100 + i * 2 + 1);
    const disagreement = ensembleDisagreement(state.predictors, a, b);
    if (best === null || disagreement > best.disagreement) {
      best = { id, a, b, disagreement };
    }
  }
  return best as Query;
}

// The oracle labels by the true summed reward of each clip, with a configurable
// chance of flipping (human error). A genuine tie gets a split label.
function oracleLabel(state: SimState, query: Query): [number, number] {
  const ra = clipTrueReturn(query.a);
  const rb = clipTrueReturn(query.b);
  if (Math.abs(ra - rb) < 1e-9) return [0.5, 0.5];
  let prefersA = ra > rb;
  if (draw(state) < state.params.mislabelRate) prefersA = !prefersA;
  return prefersA ? [1, 0] : [0, 1];
}

// --- policy: value iteration on r-hat ---

const ACTIONS = [
  { dr: -1, dc: 0 },
  { dr: 1, dc: 0 },
  { dr: 0, dc: -1 },
  { dr: 0, dc: 1 },
] as const;

function neighbor(cell: number, action: number): number {
  const { row, col } = rowCol(cell);
  const move = at(ACTIONS, action);
  return clampCell(row + move.dr, col + move.dc);
}

// Greedy action per cell from value iteration on a per-cell reward table. The
// next cell's reward plus discounted value decides the move, which is exactly
// "maximize the sum of predicted reward" at the policy level.
function valueIteration(reward: number[]): {
  policy: number[];
  values: number[];
} {
  const gamma = 0.9;
  const values = new Array<number>(CELLS).fill(0);
  for (let sweep = 0; sweep < 60; sweep++) {
    const next = values.slice();
    for (let cell = 0; cell < CELLS; cell++) {
      if (cell === GOAL) {
        next[cell] = at(reward, cell);
        continue;
      }
      let best = -Infinity;
      for (let action = 0; action < ACTIONS.length; action++) {
        const n = neighbor(cell, action);
        const q = at(reward, n) + gamma * at(values, n);
        if (q > best) best = q;
      }
      next[cell] = best;
    }
    for (let c = 0; c < CELLS; c++) values[c] = at(next, c);
  }
  const policy = new Array<number>(CELLS).fill(0);
  for (let cell = 0; cell < CELLS; cell++) {
    let best = -Infinity;
    let bestAction = 0;
    for (let action = 0; action < ACTIONS.length; action++) {
      const n = neighbor(cell, action);
      const q = at(reward, n) + gamma * at(values, n);
      if (q > best) {
        best = q;
        bestAction = action;
      }
    }
    policy[cell] = bestAction;
  }
  return { policy, values };
}

// --- alignment scoring (uses the hidden truth, for the readout only) ---

function correlation(a: number[], b: number[]): number {
  const n = a.length;
  const ma = a.reduce((s, v) => s + v, 0) / n;
  const mb = b.reduce((s, v) => s + v, 0) / n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i++) {
    const da = at(a, i) - ma;
    const db = at(b, i) - mb;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  const denom = Math.sqrt(va * vb) || 1;
  return cov / denom;
}

const TRUE_TABLE = Array.from({ length: CELLS }, (_, c) => trueReward(c));
const TRUE_POLICY = valueIteration(TRUE_TABLE).policy;

function policyShare(policy: number[]): number {
  let match = 0;
  for (let c = 0; c < CELLS; c++) {
    if (c === GOAL) continue;
    if (at(policy, c) === at(TRUE_POLICY, c)) match += 1;
  }
  return match / (CELLS - 1);
}

// --- recompute the derived fields after any change to predictors ---

function refresh(state: SimState): void {
  state.rhat = averagePredictors(state.predictors);
  const { policy } = valueIteration(state.rhat);
  state.policy = policy;
  state.loss = databaseLoss(state.predictors, state.database);
  state.alignment = correlation(state.rhat, TRUE_TABLE);
  state.optimalShare = policyShare(policy);
}

// Roll the live agent one cell forward under the current greedy policy, so the
// reader watches the trajectory the learned reward induces. Resets at the goal.
function advanceAgent(state: SimState): void {
  if (state.agentCell === GOAL) {
    state.reachedGoal = true;
    state.agentCell = 0;
    return;
  }
  const action = at(state.policy, state.agentCell);
  state.agentCell = neighbor(state.agentCell, action);
}

// --- the public reducer surface ---

export type Intervention =
  | { type: "label"; prefer: "a" | "b" | "tie" }
  | { type: "setLabeler"; value: Labeler }
  | { type: "setQueryRule"; value: QueryRule }
  | { type: "setMislabel"; value: number }
  | { type: "setQueryEvery"; value: number }
  | { type: "toggleOnline" }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: Snapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: Params;
}

export const PRESETS: Preset[] = [
  {
    id: "oracle",
    label: "Synthetic oracle",
    blurb:
      "An automatic labeler picks the clip with higher true reward, the paper's baseline. Press play and watch the reward map fill in and the policy line up with the hidden truth.",
    params: {
      labeler: "oracle",
      queryRule: "disagreement",
      mislabelRate: 0,
      queryEvery: 4,
      onlineQueries: true,
    },
  },
  {
    id: "you",
    label: "You are the labeler",
    blurb:
      "You decide which clip is better. The run pauses on each query; click the clip that ends nearer the goal, then resume.",
    params: {
      labeler: "you",
      queryRule: "disagreement",
      mislabelRate: 0,
      queryEvery: 4,
      onlineQueries: true,
    },
  },
  {
    id: "offline",
    label: "Offline only",
    blurb:
      "Gather a batch of clips at the start, then freeze querying. The reward fits a stale slice of the world and the policy games that proxy, the paper's offline failure.",
    params: {
      labeler: "oracle",
      queryRule: "disagreement",
      mislabelRate: 0,
      queryEvery: 4,
      onlineQueries: false,
    },
  },
  {
    id: "noisy",
    label: "Noisy human",
    blurb:
      "The labeler flips its choice 30% of the time, like a tired contractor. Watch the reward map and the policy share wobble instead of locking in.",
    params: {
      labeler: "oracle",
      queryRule: "disagreement",
      mislabelRate: 0.3,
      queryEvery: 4,
      onlineQueries: true,
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

function snapshotOf(state: SimState): Snapshot {
  return {
    params: { ...state.params },
    rngState: state.rngState,
    step: state.step,
    predictors: state.predictors.map((p) => ({ cell: p.cell.slice() })),
    agentCell: state.agentCell,
    database: state.database.map((c) => ({ ...c })),
    labelCount: state.labelCount,
  };
}

const BASE_SEED = 1337;

export function initialState(presetId = "oracle"): SimState {
  const params = { ...getPreset(presetId).params };
  const predictors = makeEnsemble(BASE_SEED);
  const state: SimState = {
    params,
    rngState: BASE_SEED >>> 0,
    step: 0,
    predictors,
    rhat: averagePredictors(predictors),
    agentCell: 0,
    policy: new Array<number>(CELLS).fill(3),
    database: [],
    pending: null,
    labelCount: 0,
    loss: 0,
    alignment: 0,
    optimalShare: 0,
    reachedGoal: false,
    log: emptyLog<Snapshot>(),
  };
  refresh(state);
  return state;
}

function logged(state: SimState, label: string): SimState {
  return {
    ...state,
    log: record(state.log, state.step, label, snapshotOf(state)),
  };
}

// Commit one labeled comparison into the database and train every predictor on
// it a few passes, the supervised reward-fitting process.
function commitLabel(
  state: SimState,
  query: Query,
  mu: [number, number],
): void {
  const comp: Comparison = { query, mu };
  state.database = [...state.database.slice(-49), comp];
  state.labelCount += 1;
  for (let pass = 0; pass < 4; pass++) {
    for (const predictor of state.predictors) trainStep(predictor, comp);
  }
  refresh(state);
}

function offlineBatchDone(state: SimState): boolean {
  return !state.params.onlineQueries && state.labelCount >= 12;
}

function tick(state: SimState): SimState {
  const next: SimState = { ...state, step: state.step + 1 };
  advanceAgent(next);

  const dueForQuery = next.step % Math.max(1, next.params.queryEvery) === 0;
  const stopQuerying = offlineBatchDone(next);

  if (dueForQuery && !stopQuerying && next.pending === null) {
    const query = buildQuery(next, next.step);
    if (next.params.labeler === "you") {
      next.pending = query;
      return logged(next, `query #${query.id} awaiting your label`);
    }
    const mu = oracleLabel(next, query);
    commitLabel(next, query, mu);
    const winner =
      mu[0] === mu[1] ? "tie" : mu[0] > mu[1] ? "clip A" : "clip B";
    return logged(next, `oracle labeled query #${query.id}: ${winner}`);
  }

  return next;
}

function applyManualLabel(
  state: SimState,
  prefer: "a" | "b" | "tie",
): SimState {
  if (state.pending === null) return state;
  const query = state.pending;
  const mu: [number, number] =
    prefer === "tie" ? [0.5, 0.5] : prefer === "a" ? [1, 0] : [0, 1];
  const next: SimState = { ...state, pending: null };
  commitLabel(next, query, mu);
  const which = prefer === "tie" ? "tie" : prefer === "a" ? "clip A" : "clip B";
  return logged(next, `you labeled query #${query.id}: ${which}`);
}

function withParams(state: SimState, params: Params, label: string): SimState {
  const next: SimState = { ...state, params };
  refresh(next);
  return logged(next, label);
}

function applyIntervention(state: SimState, action: Intervention): SimState {
  switch (action.type) {
    case "label":
      return applyManualLabel(state, action.prefer);
    case "setLabeler":
      return withParams(
        { ...state, pending: null },
        { ...state.params, labeler: action.value },
        `labeler -> ${action.value}`,
      );
    case "setQueryRule":
      return withParams(
        state,
        { ...state.params, queryRule: action.value },
        `query rule -> ${action.value}`,
      );
    case "setMislabel":
      return withParams(
        state,
        { ...state.params, mislabelRate: action.value },
        `mislabel rate -> ${Math.round(action.value * 100)}%`,
      );
    case "setQueryEvery":
      return withParams(
        state,
        { ...state.params, queryEvery: action.value },
        `query every -> ${action.value} steps`,
      );
    case "toggleOnline":
      return withParams(
        state,
        { ...state.params, onlineQueries: !state.params.onlineQueries },
        `online queries ${!state.params.onlineQueries ? "on" : "off"}`,
      );
    case "loadPreset": {
      const preset = getPreset(action.id);
      const fresh = initialState(action.id);
      return logged(fresh, `preset -> ${preset.label}`);
    }
    case "restore": {
      const snap = action.snapshot;
      const restored: SimState = {
        ...state,
        params: { ...snap.params },
        rngState: snap.rngState,
        step: snap.step,
        predictors: snap.predictors.map((p) => ({ cell: p.cell.slice() })),
        agentCell: snap.agentCell,
        database: snap.database.map((c) => ({ ...c })),
        pending: null,
        labelCount: snap.labelCount,
        reachedGoal: false,
      };
      refresh(restored);
      return logged(restored, action.label);
    }
  }
}

// The pure, deterministic reducer. A tick advances the live agent and, when due,
// generates and labels a query; an intervention changes one knob or commits a
// manual label. Every transition is replayable from the seed and the inputs.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return tick(state);
  return applyIntervention(state, input.action);
}

export type PreferenceEvent = SimEvent<Snapshot>;
