// A pure, deterministic model of generalized advantage estimation (GAE) from
// Schulman, Moritz, Levine, Jordan and Abbeel's 2016 paper
// "High-Dimensional Continuous Control Using Generalized Advantage Estimation".
// A softmax (Gibbs) policy with one preference theta(s,a) per state-action runs
// on a small gridworld MDP. Each tick samples a batch of episodes, estimates the
// advantage at every visited step with GAE(gamma, lambda), and takes one
// policy-gradient ascent step on the expected return.
//
// The estimator is the paper's central object, Equation (16):
//
//   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)              (the TD residual)
//   A_hat_t = sum_{l>=0} (gamma * lambda)^l * delta_{t+l}    (GAE)
//
// lambda slides between two limits the paper names explicitly:
//   - lambda = 0: A_hat_t = delta_t. Low variance, but biased unless V is exact.
//   - lambda = 1: A_hat_t = sum_l gamma^l r_{t+l} - V(s_t). Unbiased for any V
//                 (gamma-just), but high variance from the long reward sum.
// Intermediate lambda trades the two off, which is the paper's main empirical
// finding (best results at lambda in [0.92, 0.99]).
//
// The reader controls the value function V the estimator subtracts:
//   - exact:    V = V^{pi,gamma}, solved on the known MDP (the ground truth).
//   - noisy:    exact V plus seeded per-state Gaussian error, scaled by a knob.
//   - zero:     V = 0 everywhere, the worst baseline.
// When V is wrong, lambda = 0 is heavily biased and the climb stalls, while
// lambda = 1 stays unbiased and only suffers variance. The model also computes
// the EXACT advantage A^{pi,gamma} separately, used only to score the estimator's
// bias (gap to the true gradient direction) and variance across the batch.
//
// All randomness comes from a seeded stream threaded through the state, so a seed
// plus a list of inputs reproduces a run exactly. No Math.random, no wall clock.

import { at } from "@/lib/simulation/array";
import { emptyLog, record, type EventLog } from "@/lib/simulation/history";
import {
  ACTION_LABELS,
  clamp,
  colOf,
  discountedVisitation,
  enterReward,
  evaluatePolicy,
  GRID_W,
  isTerminal,
  type Mdp,
  N_ACTIONS,
  N_STATES,
  nextState,
  oneHot,
  optimalReturn,
  policyTransition,
  rowOf,
  solveLinear,
} from "@/lib/simulation/mdp";
import { nextRandom } from "@/lib/simulation/rng";

export { ACTION_LABELS, colOf, GRID_W, isTerminal, N_ACTIONS, N_STATES, rowOf };

const THETA_CLAMP = 25;
const HISTORY_LEN = 80;
const EPISODE_MAX_STEPS = 60;

export type ValueMode = "exact" | "noisy" | "zero";

export interface Params {
  lambda: number;
  gamma: number;
  valueMode: ValueMode;
  valueError: number; // std of the per-state Gaussian noise on V (noisy mode)
  episodes: number; // episodes sampled per ascent step
  alpha: number; // ascent learning rate
}

export interface SimState {
  mdp: Mdp;
  params: Params;
  theta: number[]; // length N_STATES * N_ACTIONS
  pi: number[][]; // [state][action]
  trueValues: number[]; // V^{pi,gamma} per state (ground truth)
  trueAdvantage: number[][]; // A^{pi,gamma}[state][action]
  estValue: number[]; // the V the estimator actually uses (exact / noisy / zero)
  exactGrad: number[]; // flat, the true policy gradient (for scoring only)
  expectedGrad: number[]; // flat, the estimator's expected gradient (bias scoring)
  appliedBias: number; // 1 - cosine(expected estimate, true gradient)
  appliedVariance: number; // mean per-step variance of A_hat across episodes
  optimal: number; // optimal expected return from start (cached per gamma)
  rng: number; // rollout rng cursor
  noiseSeed: number; // fixed seed for the noisy value function's per-state error
  steps: number;
  rho: number; // expected return V^{pi,gamma}(start)
  rhoHistory: number[];
  biasHistory: number[];
  varHistory: number[];
  lastTrajectory: number[] | null; // states of the most recent sampled episode
  log: EventLog<Snapshot>;
}

export type Snapshot = Omit<SimState, "log">;

export function softmax(theta: number[], state: number): number[] {
  const base = state * N_ACTIONS;
  let max = -Infinity;
  for (let a = 0; a < N_ACTIONS; a++) max = Math.max(max, at(theta, base + a));
  const exps: number[] = [];
  let sum = 0;
  for (let a = 0; a < N_ACTIONS; a++) {
    const e = Math.exp(at(theta, base + a) - max);
    exps.push(e);
    sum += e;
  }
  return exps.map((e) => e / sum);
}

// Box-Muller standard normal from two uniforms, threaded through the seeded rng.
function nextGaussian(state: number): { value: number; state: number } {
  const u1 = nextRandom(state);
  const u2 = nextRandom(u1.state);
  const r = Math.sqrt(-2 * Math.log(Math.max(u1.value, 1e-12)));
  return { value: r * Math.cos(2 * Math.PI * u2.value), state: u2.state };
}

// The value function the estimator subtracts. Exact returns the solved
// V^{pi,gamma}; noisy adds fixed seeded per-state error (drawn once per derive so
// it stays put while the policy steps); zero is the empty baseline.
function buildEstValue(
  trueValues: number[],
  params: Params,
  seed: number,
): number[] {
  if (params.valueMode === "zero") return new Array<number>(N_STATES).fill(0);
  if (params.valueMode === "exact") return [...trueValues];
  let rng = seed >>> 0;
  const out: number[] = [];
  for (let s = 0; s < N_STATES; s++) {
    const draw = nextGaussian(rng);
    rng = draw.state;
    out.push(at(trueValues, s) + draw.value * params.valueError);
  }
  return out;
}

// The policy gradient in the tabular-softmax closed form
// d^pi(s) * pi(s,a) * (score(s,a) - sum_b pi(s,b) score(s,b)). The per-state
// policy-weighted mean is subtracted because the softmax score gradient carries a
// factor (indicator - pi), which makes any state-dependent baseline cancel. With
// the true advantage A (already mean-zero per state) this is the exact gradient;
// with an uncentered score like Q it recovers the same direction, which is why a
// baseline never changes the gradient in expectation.
function policyGradient(
  mdp: Mdp,
  pi: number[][],
  score: number[][],
  visitation: number[],
): number[] {
  const grad = new Array<number>(N_STATES * N_ACTIONS).fill(0);
  for (let s = 0; s < N_STATES; s++) {
    if (isTerminal(mdp, s)) continue;
    const d = at(visitation, s);
    const probs = at(pi, s);
    const row = at(score, s);
    let mean = 0;
    for (let a = 0; a < N_ACTIONS; a++) mean += at(probs, a) * at(row, a);
    for (let a = 0; a < N_ACTIONS; a++) {
      grad[s * N_ACTIONS + a] = d * at(probs, a) * (at(row, a) - mean);
    }
  }
  return grad;
}

// The expected GAE advantage E[A_hat^{gamma,lambda} | s, a] under the MDP, solved
// in closed form so bias can be separated from sampling variance. The estimator
// averages, over runs, to a fixed gradient direction; the cosine of THAT to the
// true gradient is the bias the paper talks about, not the per-batch scatter.
//
// With the deterministic grid, define ed(s,a) = E[delta | s,a] = R + gamma * V(s')
// - V(s) and h(s) = sum_a pi(s,a) * E[A_hat | s,a]. Then h solves the linear
// system h(s) - gamma*lambda * sum_a pi(s,a) * [s' not terminal] * h(s') =
// sum_a pi(s,a) * ed(s,a), after which E[A_hat | s,a] = ed(s,a) +
// gamma*lambda * [s' not terminal] * h(s').
function expectedGaeAdvantage(
  mdp: Mdp,
  pi: number[][],
  estValue: number[],
  gamma: number,
  lambda: number,
): number[][] {
  const ed: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    for (let a = 0; a < N_ACTIONS; a++) {
      const ns = nextState(s, a);
      const vNext = isTerminal(mdp, ns) ? 0 : at(estValue, ns);
      row.push(enterReward(mdp, ns) + gamma * vNext - at(estValue, s));
    }
    ed.push(row);
  }

  const decay = gamma * lambda;
  const matrix: number[][] = [];
  const rhs = new Array<number>(N_STATES).fill(0);
  for (let s = 0; s < N_STATES; s++) {
    const mrow = new Array<number>(N_STATES).fill(0);
    mrow[s] = 1;
    if (!isTerminal(mdp, s)) {
      const probs = at(pi, s);
      const edRow = at(ed, s);
      for (let a = 0; a < N_ACTIONS; a++) {
        const ns = nextState(s, a);
        rhs[s] = at(rhs, s) + at(probs, a) * at(edRow, a);
        if (!isTerminal(mdp, ns)) {
          mrow[ns] = at(mrow, ns) - decay * at(probs, a);
        }
      }
    }
    matrix.push(mrow);
  }
  const h = solveLinear(matrix, rhs);

  const out: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    const edRow = at(ed, s);
    for (let a = 0; a < N_ACTIONS; a++) {
      const ns = nextState(s, a);
      const cont = isTerminal(mdp, ns) ? 0 : decay * at(h, ns);
      row.push(at(edRow, a) + cont);
    }
    out.push(row);
  }
  return out;
}

function pickAction(probs: number[], u: number): number {
  let acc = 0;
  for (let a = 0; a < N_ACTIONS; a++) {
    acc += at(probs, a);
    if (u < acc) return a;
  }
  return N_ACTIONS - 1;
}

interface EpisodeSample {
  states: number[];
  actions: number[];
  rewards: number[];
  path: number[];
}

function rolloutEpisode(
  mdp: Mdp,
  pi: number[][],
  seed: number,
): { episode: EpisodeSample; rng: number } {
  let rng = seed;
  const states: number[] = [];
  const actions: number[] = [];
  const rewards: number[] = [];
  let s = mdp.start;
  const path: number[] = [s];
  for (let t = 0; t < EPISODE_MAX_STEPS && !isTerminal(mdp, s); t++) {
    const draw = nextRandom(rng);
    rng = draw.state;
    const a = pickAction(at(pi, s), draw.value);
    const ns = nextState(s, a);
    states.push(s);
    actions.push(a);
    rewards.push(enterReward(mdp, ns));
    path.push(ns);
    s = ns;
  }
  return { episode: { states, actions, rewards, path }, rng };
}

// GAE applied to one rolled-out episode. Walks backward accumulating
// A_hat_t = delta_t + (gamma*lambda) * A_hat_{t+1}, which is exactly the
// exponentially-weighted sum of TD residuals from Equation (16). Returns the
// per-step advantage estimate aligned with `states`/`actions`. A terminal
// next-state bootstraps with V = 0, an absorbing-state convention; otherwise it
// bootstraps with the estimator's V at the next state.
function gaeAdvantages(
  mdp: Mdp,
  episode: EpisodeSample,
  estValue: number[],
  gamma: number,
  lambda: number,
): number[] {
  const { states, rewards, path } = episode;
  const n = states.length;
  const adv = new Array<number>(n).fill(0);
  let running = 0;
  for (let t = n - 1; t >= 0; t--) {
    const s = at(states, t);
    const ns = at(path, t + 1);
    const bootstrap = isTerminal(mdp, ns) ? 0 : at(estValue, ns);
    const delta = at(rewards, t) + gamma * bootstrap - at(estValue, s);
    running = delta + gamma * lambda * running;
    adv[t] = running;
  }
  return adv;
}

interface BatchResult {
  grad: number[];
  rng: number;
  variance: number;
  lastTrajectory: number[];
}

// Samples `episodes` rollouts, computes the GAE policy-gradient estimate as the
// batch average of A_hat_t * grad-log-pi(a_t|s_t), and measures the variance of
// the per-step advantage estimate across the batch. Threads the seeded stream.
function batchGradient(state: SimState): BatchResult {
  const { mdp, params, pi, estValue } = state;
  const grad = new Array<number>(N_STATES * N_ACTIONS).fill(0);
  let rng = state.rng;
  let count = 0;
  let advSum = 0;
  let advSqSum = 0;
  let lastTrajectory: number[] = [mdp.start];

  for (let ep = 0; ep < params.episodes; ep++) {
    const rolled = rolloutEpisode(mdp, pi, rng);
    rng = rolled.rng;
    const episode = rolled.episode;
    lastTrajectory = episode.path;
    const adv = gaeAdvantages(
      mdp,
      episode,
      estValue,
      params.gamma,
      params.lambda,
    );
    for (let t = 0; t < episode.states.length; t++) {
      const s = at(episode.states, t);
      const action = at(episode.actions, t);
      const probs = at(pi, s);
      const a_hat = at(adv, t);
      advSum += a_hat;
      advSqSum += a_hat * a_hat;
      count += 1;
      for (let a = 0; a < N_ACTIONS; a++) {
        const indicator = a === action ? 1 : 0;
        const index = s * N_ACTIONS + a;
        grad[index] = at(grad, index) + (indicator - at(probs, a)) * a_hat;
      }
    }
  }

  for (let i = 0; i < grad.length; i++) grad[i] = at(grad, i) / params.episodes;
  const mean = count > 0 ? advSum / count : 0;
  const variance = count > 0 ? Math.max(0, advSqSum / count - mean * mean) : 0;
  return { grad, rng, variance, lastTrajectory };
}

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += at(a, i) * at(b, i);
  return sum;
}

function cosine(a: number[], b: number[]): number {
  const na = Math.sqrt(dot(a, a));
  const nb = Math.sqrt(dot(b, b));
  if (na < 1e-12 || nb < 1e-12) return 0;
  return clamp(dot(a, b) / (na * nb), -1, 1);
}

// Recomputes everything that follows from theta and the params: the policy, the
// solved value function, the true advantage, the exact gradient, and the value
// function the estimator will use. Pure given the noise seed. Called after every
// theta or param change.
function derive(state: SimState): SimState {
  const { mdp, params, theta } = state;
  const pi: number[][] = [];
  for (let s = 0; s < N_STATES; s++) pi.push(softmax(theta, s));
  const { p, reward } = policyTransition(mdp, pi);
  const { values, q } = evaluatePolicy(mdp, pi, p, reward, params.gamma);
  const trueAdvantage: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    for (let a = 0; a < N_ACTIONS; a++)
      row.push(at(at(q, s), a) - at(values, s));
    trueAdvantage.push(row);
  }
  const visitation = discountedVisitation(p, params.gamma, oneHot(mdp.start));
  const exactGrad = policyGradient(mdp, pi, trueAdvantage, visitation);
  const estValue = buildEstValue(values, params, state.noiseSeed);
  const expectedAdvantage = expectedGaeAdvantage(
    mdp,
    pi,
    estValue,
    params.gamma,
    params.lambda,
  );
  const expectedGrad = policyGradient(mdp, pi, expectedAdvantage, visitation);
  return {
    ...state,
    pi,
    trueValues: values,
    trueAdvantage,
    exactGrad,
    expectedGrad,
    estValue,
    appliedBias: 1 - cosine(expectedGrad, exactGrad),
    rho: at(values, mdp.start),
  };
}

function advance(state: SimState): SimState {
  const result = batchGradient(state);
  const theta = state.theta.map((value, index) =>
    clamp(
      value + state.params.alpha * at(result.grad, index),
      -THETA_CLAMP,
      THETA_CLAMP,
    ),
  );

  // derive recomputes appliedBias from the estimator's expected gradient against
  // the true gradient, so the bias reading reflects the estimator, not the luck
  // of this batch. Variance comes from the batch we just sampled.
  const advanced = derive({
    ...state,
    theta,
    rng: result.rng,
    steps: state.steps + 1,
    appliedVariance: result.variance,
    lastTrajectory: result.lastTrajectory,
  });

  const next: SimState = {
    ...advanced,
    rhoHistory: [...state.rhoHistory, advanced.rho].slice(-HISTORY_LEN),
    biasHistory: [...state.biasHistory, advanced.appliedBias].slice(
      -HISTORY_LEN,
    ),
    varHistory: [...state.varHistory, result.variance].slice(-HISTORY_LEN),
  };
  return logMilestones(state, next);
}

// Logs the two moments worth rewinding to: the step where the policy first
// reaches near-optimal return, and the step where a biased estimator stalls the
// climb well short of it. Each fires only on the transition.
function logMilestones(prev: SimState, next: SimState): SimState {
  const optimal = next.optimal;
  if (optimal <= 1e-9) return next;
  const wasNear = prev.rho >= 0.9 * optimal;
  const isNear = next.rho >= 0.9 * optimal;
  let result = next;
  if (!wasNear && isNear) {
    result = withLog(result, `step ${next.steps}: reached near-optimal return`);
  }
  const biasedSetup =
    next.params.valueMode !== "exact" && next.params.lambda < 0.2;
  const stalled =
    biasedSetup &&
    next.steps >= 80 &&
    prev.steps < 80 &&
    next.rho < 0.5 * optimal;
  if (stalled) {
    result = withLog(
      result,
      `step ${next.steps}: biased estimator stalled the climb`,
    );
  }
  return result;
}

function snapshotOf(state: SimState): Snapshot {
  const { log: _log, ...rest } = state;
  void _log;
  return rest;
}

function withLog(state: SimState, label: string): SimState {
  return {
    ...state,
    log: record(state.log, state.steps, label, snapshotOf(state)),
  };
}

export interface Status {
  rho: number;
  optimal: number;
  fractionOfOptimal: number;
  bias: number;
  variance: number;
  nearOptimal: boolean;
  unbiased: boolean;
}

// How the run is doing, scored against the optimal return the learner never
// sees. bias is 1 - cosine(batch estimate, true gradient): it sits near 0 when
// the estimator is gamma-just (lambda = 1, or any lambda with exact V) and grows
// as lambda shrinks under a wrong value function.
export function status(state: SimState): Status {
  const optimal = state.optimal;
  const fraction = optimal <= 1e-9 ? 0 : state.rho / optimal;
  return {
    rho: state.rho,
    optimal,
    fractionOfOptimal: fraction,
    bias: state.appliedBias,
    variance: state.appliedVariance,
    nearOptimal: fraction >= 0.9,
    unbiased: state.appliedBias < 0.08,
  };
}

export type Intervention =
  | { type: "setLambda"; value: number }
  | { type: "setGamma"; value: number }
  | { type: "setValueMode"; value: ValueMode }
  | { type: "setValueError"; value: number }
  | { type: "setEpisodes"; value: number }
  | { type: "setAlpha"; value: number }
  | { type: "resetPolicy" }
  | { type: "setSeed"; value: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: Snapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

// ---- MDP and presets ----------------------------------------------------

const GRID: Mdp = {
  start: 4 * 5 + 0, // bottom-left
  goal: 0 * 5 + 4, // top-right
  trap: 2 * 5 + 2, // center
};

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: Params;
  seed: number;
}

const BASE: Params = {
  lambda: 0.96,
  gamma: 0.98,
  valueMode: "exact",
  valueError: 0.25,
  episodes: 24,
  alpha: 3,
};

export const PRESETS: Preset[] = [
  {
    id: "balanced",
    label: "Balanced (lambda 0.96)",
    blurb:
      "The paper's sweet spot. With a good value function and lambda at 0.96, the advantage estimate has low bias and low variance. Press play and watch the return climb fast and smooth to optimal while the bias light stays green.",
    params: { ...BASE, lambda: 0.96, valueMode: "exact" },
    seed: 1,
  },
  {
    id: "monte-carlo",
    label: "Monte Carlo (lambda 1)",
    blurb:
      "Set lambda to 1 and the estimate becomes the full sampled return minus the baseline. It stays unbiased no matter how wrong the value function is, but the long reward sum makes it noisy. The bias light stays green while the variance readout runs high and the climb jitters.",
    params: { ...BASE, lambda: 1, valueMode: "noisy", valueError: 0.4 },
    seed: 7,
  },
  {
    id: "one-step-broken",
    label: "One-step on a broken V (lambda 0)",
    blurb:
      "Set lambda to 0 with a zeroed value function and the estimate is a single biased TD residual. The paper found lambda 0 leads to excessive bias. Watch the bias light turn red and the return stall well short of optimal. Then drag lambda up toward 1 and watch the bias fall and the climb restart.",
    params: { ...BASE, lambda: 0, valueMode: "zero" },
    seed: 3,
  },
  {
    id: "noisy-value",
    label: "Noisy value function",
    blurb:
      "A good lambda of 0.96 but the value function is corrupted with per-state error. Lower lambda would lean harder on the bad V and bias the step; higher lambda leans on sampled rewards and stays honest. Drag the value error up and watch the bias grow, then push lambda toward 1 to wash it out.",
    params: { ...BASE, lambda: 0.96, valueMode: "noisy", valueError: 0.6 },
    seed: 5,
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

// The fixed noise seed used to draw the noisy value function's per-state error.
// Kept separate from the rollout rng so changing the rollout seed doesn't reshape
// the value function, and so the value error stays put while the policy learns.
const NOISE_SEED = 0x9e3779b9;

export function initialState(
  presetId = "balanced",
  seedOverride?: number,
): SimState {
  const preset = getPreset(presetId);
  const base: SimState = {
    mdp: GRID,
    params: { ...preset.params },
    theta: new Array<number>(N_STATES * N_ACTIONS).fill(0),
    pi: [],
    trueValues: [],
    trueAdvantage: [],
    estValue: [],
    exactGrad: [],
    expectedGrad: [],
    appliedBias: 0,
    appliedVariance: 0,
    optimal: optimalReturn(GRID, preset.params.gamma),
    rng: (seedOverride ?? preset.seed) >>> 0,
    noiseSeed: NOISE_SEED,
    steps: 0,
    rho: 0,
    rhoHistory: [],
    biasHistory: [],
    varHistory: [],
    lastTrajectory: null,
    log: emptyLog<Snapshot>(),
  };
  return derive(base);
}

function applyIntervention(
  state: SimState,
  action: Intervention,
): { state: SimState; label: string } {
  switch (action.type) {
    case "setLambda":
      return {
        state: derive({
          ...state,
          params: { ...state.params, lambda: action.value },
        }),
        label: `lambda -> ${action.value.toFixed(2)}`,
      };
    case "setGamma":
      return {
        state: derive({
          ...state,
          params: { ...state.params, gamma: action.value },
          optimal: optimalReturn(state.mdp, action.value),
        }),
        label: `gamma -> ${action.value.toFixed(3)}`,
      };
    case "setValueMode":
      return {
        state: derive({
          ...state,
          params: { ...state.params, valueMode: action.value },
        }),
        label: `value -> ${action.value}`,
      };
    case "setValueError":
      return {
        state: derive({
          ...state,
          params: { ...state.params, valueError: action.value },
        }),
        label: `value error -> ${action.value.toFixed(2)}`,
      };
    case "setEpisodes":
      return {
        state: {
          ...state,
          params: { ...state.params, episodes: action.value },
        },
        label: `episodes -> ${action.value}`,
      };
    case "setAlpha":
      return {
        state: { ...state, params: { ...state.params, alpha: action.value } },
        label: `alpha -> ${action.value.toFixed(1)}`,
      };
    case "resetPolicy":
      return {
        state: derive({
          ...state,
          theta: new Array<number>(N_STATES * N_ACTIONS).fill(0),
          appliedBias: 0,
          appliedVariance: 0,
          rhoHistory: [],
          biasHistory: [],
          varHistory: [],
          lastTrajectory: null,
          steps: 0,
        }),
        label: "policy reset to uniform",
      };
    case "setSeed":
      return {
        state: { ...state, rng: action.value >>> 0 },
        label: `seed -> ${action.value}`,
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

// The pure reducer. A tick samples a batch and runs one GAE policy-gradient
// ascent step; an intervention changes a knob or the policy, logging the moment
// just before it took effect so the event log can rewind to it.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return advance(state);
  const before = snapshotOf(state);
  const { state: changed, label } = applyIntervention(state, input.action);
  return { ...changed, log: record(state.log, state.steps, label, before) };
}
