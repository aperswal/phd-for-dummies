// A pure, deterministic model of the policy gradient theorem from Sutton,
// McAllester, Singh and Mansour's 2000 paper "Policy Gradient Methods for
// Reinforcement Learning with Function Approximation". A softmax (Gibbs) policy
// with one preference theta(s,a) per state-action runs on a small gridworld MDP
// the model solves exactly. Each tick takes one gradient-ascent step on the
// expected return rho = V^pi(start).
//
// Theorem 1 gives the gradient in closed form, and for a tabular softmax it
// collapses to a clean per-parameter rule:
//
//   d rho / d theta(s,a) = d^pi(s) * pi(s,a) * (Q^pi(s,a) - V^pi(s))
//                        = d^pi(s) * pi(s,a) * A^pi(s,a)
//
// where d^pi is the discounted state-visitation from the start. The crucial fact
// the paper proves is that this gradient has no term for how theta reshapes
// d^pi: the visitation appears, but never its derivative.
//
// The reader swaps the critic that supplies the value used in the step:
//   - exact:        the true advantage A^pi (Theorem 1, the ground truth).
//   - compatible:   a critic linear in the policy's own features, which for a
//                   softmax can only represent mean-zero-per-state functions, so
//                   it fits exactly to A^pi and reproduces the exact gradient
//                   with zero bias (Theorem 2).
//   - incompatible: a critic that lumps {left,right} and {up,down} together, so
//                   it cannot tell apart actions the policy tells apart. Its
//                   gradient is biased and the policy never learns left from
//                   right, which is what the compatibility condition forbids.
//   - reinforce:    Williams' Monte Carlo estimate from sampled episode returns,
//                   unbiased but high-variance, optionally with a value baseline.
//
// The model always computes the exact gradient too, used only to score how well
// the critic's step aligns with it (the cosine). All randomness comes from a
// seeded stream threaded through the state, so a seed plus a list of inputs
// reproduces a run exactly.

import { at } from "@/lib/simulation/array";
import { emptyLog, record, type EventLog } from "@/lib/simulation/history";
import {
  clamp,
  discountedVisitation,
  enterReward,
  evaluatePolicy,
  isTerminal,
  type Mdp,
  N_ACTIONS,
  N_STATES,
  nextState,
  oneHot,
  optimalReturn,
  policyTransition,
} from "@/lib/simulation/mdp";
import { nextRandom } from "@/lib/simulation/rng";

export {
  ACTION_LABELS,
  colOf,
  GRID_W,
  isTerminal,
  N_ACTIONS,
  N_STATES,
  rowOf,
} from "@/lib/simulation/mdp";

// Grouped by axis for the incompatible critic: {up,down}=0, {left,right}=1.
const ACTION_GROUP = [0, 0, 1, 1];

const THETA_CLAMP = 25;
const HISTORY_LEN = 80;
const EPISODE_MAX_STEPS = 80;

export type CriticMode = "exact" | "compatible" | "incompatible" | "reinforce";

export interface Params {
  critic: CriticMode;
  baseline: boolean;
  alpha: number;
  gamma: number;
  mcEpisodes: number;
}

export interface SimState {
  mdp: Mdp;
  params: Params;
  theta: number[]; // length N_STATES * N_ACTIONS
  pi: number[][]; // [state][action]
  values: number[]; // V^pi per state
  q: number[][]; // Q^pi[state][action]
  advantage: number[][]; // A^pi[state][action]
  visitation: number[]; // d^pi per state
  exactGrad: number[]; // flat N_STATES * N_ACTIONS, the true gradient
  appliedCos: number; // cosine of the last applied step against the true gradient
  optimal: number; // optimal expected return from the start (cached, depends on gamma)
  rng: number;
  steps: number;
  rho: number; // expected return V^pi(start)
  rhoHistory: number[];
  cosHistory: number[];
  lastTrajectory: number[] | null; // states of the last sampled episode (reinforce)
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

// The exact policy gradient (Theorem 1) per parameter, in the tabular-softmax
// closed form d^pi(s) * pi(s,a) * A^pi(s,a).
function exactGradient(
  mdp: Mdp,
  pi: number[][],
  advantage: number[][],
  visitation: number[],
): number[] {
  const grad = new Array<number>(N_STATES * N_ACTIONS).fill(0);
  for (let s = 0; s < N_STATES; s++) {
    if (isTerminal(mdp, s)) continue;
    const d = at(visitation, s);
    const probs = at(pi, s);
    const adv = at(advantage, s);
    for (let a = 0; a < N_ACTIONS; a++) {
      grad[s * N_ACTIONS + a] = d * at(probs, a) * at(adv, a);
    }
  }
  return grad;
}

// The value the critic feeds into the gradient in place of the true advantage.
// exact and compatible both return A^pi (the compatible critic, being linear in
// the policy's features, fits A^pi exactly); incompatible collapses each axis to
// one shared value, losing the within-axis signal.
function criticAdvantage(
  state: SimState,
  pi: number[][],
  advantage: number[][],
): number[][] {
  if (state.params.critic === "incompatible") {
    const result: number[][] = [];
    for (let s = 0; s < N_STATES; s++) {
      const probs = at(pi, s);
      const adv = at(advantage, s);
      const groupNum = [0, 0];
      const groupDen = [0, 0];
      for (let a = 0; a < N_ACTIONS; a++) {
        const g = at(ACTION_GROUP, a);
        groupNum[g] = at(groupNum, g) + at(probs, a) * at(adv, a);
        groupDen[g] = at(groupDen, g) + at(probs, a);
      }
      const row: number[] = [];
      for (let a = 0; a < N_ACTIONS; a++) {
        const g = at(ACTION_GROUP, a);
        const den = at(groupDen, g);
        row.push(den < 1e-12 ? 0 : at(groupNum, g) / den);
      }
      result.push(row);
    }
    return result;
  }
  return advantage;
}

function analyticStep(
  state: SimState,
  pi: number[][],
  advantage: number[][],
  visitation: number[],
): number[] {
  const adv = criticAdvantage(state, pi, advantage);
  const grad = new Array<number>(N_STATES * N_ACTIONS).fill(0);
  for (let s = 0; s < N_STATES; s++) {
    if (isTerminal(state.mdp, s)) continue;
    const d = at(visitation, s);
    const probs = at(pi, s);
    const row = at(adv, s);
    for (let a = 0; a < N_ACTIONS; a++) {
      grad[s * N_ACTIONS + a] = d * at(probs, a) * at(row, a);
    }
  }
  return grad;
}

function pickAction(probs: number[], u: number): number {
  let acc = 0;
  for (let a = 0; a < N_ACTIONS; a++) {
    acc += at(probs, a);
    if (u < acc) return a;
  }
  return N_ACTIONS - 1;
}

// Williams' REINFORCE estimate of the same gradient, built from sampled episode
// returns rather than the solved value function. For each visited (s_t, a_t) it
// adds grad-log-pi times the return-to-go minus an optional value baseline,
// averaged over mcEpisodes rollouts. Unbiased, but its variance is the whole
// reason a critic helps. Threads the seeded stream so the run stays replayable.
function reinforceStep(
  state: SimState,
  pi: number[][],
  values: number[],
): { grad: number[]; rng: number; trajectory: number[] } {
  const { mdp, params } = state;
  const grad = new Array<number>(N_STATES * N_ACTIONS).fill(0);
  let rng = state.rng;
  let lastTrajectory: number[] = [mdp.start];

  for (let ep = 0; ep < params.mcEpisodes; ep++) {
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
    lastTrajectory = path;

    let ret = 0;
    for (let t = rewards.length - 1; t >= 0; t--) {
      ret = at(rewards, t) + params.gamma * ret;
      const st = at(states, t);
      const at_ = at(actions, t);
      const probs = at(pi, st);
      const baseline = params.baseline ? at(values, st) : 0;
      const weight = ret - baseline;
      for (let a = 0; a < N_ACTIONS; a++) {
        const indicator = a === at_ ? 1 : 0;
        const index = st * N_ACTIONS + a;
        grad[index] =
          at(grad, index) +
          ((indicator - at(probs, a)) * weight) / params.mcEpisodes;
      }
    }
  }
  return { grad, rng, trajectory: lastTrajectory };
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
// solved value function, the visitation, the true advantage, and the exact
// gradient. Pure, no rng. Called after every theta or param change.
function derive(state: SimState): SimState {
  const { mdp, params, theta } = state;
  const pi: number[][] = [];
  for (let s = 0; s < N_STATES; s++) pi.push(softmax(theta, s));
  const { p, reward } = policyTransition(mdp, pi);
  const { values, q } = evaluatePolicy(mdp, pi, p, reward, params.gamma);
  const advantage: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    for (let a = 0; a < N_ACTIONS; a++)
      row.push(at(at(q, s), a) - at(values, s));
    advantage.push(row);
  }
  const visitation = discountedVisitation(p, params.gamma, oneHot(mdp.start));
  const exactGrad = exactGradient(mdp, pi, advantage, visitation);
  return {
    ...state,
    pi,
    values,
    q,
    advantage,
    visitation,
    exactGrad,
    rho: at(values, mdp.start),
  };
}

function advance(state: SimState): SimState {
  let step: number[];
  let rng = state.rng;
  let trajectory = state.lastTrajectory;

  if (state.params.critic === "reinforce") {
    const result = reinforceStep(state, state.pi, state.values);
    step = result.grad;
    rng = result.rng;
    trajectory = result.trajectory;
  } else {
    step = analyticStep(state, state.pi, state.advantage, state.visitation);
    trajectory = null;
  }

  const appliedCos = cosine(step, state.exactGrad);
  const theta = state.theta.map((value, index) =>
    clamp(
      value + state.params.alpha * at(step, index),
      -THETA_CLAMP,
      THETA_CLAMP,
    ),
  );

  const advanced = derive({
    ...state,
    theta,
    rng,
    steps: state.steps + 1,
    appliedCos,
    lastTrajectory: trajectory,
    rhoHistory: state.rhoHistory,
    cosHistory: state.cosHistory,
  });

  const next: SimState = {
    ...advanced,
    rhoHistory: [...state.rhoHistory, advanced.rho].slice(-HISTORY_LEN),
    cosHistory: [...state.cosHistory, appliedCos].slice(-HISTORY_LEN),
  };
  return logMilestones(state, next);
}

// Logs the two moments worth rewinding to: the step where the policy first
// reaches near-optimal return, and the step where a biased critic stalls the
// climb well short of it. Each fires only on the transition.
function logMilestones(prev: SimState, next: SimState): SimState {
  const optimal = next.optimal;
  if (optimal <= 1e-9) return next;
  const wasNear = prev.rho >= 0.92 * optimal;
  const isNear = next.rho >= 0.92 * optimal;
  let result = next;
  if (!wasNear && isNear) {
    result = withLog(result, `step ${next.steps}: reached near-optimal return`);
  }
  const biased = next.params.critic === "incompatible";
  const stalled =
    biased && next.steps >= 60 && prev.steps < 60 && next.rho < 0.6 * optimal;
  if (stalled) {
    result = withLog(
      result,
      `step ${next.steps}: biased critic stalled the climb`,
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
  cos: number;
  aligned: boolean;
  biased: boolean;
  nearOptimal: boolean;
}

// How the run is doing, scored against the optimal return the learner never
// sees. cos is the alignment of the last applied step with the true gradient:
// it sits at 1 for the exact and compatible critics (Theorems 1 and 2), drops
// below 1 for the incompatible critic, and jitters for sampled returns.
export function status(state: SimState): Status {
  const optimal = state.optimal;
  const fraction = optimal <= 1e-9 ? 0 : state.rho / optimal;
  const cos = state.appliedCos;
  return {
    rho: state.rho,
    optimal,
    fractionOfOptimal: fraction,
    cos,
    aligned: cos > 0.98,
    biased: cos > 0 && cos <= 0.9,
    nearOptimal: fraction >= 0.92,
  };
}

export type Intervention =
  | { type: "setCritic"; value: CriticMode }
  | { type: "setBaseline"; value: boolean }
  | { type: "setAlpha"; value: number }
  | { type: "setGamma"; value: number }
  | { type: "setMcEpisodes"; value: number }
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
  critic: "exact",
  baseline: true,
  alpha: 2.5,
  gamma: 0.96,
  mcEpisodes: 24,
};

export const PRESETS: Preset[] = [
  {
    id: "exact",
    label: "Follow the true gradient",
    blurb:
      "The step uses the exact advantage, so it is the policy gradient itself (Theorem 1). Press play and watch the arrows sharpen toward the goal, the visitation shading flow along the path, and the return climb to optimal while the alignment light holds at 1.00.",
    params: { ...BASE, critic: "exact" },
    seed: 1,
  },
  {
    id: "compatible",
    label: "Swap in a compatible critic",
    blurb:
      "Now the step uses a critic linear in the policy's own features. It can only store advantages, never the absolute value, yet it reproduces the exact gradient with zero bias (Theorem 2). The alignment light stays at 1.00 and the climb matches the true gradient step for step.",
    params: { ...BASE, critic: "compatible" },
    seed: 1,
  },
  {
    id: "reinforce",
    label: "Sampled returns (REINFORCE)",
    blurb:
      "No solved value function here, just Williams' estimate from sampled episode returns. It is unbiased but noisy, so the alignment light jitters and the climb is slow. Turn the value baseline off and watch the variance explode, then back on to steady it.",
    params: { ...BASE, critic: "reinforce", baseline: true, alpha: 1.2 },
    seed: 7,
  },
  {
    id: "incompatible",
    label: "An incompatible critic biases it",
    blurb:
      "This critic lumps left with right and up with down, so it cannot tell apart actions the policy tells apart. The alignment light drops below 1.00, the gradient is biased, and the policy never learns which way to go. Switch the critic to compatible mid-run and watch the light snap to 1.00 and the climb start.",
    params: { ...BASE, critic: "incompatible", alpha: 2.5 },
    seed: 3,
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(
  presetId = "exact",
  seedOverride?: number,
): SimState {
  const preset = getPreset(presetId);
  const base: SimState = {
    mdp: GRID,
    params: { ...preset.params },
    theta: new Array<number>(N_STATES * N_ACTIONS).fill(0),
    pi: [],
    values: [],
    q: [],
    advantage: [],
    visitation: [],
    exactGrad: [],
    appliedCos: 1,
    optimal: optimalReturn(GRID, preset.params.gamma),
    rng: (seedOverride ?? preset.seed) >>> 0,
    steps: 0,
    rho: 0,
    rhoHistory: [],
    cosHistory: [],
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
    case "setCritic":
      return {
        state: derive({
          ...state,
          params: { ...state.params, critic: action.value },
          appliedCos: action.value === "incompatible" ? state.appliedCos : 1,
        }),
        label: `critic -> ${action.value}`,
      };
    case "setBaseline":
      return {
        state: {
          ...state,
          params: { ...state.params, baseline: action.value },
        },
        label: `baseline ${action.value ? "on" : "off"}`,
      };
    case "setAlpha":
      return {
        state: { ...state, params: { ...state.params, alpha: action.value } },
        label: `alpha -> ${action.value.toFixed(2)}`,
      };
    case "setGamma":
      return {
        state: derive({
          ...state,
          params: { ...state.params, gamma: action.value },
          optimal: optimalReturn(state.mdp, action.value),
        }),
        label: `gamma -> ${action.value.toFixed(2)}`,
      };
    case "setMcEpisodes":
      return {
        state: {
          ...state,
          params: { ...state.params, mcEpisodes: action.value },
        },
        label: `episodes per step -> ${action.value}`,
      };
    case "resetPolicy":
      return {
        state: derive({
          ...state,
          theta: new Array<number>(N_STATES * N_ACTIONS).fill(0),
          appliedCos: 1,
          rhoHistory: [],
          cosHistory: [],
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

// The pure reducer. A tick runs one gradient-ascent step under the chosen
// critic; an intervention changes a knob or the policy, logging the moment just
// before it took effect so the event log can rewind to it.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return advance(state);
  const before = snapshotOf(state);
  const { state: changed, label } = applyIntervention(state, input.action);
  return { ...changed, log: record(state.log, state.steps, label, before) };
}
