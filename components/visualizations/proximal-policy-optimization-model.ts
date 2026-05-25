// A pure, deterministic model of the clipped surrogate objective from Schulman,
// Wolski, Dhariwal, Radford and Klimov's 2017 paper "Proximal Policy
// Optimization Algorithms". A tabular softmax policy runs on a small chain MDP
// the model solves exactly. One "round" of PPO does what Algorithm 1 does:
// freeze a snapshot policy pi_old, compute the advantages under pi_old, then run
// K epochs of gradient ascent on a surrogate objective built from the
// probability ratio r(theta) = pi_theta(a|s) / pi_old(a|s), and finally set
// pi_old <- pi.
//
// The objective the reader swaps is the whole point:
//   - cpi:  L = E[ r * A ]                      (no clipping, eq 6)
//   - clip: L = E[ min(r*A, clip(r,1-e,1+e)*A) ] (the paper's eq 7)
//   - kl:   L = E[ r*A - beta * KL(pi_old, pi) ] (the penalty variant, eq 5/8)
//
// Run many epochs (K) on one batch and cpi keeps pushing the ratio in the same
// direction, so the policy overshoots and the return collapses, which is the
// paper's central failure (Section 2.1, "destructively large policy updates").
// clip flattens the gradient once the ratio leaves [1-e, 1+e] in the helpful
// direction, so the same K epochs stay bounded and the return climbs. That
// contrast is the thing the reader perturbs.
//
// The advantage is the true A^pi solved on the MDP, so the surrogate is exact up
// to the policy class. All randomness (optional minibatch sampling, the optional
// observation noise on the advantage) comes from a seeded stream threaded through
// the state, so a seed plus a list of inputs reproduces a run exactly.

import { at } from "@/lib/simulation/array";
import { emptyLog, record, type EventLog } from "@/lib/simulation/history";
import { nextRandom } from "@/lib/simulation/rng";

// A short corridor of states laid out left to right. The walkable states are
// START at the left up through the GOAL at the right. Going LEFT from any state
// drops into a single TRAP terminal that pays a small reward; going RIGHT walks
// one step toward the GOAL terminal that pays a large reward. So at every state
// the policy faces a real decision with a clear right answer (keep going right),
// the visitation spreads along the corridor, and "always right" is genuinely
// optimal. Two actions keep the softmax legible and the ratio easy to read.
export const N_STATES = 6; // 0..4 walkable, 5 = GOAL terminal, plus a TRAP
export const N_ACTIONS = 2; // 0 = left (toward the trap), 1 = right (toward the goal)
export const ACTION_LABELS = ["left", "right"] as const;

export const START = 0;
export const GOAL = N_STATES - 1;
export const TRAP = -1; // a virtual terminal; not a grid cell

const GOAL_REWARD = 1;
const TRAP_REWARD = 0.1; // the tempting-but-worse exit
const STEP_COST = -0.01;

function nextState(state: number, action: number): number {
  if (action === 0) return TRAP; // left always falls into the trap
  return Math.min(state + 1, GOAL); // right walks toward the goal
}

function isTerminal(state: number): boolean {
  return state === GOAL || state === TRAP;
}

// Reward for taking `action` in `state`. Falling into the trap pays TRAP_REWARD
// once; stepping into the goal pays GOAL_REWARD; every other rightward step pays
// a tiny cost so dithering is discouraged.
function enterReward(state: number, action: number, ns: number): number {
  if (ns === TRAP) return TRAP_REWARD;
  if (ns === GOAL) return GOAL_REWARD;
  void state;
  return STEP_COST;
}

// Value of a terminal is zero (the reward is paid on entry). The trap is a
// virtual index, so any value lookup of it returns 0.
function valueOf(values: number[], state: number): number {
  if (state === TRAP || state === GOAL) return 0;
  return at(values, state);
}

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

// Exact policy evaluation by value iteration on the policy-averaged Bellman
// equation. Tiny state space, so a fixed sweep count converges well past
// machine-visible precision. Returns V^pi and Q^pi.
function evaluatePolicy(
  pi: number[][],
  gamma: number,
): { values: number[]; q: number[][] } {
  const values = new Array<number>(N_STATES).fill(0);
  for (let sweep = 0; sweep < 400; sweep++) {
    const next = new Array<number>(N_STATES).fill(0);
    for (let s = 0; s < N_STATES; s++) {
      if (isTerminal(s)) continue;
      const probs = at(pi, s);
      let v = 0;
      for (let a = 0; a < N_ACTIONS; a++) {
        const ns = nextState(s, a);
        const r = enterReward(s, a, ns);
        v += at(probs, a) * (r + gamma * valueOf(values, ns));
      }
      next[s] = v;
    }
    for (let s = 0; s < N_STATES; s++) values[s] = at(next, s);
  }
  const q: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    for (let a = 0; a < N_ACTIONS; a++) {
      const ns = nextState(s, a);
      const r = enterReward(s, a, ns);
      row.push(isTerminal(s) ? 0 : r + gamma * valueOf(values, ns));
    }
    q.push(row);
  }
  return { values, q };
}

// Discounted state-visitation d^pi(s) from the start, the weight Theorem 1 of the
// policy-gradient paper puts on each state. PPO inherits it: the surrogate is an
// expectation over states the snapshot policy actually visits.
function discountedVisitation(pi: number[][], gamma: number): number[] {
  const d = new Array<number>(N_STATES).fill(0);
  let dist = new Array<number>(N_STATES).fill(0);
  dist[START] = 1;
  for (let t = 0; t < 200; t++) {
    for (let s = 0; s < N_STATES; s++)
      d[s] = at(d, s) + Math.pow(gamma, t) * at(dist, s);
    const next = new Array<number>(N_STATES).fill(0);
    for (let s = 0; s < N_STATES; s++) {
      if (isTerminal(s)) continue;
      const probs = at(pi, s);
      for (let a = 0; a < N_ACTIONS; a++) {
        const ns = nextState(s, a);
        if (ns === TRAP || ns === GOAL) continue; // mass leaves the walkable set
        next[ns] = at(next, ns) + at(dist, s) * at(probs, a);
      }
    }
    dist = next;
  }
  const total = d.reduce((sum, v) => sum + v, 0) || 1;
  return d.map((v) => v / total);
}

export function optimalReturn(gamma: number): number {
  // The best policy always goes right. Return from START is the discounted sum
  // of step costs then the goal reward at the far end.
  let v = 0;
  for (let s = GOAL - 1; s >= START; s--) {
    const ns = s + 1;
    const r = enterReward(s, 1, ns);
    v = r + gamma * v;
  }
  return v;
}

export type Objective = "cpi" | "clip" | "kl";

export interface Params {
  objective: Objective;
  epsilon: number; // clip half-width
  epochs: number; // K, gradient steps per round on the same batch
  alpha: number; // learning rate
  gamma: number;
  beta: number; // KL penalty coefficient (kl objective only)
  minibatch: boolean; // sample states by visitation instead of weighting by it
  noise: number; // observation noise added to the advantage estimate
}

export interface SimState {
  params: Params;
  theta: number[]; // length N_STATES * N_ACTIONS, the live policy
  thetaOld: number[]; // the snapshot policy this round optimizes against
  pi: number[][]; // softmax(theta)
  piOld: number[][]; // softmax(thetaOld)
  values: number[]; // V^pi
  q: number[][]; // Q^pi
  advantageOld: number[][]; // A computed under piOld, frozen for the round
  visitationOld: number[]; // d^piOld, frozen for the round
  ratio: number[][]; // r(theta) per state-action, against piOld
  clippedFraction: number; // fraction of state-actions whose ratio is clipped
  klOldToNew: number; // E_s[ KL(piOld(.|s), pi(.|s)) ], the thing clip bounds
  rho: number; // expected return V^pi(START)
  optimal: number;
  rng: number;
  round: number; // PPO outer iterations
  epoch: number; // inner gradient step within the current round
  rhoHistory: number[];
  klHistory: number[];
  log: EventLog<Snapshot>;
}

export type Snapshot = Omit<SimState, "log">;

const THETA_CLAMP = 30;
const HISTORY_LEN = 120;

function klPerState(p: number[], qd: number[]): number {
  let kl = 0;
  for (let a = 0; a < p.length; a++) {
    const pa = at(p, a);
    if (pa > 1e-12) kl += pa * Math.log(pa / Math.max(at(qd, a), 1e-12));
  }
  return kl;
}

// Recomputes everything that follows from theta and thetaOld: both policies, the
// current value function, the probability ratio against the snapshot, the
// clipped fraction, and KL(piOld || pi). Pure, no rng.
function derive(state: SimState): SimState {
  const { theta, thetaOld, params } = state;
  const pi: number[][] = [];
  const piOld: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    pi.push(softmax(theta, s));
    piOld.push(softmax(thetaOld, s));
  }
  const { values, q } = evaluatePolicy(pi, params.gamma);

  const ratio: number[][] = [];
  let clipped = 0;
  let counted = 0;
  let klSum = 0;
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    const cur = at(pi, s);
    const old = at(piOld, s);
    for (let a = 0; a < N_ACTIONS; a++) {
      const r = at(cur, a) / Math.max(at(old, a), 1e-12);
      row.push(r);
      if (!isTerminal(s)) {
        counted += 1;
        if (r > 1 + params.epsilon || r < 1 - params.epsilon) clipped += 1;
      }
    }
    ratio.push(row);
    if (!isTerminal(s)) klSum += klPerState(old, cur);
  }

  return {
    ...state,
    pi,
    piOld,
    values,
    q,
    ratio,
    clippedFraction: counted === 0 ? 0 : clipped / counted,
    klOldToNew: klSum / Math.max(N_STATES - 1, 1),
    rho: at(values, START),
  };
}

// Freezes a fresh snapshot at the top of a PPO round: piOld <- pi, and the
// advantage and visitation are recomputed under piOld and held fixed for all K
// epochs (Algorithm 1 collects the batch once per round). Optional seeded noise
// stands in for the fact that real PPO estimates A from samples, not exactly.
function freezeSnapshot(state: SimState): SimState {
  const thetaOld = [...state.theta];
  const piOld: number[][] = [];
  for (let s = 0; s < N_STATES; s++) piOld.push(softmax(thetaOld, s));
  const { values, q } = evaluatePolicy(piOld, state.params.gamma);

  let rng = state.rng;
  const advantageOld: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    for (let a = 0; a < N_ACTIONS; a++) {
      let adv = at(at(q, s), a) - at(values, s);
      if (state.params.noise > 0) {
        const draw = nextRandom(rng);
        rng = draw.state;
        adv += (draw.value - 0.5) * 2 * state.params.noise;
      }
      row.push(adv);
    }
    advantageOld.push(row);
  }
  const visitationOld = discountedVisitation(piOld, state.params.gamma);

  return derive({ ...state, thetaOld, advantageOld, visitationOld, rng });
}

// The gradient of the chosen surrogate objective with respect to theta, for the
// tabular softmax. The surrogate is sum over s,a of weight(s) * pi_old(s,a) *
// objectiveTerm(r(s,a), A(s,a)), and because d r / d theta runs through the
// softmax jacobian, the per-parameter gradient lands in closed form. Each
// objective differs only in how the ratio enters.
function surrogateGradient(state: SimState): {
  grad: number[];
  rng: number;
} {
  const { params, pi, piOld, advantageOld, visitationOld, ratio } = state;
  const grad = new Array<number>(N_STATES * N_ACTIONS).fill(0);
  let rng = state.rng;

  for (let s = 0; s < N_STATES; s++) {
    if (isTerminal(s)) continue;

    // Either weight each state by its visitation (the expectation), or sample
    // which states the batch contains (minibatch realism). Sampling keeps the
    // run reproducible because it draws from the seeded stream.
    let weight = at(visitationOld, s);
    if (params.minibatch) {
      const draw = nextRandom(rng);
      rng = draw.state;
      weight = draw.value < at(visitationOld, s) * N_STATES ? 1 / N_STATES : 0;
    }
    if (weight <= 0) continue;

    const curr = at(pi, s);
    const oldp = at(piOld, s);
    const adv = at(advantageOld, s);
    const r = at(ratio, s);

    // dL/dtheta(s,a) = weight * sum_b pi(s,b) * (indicator(a==b) - pi(s,a)) * term'(b)
    // where term'(b) is the derivative of the objective term in pi(s,b), since
    // r(s,b) = pi(s,b)/pi_old(s,b).
    const termDeriv: number[] = [];
    for (let b = 0; b < N_ACTIONS; b++) {
      termDeriv.push(
        objectiveTermDeriv(params, at(r, b), at(adv, b), at(oldp, b), curr, b),
      );
    }

    for (let a = 0; a < N_ACTIONS; a++) {
      let g = 0;
      for (let b = 0; b < N_ACTIONS; b++) {
        const indicator = a === b ? 1 : 0;
        // d pi(s,b) / d theta(s,a) = pi(s,b) * (indicator - pi(s,a))
        g += at(curr, b) * (indicator - at(curr, a)) * at(termDeriv, b);
      }
      grad[s * N_ACTIONS + a] = weight * g;
    }
  }
  return { grad, rng };
}

// d(objective term)/d pi(s,b). The objective term is a function of the ratio
// r = pi/pi_old, so its derivative in pi is term'(r) / pi_old.
//   - cpi:  term = r*A           -> dterm/dpi = A / pi_old
//   - clip: term = min(r*A, clip(r,1-e,1+e)*A); in the flat region the gradient
//           is exactly zero, which is what stops the overshoot. Elsewhere it is
//           the cpi gradient.
//   - kl:   term = r*A - beta*KL(pi_old||pi); d(-KL)/dpi = beta * pi_old/pi.
function objectiveTermDeriv(
  params: Params,
  r: number,
  advantage: number,
  piOld: number,
  curr: number[],
  b: number,
): number {
  const safeOld = Math.max(piOld, 1e-12);
  if (params.objective === "cpi") {
    return advantage / safeOld;
  }
  if (params.objective === "kl") {
    const cpiGrad = advantage / safeOld;
    // d/dpi(s,b) of -beta * sum_a pi_old(a) ln(pi_old(a)/pi(a)) = beta*pi_old(b)/pi(b)
    const klGrad = (params.beta * safeOld) / Math.max(at(curr, b), 1e-12);
    return cpiGrad + klGrad;
  }
  // clip: the min picks the clipped branch exactly when that branch is smaller.
  const lo = 1 - params.epsilon;
  const hi = 1 + params.epsilon;
  const unclipped = r * advantage;
  const clippedR = Math.min(Math.max(r, lo), hi);
  const clippedTerm = clippedR * advantage;
  // The objective is the min of the two terms. Its gradient is the gradient of
  // whichever term is currently active. The clipped term is flat in r whenever r
  // sits outside [lo, hi], so its derivative is 0 there.
  if (clippedTerm <= unclipped) {
    const flat = r > hi || r < lo;
    return flat ? 0 : advantage / safeOld;
  }
  return advantage / safeOld;
}

// One inner epoch: a single gradient-ascent step on the surrogate, against the
// frozen snapshot. After `epochs` of these, the round ends and a new snapshot is
// frozen.
function epochStep(state: SimState): SimState {
  const { grad, rng } = surrogateGradient(state);
  const theta = state.theta.map((value, index) =>
    Math.min(
      Math.max(value + state.params.alpha * at(grad, index), -THETA_CLAMP),
      THETA_CLAMP,
    ),
  );
  return derive({ ...state, theta, rng, epoch: state.epoch + 1 });
}

// One tick advances the simulation by one inner epoch. When a round's epochs are
// spent, the next tick first closes the round (logs it, freezes a new snapshot)
// and then takes the first epoch of the new round, so a tick always makes
// visible progress.
function advance(state: SimState): SimState {
  let working = state;
  if (working.epoch >= working.params.epochs) {
    working = closeRound(working);
    working = freezeSnapshot(working);
    working = { ...working, epoch: 0 };
  }
  const stepped = epochStep(working);
  return {
    ...stepped,
    rhoHistory: [...state.rhoHistory, stepped.rho].slice(-HISTORY_LEN),
    klHistory: [...state.klHistory, stepped.klOldToNew].slice(-HISTORY_LEN),
  };
}

// Closes a PPO round: bumps the round counter and logs the moment worth
// rewinding to. A round that blew the policy far past the trust region (large
// KL) or that reached near-optimal return is what the reader wants to jump back
// to, so each logs on its transition.
function closeRound(state: SimState): SimState {
  const next: SimState = { ...state, round: state.round + 1 };
  const optimal = next.optimal;
  if (optimal <= 1e-9) return next;
  const blew = next.klOldToNew > 0.5;
  if (blew) {
    return withLog(
      next,
      `round ${next.round}: KL ${next.klOldToNew.toFixed(2)} blew past the band`,
    );
  }
  if (next.rho >= 0.95 * optimal && state.rho < 0.95 * optimal) {
    return withLog(next, `round ${next.round}: reached near-optimal return`);
  }
  return next;
}

function snapshotOf(state: SimState): Snapshot {
  const { log: _log, ...rest } = state;
  void _log;
  return rest;
}

function withLog(state: SimState, label: string): SimState {
  return {
    ...state,
    log: record(state.log, state.round, label, snapshotOf(state)),
  };
}

export interface Status {
  rho: number;
  optimal: number;
  fractionOfOptimal: number;
  kl: number;
  clippedFraction: number;
  inTrustRegion: boolean; // KL stayed small, the property clip is meant to keep
  diverged: boolean; // the policy blew up, the cpi failure mode
  nearOptimal: boolean;
}

const TRUST_KL = 0.08;
const DIVERGE_KL = 0.5;

// How the run is doing, scored against the optimal return the learner never sees
// and against the KL the clip is supposed to bound. inTrustRegion is the safety
// light: clip keeps it lit, many epochs of cpi flip it.
export function status(state: SimState): Status {
  const optimal = state.optimal;
  const fraction = optimal <= 1e-9 ? 0 : state.rho / optimal;
  return {
    rho: state.rho,
    optimal,
    fractionOfOptimal: fraction,
    kl: state.klOldToNew,
    clippedFraction: state.clippedFraction,
    inTrustRegion: state.klOldToNew <= TRUST_KL,
    diverged: state.klOldToNew > DIVERGE_KL,
    nearOptimal: fraction >= 0.95,
  };
}

export type Intervention =
  | { type: "setObjective"; value: Objective }
  | { type: "setEpsilon"; value: number }
  | { type: "setEpochs"; value: number }
  | { type: "setAlpha"; value: number }
  | { type: "setBeta"; value: number }
  | { type: "setMinibatch"; value: boolean }
  | { type: "setNoise"; value: number }
  | { type: "resetPolicy" }
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
  seed: number;
}

const BASE: Params = {
  objective: "clip",
  epsilon: 0.2,
  epochs: 10,
  alpha: 3,
  gamma: 0.97,
  beta: 1,
  minibatch: false,
  noise: 0,
};

export const PRESETS: Preset[] = [
  {
    id: "clip",
    label: "Clipped objective",
    blurb:
      "The paper's main method, epsilon 0.2 with 10 reuse epochs per round. Press play and watch the return climb to optimal while the trust-region light stays lit, because the clip flattens the gradient the instant a ratio leaves the band.",
    params: { ...BASE, objective: "clip", epsilon: 0.2, epochs: 10, alpha: 3 },
    seed: 1,
  },
  {
    id: "overshoot",
    label: "No clip, many epochs",
    blurb:
      "The plain L_CPI objective with 40 reuse epochs and a big step, nothing holding the ratio. The policy chews one batch far too greedily and the KL blows past the band, the destructive update the paper warns about. Switch the objective to clip mid-run and watch the trust-region light come back on.",
    params: { ...BASE, objective: "cpi", epochs: 40, alpha: 6 },
    seed: 1,
  },
  {
    id: "kl",
    label: "KL penalty",
    blurb:
      "The penalty variant from Section 4, a soft beta times KL term instead of the hard clip. It keeps the policy near the snapshot but is touchier than clip: turn beta down toward zero and watch it slide toward the overshoot.",
    params: { ...BASE, objective: "kl", epochs: 10, beta: 3, alpha: 2 },
    seed: 1,
  },
  {
    id: "noisy",
    label: "Noisy advantages",
    blurb:
      "The clip objective again, but now the advantage carries the sampling noise real PPO sees. The clip still bounds each step, so the trust-region light holds and the return climbs through the jitter. Raise the noise and watch the climb get bumpier without diverging.",
    params: {
      ...BASE,
      objective: "clip",
      epsilon: 0.2,
      epochs: 10,
      alpha: 3,
      noise: 0.4,
    },
    seed: 7,
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(
  presetId = "clip",
  seedOverride?: number,
): SimState {
  const preset = getPreset(presetId);
  const base: SimState = {
    params: { ...preset.params },
    theta: new Array<number>(N_STATES * N_ACTIONS).fill(0),
    thetaOld: new Array<number>(N_STATES * N_ACTIONS).fill(0),
    pi: [],
    piOld: [],
    values: [],
    q: [],
    advantageOld: [],
    visitationOld: [],
    ratio: [],
    clippedFraction: 0,
    klOldToNew: 0,
    rho: 0,
    optimal: optimalReturn(preset.params.gamma),
    rng: (seedOverride ?? preset.seed) >>> 0,
    round: 0,
    epoch: preset.params.epochs, // forces the first tick to freeze a snapshot
    rhoHistory: [],
    klHistory: [],
    log: emptyLog<Snapshot>(),
  };
  return derive(freezeSnapshotShallow(base));
}

// Like freezeSnapshot but without consuming the rng (used only at construction,
// where there is no noise to draw yet because epoch is forced and the first real
// tick will freeze properly).
function freezeSnapshotShallow(state: SimState): SimState {
  const thetaOld = [...state.theta];
  const piOld: number[][] = [];
  for (let s = 0; s < N_STATES; s++) piOld.push(softmax(thetaOld, s));
  const { values, q } = evaluatePolicy(piOld, state.params.gamma);
  const advantageOld: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    for (let a = 0; a < N_ACTIONS; a++)
      row.push(at(at(q, s), a) - at(values, s));
    advantageOld.push(row);
  }
  const visitationOld = discountedVisitation(piOld, state.params.gamma);
  return { ...state, thetaOld, advantageOld, visitationOld };
}

function withParams(state: SimState, patch: Partial<Params>): SimState {
  const params = { ...state.params, ...patch };
  return derive({ ...state, params, optimal: optimalReturn(params.gamma) });
}

function applyIntervention(
  state: SimState,
  action: Intervention,
): { state: SimState; label: string } {
  switch (action.type) {
    case "setObjective":
      return {
        state: withParams(state, { objective: action.value }),
        label: `objective -> ${action.value}`,
      };
    case "setEpsilon":
      return {
        state: withParams(state, { epsilon: action.value }),
        label: `epsilon -> ${action.value.toFixed(2)}`,
      };
    case "setEpochs":
      return {
        state: { ...state, params: { ...state.params, epochs: action.value } },
        label: `epochs -> ${action.value}`,
      };
    case "setAlpha":
      return {
        state: { ...state, params: { ...state.params, alpha: action.value } },
        label: `step size -> ${action.value.toFixed(2)}`,
      };
    case "setBeta":
      return {
        state: withParams(state, { beta: action.value }),
        label: `beta -> ${action.value.toFixed(2)}`,
      };
    case "setMinibatch":
      return {
        state: {
          ...state,
          params: { ...state.params, minibatch: action.value },
        },
        label: `minibatch sampling ${action.value ? "on" : "off"}`,
      };
    case "setNoise":
      return {
        state: { ...state, params: { ...state.params, noise: action.value } },
        label: `advantage noise -> ${action.value.toFixed(2)}`,
      };
    case "resetPolicy":
      return {
        state: derive(
          freezeSnapshotShallow({
            ...state,
            theta: new Array<number>(N_STATES * N_ACTIONS).fill(0),
            epoch: state.params.epochs,
            round: 0,
            rhoHistory: [],
            klHistory: [],
          }),
        ),
        label: "policy reset to uniform",
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

// The pure reducer. A tick runs one PPO inner epoch (closing and re-snapshotting
// the round when its epochs are spent); an intervention changes a knob or the
// policy, logging the moment just before it took effect so the event log can
// rewind to it.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return advance(state);
  const before = snapshotOf(state);
  const { state: changed, label } = applyIntervention(state, input.action);
  return { ...changed, log: record(state.log, state.round, label, before) };
}

// The two functions the view uses to label the ratio against the clip band.
export function clipLow(epsilon: number): number {
  return 1 - epsilon;
}
export function clipHigh(epsilon: number): number {
  return 1 + epsilon;
}
