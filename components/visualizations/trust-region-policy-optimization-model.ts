// A pure, deterministic model of Trust Region Policy Optimization (Schulman et
// al., 2015) on a small gridworld the model solves exactly. A softmax policy
// pi_theta(a;s) runs on a 5x5 MDP with a goal worth +1 and a trap worth -1.
// Each tick is one TRPO policy-improvement round.
//
// The round follows the paper's practical algorithm:
//
//   1. Solve the current policy exactly for V^pi, Q^pi, the advantage A^pi, and
//      the normalized discounted state visitation rho under the start state.
//
//   2. Form the surrogate objective L_theta_old(theta), the local linear model
//      of how much a new policy improves the true return eta. Equation 3/14 of
//      the paper, L(theta) = eta(old) + sum_s rho(s) sum_a pi_theta(a;s) A(s,a).
//      Its gradient in the policy logits is g.
//
//   3. Compute the natural-gradient direction A^-1 g, where A is the Fisher
//      information of the policy (the Hessian of the KL divergence). For a
//      softmax over a finite MDP the per-state Fisher is exact: F_s = diag(p) -
//      p p^T, and the mean-KL Hessian is rho-weighted. Section 6 / equation 17.
//
//   4. Scale the natural step so the predicted mean KL equals the trust-region
//      radius delta, then run TRPO's backtracking line search: halve the step
//      until the *true* surrogate improves and the measured mean KL stays under
//      delta. Equation 11/12, the constraint max_theta L subject to
//      mean-KL(old, theta) <= delta.
//
// Three update modes make the central claim pokeable:
//
//   - trustRegion: the TRPO step above. The KL constraint keeps the new policy
//     close enough that the surrogate stays honest, so the true return rises
//     almost every round and the climb is fast.
//   - penalty: the theory's KL-penalty form, max_theta L - C * KL with the large
//     C the bound recommends. The steps are tiny and the climb crawls (the
//     paper's reason for using a constraint instead of a penalty).
//   - unconstrained: a plain gradient step with a large fixed size and no KL
//     limit. It steps past where the surrogate is trustworthy, so the surrogate
//     keeps predicting gain while the true return lurches and drops.
//
// The model is deterministic. No Math.random and no wall-clock reach the math;
// a seed only sets a fixed advantage-estimate bias pattern, so the same seed and
// inputs reproduce a run exactly.

// ---- inlined deterministic helpers (kept self-contained on purpose) --------

function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) throw new RangeError(`index ${index} out of range`);
  return value;
}

// One step of the mulberry32 recurrence as a pure function: given the
// generator's state, return the next value in [0, 1) and the next state.
function nextRandom(state: number): { value: number; state: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, state: a };
}

export interface SimEvent<TSnapshot> {
  id: number;
  tick: number;
  label: string;
  snapshot: TSnapshot;
}

export interface EventLog<TSnapshot> {
  events: SimEvent<TSnapshot>[];
  nextEventId: number;
}

function emptyLog<TSnapshot>(): EventLog<TSnapshot> {
  return { events: [], nextEventId: 1 };
}

const LOG_LIMIT = 60;

function record<TSnapshot>(
  log: EventLog<TSnapshot>,
  tick: number,
  label: string,
  snapshot: TSnapshot,
): EventLog<TSnapshot> {
  const event: SimEvent<TSnapshot> = {
    id: log.nextEventId,
    tick,
    label,
    snapshot,
  };
  return {
    events: [...log.events, event].slice(-LOG_LIMIT),
    nextEventId: log.nextEventId + 1,
  };
}

// ---- gridworld MDP ---------------------------------------------------------

export const GRID_W = 5;
export const GRID_H = 5;
export const N_STATES = GRID_W * GRID_H;
export const N_ACTIONS = 4;

// 0 up, 1 down, 2 left, 3 right.
export const ACTION_LABELS = ["up", "down", "left", "right"] as const;
const ACTION_DROW = [-1, 1, 0, 0];
const ACTION_DCOL = [0, 0, -1, 1];

export interface Mdp {
  start: number;
  goal: number;
  trap: number;
}

const GRID: Mdp = {
  start: 4 * GRID_W + 0, // bottom-left
  goal: 0 * GRID_W + 4, // top-right
  trap: 2 * GRID_W + 2, // center
};

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

export function rowOf(state: number): number {
  return Math.floor(state / GRID_W);
}

export function colOf(state: number): number {
  return state % GRID_W;
}

export function isTerminal(mdp: Mdp, state: number): boolean {
  return state === mdp.goal || state === mdp.trap;
}

function enterReward(mdp: Mdp, state: number): number {
  if (state === mdp.goal) return 1;
  if (state === mdp.trap) return -1;
  return 0;
}

function nextState(state: number, action: number): number {
  const row = clamp(rowOf(state) + at(ACTION_DROW, action), 0, GRID_H - 1);
  const col = clamp(colOf(state) + at(ACTION_DCOL, action), 0, GRID_W - 1);
  return row * GRID_W + col;
}

// Gaussian elimination with partial pivoting. The MDP is tiny (25 states), so a
// direct solve is exact and faster than iterating the Bellman map near gamma 1.
function solveLinear(a: number[][], b: number[]): number[] {
  const n = b.length;
  const m = a.map((row, i) => [...row, at(b, i)]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(at(at(m, row), col)) > Math.abs(at(at(m, pivot), col))) {
        pivot = row;
      }
    }
    const tmp = at(m, col);
    m[col] = at(m, pivot);
    m[pivot] = tmp;
    const pivotVal = at(at(m, col), col);
    if (Math.abs(pivotVal) < 1e-14) continue;
    const pivotRow = at(m, col);
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const targetRow = at(m, row);
      const factor = at(targetRow, col) / pivotVal;
      if (factor === 0) continue;
      for (let k = col; k <= n; k++) {
        targetRow[k] = at(targetRow, k) - factor * at(pivotRow, k);
      }
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    const pivotVal = at(at(m, i), i);
    x[i] = Math.abs(pivotVal) < 1e-14 ? 0 : at(at(m, i), n) / pivotVal;
  }
  return x;
}

function policyTransition(
  mdp: Mdp,
  pi: number[][],
): { p: number[][]; reward: number[] } {
  const p: number[][] = [];
  const reward = new Array<number>(N_STATES).fill(0);
  for (let s = 0; s < N_STATES; s++) {
    const row = new Array<number>(N_STATES).fill(0);
    if (!isTerminal(mdp, s)) {
      const probs = at(pi, s);
      for (let a = 0; a < N_ACTIONS; a++) {
        const ns = nextState(s, a);
        row[ns] = at(row, ns) + at(probs, a);
        reward[s] = at(reward, s) + at(probs, a) * enterReward(mdp, ns);
      }
    }
    p.push(row);
  }
  return { p, reward };
}

// Exact policy evaluation: V solves (I - gamma P^pi) V = R^pi, then Q reads off.
function evaluatePolicy(
  mdp: Mdp,
  p: number[][],
  reward: number[],
  gamma: number,
): { values: number[]; q: number[][] } {
  const a: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row = new Array<number>(N_STATES).fill(0);
    for (let s2 = 0; s2 < N_STATES; s2++) {
      row[s2] = (s === s2 ? 1 : 0) - gamma * at(at(p, s), s2);
    }
    a.push(row);
  }
  const values = solveLinear(a, reward);
  const q: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    for (let action = 0; action < N_ACTIONS; action++) {
      const ns = nextState(s, action);
      row.push(
        enterReward(mdp, ns) +
          (isTerminal(mdp, ns) ? 0 : gamma * at(values, ns)),
      );
    }
    q.push(row);
  }
  return { values, q };
}

// Normalized discounted state visitation rho(s) = (1-gamma) sum_t gamma^t
// P(s_t=s) from the start, solving (I - gamma P^T) d = e_start. The surrogate
// weights advantages by this distribution (the paper's rho_theta_old).
function startVisitation(
  p: number[][],
  gamma: number,
  start: number,
): number[] {
  const a: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row = new Array<number>(N_STATES).fill(0);
    for (let s2 = 0; s2 < N_STATES; s2++) {
      row[s2] = (s === s2 ? 1 : 0) - gamma * at(at(p, s2), s);
    }
    a.push(row);
  }
  const b = new Array<number>(N_STATES).fill(0);
  b[start] = 1;
  const d = solveLinear(a, b);
  const sum = d.reduce((acc, x) => acc + x, 0);
  return d.map((x) => (sum < 1e-12 ? 0 : x / sum));
}

function optimalValues(mdp: Mdp, gamma: number): number[] {
  const values = new Array<number>(N_STATES).fill(0);
  for (let iter = 0; iter < 2000; iter++) {
    let delta = 0;
    for (let s = 0; s < N_STATES; s++) {
      if (isTerminal(mdp, s)) continue;
      let best = -Infinity;
      for (let a = 0; a < N_ACTIONS; a++) {
        const ns = nextState(s, a);
        const r = enterReward(mdp, ns);
        best = Math.max(
          best,
          r + (isTerminal(mdp, ns) ? 0 : gamma * at(values, ns)),
        );
      }
      delta = Math.max(delta, Math.abs(best - at(values, s)));
      values[s] = best;
    }
    if (delta < 1e-12) break;
  }
  return values;
}

// ---- softmax policy --------------------------------------------------------

// The policy is parameterized by logits theta[state][action]; pi = softmax(theta)
// per state. This is the differentiable, general stochastic policy class TRPO
// targets, in contrast to the conservative-policy-iteration mixture.
function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const total = exps.reduce((s, v) => s + v, 0) || 1;
  return exps.map((v) => v / total);
}

function policyFromLogits(mdp: Mdp, theta: number[][]): number[][] {
  return theta.map((logits, s) =>
    isTerminal(mdp, s) ? new Array<number>(N_ACTIONS).fill(0) : softmax(logits),
  );
}

function meanKL(
  mdp: Mdp,
  rho: number[],
  piOld: number[][],
  piNew: number[][],
): number {
  let kl = 0;
  for (let s = 0; s < N_STATES; s++) {
    if (isTerminal(mdp, s)) continue;
    const oldP = at(piOld, s);
    const newP = at(piNew, s);
    let stateKL = 0;
    for (let a = 0; a < N_ACTIONS; a++) {
      const po = at(oldP, a);
      const pn = at(newP, a);
      if (po > 1e-12 && pn > 1e-12) stateKL += po * Math.log(po / pn);
    }
    kl += at(rho, s) * stateKL;
  }
  return Math.max(0, kl);
}

function maxKL(mdp: Mdp, piOld: number[][], piNew: number[][]): number {
  let worst = 0;
  for (let s = 0; s < N_STATES; s++) {
    if (isTerminal(mdp, s)) continue;
    const oldP = at(piOld, s);
    const newP = at(piNew, s);
    let stateKL = 0;
    for (let a = 0; a < N_ACTIONS; a++) {
      const po = at(oldP, a);
      const pn = at(newP, a);
      if (po > 1e-12 && pn > 1e-12) stateKL += po * Math.log(po / pn);
    }
    worst = Math.max(worst, stateKL);
  }
  return worst;
}

// ---- model types -----------------------------------------------------------

export type UpdateMode = "trustRegion" | "penalty" | "unconstrained";

export interface Params {
  update: UpdateMode;
  delta: number; // trust-region radius (mean-KL budget), the paper's delta
  gamma: number;
  estError: number; // l-infinity bias on the advantage estimate (seeded, fixed per state-action)
}

const HISTORY_LEN = 200;
// Once the surrogate gradient is this small the policy has converged and the run
// stops on its own.
const SETTLE_GRAD = 2e-3;
// The penalty form's coefficient, large the way the theory's C is, so its steps
// are tiny. C ~ 4 eps gamma / (1-gamma)^2 in the paper; here a fixed large value.
const PENALTY_C = 60;
// The logit push the unconstrained mode adds toward the surrogate-greedy action
// each round. It is large enough to drive the policy almost fully to the action
// the (error-laden) advantage estimate likes best, the uncontrolled full step
// TRPO's trust region replaces.
const UNCONSTRAINED_PUSH = 6;

export interface SimState {
  mdp: Mdp;
  params: Params;
  theta: number[][]; // policy logits [state][action]
  pi: number[][]; // softmax(theta), the current policy
  values: number[]; // V^pi exact
  advantage: number[][]; // A^pi exact (true)
  rho: number[]; // normalized discounted visitation from the start
  gradNorm: number; // norm of the surrogate gradient at the current policy
  // The quantities of the last applied update:
  appliedKL: number; // measured mean KL of the step actually taken
  appliedMaxKL: number; // measured max KL of the step taken
  surrogateGain: number; // predicted improvement L(new) - L(old)
  realGain: number; // true improvement eta(new) - eta(old)
  backtracks: number; // line-search halvings the step needed (trust region only)
  eta: number; // true objective eta = V^pi(start)
  optimal: number; // V*(start), for scoring
  seed: number;
  steps: number;
  done: boolean;
  etaHistory: number[]; // true return per round
  surrHistory: number[]; // surrogate-predicted return per round (cumulative)
  log: EventLog<Snapshot>;
}

export type Snapshot = Omit<SimState, "log">;

export type Intervention =
  | { type: "setUpdate"; value: UpdateMode }
  | { type: "setDelta"; value: number }
  | { type: "setGamma"; value: number }
  | { type: "setEstError"; value: number }
  | { type: "setSeed"; value: number }
  | { type: "resetPolicy" }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: Snapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

// ---- the surrogate, its gradient, and the natural step --------------------

// A fixed, seeded per-(state,action) bias on the advantage estimate, the way a
// learned value function carries error. Deterministic in the seed.
function estimationBias(seed: number, estError: number): number[][] {
  if (estError === 0 || seed === 0) {
    return Array.from({ length: N_STATES }, () =>
      new Array<number>(N_ACTIONS).fill(0),
    );
  }
  let rng = seed >>> 0;
  const bias: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    for (let a = 0; a < N_ACTIONS; a++) {
      const draw = nextRandom(rng);
      rng = draw.state;
      row.push(estError * (2 * draw.value - 1));
    }
    bias.push(row);
  }
  return bias;
}

// The surrogate gradient in the logits. L(theta) = const + sum_s rho(s) sum_a
// pi_theta(a;s) Ahat(s,a). For softmax logits the gradient of pi(a;s) w.r.t.
// theta(s,b) is pi(a;s)(delta_ab - pi(b;s)), so dL/dtheta(s,b) = rho(s) pi(b;s)
// (Ahat(s,b) - sum_a pi(a;s) Ahat(s,a)). That inner term is the (estimated)
// advantage of action b centered by the policy's mean advantage.
function surrogateGradient(state: SimState, bias: number[][]): number[][] {
  const { mdp, pi, advantage, rho } = state;
  const grad: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row = new Array<number>(N_ACTIONS).fill(0);
    if (!isTerminal(mdp, s)) {
      const p = at(pi, s);
      const advRow = at(advantage, s);
      const biasRow = at(bias, s);
      let mean = 0;
      for (let a = 0; a < N_ACTIONS; a++) {
        mean += at(p, a) * (at(advRow, a) + at(biasRow, a));
      }
      for (let b = 0; b < N_ACTIONS; b++) {
        const adv = at(advRow, b) + at(biasRow, b);
        row[b] = at(rho, s) * at(p, b) * (adv - mean);
      }
    }
    grad.push(row);
  }
  return grad;
}

// Apply the per-state Fisher inverse to a gradient direction. For a softmax the
// per-state Fisher is F_s = diag(p) - p p^T, singular along the all-ones
// direction. The gradient already lives in the tangent space (sum_b grad_b = 0
// because it is rho * p * (adv - mean)), so the natural direction that solves
// F_s x = g_s in that subspace is x_b = g_b / p_b (mean-centered). That is the
// exact natural gradient for this policy class, the analytic Fisher of
// equation 17 specialized to a tabular softmax.
function naturalDirection(state: SimState, grad: number[][]): number[][] {
  const { mdp, pi, rho } = state;
  const dir: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row = new Array<number>(N_ACTIONS).fill(0);
    const r = at(rho, s);
    if (!isTerminal(mdp, s) && r > 1e-12) {
      const p = at(pi, s);
      const g = at(grad, s);
      let mean = 0;
      for (let a = 0; a < N_ACTIONS; a++) {
        const pa = at(p, a);
        if (pa > 1e-12) mean += g[a] !== undefined ? at(g, a) / r : 0;
      }
      mean /= N_ACTIONS;
      for (let b = 0; b < N_ACTIONS; b++) {
        const pb = at(p, b);
        // g_b = r * p_b * centeredAdv_b, so g_b / (r * p_b) recovers the
        // centered advantage, which is exactly F_s^+ applied in this subspace.
        const centeredAdv = pb > 1e-12 ? at(g, b) / (r * pb) : 0;
        row[b] = centeredAdv - mean;
      }
    }
    dir.push(row);
  }
  return dir;
}

function gradientNorm(grad: number[][]): number {
  let sum = 0;
  for (const row of grad) for (const v of row) sum += v * v;
  return Math.sqrt(sum);
}

// The quadratic mean-KL of a step beta * dir from the current policy, using the
// Fisher: KL ~ (1/2) beta^2 dir^T F dir with F the rho-weighted per-state
// Fisher. This is what TRPO uses to scale the natural step so the predicted KL
// equals delta before the line search refines it.
function quadraticKL(state: SimState, dir: number[][]): number {
  const { mdp, pi, rho } = state;
  let quad = 0;
  for (let s = 0; s < N_STATES; s++) {
    if (isTerminal(mdp, s)) continue;
    const p = at(pi, s);
    const d = at(dir, s);
    let dp = 0;
    let dFd = 0;
    for (let a = 0; a < N_ACTIONS; a++) dp += at(p, a) * at(d, a);
    for (let a = 0; a < N_ACTIONS; a++) {
      const da = at(d, a);
      dFd += at(p, a) * da * da;
    }
    dFd -= dp * dp; // d^T (diag(p) - p p^T) d
    quad += at(rho, s) * dFd;
  }
  return 0.5 * quad;
}

function stepLogits(
  theta: number[][],
  dir: number[][],
  beta: number,
): number[][] {
  return theta.map((row, s) =>
    row.map((v, a) => v + beta * (dir[s]?.[a] ?? 0)),
  );
}

// The surrogate evaluated at a candidate policy: sum_s rho_old(s) sum_a
// pi_cand(a;s) Ahat_old(s,a), using the old visitation and old advantage (the
// surrogate freezes both at theta_old, which is exactly why it stops being
// trustworthy once the policy moves far).
function surrogateValue(
  state: SimState,
  piCand: number[][],
  bias: number[][],
): number {
  const { mdp, advantage, rho } = state;
  let value = 0;
  for (let s = 0; s < N_STATES; s++) {
    if (isTerminal(mdp, s)) continue;
    const cand = at(piCand, s);
    const advRow = at(advantage, s);
    const biasRow = at(bias, s);
    for (let a = 0; a < N_ACTIONS; a++) {
      value += at(rho, s) * at(cand, a) * (at(advRow, a) + at(biasRow, a));
    }
  }
  return value;
}

// ---- the round -------------------------------------------------------------

// Recompute every exact quantity that follows from the current logits: the
// policy, its value and advantage, the discounted visitation, and the surrogate
// gradient norm. Pure and seed-only.
function derive(state: SimState): SimState {
  const { mdp, params, theta } = state;
  const pi = policyFromLogits(mdp, theta);
  const { p, reward } = policyTransition(mdp, pi);
  const { values, q } = evaluatePolicy(mdp, p, reward, params.gamma);
  const advantage: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    for (let a = 0; a < N_ACTIONS; a++)
      row.push(at(at(q, s), a) - at(values, s));
    advantage.push(row);
  }
  const rho = startVisitation(p, params.gamma, mdp.start);
  const next: SimState = {
    ...state,
    pi,
    values,
    advantage,
    rho,
    eta: at(values, mdp.start),
  };
  const bias = estimationBias(state.seed, params.estError);
  next.gradNorm = gradientNorm(surrogateGradient(next, bias));
  return next;
}

interface StepResult {
  theta: number[][];
  kl: number;
  maxKLValue: number;
  surrogateGain: number;
  backtracks: number;
}

// TRPO's trust-region step: natural-gradient direction, scaled so the quadratic
// KL equals delta, then backtracking line search to the largest fraction whose
// measured mean KL stays under delta and whose surrogate actually improves.
function trustRegionStep(state: SimState, bias: number[][]): StepResult {
  const grad = surrogateGradient(state, bias);
  const dir = naturalDirection(state, grad);
  const quad = quadraticKL(state, dir);
  const baseL = surrogateValue(state, state.pi, bias);
  if (quad <= 1e-12) {
    return {
      theta: state.theta,
      kl: 0,
      maxKLValue: 0,
      surrogateGain: 0,
      backtracks: 0,
    };
  }
  // beta so that (1/2) beta^2 dir^T F dir = delta.
  const fullBeta = Math.sqrt(state.params.delta / quad);
  let frac = 1;
  for (let i = 0; i < 12; i++) {
    const beta = fullBeta * frac;
    const candTheta = stepLogits(state.theta, dir, beta);
    const candPi = policyFromLogits(state.mdp, candTheta);
    const kl = meanKL(state.mdp, state.rho, state.pi, candPi);
    const gain = surrogateValue(state, candPi, bias) - baseL;
    if (kl <= state.params.delta + 1e-9 && gain > 0) {
      return {
        theta: candTheta,
        kl,
        maxKLValue: maxKL(state.mdp, state.pi, candPi),
        surrogateGain: gain,
        backtracks: i,
      };
    }
    frac *= 0.6;
  }
  return {
    theta: state.theta,
    kl: 0,
    maxKLValue: 0,
    surrogateGain: 0,
    backtracks: 12,
  };
}

// The KL-penalty form the theory justifies: take a natural step that maximizes
// L - C * KL. For the quadratic KL model the optimal step scales the natural
// direction by 1 / C, so a large C (as the bound demands) gives a tiny step.
function penaltyStep(state: SimState, bias: number[][]): StepResult {
  const grad = surrogateGradient(state, bias);
  const dir = naturalDirection(state, grad);
  const beta = 1 / PENALTY_C;
  const candTheta = stepLogits(state.theta, dir, beta);
  const candPi = policyFromLogits(state.mdp, candTheta);
  const baseL = surrogateValue(state, state.pi, bias);
  return {
    theta: candTheta,
    kl: meanKL(state.mdp, state.rho, state.pi, candPi),
    maxKLValue: maxKL(state.mdp, state.pi, candPi),
    surrogateGain: surrogateValue(state, candPi, bias) - baseL,
    backtracks: 0,
  };
}

// An uncontrolled full step with no KL limit, the kind of update TRPO replaces.
// It pushes the policy almost fully onto the action the (error-laden) advantage
// estimate likes best in every visited state, the analog of the greedy alpha = 1
// step. Because the estimate carries error and the big move shifts which states
// the agent then visits, the surrogate (frozen at the old visitation and the
// same biased estimate) keeps predicting a gain while the true return can fall.
function unconstrainedStep(state: SimState, bias: number[][]): StepResult {
  const { mdp, pi, advantage, rho } = state;
  const candTheta = state.theta.map((logits, s) => {
    if (isTerminal(mdp, s) || at(rho, s) <= 1e-12) return [...logits];
    const advRow = at(advantage, s);
    const biasRow = at(bias, s);
    let bestA = 0;
    let bestVal = -Infinity;
    for (let a = 0; a < N_ACTIONS; a++) {
      const est = at(advRow, a) + at(biasRow, a);
      if (est > bestVal) {
        bestVal = est;
        bestA = a;
      }
    }
    return logits.map((v, a) => v + (a === bestA ? UNCONSTRAINED_PUSH : 0));
  });
  const candPi = policyFromLogits(mdp, candTheta);
  const baseL = surrogateValue(state, pi, bias);
  return {
    theta: candTheta,
    kl: meanKL(mdp, rho, pi, candPi),
    maxKLValue: maxKL(mdp, pi, candPi),
    surrogateGain: surrogateValue(state, candPi, bias) - baseL,
    backtracks: 0,
  };
}

function computeStep(state: SimState, bias: number[][]): StepResult {
  if (state.params.update === "penalty") return penaltyStep(state, bias);
  if (state.params.update === "unconstrained")
    return unconstrainedStep(state, bias);
  return trustRegionStep(state, bias);
}

function advance(state: SimState): SimState {
  if (state.done) return state;
  const bias = estimationBias(state.seed, state.params.estError);
  const etaBefore = state.eta;
  const result = computeStep(state, bias);

  const moved = derive({
    ...state,
    theta: result.theta,
    steps: state.steps + 1,
  });
  const realGain = moved.eta - etaBefore;

  const next: SimState = {
    ...moved,
    appliedKL: result.kl,
    appliedMaxKL: result.maxKLValue,
    surrogateGain: result.surrogateGain,
    realGain,
    backtracks: result.backtracks,
    etaHistory: [...state.etaHistory, moved.eta].slice(-HISTORY_LEN),
    surrHistory: [
      ...state.surrHistory,
      (state.surrHistory[state.surrHistory.length - 1] ?? etaBefore) +
        result.surrogateGain,
    ].slice(-HISTORY_LEN),
    // The smooth modes settle when the surrogate gradient vanishes. The
    // uncontrolled mode never settles on a small gradient (it keeps re-jumping to
    // the biased greedy action), so it only stops once it is genuinely near
    // optimal, which lets the reader watch it keep lurching.
    done:
      state.params.update === "unconstrained"
        ? moved.eta >= moved.optimal - 1e-3
        : moved.gradNorm < SETTLE_GRAD,
  };
  return logMilestones(state, next);
}

// Logs the moments worth rewinding to: a round where the surrogate promised gain
// but the true return fell (the overshoot TRPO's trust region prevents), and the
// round the policy settles.
function logMilestones(prev: SimState, next: SimState): SimState {
  let result = next;
  if (next.surrogateGain > 1e-4 && next.realGain < -1e-4) {
    result = withLog(
      result,
      `round ${next.steps}: surrogate +${next.surrogateGain.toFixed(3)} but true ${next.realGain.toFixed(3)}`,
    );
  } else if (next.backtracks > 0 && next.params.update === "trustRegion") {
    result = withLog(
      result,
      `round ${next.steps}: line search backtracked ${next.backtracks}x to stay in the region`,
    );
  }
  if (!prev.done && next.done) {
    result = withLog(result, `round ${next.steps}: policy settled, stopped`);
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

// ---- status, presets, reducer ---------------------------------------------

export interface Status {
  eta: number;
  optimal: number;
  fraction: number;
  appliedKL: number;
  delta: number;
  surrogateGain: number;
  realGain: number;
  overshot: boolean; // surrogate predicted gain but the true return fell
  done: boolean;
}

export function status(state: SimState): Status {
  const fraction =
    Math.abs(state.optimal) < 1e-9 ? 0 : state.eta / state.optimal;
  return {
    eta: state.eta,
    optimal: state.optimal,
    fraction,
    appliedKL: state.appliedKL,
    delta: state.params.delta,
    surrogateGain: state.surrogateGain,
    realGain: state.realGain,
    overshot: state.surrogateGain > 1e-4 && state.realGain < -1e-4,
    done: state.done,
  };
}

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: Params;
  seed: number;
}

export const PRESETS: Preset[] = [
  {
    id: "trust-region",
    label: "Trust region: steady climb",
    blurb:
      "Each round takes the natural-gradient step scaled to a mean KL of delta, then backtracks until the surrogate truly improves. The new policy stays close enough that the surrogate stays honest, so the true return climbs almost every round. Press play and watch the two curves rise together toward optimal.",
    params: { update: "trustRegion", delta: 0.02, gamma: 0.95, estError: 0.02 },
    seed: 7,
  },
  {
    id: "overshoot",
    label: "No trust region: overshoot",
    blurb:
      "Now each round takes a big plain gradient step with no KL limit, so the policy leaps past where the surrogate is trustworthy. Watch the surrogate keep predicting gain (clay-orange) while the true return (sage) lurches and drops below it. Switch the update to trust region mid-run and watch the true curve settle and climb.",
    params: {
      update: "unconstrained",
      delta: 0.02,
      gamma: 0.95,
      estError: 0.2,
    },
    seed: 7,
  },
  {
    id: "penalty",
    label: "KL penalty: crawl",
    blurb:
      "The KL-penalty form the theory justifies, max L minus a large C times KL. The penalty coefficient the bound demands is so big that every step is tiny, so the climb crawls. This is why the paper constrains the KL with a fixed budget delta instead of penalizing it. Switch to trust region mid-run and watch the steps grow.",
    params: { update: "penalty", delta: 0.02, gamma: 0.95, estError: 0.02 },
    seed: 7,
  },
  {
    id: "wide",
    label: "Trust region too wide",
    blurb:
      "Trust region again, but with delta cranked up while the advantage estimate carries error. A budget this loose lets the policy move so far each round that the surrogate stops matching reality, so the true return wobbles and can settle well short of optimal. Drag delta back down mid-run and watch the climb steady.",
    params: { update: "trustRegion", delta: 0.4, gamma: 0.95, estError: 0.25 },
    seed: 7,
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

function uniformLogits(): number[][] {
  return Array.from({ length: N_STATES }, () =>
    new Array<number>(N_ACTIONS).fill(0),
  );
}

export function initialState(
  presetId = "trust-region",
  seedOverride?: number,
): SimState {
  const preset = getPreset(presetId);
  const vStar = optimalValues(GRID, preset.params.gamma);
  const base: SimState = {
    mdp: GRID,
    params: { ...preset.params },
    theta: uniformLogits(),
    pi: [],
    values: [],
    advantage: [],
    rho: [],
    gradNorm: 0,
    appliedKL: 0,
    appliedMaxKL: 0,
    surrogateGain: 0,
    realGain: 0,
    backtracks: 0,
    eta: 0,
    optimal: at(vStar, GRID.start),
    seed: (seedOverride ?? preset.seed) >>> 0,
    steps: 0,
    done: false,
    etaHistory: [],
    surrHistory: [],
    log: emptyLog<Snapshot>(),
  };
  const derived = derive(base);
  return {
    ...derived,
    etaHistory: [derived.eta],
    surrHistory: [derived.eta],
  };
}

function refreshOptimal(state: SimState): SimState {
  const vStar = optimalValues(state.mdp, state.params.gamma);
  return { ...state, optimal: at(vStar, state.mdp.start) };
}

function applyIntervention(
  state: SimState,
  action: Intervention,
): { state: SimState; label: string } {
  switch (action.type) {
    case "setUpdate":
      return {
        state: {
          ...state,
          params: { ...state.params, update: action.value },
          done: false,
        },
        label: `update -> ${action.value}`,
      };
    case "setDelta":
      return {
        state: {
          ...state,
          params: { ...state.params, delta: action.value },
          done: false,
        },
        label: `delta -> ${action.value.toFixed(3)}`,
      };
    case "setGamma":
      return {
        state: refreshOptimal(
          derive({
            ...state,
            params: { ...state.params, gamma: action.value },
            done: false,
          }),
        ),
        label: `gamma -> ${action.value.toFixed(2)}`,
      };
    case "setEstError":
      return {
        state: derive({
          ...state,
          params: { ...state.params, estError: action.value },
          done: false,
        }),
        label: `estimate error -> ${action.value.toFixed(2)}`,
      };
    case "setSeed":
      return {
        state: derive({ ...state, seed: action.value >>> 0, done: false }),
        label: `seed -> ${action.value}`,
      };
    case "resetPolicy":
      return {
        state: {
          ...derive({
            ...state,
            theta: uniformLogits(),
            steps: 0,
            done: false,
          }),
          etaHistory: [],
          surrHistory: [],
          appliedKL: 0,
          appliedMaxKL: 0,
          surrogateGain: 0,
          realGain: 0,
          backtracks: 0,
        },
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

// The pure reducer. A tick runs one TRPO round under the chosen update rule; an
// intervention changes a knob and logs the moment just before it took effect so
// the event log can rewind there.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return advance(state);
  const before = snapshotOf(state);
  const { state: changed, label } = applyIntervention(state, input.action);
  return { ...changed, log: record(state.log, state.steps, label, before) };
}
