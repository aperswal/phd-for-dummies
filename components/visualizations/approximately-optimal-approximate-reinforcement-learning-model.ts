// A pure, deterministic model of Conservative Policy Iteration from Kakade and
// Langford's 2002 paper "Approximately Optimal Approximate Reinforcement
// Learning". A stochastic policy pi(a;s) runs on a small gridworld MDP the model
// solves exactly. Each tick is one policy-improvement round.
//
// The round has three parts, all from the paper:
//
//   1. An approximate greedy policy chooser proposes a deterministic policy pi'.
//      It is greedy with respect to value estimates carrying an l-infinity error
//      of size epsilon (a fixed per-state bias set by the seed), so it makes the
//      kind of mistakes a real value-function approximator makes.
//
//   2. The policy advantage A = E_{s ~ d_{pi,mu}}[ A^pi(s, pi'(s)) ] is measured
//      with the *true* advantage A^pi. This is how much pi' picks high-advantage
//      actions, averaged over the states pi visits under the restart
//      distribution mu. When A <= epsilon the chooser can't find a real
//      improvement and the algorithm stops (the break point, OPT(A) < 2 epsilon).
//
//   3. The update blends, rather than replaces:  pi_new = (1 - alpha) pi + alpha pi'.
//      Greedy iteration takes alpha = 1 and can degrade, because committing fully
//      to pi' shifts which states you visit and the gain you measured no longer
//      holds. Conservative iteration takes the step that maximizes Theorem 4.1's
//      guaranteed gain, alpha* = (1 - gamma) A / (4 gamma eps) with eps the
//      per-state max advantage, so eta_mu rises every round (Corollary 4.2 gives
//      the cruder R-based form of the same step).
//
// The restart distribution mu is the other lever. Optimizing eta_mu = E_mu[V^pi]
// under a uniform mu scatters the agent across the whole map so rarely-visited
// states still count toward the policy advantage; optimizing under mu = D (the
// start state only) ignores them, so the climb stalls far from optimal under the
// true objective eta_D and the distribution mismatch ||d_{pi*,D} / mu||_inf
// blows up. The model is deterministic: the seed fixes the chooser's error
// pattern, so the same seed and inputs reproduce a run exactly.

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
  optimalValues,
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

const HISTORY_LEN = 160;
// The break point: once the policy advantage falls this low the chooser can no
// longer find a real improvement and the run stops (the paper's OPT(A) < 2 eps).
const BREAK_EPS = 0.012;

export type UpdateMode = "conservative" | "greedy" | "manual";
export type RestartMode = "uniform" | "start" | "corners";

export interface Params {
  update: UpdateMode;
  alpha: number; // manual mixture weight, used only in manual mode
  gamma: number;
  chooserError: number; // epsilon: l-infinity error of the greedy chooser and the break-point threshold
  restart: RestartMode; // the restart distribution mu
}

export interface SimState {
  mdp: Mdp;
  params: Params;
  pi: number[][]; // current stochastic policy [state][action]
  values: number[]; // V^pi per state (exact)
  q: number[][]; // Q^pi[state][action]
  advantage: number[][]; // A^pi[state][action] (true)
  visitMu: number[]; // normalized discounted visitation d_{pi,mu} (sums to 1)
  greedy: number[]; // the chooser's deterministic pi': action per non-terminal state, -1 at terminals
  policyAdvantage: number; // A = sum_s d_{pi,mu}(s) A^pi(s, pi'(s))
  maxStateAdv: number; // max_s |A^pi(s, pi'(s))|, the epsilon in Theorem 4.1's penalty
  appliedAlpha: number; // the alpha used on the last step
  etaMu: number; // E_mu[V^pi], what conservative PI improves
  etaD: number; // V^pi(start), the true objective
  optimalMu: number; // E_mu[V*]
  optimalD: number; // V*(start)
  mismatch: number; // ||d_{pi*,D} / mu||_inf, Infinity when mu leaves an optimal-visited state uncovered
  rng: number; // the seed; fixes the chooser's error pattern
  steps: number;
  done: boolean; // policy advantage <= epsilon, no improving direction left
  etaMuHistory: number[];
  etaDHistory: number[];
  log: EventLog<Snapshot>;
}

export type Snapshot = Omit<SimState, "log">;

// The restart distribution mu over start states (sums to 1). uniform spreads
// over every non-terminal cell, start puts all mass on the true start (mu = D),
// corners spreads over the non-terminal corner cells.
function restartDistribution(mdp: Mdp, mode: RestartMode): number[] {
  const mu = new Array<number>(N_STATES).fill(0);
  if (mode === "start") {
    mu[mdp.start] = 1;
    return mu;
  }
  const cells: number[] = [];
  if (mode === "corners") {
    const corners = [0, 4, 20, 24];
    for (const c of corners) if (!isTerminal(mdp, c)) cells.push(c);
  } else {
    for (let s = 0; s < N_STATES; s++) if (!isTerminal(mdp, s)) cells.push(s);
  }
  for (const c of cells) mu[c] = 1 / cells.length;
  return mu;
}

// The approximate greedy policy chooser. It is greedy with respect to value
// estimates V~(s) = V(s) + chooserError * noise(s), where each round's noise is a
// fresh draw from the seeded stream, so the chooser makes the kind of mistakes a
// re-fit value-function approximator makes. With zero error it is the exact
// greedy policy; with error it can pick wrong actions, which is what makes a full
// (alpha = 1) commitment able to drop performance. It also reads off the policy
// advantage A = sum_s d_{pi,mu}(s) A^pi(s, pi'(s)) and its per-state max, both
// from the *true* advantage, so the conservative wrapper can size a safe step
// and decide when to stop. Consumes the rng (one draw per non-terminal state),
// returning the advanced stream so the run stays replayable.
function chooseGreedy(state: SimState): SimState {
  const { mdp, params, values, advantage, visitMu } = state;
  const g = new Array<number>(N_STATES).fill(-1);
  let rng = state.rng;
  for (let s = 0; s < N_STATES; s++) {
    if (isTerminal(mdp, s)) continue;
    let bestA = 0;
    let bestVal = -Infinity;
    for (let a = 0; a < N_ACTIONS; a++) {
      const ns = nextState(s, a);
      const draw = nextRandom(rng);
      rng = draw.state;
      const noise = params.chooserError * (2 * draw.value - 1);
      const vEst = isTerminal(mdp, ns) ? 0 : at(values, ns) + noise;
      const qEst = enterReward(mdp, ns) + params.gamma * vEst;
      if (qEst > bestVal) {
        bestVal = qEst;
        bestA = a;
      }
    }
    g[s] = bestA;
  }

  let policyAdvantage = 0;
  let maxStateAdv = 0;
  for (let s = 0; s < N_STATES; s++) {
    const a = at(g, s);
    if (a < 0) continue;
    const adv = at(at(advantage, s), a);
    policyAdvantage += at(visitMu, s) * adv;
    maxStateAdv = Math.max(maxStateAdv, Math.abs(adv));
  }

  return { ...state, greedy: g, policyAdvantage, maxStateAdv, rng };
}

// The exact discounted state-distribution of an optimal policy started from D,
// used only to score the mismatch ||d_{pi*,D} / mu||_inf, never by the learner.
function optimalVisitation(mdp: Mdp, gamma: number): number[] {
  const vStar = optimalValues(mdp, gamma);
  const piStar: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row = new Array<number>(N_ACTIONS).fill(0);
    if (!isTerminal(mdp, s)) {
      let bestA = 0;
      let bestVal = -Infinity;
      for (let a = 0; a < N_ACTIONS; a++) {
        const ns = nextState(s, a);
        const val =
          enterReward(mdp, ns) +
          (isTerminal(mdp, ns) ? 0 : gamma * at(vStar, ns));
        if (val > bestVal) {
          bestVal = val;
          bestA = a;
        }
      }
      row[bestA] = 1;
    }
    piStar.push(row);
  }
  const { p } = policyTransition(mdp, piStar);
  const d = discountedVisitation(p, gamma, oneHot(mdp.start));
  const sum = d.reduce((acc, x) => acc + x, 0);
  return d.map((x) => (sum < 1e-12 ? 0 : x / sum));
}

function mismatchCoefficient(mdp: Mdp, gamma: number, mu: number[]): number {
  const dStar = optimalVisitation(mdp, gamma);
  let worst = 0;
  for (let s = 0; s < N_STATES; s++) {
    // Terminals absorb arrival mass but the agent never acts there, so only the
    // decision states need coverage by mu.
    if (isTerminal(mdp, s)) continue;
    const d = at(dStar, s);
    if (d <= 1e-9) continue;
    const m = at(mu, s);
    if (m <= 1e-12) return Infinity;
    worst = Math.max(worst, d / m);
  }
  return worst;
}

// Recomputes the exact quantities that follow from the policy and the params:
// the value function, the advantage, the discounted visitation under mu, and the
// two performance measures. Pure and rng-free. The greedy proposal that uses the
// stochastic chooser is layered on top by chooseGreedy.
function derive(state: SimState): SimState {
  const { mdp, params, pi } = state;
  const { p, reward } = policyTransition(mdp, pi);
  const { values, q } = evaluatePolicy(mdp, pi, p, reward, params.gamma);

  const advantage: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const row: number[] = [];
    for (let a = 0; a < N_ACTIONS; a++)
      row.push(at(at(q, s), a) - at(values, s));
    advantage.push(row);
  }

  const mu = restartDistribution(mdp, params.restart);
  const dRaw = discountedVisitation(p, params.gamma, mu);
  const dSum = dRaw.reduce((acc, x) => acc + x, 0);
  const visitMu = dRaw.map((x) => (dSum < 1e-12 ? 0 : x / dSum));

  let etaMu = 0;
  for (let s = 0; s < N_STATES; s++) etaMu += at(mu, s) * at(values, s);
  const etaD = at(values, mdp.start);

  return { ...state, values, q, advantage, visitMu, etaMu, etaD };
}

// Solve the exact quantities and then run the chooser on top, the pair used after
// any policy or parameter change.
function deriveAndChoose(state: SimState): SimState {
  return chooseGreedy(derive(state));
}

// The mixture policy (1 - alpha) pi + alpha pi', the heart of the conservative
// update (equation 4.1). pi' is deterministic, so it adds alpha to the chosen
// action's probability and scales the rest by (1 - alpha).
function mixPolicy(
  pi: number[][],
  greedy: number[],
  alpha: number,
): number[][] {
  return pi.map((probs, s) => {
    const g = at(greedy, s);
    if (g < 0) return [...probs];
    return probs.map(
      (value, a) => (1 - alpha) * value + alpha * (a === g ? 1 : 0),
    );
  });
}

// The step that maximizes Theorem 4.1's guaranteed gain,
// alpha* = (1 - gamma) A / (4 gamma eps) with eps the per-state max advantage.
// It sits at half the bound's zero crossing, so the step stays inside the region
// where improvement is guaranteed and eta_mu can only rise.
export function safeStep(state: SimState): number {
  const { params, policyAdvantage, maxStateAdv } = state;
  if (maxStateAdv < 1e-9 || policyAdvantage <= 0) return 0;
  return clamp(
    ((1 - params.gamma) * policyAdvantage) / (4 * params.gamma * maxStateAdv),
    0,
    1,
  );
}

// The step size at which Theorem 4.1's guaranteed gain returns to zero, twice the
// safe step. Past it the distribution-shift penalty outweighs the gain and the
// guarantee goes negative. The view zooms its bound chart to this window.
export function boundZeroCrossing(state: SimState): number {
  const { params, policyAdvantage, maxStateAdv } = state;
  if (maxStateAdv < 1e-9 || policyAdvantage <= 0) return 0;
  return (
    ((1 - params.gamma) * policyAdvantage) / (2 * params.gamma * maxStateAdv)
  );
}

// The step size for the current round. Greedy commits fully; manual uses the
// slider; conservative takes the safe step.
export function chosenAlpha(state: SimState): number {
  if (state.params.update === "greedy") return 1;
  if (state.params.update === "manual") return state.params.alpha;
  return safeStep(state);
}

// Theorem 4.1's guaranteed improvement in eta_mu as a function of the step size:
// (alpha / (1 - gamma)) (A - 2 alpha gamma eps / (1 - gamma)), with eps the
// per-state max advantage. It is an upside-down parabola in alpha: positive in a
// window, peaking at the safe step, and turning negative once the distribution
// shift outweighs the gain.
export function improvementBound(state: SimState, alpha: number): number {
  const { params, policyAdvantage, maxStateAdv } = state;
  const g = params.gamma;
  const penalty = (2 * alpha * g * maxStateAdv) / (1 - g);
  return (alpha / (1 - g)) * (policyAdvantage - penalty);
}

// The actual change in eta_mu from taking the mixture step at this alpha,
// computed exactly by solving the mixed policy. The view overlays this on the
// guaranteed-bound curve so the reader sees the real gain sit above the
// guarantee and watches both dip negative when alpha is too big.
export function actualImprovement(state: SimState, alpha: number): number {
  const mu = restartDistribution(state.mdp, state.params.restart);
  const mixed = mixPolicy(state.pi, state.greedy, alpha);
  const { p, reward } = policyTransition(state.mdp, mixed);
  const { values } = evaluatePolicy(
    state.mdp,
    mixed,
    p,
    reward,
    state.params.gamma,
  );
  let etaMu = 0;
  for (let s = 0; s < N_STATES; s++) etaMu += at(mu, s) * at(values, s);
  return etaMu - state.etaMu;
}

function advance(state: SimState): SimState {
  if (state.done) return state;

  const alpha = chosenAlpha(state);
  const pi = mixPolicy(state.pi, state.greedy, alpha);
  const advanced = deriveAndChoose({
    ...state,
    pi,
    steps: state.steps + 1,
    appliedAlpha: alpha,
  });

  const done = advanced.policyAdvantage <= BREAK_EPS;
  const next: SimState = {
    ...advanced,
    done,
    etaMuHistory: [...state.etaMuHistory, advanced.etaMu].slice(-HISTORY_LEN),
    etaDHistory: [...state.etaDHistory, advanced.etaD].slice(-HISTORY_LEN),
  };
  return logMilestones(state, next);
}

// Logs the moments worth rewinding to: a round where greedy commitment dropped
// eta_mu (the degradation the conservative step exists to prevent), and the
// round the algorithm reaches its break point and stops.
function logMilestones(prev: SimState, next: SimState): SimState {
  let result = next;
  if (next.etaMu < prev.etaMu - 1e-9) {
    result = withLog(
      result,
      `step ${next.steps}: eta_mu dropped ${(prev.etaMu - next.etaMu).toFixed(3)}`,
    );
  }
  if (!prev.done && next.done) {
    result = withLog(
      result,
      `step ${next.steps}: reached the break point, stopped`,
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
  etaMu: number;
  etaD: number;
  optimalMu: number;
  optimalD: number;
  fractionMu: number;
  fractionD: number;
  policyAdvantage: number;
  mismatch: number;
  done: boolean;
  nearOptimalD: boolean;
}

export function status(state: SimState): Status {
  const fractionMu =
    Math.abs(state.optimalMu) < 1e-9 ? 0 : state.etaMu / state.optimalMu;
  const fractionD =
    Math.abs(state.optimalD) < 1e-9 ? 0 : state.etaD / state.optimalD;
  return {
    etaMu: state.etaMu,
    etaD: state.etaD,
    optimalMu: state.optimalMu,
    optimalD: state.optimalD,
    fractionMu,
    fractionD,
    policyAdvantage: state.policyAdvantage,
    mismatch: state.mismatch,
    done: state.done,
    nearOptimalD: fractionD >= 0.92,
  };
}

export type Intervention =
  | { type: "setUpdate"; value: UpdateMode }
  | { type: "setAlpha"; value: number }
  | { type: "setGamma"; value: number }
  | { type: "setChooserError"; value: number }
  | { type: "setRestart"; value: RestartMode }
  | { type: "resetPolicy" }
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

export const PRESETS: Preset[] = [
  {
    id: "conservative",
    label: "Conservative climb",
    blurb:
      "Each round takes the safe mixture step that maximizes Theorem 4.1's guaranteed gain, so eta_mu rises every single round and never dips. Press play and watch the policy curve climb monotonically toward optimal, the policy advantage shrink to the break point, and the run stop on its own.",
    params: {
      update: "conservative",
      alpha: 0.1,
      gamma: 0.9,
      chooserError: 0.03,
      restart: "uniform",
    },
    seed: 5,
  },
  {
    id: "greedy",
    label: "Greedy overcommits",
    blurb:
      "Now each round jumps all the way to the approximate greedy policy (alpha = 1). Because the value estimates carry error and committing fully shifts which states the agent visits, eta_mu lurches and drops on some rounds instead of climbing. Switch the update to conservative mid-run and watch the curve settle and rise.",
    params: {
      update: "greedy",
      alpha: 1,
      gamma: 0.9,
      chooserError: 0.4,
      restart: "uniform",
    },
    seed: 1,
  },
  {
    id: "restart",
    label: "Wrong restart distribution",
    blurb:
      "Conservative steps again, but the restart distribution is the start state alone, so the policy advantage barely counts the far states the agent rarely reaches. The climb under the true objective eta_D stalls short of optimal and the distribution mismatch reads off the chart. Switch the restart to uniform mid-run and watch the far cells light up and eta_D resume climbing.",
    params: {
      update: "conservative",
      alpha: 0.1,
      gamma: 0.9,
      chooserError: 0.03,
      restart: "start",
    },
    seed: 5,
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

function uniformPolicy(): number[][] {
  const pi: number[][] = [];
  for (let s = 0; s < N_STATES; s++) {
    pi.push(new Array<number>(N_ACTIONS).fill(1 / N_ACTIONS));
  }
  return pi;
}

export function initialState(
  presetId = "conservative",
  seedOverride?: number,
): SimState {
  const preset = getPreset(presetId);
  const mu = restartDistribution(GRID, preset.params.restart);
  const vStar = optimalValues(GRID, preset.params.gamma);
  let optimalMu = 0;
  for (let s = 0; s < N_STATES; s++) optimalMu += at(mu, s) * at(vStar, s);

  const base: SimState = {
    mdp: GRID,
    params: { ...preset.params },
    pi: uniformPolicy(),
    values: [],
    q: [],
    advantage: [],
    visitMu: [],
    greedy: [],
    policyAdvantage: 0,
    maxStateAdv: 0,
    appliedAlpha: 0,
    etaMu: 0,
    etaD: 0,
    optimalMu,
    optimalD: at(vStar, GRID.start),
    mismatch: mismatchCoefficient(GRID, preset.params.gamma, mu),
    rng: (seedOverride ?? preset.seed) >>> 0,
    steps: 0,
    done: false,
    etaMuHistory: [],
    etaDHistory: [],
    log: emptyLog<Snapshot>(),
  };
  return deriveAndChoose(base);
}

// Recomputes the optimal-return cache and the mismatch when gamma or the restart
// distribution changes, since both depend on them.
function refreshOptima(state: SimState): SimState {
  const mu = restartDistribution(state.mdp, state.params.restart);
  const vStar = optimalValues(state.mdp, state.params.gamma);
  let optimalMu = 0;
  for (let s = 0; s < N_STATES; s++) optimalMu += at(mu, s) * at(vStar, s);
  return {
    ...state,
    optimalMu,
    optimalD: at(vStar, state.mdp.start),
    mismatch: mismatchCoefficient(state.mdp, state.params.gamma, mu),
  };
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
    case "setAlpha":
      return {
        state: { ...state, params: { ...state.params, alpha: action.value } },
        label: `alpha -> ${action.value.toFixed(2)}`,
      };
    case "setGamma":
      return {
        state: refreshOptima(
          deriveAndChoose({
            ...state,
            params: { ...state.params, gamma: action.value },
            done: false,
          }),
        ),
        label: `gamma -> ${action.value.toFixed(2)}`,
      };
    case "setChooserError":
      return {
        state: deriveAndChoose({
          ...state,
          params: { ...state.params, chooserError: action.value },
          done: false,
        }),
        label: `chooser error -> ${action.value.toFixed(2)}`,
      };
    case "setRestart":
      return {
        state: refreshOptima(
          deriveAndChoose({
            ...state,
            params: { ...state.params, restart: action.value },
            done: false,
          }),
        ),
        label: `restart -> ${action.value}`,
      };
    case "resetPolicy":
      return {
        state: deriveAndChoose({
          ...state,
          pi: uniformPolicy(),
          steps: 0,
          appliedAlpha: 0,
          done: false,
          etaMuHistory: [],
          etaDHistory: [],
        }),
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

// The pure reducer. A tick runs one policy-improvement round under the chosen
// update rule; an intervention changes a knob, the restart distribution, or the
// policy, logging the moment just before it took effect so the event log can
// rewind there.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return advance(state);
  const before = snapshotOf(state);
  const { state: changed, label } = applyIntervention(state, input.action);
  return { ...changed, log: record(state.log, state.steps, label, before) };
}
