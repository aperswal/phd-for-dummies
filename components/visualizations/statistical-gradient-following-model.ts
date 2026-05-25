// A pure, deterministic model of REINFORCE from Williams' 1992 paper "Simple
// Statistical Gradient-Following Algorithms for Connectionist Reinforcement
// Learning". One Gaussian unit outputs a 2D point y ~ N(mu, sigma^2 I) on a
// reward landscape it cannot see, collects the scalar reward r at that point,
// and adjusts its parameters by the paper's rule:
//
//   Delta mu    = alpha (r - b) (y - mu)                              (eq. 13)
//   Delta sigma = alpha (r - b) (||y - mu||^2 - DIM*sigma^2) / sigma  (eq. 14)
//
// using the paper's choice alpha_mu = alpha_sigma = alpha*sigma^2, which cancels
// the sigma^2 in the denominators. b is the reinforcement baseline; with
// reinforcement comparison it is an exponential running average of past reward,
// otherwise 0. Each (r-b)(d ln g / d w) is an unbiased estimate of the gradient
// of expected reward, so the averaged update climbs E{r} without ever measuring
// a slope (Theorem 1). The expected-reward landscape and its gradient are
// available in closed form here, used only to score how well the averaged update
// tracks the true gradient. All randomness comes from a seeded stream threaded
// through the state, so a seed plus a list of inputs reproduces a run exactly.

import { at } from "@/lib/simulation/array";
import { emptyLog, record, type EventLog } from "@/lib/simulation/history";
import { nextRandom } from "@/lib/simulation/rng";

export const DIM = 2;
const SIGMA_MIN = 0.012;
const SIGMA_MAX = 0.45;
const HISTORY_LEN = 80;
const DOMAIN_MIN = 0;
const DOMAIN_MAX = 1;

export interface Vec2 {
  x: number;
  y: number;
}

// One hill on the landscape: a Gaussian bump of the given height and width
// centered at a point. The reward at any point is the floor plus the sum of the
// bumps, which keeps both the raw reward and its smoothed expectation in closed
// form.
export interface Bump {
  center: Vec2;
  height: number;
  width: number;
}

export interface Landscape {
  bumps: Bump[];
  floor: number;
}

export type BaselineMode = "comparison" | "none";
export type SigmaMode = "adapt" | "fixed";

// The knobs the reader turns live. alpha is the learning rate, baselineMode
// switches reinforcement comparison on or off, sigmaMode lets the unit adapt its
// own exploration width or pins it, baselineDecay is the weight on the newest
// reward in the running average, rewardNoise corrupts the reward signal, and
// batchSize is how many tries make up one averaged update.
export interface Params {
  alpha: number;
  baselineMode: BaselineMode;
  sigmaMode: SigmaMode;
  baselineDecay: number;
  rewardNoise: number;
  batchSize: number;
}

// One try in the current batch: where the unit sampled, the reward it got, and
// how that reward compared to the baseline (positive pulls mu toward it).
export interface TrialSample {
  pos: Vec2;
  reward: number;
  advantage: number;
}

export interface UpdateVec {
  mu: Vec2;
  sigma: number;
}

export interface SimState {
  landscape: Landscape;
  params: Params;
  mu: Vec2;
  sigma: number;
  baseline: number;
  rng: number;
  trials: number;
  batches: number;
  samples: TrialSample[];
  lastUpdate: UpdateVec | null;
  avgUpdateMu: Vec2;
  rewardHistory: number[];
  log: EventLog<Snapshot>;
}

export type Snapshot = Omit<SimState, "log">;

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

function dist2(a: Vec2, b: Vec2): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

// The reward a try collects at a point: the floor plus every hill's Gaussian
// bump. This is the landscape the unit samples but never sees the shape of.
export function rawReward(landscape: Landscape, p: Vec2): number {
  let reward = landscape.floor;
  for (const bump of landscape.bumps) {
    reward +=
      bump.height *
      Math.exp(-dist2(p, bump.center) / (2 * bump.width * bump.width));
  }
  return reward;
}

// Expected reward when the unit sits at mu and scatters its tries with width
// sigma: the landscape convolved with the unit's Gaussian. This is the true
// objective J(mu, sigma) that REINFORCE climbs, and a Gaussian bump convolved
// with a Gaussian stays a Gaussian, so it is exact.
export function expectedReward(
  landscape: Landscape,
  mu: Vec2,
  sigma: number,
): number {
  const s2 = sigma * sigma;
  let value = landscape.floor;
  for (const bump of landscape.bumps) {
    const l2 = bump.width * bump.width;
    const denom = l2 + s2;
    value +=
      bump.height *
      (l2 / denom) *
      Math.exp(-dist2(mu, bump.center) / (2 * denom));
  }
  return value;
}

// The true gradient of expected reward with respect to mu, in closed form. The
// reader never needs this to learn; it exists only to check that the averaged
// update points the same way (Theorem 1).
export function gradientMu(
  landscape: Landscape,
  mu: Vec2,
  sigma: number,
): Vec2 {
  const s2 = sigma * sigma;
  let gx = 0;
  let gy = 0;
  for (const bump of landscape.bumps) {
    const l2 = bump.width * bump.width;
    const denom = l2 + s2;
    const weight =
      bump.height *
      (l2 / denom) *
      Math.exp(-dist2(mu, bump.center) / (2 * denom));
    gx += weight * (-(mu.x - bump.center.x) / denom);
    gy += weight * (-(mu.y - bump.center.y) / denom);
  }
  return { x: gx, y: gy };
}

// A standard normal deviate from the seeded stream via Box-Muller, returning the
// advanced stream state so the reducer stays pure.
function nextNormal(rng: number): { value: number; rng: number } {
  const u1 = nextRandom(rng);
  const u2 = nextRandom(u1.state);
  const mag = Math.sqrt(-2 * Math.log(Math.max(u1.value, 1e-12)));
  return { value: mag * Math.cos(2 * Math.PI * u2.value), rng: u2.state };
}

export function globalPeak(landscape: Landscape): Bump {
  let best = at(landscape.bumps, 0);
  for (const bump of landscape.bumps) {
    if (bump.height > best.height) best = bump;
  }
  return best;
}

// One batch of tries folded into a single averaged update. Each try samples a
// point, scores it, and contributes its eligibility weighted by how far the
// reward beat the baseline. The baseline used is the running average from before
// this batch, so it stays independent of the tries it weights, which the paper
// requires. The baseline is refreshed afterward from this batch's mean reward.
function advance(state: SimState): SimState {
  const { params, landscape } = state;
  const batch = params.batchSize;
  const base = params.baselineMode === "comparison" ? state.baseline : 0;
  const sigma = state.sigma;
  const s2 = sigma * sigma;

  let rng = state.rng;
  const samples: TrialSample[] = [];
  let sumReward = 0;
  let gradMuX = 0;
  let gradMuY = 0;
  let gradSigma = 0;

  for (let i = 0; i < batch; i++) {
    const drawX = nextNormal(rng);
    const drawY = nextNormal(drawX.rng);
    rng = drawY.rng;
    const pos: Vec2 = {
      x: state.mu.x + sigma * drawX.value,
      y: state.mu.y + sigma * drawY.value,
    };
    let reward = rawReward(landscape, pos);
    if (params.rewardNoise > 0) {
      const noise = nextNormal(rng);
      rng = noise.rng;
      reward += params.rewardNoise * noise.value;
    }
    const advantage = reward - base;
    samples.push({ pos, reward, advantage });
    sumReward += reward;

    gradMuX += advantage * (pos.x - state.mu.x);
    gradMuY += advantage * (pos.y - state.mu.y);
    gradSigma += (advantage * (dist2(pos, state.mu) - DIM * s2)) / sigma;
  }

  const scale = params.alpha / batch;
  const dMu: Vec2 = { x: scale * gradMuX, y: scale * gradMuY };
  const dSigma = params.sigmaMode === "adapt" ? scale * gradSigma : 0;

  const mu: Vec2 = {
    x: clamp(state.mu.x + dMu.x, DOMAIN_MIN, DOMAIN_MAX),
    y: clamp(state.mu.y + dMu.y, DOMAIN_MIN, DOMAIN_MAX),
  };
  const nextSigma = clamp(state.sigma + dSigma, SIGMA_MIN, SIGMA_MAX);

  const batchMean = sumReward / batch;
  const baseline =
    state.baseline + params.baselineDecay * (batchMean - state.baseline);

  // An exponential average of the applied mu-step, so the alignment readout can
  // compare the direction the unit actually moves against the true gradient.
  const blend = 0.8;
  const avgUpdateMu: Vec2 = {
    x: blend * state.avgUpdateMu.x + (1 - blend) * dMu.x,
    y: blend * state.avgUpdateMu.y + (1 - blend) * dMu.y,
  };

  const next: SimState = {
    ...state,
    mu,
    sigma: nextSigma,
    baseline,
    rng,
    trials: state.trials + batch,
    batches: state.batches + 1,
    samples,
    lastUpdate: { mu: dMu, sigma: dSigma },
    avgUpdateMu,
    rewardHistory: [...state.rewardHistory, batchMean].slice(-HISTORY_LEN),
  };

  return logMilestones(state, next);
}

// Logs the two moments worth rewinding to: the first batch that lands the unit
// on the best hill, and the batch where exploration collapses to the floor.
// Both fire only on the transition, so the log does not flood.
function logMilestones(prev: SimState, next: SimState): SimState {
  const best = globalPeak(next.landscape);
  const ceiling = next.landscape.floor + best.height;
  const wasNear =
    expectedReward(prev.landscape, prev.mu, prev.sigma) >= 0.9 * ceiling;
  const isNear =
    expectedReward(next.landscape, next.mu, next.sigma) >= 0.9 * ceiling;
  let result = next;
  if (!wasNear && isNear && distTo(next.mu, best.center) < 0.12) {
    result = withLog(result, `batch ${next.batches}: reached the peak`);
  }
  const collapseLine = SIGMA_MIN * 1.6;
  if (prev.sigma > collapseLine && next.sigma <= collapseLine) {
    result = withLog(result, `batch ${next.batches}: exploration collapsed`);
  }
  return result;
}

function distTo(a: Vec2, b: Vec2): number {
  return Math.sqrt(dist2(a, b));
}

function snapshotOf(state: SimState): Snapshot {
  return {
    landscape: state.landscape,
    params: state.params,
    mu: state.mu,
    sigma: state.sigma,
    baseline: state.baseline,
    rng: state.rng,
    trials: state.trials,
    batches: state.batches,
    samples: state.samples,
    lastUpdate: state.lastUpdate,
    avgUpdateMu: state.avgUpdateMu,
    rewardHistory: state.rewardHistory,
  };
}

function withLog(state: SimState, label: string): SimState {
  return {
    ...state,
    log: record(state.log, state.batches, label, snapshotOf(state)),
  };
}

export interface Status {
  expected: number;
  ceiling: number;
  fractionOfBest: number;
  atGlobalPeak: boolean;
  gradientCos: number;
  trappedOnLesserPeak: boolean;
}

// How well the run is doing, scored against the closed-form objective the unit
// never sees. gradientCos is the cosine between the averaged update and the true
// gradient: positive means the unit is climbing E{r}, which is the content of
// Theorem 1. atGlobalPeak means mu has settled tightly on the tallest hill.
export function status(state: SimState): Status {
  const { landscape, mu, sigma } = state;
  const best = globalPeak(landscape);
  const ceiling = landscape.floor + best.height;
  const expected = expectedReward(landscape, mu, sigma);

  const grad = gradientMu(landscape, mu, sigma);
  const avg = state.avgUpdateMu;
  const gMag = Math.hypot(grad.x, grad.y);
  const aMag = Math.hypot(avg.x, avg.y);
  const gradientCos =
    gMag < 1e-9 || aMag < 1e-9
      ? 0
      : (grad.x * avg.x + grad.y * avg.y) / (gMag * aMag);

  const nearBest = distTo(mu, best.center) < 0.08 && sigma < 0.09;
  const onSomePeak = sigma < 0.09 && expected >= 0.55 * ceiling;
  return {
    expected,
    ceiling,
    fractionOfBest: ceiling === 0 ? 0 : expected / ceiling,
    atGlobalPeak: nearBest,
    gradientCos,
    trappedOnLesserPeak: onSomePeak && !nearBest && landscape.bumps.length > 1,
  };
}

export type Intervention =
  | { type: "setAlpha"; value: number }
  | { type: "setBaselineMode"; value: BaselineMode }
  | { type: "setSigmaMode"; value: SigmaMode }
  | { type: "setSigma"; value: number }
  | { type: "setBaselineDecay"; value: number }
  | { type: "setRewardNoise"; value: number }
  | { type: "moveMu"; pos: Vec2 }
  | { type: "resetLearning" }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: Snapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

// ---- Landscapes and presets --------------------------------------------

const SINGLE: Landscape = {
  floor: 0,
  bumps: [{ center: { x: 0.72, y: 0.34 }, height: 1, width: 0.22 }],
};

// A tall global peak and a shorter local one. An adapting unit that starts near
// the small hill narrows onto it and never finds the tall one, which is the
// false-optimum trap the paper warns gradient methods fall into.
const TWO_PEAKS: Landscape = {
  floor: 0,
  bumps: [
    { center: { x: 0.76, y: 0.7 }, height: 1, width: 0.13 },
    { center: { x: 0.24, y: 0.3 }, height: 0.6, width: 0.11 },
  ],
};

// One peak sitting on a high positive floor, so every reward is large and
// positive. With no baseline to subtract, the sigma updates are dominated by
// noise and the search width drifts down before mu climbs, the collapse the
// paper describes for nonnegative reward without reinforcement comparison.
const POSITIVE_FLOOR: Landscape = {
  floor: 1.2,
  bumps: [{ center: { x: 0.72, y: 0.34 }, height: 1, width: 0.22 }],
};

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  landscape: Landscape;
  mu: Vec2;
  sigma: number;
  params: Params;
}

const BASE: Params = {
  alpha: 0.12,
  baselineMode: "comparison",
  sigmaMode: "adapt",
  baselineDecay: 0.15,
  rewardNoise: 0,
  batchSize: 24,
};

export const PRESETS: Preset[] = [
  {
    id: "climb",
    label: "Climb the hill",
    blurb:
      "One Gaussian unit on a single hill it cannot see. Run it and watch the cloud of tries drift uphill while its spread tightens onto the peak. The reward climbs, the averaged step lines up with the true gradient, and the optimal light turns on.",
    landscape: SINGLE,
    mu: { x: 0.2, y: 0.78 },
    sigma: 0.28,
    params: { ...BASE },
  },
  {
    id: "trapped",
    label: "Trapped on a lesser peak",
    blurb:
      "A tall hill and a shorter one, with the unit starting beside the short one. It tightens onto the lesser peak and never discovers the taller one, the false optimum every gradient method risks. Drag the aim onto the far slope, or push exploration back up, to break it out.",
    landscape: TWO_PEAKS,
    mu: { x: 0.23, y: 0.3 },
    sigma: 0.12,
    params: { ...BASE, alpha: 0.1 },
  },
  {
    id: "nobaseline",
    label: "No baseline, noisy climb",
    blurb:
      "The same hill on a high floor, so every reward is large and positive. With no baseline subtracted, the big constant reward swamps each step, so the spread balloons and the climb stalls and lurches. Switch the baseline to reinforcement comparison and watch the same run steady and speed up.",
    landscape: POSITIVE_FLOOR,
    mu: { x: 0.2, y: 0.78 },
    sigma: 0.26,
    params: { ...BASE, baselineMode: "none", alpha: 0.1 },
  },
  {
    id: "narrow",
    label: "Exploration too narrow",
    blurb:
      "The unit starts far from the hill with its spread pinned tiny, so no try ever lands on the slope and the reward stays flat. Nothing to learn from, so the aim sits frozen in the dark. Widen the spread, or switch it back to adapt, and the tries reach the hill and the climb begins.",
    landscape: SINGLE,
    mu: { x: 0.2, y: 0.78 },
    sigma: 0.045,
    params: { ...BASE, sigmaMode: "fixed", alpha: 0.16 },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(presetId = "climb", seed = 1): SimState {
  const preset = getPreset(presetId);
  return {
    landscape: preset.landscape,
    params: { ...preset.params },
    mu: { ...preset.mu },
    sigma: preset.sigma,
    baseline: 0,
    rng: seed >>> 0,
    trials: 0,
    batches: 0,
    samples: [],
    lastUpdate: null,
    avgUpdateMu: { x: 0, y: 0 },
    rewardHistory: [],
    log: emptyLog<Snapshot>(),
  };
}

function resetLearningState(state: SimState): SimState {
  const preset = PRESETS.find(
    (candidate) => candidate.landscape === state.landscape,
  );
  return {
    ...state,
    mu: preset ? { ...preset.mu } : { x: 0.16, y: 0.82 },
    sigma: preset ? preset.sigma : 0.28,
    baseline: 0,
    trials: 0,
    batches: 0,
    samples: [],
    lastUpdate: null,
    avgUpdateMu: { x: 0, y: 0 },
    rewardHistory: [],
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
    case "setBaselineMode":
      return {
        state: {
          ...state,
          params: { ...state.params, baselineMode: action.value },
        },
        label: `baseline -> ${action.value}`,
      };
    case "setSigmaMode":
      return {
        state: {
          ...state,
          params: { ...state.params, sigmaMode: action.value },
        },
        label: `sigma -> ${action.value}`,
      };
    case "setSigma":
      return {
        state: { ...state, sigma: clamp(action.value, SIGMA_MIN, SIGMA_MAX) },
        label: `sigma set to ${action.value.toFixed(3)}`,
      };
    case "setBaselineDecay":
      return {
        state: {
          ...state,
          params: { ...state.params, baselineDecay: action.value },
        },
        label: `baseline decay -> ${action.value.toFixed(2)}`,
      };
    case "setRewardNoise":
      return {
        state: {
          ...state,
          params: { ...state.params, rewardNoise: action.value },
        },
        label: `reward noise -> ${action.value.toFixed(2)}`,
      };
    case "moveMu":
      return {
        state: {
          ...state,
          mu: {
            x: clamp(action.pos.x, DOMAIN_MIN, DOMAIN_MAX),
            y: clamp(action.pos.y, DOMAIN_MIN, DOMAIN_MAX),
          },
          avgUpdateMu: { x: 0, y: 0 },
        },
        label: `aim moved to (${action.pos.x.toFixed(2)}, ${action.pos.y.toFixed(2)})`,
      };
    case "resetLearning":
      return { state: resetLearningState(state), label: "learning reset" };
    case "loadPreset": {
      const preset = getPreset(action.id);
      return {
        state: { ...initialState(action.id, state.rng), log: state.log },
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

// The pure reducer. A tick runs one batch of tries and one averaged update; an
// intervention changes a knob, the world, or the aim, logging the moment just
// before it took effect so the event log can rewind to it.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return advance(state);
  const before = snapshotOf(state);
  const { state: changed, label } = applyIntervention(state, input.action);
  return { ...changed, log: record(state.log, state.batches, label, before) };
}
