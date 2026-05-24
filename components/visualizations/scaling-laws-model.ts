// A pure, deterministic model of the empirical scaling laws from "Scaling Laws
// for Neural Language Models". The math is the paper's own: the cross-entropy
// test loss falls as a power law in each scale factor (parameters N, dataset D,
// compute C), the combined law L(N, D) governs overfitting, and a fixed compute
// budget has one optimal split N* ~ C^0.73. The reader sets a budget and a way
// to spend it, and these equations return the loss they would get. The only
// randomness is a small seeded jitter on the plotted empirical points so the
// curves look like real noisy measurements while still reproducing exactly for
// a given seed.
//
// This file is intentionally self-contained: it inlines its own seeded rng and
// event log instead of importing the shared lib/simulation modules, so the
// simulation is one unit a reader can read top to bottom.

// One step of the mulberry32 recurrence. Threading a seed through the model
// instead of calling Math.random is what makes a run reproducible.
function nextRandom(state: number): { value: number; state: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, state: a };
}

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    const next = nextRandom(state);
    state = next.state;
    return next.value;
  };
}

// Indexed access that throws on a genuine out-of-range read instead of quietly
// returning undefined, so the strict compiler keeps `arr[i]` typed as T.
function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) {
    throw new RangeError(`index ${index} out of range`);
  }
  return value;
}

// The fitted constants from the paper. The three single-factor power laws use
// the Section 1.2 values; the combined L(N, D) uses the Table 2 fit; the optimal
// allocation uses the Section 6 exponents.
export const ALPHA_N = 0.076; // L(N) exponent, eq 1.1
export const ALPHA_D = 0.095; // L(D) exponent, eq 1.2
export const ALPHA_C = 0.05; // L(C_min) exponent, eq 1.3 / 6.3
export const N_C = 8.8e13; // eq 1.1 constant (non-embedding params)
export const D_C = 5.4e13; // eq 1.2 constant (tokens)
export const C_C = 3.1e8; // eq 1.3 constant (PF-days)

// The combined fit (Table 2), used for L(N, D) and the overfitting penalty.
export const ALPHA_N_FIT = 0.076;
export const ALPHA_D_FIT = 0.103;
export const N_C_FIT = 6.4e13;
export const D_C_FIT = 1.8e13;

// Optimal-allocation exponents, Section 6. With a fixed compute budget the loss
// is minimized by N* ~ C^0.73 (eq 6.1), and the data processed grows as the
// remainder. Compute is spent C = 6 N B S; with the optimal split most of any
// new compute goes to a bigger model, only a little to more data.
export const N_EXP = 0.73; // N* ~ C^0.73, eq 6.1
export const D_EXP = 0.27; // D ~ C^0.27, the remainder of C = 6 N D-ish
export const C_PER_PARAM_TOKEN = 6; // C ~ 6 N D in PF-flop accounting

// L(N): the loss of a model with N non-embedding parameters trained to
// convergence on enough data. Equation 1.1.
export function lossFromParams(n: number): number {
  return Math.pow(N_C / Math.max(n, 1), ALPHA_N);
}

// L(D): the loss of a large model trained on D tokens with early stopping.
// Equation 1.2.
export function lossFromData(d: number): number {
  return Math.pow(D_C / Math.max(d, 1), ALPHA_D);
}

// L(C_min): the loss reachable with an optimally allocated compute budget C
// (in PF-days). Equation 1.3 / 6.3.
export function lossFromCompute(c: number): number {
  return Math.pow(C_C / Math.max(c, 1e-12), ALPHA_C);
}

// L(N, D): the early-stopped loss as one law over both model size and data.
// Equation 1.5 / 4.1. Sending D to infinity recovers L(N); sending N to
// infinity recovers L(D). The cross term is what produces overfitting.
export function lossFromParamsAndData(n: number, d: number): number {
  const nTerm = Math.pow(N_C_FIT / Math.max(n, 1), ALPHA_N_FIT / ALPHA_D_FIT);
  const dTerm = D_C_FIT / Math.max(d, 1);
  return Math.pow(nTerm + dTerm, ALPHA_D_FIT);
}

// The overfitting penalty delta-L from equation 4.2: how much worse the finite
// dataset makes the loss compared with the infinite-data limit L(N, infinity).
export function overfitPenalty(n: number, d: number): number {
  const finite = lossFromParamsAndData(n, d);
  const infinite = lossFromParamsAndData(n, Number.POSITIVE_INFINITY);
  return finite / infinite - 1;
}

// The optimal model size for a compute budget C (PF-days). Equation 6.1.
export function optimalParams(c: number): number {
  return 1.3e9 * Math.pow(Math.max(c, 1e-12), N_EXP);
}

// Given a compute budget and a model size, recover how many tokens that leaves
// for training. The non-embedding training compute is roughly C ~ 6 N D
// PF-flop, and one PF-day is 8.64e19 flop, so C[PF-days] ~ 6 N D / 8.64e19.
const FLOP_PER_PF_DAY = 8.64e19;

export function tokensFromComputeAndParams(c: number, n: number): number {
  const flop = c * FLOP_PER_PF_DAY;
  return flop / (C_PER_PARAM_TOKEN * Math.max(n, 1));
}

export type AxisId = "params" | "data" | "compute";

export interface AxisSpec {
  id: AxisId;
  label: string;
  unit: string;
  blurb: string;
  min: number; // log10 lower bound of the swept factor
  max: number; // log10 upper bound
  alpha: number;
  loss: (x: number) => number;
}

export const AXES: AxisSpec[] = [
  {
    id: "params",
    label: "Model size N",
    unit: "non-embedding params",
    blurb:
      "Hold data and compute out of the way and grow only the model. The loss tracks a straight power law in N with slope 0.076.",
    min: 3,
    max: 9.3,
    alpha: ALPHA_N,
    loss: lossFromParams,
  },
  {
    id: "data",
    label: "Dataset size D",
    unit: "tokens",
    blurb:
      "Hold the model huge and grow only the data. The loss tracks a straight power law in D with slope 0.095.",
    min: 7,
    max: 10.4,
    alpha: ALPHA_D,
    loss: lossFromData,
  },
  {
    id: "compute",
    label: "Compute C",
    unit: "PF-days",
    blurb:
      "Spend a growing, optimally split compute budget. The loss tracks a straight power law in C with slope 0.050.",
    min: -8,
    max: 4,
    alpha: ALPHA_C,
    loss: lossFromCompute,
  },
];

export function getAxis(id: AxisId): AxisSpec {
  return AXES.find((axis) => axis.id === id) ?? at(AXES, 0);
}

export interface DataPoint {
  logX: number; // log10 of the swept scale factor
  logLoss: number; // log10 of the loss at that point (the power law is a line here)
  loss: number;
  x: number;
}

// Build one point on the log-log curve. The clean power law is exactly a
// straight line in log-log space; the seeded jitter nudges each point off the
// line a little so the curve reads like real measurements, the way the paper's
// figures look.
function buildPoint(axis: AxisSpec, logX: number, jitter: number): DataPoint {
  const x = Math.pow(10, logX);
  const cleanLoss = axis.loss(x);
  const loss = cleanLoss * (1 + jitter);
  return { logX, logLoss: Math.log10(loss), loss, x };
}

// Sweep the chosen axis from its min to its current reveal point, producing the
// line of empirical points. The reveal grows one tick at a time so the curve
// draws itself out left to right as the run plays.
export function buildCurve(
  axisId: AxisId,
  revealLog: number,
  seed: number,
  pointCount: number,
): DataPoint[] {
  const axis = getAxis(axisId);
  const rng = mulberry32(seed === 0 ? 1 : seed);
  const points: DataPoint[] = [];
  for (let i = 0; i < pointCount; i++) {
    const logX = axis.min + ((axis.max - axis.min) * i) / (pointCount - 1);
    const jitter = seed === 0 ? 0 : (rng() - 0.5) * 0.035;
    if (logX > revealLog + 1e-9) break;
    points.push(buildPoint(axis, logX, jitter));
  }
  return points;
}

// A bounded event log. Each entry carries a snapshot the reader can rewind to.
export interface SimEvent {
  id: number;
  tick: number;
  label: string;
  snapshot: ScalingParams;
}

export interface EventLog {
  events: SimEvent[];
  nextEventId: number;
}

export function emptyLog(): EventLog {
  return { events: [], nextEventId: 1 };
}

const LOG_LIMIT = 60;

function record(
  log: EventLog,
  tick: number,
  label: string,
  snapshot: ScalingParams,
): EventLog {
  const event: SimEvent = { id: log.nextEventId, tick, label, snapshot };
  return {
    events: [...log.events, event].slice(-LOG_LIMIT),
    nextEventId: log.nextEventId + 1,
  };
}

export interface ScalingParams {
  axisId: AxisId;
  seed: number;
  // The compute-budget allocator. logCompute is the budget (log10 PF-days);
  // allocation in [0, 1] sets how much of the model size the reader uses
  // relative to the optimal N*(C): 1 is the optimal split, below 1 starves the
  // model (the small-model instinct), above 1 over-sizes it and starves data.
  logCompute: number;
  allocation: number;
}

export type Intervention =
  | { type: "setAxis"; id: AxisId }
  | { type: "setSeed"; value: number }
  | { type: "setCompute"; value: number }
  | { type: "setAllocation"; value: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; params: ScalingParams; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface SimState {
  params: ScalingParams;
  tick: number;
  revealLog: number; // how far along the swept axis the curve has drawn
  log: EventLog;
}

export const POINT_COUNT = 26;
const REVEAL_STEP = 0.55; // how much of a decade each tick reveals

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: ScalingParams;
}

export const PRESETS: Preset[] = [
  {
    id: "params-law",
    label: "Loss vs model size",
    blurb:
      "Grow only the model and watch the loss fall along a dead-straight line in log-log. Press play and the curve draws itself out over six orders of magnitude.",
    params: { axisId: "params", seed: 7, logCompute: 0, allocation: 1 },
  },
  {
    id: "compute-law",
    label: "Loss vs compute",
    blurb:
      "Spend a growing, optimally split compute budget. Same straight line, shallower slope, because compute buys both a bigger model and a little more data.",
    params: { axisId: "compute", seed: 7, logCompute: 0, allocation: 1 },
  },
  {
    id: "optimal-split",
    label: "Optimal budget split",
    blurb:
      "Pick a compute budget, then drag allocation. It sits at 1, the optimal split N* ~ C^0.73. The realized loss is at its floor; any move makes it worse.",
    params: { axisId: "compute", seed: 7, logCompute: 1, allocation: 1 },
  },
  {
    id: "small-model",
    label: "Small-model mistake",
    blurb:
      "The pre-2020 instinct: spend the budget on a small model trained on lots of data. Drag allocation down and watch the realized loss climb above the optimal floor.",
    params: { axisId: "compute", seed: 7, logCompute: 1, allocation: 0.18 },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(presetId = "params-law"): SimState {
  const params = { ...getPreset(presetId).params };
  return {
    params,
    tick: 0,
    revealLog: getAxis(params.axisId).min,
    log: emptyLog(),
  };
}

// Map an allocation knob in [0, 1.5]-ish to a model size relative to N*(C).
// Allocation 1 is the optimal split; below 1 starves the model, above 1
// over-sizes it and starves data.
function paramsForAllocation(compute: number, allocation: number): number {
  return optimalParams(compute) * Math.pow(10, (allocation - 1) * 2.2);
}

// The realized loss the reader gets at a given budget and allocation: train a
// model of that size on the tokens the budget buys, scored by L(N, D).
function realizedLossAt(compute: number, allocation: number): number {
  const n = paramsForAllocation(compute, allocation);
  return lossFromParamsAndData(n, tokensFromComputeAndParams(compute, n));
}

// The best loss reachable at this budget by sweeping the allocation knob. This
// is the simulation's own optimal floor, so "excess" stays internally
// consistent: it measures how much worse the reader's split is than the best
// split this same L(N, D) model can reach, never against a separately fit law.
function optimalRealizedLoss(compute: number): number {
  let best = Number.POSITIVE_INFINITY;
  for (let a = 0; a <= 200; a++) {
    const allocation = (a / 200) * 1.8;
    best = Math.min(best, realizedLossAt(compute, allocation));
  }
  return best;
}

// A snapshot of what the allocator currently produces, for the readouts.
export interface AllocationResult {
  compute: number;
  optimalN: number;
  chosenN: number;
  tokens: number;
  realizedLoss: number;
  optimalLoss: number;
  excess: number; // how much worse than the best split this budget can reach
  overfit: number; // the overfitting penalty at this N, D
}

export function computeAllocation(params: ScalingParams): AllocationResult {
  const compute = Math.pow(10, params.logCompute);
  const optimalN = optimalParams(compute);
  const chosenN = paramsForAllocation(compute, params.allocation);
  const tokens = tokensFromComputeAndParams(compute, chosenN);
  const realized = lossFromParamsAndData(chosenN, tokens);
  const optimalLoss = optimalRealizedLoss(compute);
  const overfit = overfitPenalty(chosenN, tokens);
  return {
    compute,
    optimalN,
    chosenN,
    tokens,
    realizedLoss: realized,
    optimalLoss,
    excess: realized / optimalLoss - 1,
    overfit,
  };
}

function logged(
  state: SimState,
  label: string,
  params: ScalingParams,
  revealLog: number,
): SimState {
  return {
    ...state,
    params,
    revealLog,
    log: record(state.log, state.tick, label, params),
  };
}

function applyIntervention(
  state: SimState,
  action: Intervention,
): { params: ScalingParams; label: string; revealLog: number } {
  const { params } = state;
  switch (action.type) {
    case "setAxis": {
      const next = { ...params, axisId: action.id };
      return {
        params: next,
        label: `axis -> ${getAxis(action.id).label}`,
        revealLog: getAxis(action.id).min,
      };
    }
    case "setSeed":
      return {
        params: { ...params, seed: action.value },
        label: `seed -> ${action.value}`,
        revealLog: state.revealLog,
      };
    case "setCompute":
      return {
        params: { ...params, logCompute: action.value },
        label: `compute -> 1e${action.value.toFixed(1)} PF-days`,
        revealLog: state.revealLog,
      };
    case "setAllocation":
      return {
        params: { ...params, allocation: action.value },
        label: `allocation -> ${action.value.toFixed(2)}x optimal`,
        revealLog: state.revealLog,
      };
    case "loadPreset": {
      const preset = getPreset(action.id);
      return {
        params: { ...preset.params },
        label: `preset -> ${preset.label}`,
        revealLog: getAxis(preset.params.axisId).min,
      };
    }
    case "restore":
      return {
        params: { ...action.params },
        label: action.label,
        revealLog: getAxis(action.params.axisId).min,
      };
  }
}

// The pure reducer. A tick draws the curve a little further along the swept
// axis; an intervention changes one parameter, resets the reveal when the axis
// changes, and records what happened. Every step is deterministic given the
// state, which carries the seed.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") {
    const axis = getAxis(state.params.axisId);
    const revealLog = Math.min(state.revealLog + REVEAL_STEP, axis.max);
    return { ...state, tick: state.tick + 1, revealLog };
  }

  const { params, label, revealLog } = applyIntervention(state, input.action);
  return logged({ ...state }, label, params, revealLog);
}
