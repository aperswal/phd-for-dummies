// A pure, deterministic model of the gradient geometry from Will Brown's
// "On SFT, RL, and on-policy distillation". The post's claim is that SFT, RL,
// on-policy distillation (OPD), and on-policy self-distillation (OPSD) are the
// same token-level policy gradient, and that the SHAPE of the per-token gradient
// cloud in parameter space is what decides whether a training step is fast,
// safe, or collapses.
//
// We project that high-dimensional cloud onto a 2D plane so a reader can see it.
// Each method draws a batch of per-token gradient vectors with a characteristic
// distribution, the model averages them into one update vector, and reports the
// three quantities the post cares about: the resultant magnitude (how far the
// model moves), the spread (how diffuse the signal is across tokens), and the
// concentration ratio (how much one token dominates). When the averaged update
// overshoots a "safe region" radius, the run is in collapse, exactly the OPSD
// failure the post warns about and the per-token KL clip prevents.
//
// All randomness comes from a seeded generator threaded through the state, so a
// given seed plus the same inputs reproduces the same cloud, which is what makes
// rewind and "what just happened?" honest.

export type Method = "rl" | "sft" | "opd" | "opsd";

export interface Vec2 {
  x: number;
  y: number;
}

// One per-token gradient vector in the projected parameter plane, tagged so the
// view can color the pivot token and the reward-aligned survivors.
export interface TokenGradient {
  vec: Vec2;
  kind: "noise" | "biased" | "pivot";
}

// ---- seeded rng (inline, kept self-contained) -----------------------------

function nextRandom(state: number): { value: number; state: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, state: a };
}

// A small generator object that advances its own cursor. Built fresh from a
// seed on every compute so the cloud is a pure function of the parameters.
function makeRng(seed: number): { next: () => number; gauss: () => number } {
  let state = seed >>> 0 || 1;
  const next = (): number => {
    const result = nextRandom(state);
    state = result.state;
    return result.value;
  };
  // Box-Muller for roughly normal noise, which is what a real gradient cloud
  // looks like.
  const gauss = (): number => {
    const u = Math.max(next(), 1e-9);
    const v = next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return { next, gauss };
}

// ---- the gradient parameters the reader pokes ------------------------------

export interface GeometryParams {
  method: Method;
  batchSize: number;
  // RL: strength of the consistent reward-aligned component relative to noise.
  rewardBias: number;
  // OPSD: how many times larger the pivot token's gradient is than a typical
  // token. The post estimates ~100x from log(0.6/0.01) ~= 4.1 nats.
  pivotStrength: number;
  // OPSD defense: cap a single token's contribution so a few stylistic tokens
  // can't dominate the signal. This is the post's per-token point-wise KL clip.
  clipping: boolean;
  // SFT/OPD: how wide the fan of biased directions is. A varied dataset (or a
  // same-family teacher) spreads the bias, which is what keeps it forgiving.
  spread: number;
  seed: number;
}

export interface GeometryResult {
  method: Method;
  gradients: TokenGradient[];
  update: Vec2;
  updateMagnitude: number;
  // Mean per-token vector magnitude, a stand-in for raw gradient density.
  meanMagnitude: number;
  // The largest single contribution divided by the mean: how much one token
  // dominates. High concentration is the OPSD danger.
  concentration: number;
  // Angular spread of the biased vectors in radians, a stand-in for how diffuse
  // the signal is across tokens. Wider is safer.
  diffuseness: number;
  // The averaged update collapses the model when it lands outside this radius.
  safeRadius: number;
  collapsed: boolean;
}

const PLANE = 100; // half-extent of the projected parameter plane
export const SAFE_RADIUS = 34;

function magnitude(v: Vec2): number {
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

function clampVec(v: Vec2, cap: number): Vec2 {
  const m = magnitude(v);
  if (m <= cap || m === 0) return v;
  const scale = cap / m;
  return { x: v.x * scale, y: v.y * scale };
}

// The single fixed "reward direction" (RL) / "data manifold direction"
// (SFT/OPD) / "self+hint direction" (OPSD) the bias points along, so the picture
// is stable across seeds and the reader compares shapes, not random rotations.
const BIAS_ANGLE = -Math.PI / 5;
const BIAS_DIR: Vec2 = { x: Math.cos(BIAS_ANGLE), y: Math.sin(BIAS_ANGLE) };

function buildRl(params: GeometryParams): TokenGradient[] {
  const rng = makeRng(params.seed);
  const out: TokenGradient[] = [];
  for (let i = 0; i < params.batchSize; i++) {
    // Sparse and broadcast-assigned: most tokens are pure noise, a minority
    // carry a small consistent reward component. Unbiased in expectation; the
    // noise cancels as the batch grows.
    const noise: Vec2 = { x: rng.gauss() * 26, y: rng.gauss() * 26 };
    const aligned = rng.next() < 0.35;
    if (aligned) {
      const push = params.rewardBias * 14;
      out.push({
        vec: { x: noise.x + BIAS_DIR.x * push, y: noise.y + BIAS_DIR.y * push },
        kind: "biased",
      });
    } else {
      out.push({ vec: noise, kind: "noise" });
    }
  }
  return out;
}

function buildFan(params: GeometryParams, reach: number): TokenGradient[] {
  const rng = makeRng(params.seed);
  const out: TokenGradient[] = [];
  for (let i = 0; i < params.batchSize; i++) {
    // Dense and biased: every token pulls toward the same general direction,
    // fanned out by `spread`. No cancellation, just a large well-spread pull.
    const angle = BIAS_ANGLE + (rng.next() - 0.5) * params.spread * 2;
    const len = reach * (0.55 + rng.next() * 0.7);
    out.push({
      vec: { x: Math.cos(angle) * len, y: Math.sin(angle) * len },
      kind: "biased",
    });
  }
  return out;
}

function buildOpsd(params: GeometryParams): TokenGradient[] {
  // Start from the same diffuse, biased fan as SFT/OPD, then add one pivot token
  // whose gradient is `pivotStrength` times a typical token and points off into
  // a region the model didn't previously believe in.
  const out = buildFan({ ...params }, 30);
  const rng = makeRng(params.seed ^ 0x9e3779b9);
  const pivotAngle = BIAS_ANGLE + Math.PI * 0.42;
  const pivotLen = 30 * params.pivotStrength * (0.85 + rng.next() * 0.3);
  out.push({
    vec: {
      x: Math.cos(pivotAngle) * pivotLen,
      y: Math.sin(pivotAngle) * pivotLen,
    },
    kind: "pivot",
  });
  return out;
}

function buildGradients(params: GeometryParams): TokenGradient[] {
  switch (params.method) {
    case "rl":
      return buildRl(params);
    case "sft":
      return buildFan(params, 30);
    case "opd":
      return buildFan(params, 28);
    case "opsd":
      return buildOpsd(params);
  }
}

function averageUpdate(gradients: TokenGradient[], clipping: boolean): Vec2 {
  const count = gradients.length || 1;
  // Per-token point-wise clip: cap any single contribution so a pivot token
  // can't dominate. This is the OPSD defense the post ships.
  const cap = clipping ? 46 : Infinity;
  let sx = 0;
  let sy = 0;
  for (const g of gradients) {
    const v = clampVec(g.vec, cap);
    sx += v.x;
    sy += v.y;
  }
  return { x: sx / count, y: sy / count };
}

function angularSpread(gradients: TokenGradient[]): number {
  const biased = gradients.filter((g) => g.kind === "biased");
  if (biased.length < 2) return 0;
  const angles = biased.map((g) => Math.atan2(g.vec.y, g.vec.x));
  let cos = 0;
  let sin = 0;
  for (const a of angles) {
    cos += Math.cos(a);
    sin += Math.sin(a);
  }
  const r = Math.sqrt(cos * cos + sin * sin) / angles.length;
  // Circular standard deviation: 0 when all aligned, grows as they spread.
  return Math.sqrt(-2 * Math.log(Math.max(r, 1e-6)));
}

export function computeGeometry(input: GeometryParams): GeometryResult {
  const params: GeometryParams = {
    ...input,
    batchSize: Math.max(2, Math.min(Math.round(input.batchSize), 220)),
    pivotStrength: Math.max(1, input.pivotStrength),
    rewardBias: Math.max(0, input.rewardBias),
    spread: Math.max(0.05, input.spread),
  };
  const gradients = buildGradients(params);
  const update = averageUpdate(gradients, params.clipping);
  const updateMagnitude = magnitude(update);

  const mags = gradients.map((g) => magnitude(g.vec));
  const meanMagnitude =
    mags.reduce((sum, m) => sum + m, 0) / (mags.length || 1);
  const maxMag = mags.reduce((max, m) => Math.max(max, m), 0);
  const concentration = meanMagnitude > 0 ? maxMag / meanMagnitude : 0;
  const diffuseness = angularSpread(gradients);
  const collapsed = updateMagnitude > SAFE_RADIUS;

  return {
    method: params.method,
    gradients,
    update,
    updateMagnitude,
    meanMagnitude,
    concentration,
    diffuseness,
    safeRadius: SAFE_RADIUS,
    collapsed,
  };
}

export const PLANE_EXTENT = PLANE;

// ---- event log (inline, kept self-contained) ------------------------------

export interface GeometryEvent {
  id: number;
  tick: number;
  label: string;
  snapshot: GeometryParams;
}

export interface GeometryLog {
  events: GeometryEvent[];
  nextId: number;
}

const LOG_LIMIT = 60;

function emptyLog(): GeometryLog {
  return { events: [], nextId: 1 };
}

function recordEvent(
  log: GeometryLog,
  tick: number,
  label: string,
  snapshot: GeometryParams,
): GeometryLog {
  const event: GeometryEvent = { id: log.nextId, tick, label, snapshot };
  return {
    events: [...log.events, event].slice(-LOG_LIMIT),
    nextId: log.nextId + 1,
  };
}

// ---- sim state, interventions, presets -------------------------------------

export interface SimState {
  params: GeometryParams;
  tick: number;
  log: GeometryLog;
}

export type Intervention =
  | { type: "setMethod"; method: Method }
  | { type: "toggleClipping" }
  | { type: "setBatchSize"; value: number }
  | { type: "setRewardBias"; value: number }
  | { type: "setPivotStrength"; value: number }
  | { type: "setSpread"; value: number }
  | { type: "setSeed"; value: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; params: GeometryParams; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: GeometryParams;
}

export const PRESETS: Preset[] = [
  {
    id: "rl-cancels",
    label: "RL: noise cancels",
    blurb:
      "A large batch of sparse RL gradients. Most are noise, a few carry the reward signal. Step the seed and watch the update stay small and trustworthy while the cloud reshuffles.",
    params: {
      method: "rl",
      batchSize: 140,
      rewardBias: 1,
      pivotStrength: 1,
      clipping: false,
      spread: 0.4,
      seed: 7,
    },
  },
  {
    id: "rl-small-batch",
    label: "RL: batch too small",
    blurb:
      "Drop the batch and the noise stops cancelling. The averaged update points somewhere random and can land in the collapse zone, which is why RL wants large batches and patience.",
    params: {
      method: "rl",
      batchSize: 10,
      rewardBias: 1,
      pivotStrength: 1,
      clipping: false,
      spread: 0.4,
      seed: 3,
    },
  },
  {
    id: "sft-diffuse",
    label: "SFT: dense and diffuse",
    blurb:
      "Every token pulls toward the data, fanned across many directions. The update is large but well-spread, so SFT is forgiving. Narrow the spread and it starts to concentrate.",
    params: {
      method: "sft",
      batchSize: 90,
      rewardBias: 1,
      pivotStrength: 1,
      clipping: false,
      spread: 0.95,
      seed: 11,
    },
  },
  {
    id: "opsd-collapse",
    label: "OPSD: collapse",
    blurb:
      "One pivot token the teacher loves and the student barely supports, 100x a typical token, with clipping off. The single tug drags the update past the safe radius. Turn clipping on and the update snaps back inside.",
    params: {
      method: "opsd",
      batchSize: 60,
      rewardBias: 1,
      pivotStrength: 100,
      clipping: false,
      spread: 0.9,
      seed: 5,
    },
  },
  {
    id: "opsd-clipped",
    label: "OPSD: clipped",
    blurb:
      "The same pivot token, but the per-token KL clip caps its contribution. The update stays inside the safe radius and the step is usable.",
    params: {
      method: "opsd",
      batchSize: 60,
      rewardBias: 1,
      pivotStrength: 100,
      clipping: true,
      spread: 0.9,
      seed: 5,
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? PRESETS[0]!;
}

export function initialState(presetId = "rl-cancels"): SimState {
  return {
    params: { ...getPreset(presetId).params },
    tick: 0,
    log: emptyLog(),
  };
}

const METHOD_NAMES: Record<Method, string> = {
  rl: "RL",
  sft: "SFT",
  opd: "OPD",
  opsd: "OPSD",
};

export function methodName(method: Method): string {
  return METHOD_NAMES[method];
}

function applyIntervention(
  params: GeometryParams,
  action: Intervention,
): { params: GeometryParams; label: string } {
  switch (action.type) {
    case "setMethod":
      return {
        params: { ...params, method: action.method },
        label: `method -> ${METHOD_NAMES[action.method]}`,
      };
    case "toggleClipping": {
      const next = { ...params, clipping: !params.clipping };
      return { params: next, label: `KL clip ${next.clipping ? "on" : "off"}` };
    }
    case "setBatchSize":
      return {
        params: { ...params, batchSize: action.value },
        label: `batch -> ${Math.round(action.value)}`,
      };
    case "setRewardBias":
      return {
        params: { ...params, rewardBias: action.value },
        label: `reward bias -> ${action.value.toFixed(2)}`,
      };
    case "setPivotStrength":
      return {
        params: { ...params, pivotStrength: action.value },
        label: `pivot -> ${Math.round(action.value)}x`,
      };
    case "setSpread":
      return {
        params: { ...params, spread: action.value },
        label: `spread -> ${action.value.toFixed(2)}`,
      };
    case "setSeed":
      return {
        params: { ...params, seed: action.value },
        label: `seed -> ${action.value}`,
      };
    case "loadPreset": {
      const preset = getPreset(action.id);
      return {
        params: { ...preset.params },
        label: `preset -> ${preset.label}`,
      };
    }
    case "restore":
      return { params: { ...action.params }, label: action.label };
  }
}

function logged(
  state: SimState,
  label: string,
  params: GeometryParams,
): SimState {
  return {
    ...state,
    params,
    log: recordEvent(state.log, state.tick, label, params),
  };
}

// The pure reducer. A tick redraws the gradient cloud by advancing the seed, so
// pressing play animates fresh draws of the same method and the reader watches
// the averaged update hold steady (RL, SFT) or stay collapsed (OPSD unclipped).
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") {
    const seed = (state.params.seed + 1) % 1000;
    const params = { ...state.params, seed };
    const advanced: SimState = { ...state, tick: state.tick + 1 };
    return logged(advanced, `redraw (seed ${seed})`, params);
  }
  const { params, label } = applyIntervention(state.params, input.action);
  return logged(state, label, params);
}
