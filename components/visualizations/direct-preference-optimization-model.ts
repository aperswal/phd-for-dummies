// A pure, deterministic model of Direct Preference Optimization from "Your
// Language Model is Secretly a Reward Model". The world is a tiny preference
// world: a handful of prompts, each with a fixed set of candidate completions.
// A "policy" is a softmax over per-completion logits, one logit per completion
// per prompt. A frozen "reference" policy holds the starting logits. The
// simulation runs DPO's actual gradient step on a dataset of preference pairs.
//
// For one pair (prompt x, preferred y_w, dispreferred y_l) the DPO loss is
//   L = -log sigma( beta * (logratio_w - logratio_l) )
// where logratio = log pi_theta(y) - log pi_ref(y). The implicit reward of a
// completion is r_hat = beta * logratio. The gradient of L with respect to a
// completion's logit is, after working through softmax, the per-example weight
//   g = beta * sigma( r_hat_l - r_hat_w )
// times the change in log-prob each completion contributes. That sigma weight is
// the heart of the paper: pairs the model already orders correctly push almost
// no gradient; pairs it gets wrong push hard. Turn the weight off and the naive
// objective shoves preferred probability toward 1 without restraint, which
// degenerates the policy. Everything here is deterministic; the only randomness
// is an optional seeded jitter on the starting logits, drawn from a seeded rng.

export type Rng = () => number;

// One step of mulberry32 as a pure function so the model can thread the rng
// state through immutable state and stay replayable.
function nextRandom(state: number): { value: number; state: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, state: a };
}

function mulberry32(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    const next = nextRandom(state);
    state = next.state;
    return next.value;
  };
}

// A bounded, append-only log so the reader can rewind to any earlier moment.
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

function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) throw new RangeError(`index ${index} out of range`);
  return value;
}

// A completion the policy can produce for a prompt. `quality` is a hidden
// ground-truth score used only to set the preference label of a fresh dataset
// and to draw the "good vs bad region" backdrop; the policy never sees it.
export interface Completion {
  text: string;
  quality: number;
}

export interface Prompt {
  id: string;
  text: string;
  completions: Completion[];
}

// The fixed completion world. Each prompt offers four candidate replies that
// run from clearly preferred to clearly dispreferred. The numbers are hand-set
// so the patterns are legible; a real DPO run works over a full vocabulary.
export const PROMPTS: Prompt[] = [
  {
    id: "review",
    text: "write a movie review",
    completions: [
      { text: "warm, vivid praise", quality: 1.0 },
      { text: "a fair, mild take", quality: 0.55 },
      { text: "a flat one-liner", quality: 0.3 },
      { text: "a rude pan", quality: 0.05 },
    ],
  },
  {
    id: "advice",
    text: "give relationship advice",
    completions: [
      { text: "kind, specific help", quality: 1.0 },
      { text: "generic platitudes", quality: 0.5 },
      { text: "a curt brush-off", quality: 0.25 },
      { text: "an insult", quality: 0.05 },
    ],
  },
];

export function getPrompt(id: string): Prompt {
  return PROMPTS.find((prompt) => prompt.id === id) ?? at(PROMPTS, 0);
}

// A preference pair: for one prompt, completion `winner` was chosen over
// completion `loser` by the labeler. Indices point into the prompt's
// completions array. `flipped` records that the reader hand-mislabeled the pair.
export interface PreferencePair {
  promptId: string;
  winner: number;
  loser: number;
  flipped: boolean;
}

// The base dataset: for each prompt, the clear best is preferred over the clear
// worst. This is the honest, well-labeled preference set DPO is meant to fit.
function baseDataset(): PreferencePair[] {
  return PROMPTS.map((prompt) => ({
    promptId: prompt.id,
    winner: 0,
    loser: prompt.completions.length - 1,
    flipped: false,
  }));
}

export interface DpoParams {
  beta: number;
  learningRate: number;
  useAdaptiveWeight: boolean;
  seed: number;
}

// The full mutable state. `logits[promptId]` is the policy's current logit per
// completion; `refLogits[promptId]` is the frozen reference's logits. They start
// equal (DPO initializes pi_theta = pi_ref), so the policy begins as a copy of
// the SFT model and drifts only as far as the loss and beta allow.
export interface SimState {
  params: DpoParams;
  dataset: PreferencePair[];
  logits: Record<string, number[]>;
  refLogits: Record<string, number[]>;
  tick: number;
  loss: number;
  log: EventLog<Snapshot>;
}

export interface Snapshot {
  params: DpoParams;
  dataset: PreferencePair[];
  logits: Record<string, number[]>;
  tick: number;
}

function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

export function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0) || 1;
  return exps.map((value) => value / total);
}

// log pi(y) for a single completion under a softmax over the prompt's logits.
function logProb(logits: number[], index: number): number {
  const max = Math.max(...logits);
  const total = logits.reduce((sum, value) => sum + Math.exp(value - max), 0);
  return at(logits, index) - max - Math.log(total);
}

// The implicit reward DPO assigns a completion: beta times the log-ratio of the
// policy to the reference. This is the reward model the paper says is hiding
// inside the language model, recovered without ever training one.
export function implicitReward(
  state: SimState,
  promptId: string,
  index: number,
): number {
  const logits = stateLogits(state, promptId);
  const ref = state.refLogits[promptId] ?? [];
  return state.params.beta * (logProb(logits, index) - logProb(ref, index));
}

function stateLogits(state: SimState, promptId: string): number[] {
  return state.logits[promptId] ?? [];
}

// The DPO loss over the whole dataset, averaged. Each pair contributes
// -log sigma( beta * (logratio_w - logratio_l) ).
export function datasetLoss(state: SimState): number {
  if (state.dataset.length === 0) return 0;
  let total = 0;
  for (const pair of state.dataset) {
    const logits = stateLogits(state, pair.promptId);
    const ref = state.refLogits[pair.promptId] ?? [];
    const w = pair.flipped ? pair.loser : pair.winner;
    const l = pair.flipped ? pair.winner : pair.loser;
    const ratioW = logProb(logits, w) - logProb(ref, w);
    const ratioL = logProb(logits, l) - logProb(ref, l);
    const margin = state.params.beta * (ratioW - ratioL);
    total += -Math.log(Math.max(sigmoid(margin), 1e-12));
  }
  return total / state.dataset.length;
}

// The KL divergence of the policy from the reference, summed over prompts. This
// is the quantity beta is meant to keep small; the reader watches it grow as the
// policy drifts and watches a bigger beta hold it back.
export function policyKl(state: SimState): number {
  let total = 0;
  for (const prompt of PROMPTS) {
    const p = softmax(stateLogits(state, prompt.id));
    const ref = softmax(state.refLogits[prompt.id] ?? []);
    for (let i = 0; i < p.length; i++) {
      const pi = at(p, i);
      const ri = at(ref, i);
      if (pi > 0) total += pi * Math.log(pi / Math.max(ri, 1e-12));
    }
  }
  return total;
}

// One DPO gradient step on every pair, accumulated into the policy logits.
//
// For a softmax policy, d log pi(y_k) / d logit_j = [j == k] - pi(j). So a pair
// (w, l) pushes logit_j of its prompt by the per-example weight times
//   ( ([j==w] - pi_j) - ([j==l] - pi_j) ) = [j==w] - [j==l].
// The pi_j terms cancel, which is why a preference pair only ever raises the
// winner's logit and lowers the loser's by the same amount. The weight is
//   beta * sigma( r_hat_l - r_hat_w )      (adaptive, the paper's update)
// or a constant beta when the reader turns the adaptive weight off (the naive
// objective from Appendix Table 3 that degenerates).
function gradientStep(state: SimState): {
  logits: Record<string, number[]>;
  loss: number;
} {
  const next: Record<string, number[]> = {};
  for (const prompt of PROMPTS) {
    next[prompt.id] = [...stateLogits(state, prompt.id)];
  }

  for (const pair of state.dataset) {
    const logits = next[pair.promptId] ?? [];
    const ref = state.refLogits[pair.promptId] ?? [];
    const w = pair.flipped ? pair.loser : pair.winner;
    const l = pair.flipped ? pair.winner : pair.loser;

    const rewardW = state.params.beta * (logProb(logits, w) - logProb(ref, w));
    const rewardL = state.params.beta * (logProb(logits, l) - logProb(ref, l));

    const weight = state.params.useAdaptiveWeight
      ? state.params.beta * sigmoid(rewardL - rewardW)
      : state.params.beta;

    const update = weight * state.params.learningRate;
    logits[w] = at(logits, w) + update;
    logits[l] = at(logits, l) - update;
    next[pair.promptId] = logits;
  }

  const stepped: SimState = { ...state, logits: next };
  return { logits: next, loss: datasetLoss(stepped) };
}

export type Intervention =
  | { type: "setBeta"; value: number }
  | { type: "setLearningRate"; value: number }
  | { type: "toggleAdaptiveWeight" }
  | { type: "setSeed"; value: number }
  | { type: "flipPair"; index: number }
  | { type: "removePair"; index: number }
  | { type: "addPair"; promptId: string; winner: number; loser: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: Snapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: DpoParams;
  dataset: PreferencePair[];
}

function pair(
  promptId: string,
  winner: number,
  loser: number,
  flipped = false,
): PreferencePair {
  return { promptId, winner, loser, flipped };
}

export const PRESETS: Preset[] = [
  {
    id: "align",
    label: "Align to preferences",
    blurb:
      "The honest dataset and a steady beta. Play it and watch the preferred reply climb while the policy stays close to the reference.",
    params: { beta: 0.3, learningRate: 1, useAdaptiveWeight: true, seed: 0 },
    dataset: baseDataset(),
  },
  {
    id: "degenerate",
    label: "Drop the sigma weight",
    blurb:
      "Same data, but the adaptive sigma weight is off. The naive objective shoves the winner toward probability one and the KL runs away.",
    params: { beta: 0.3, learningRate: 1, useAdaptiveWeight: false, seed: 0 },
    dataset: baseDataset(),
  },
  {
    id: "tight",
    label: "Tighten beta",
    blurb:
      "A large beta clamps the policy near the reference, so even with clear preferences the probabilities barely move.",
    params: { beta: 1.2, learningRate: 1, useAdaptiveWeight: true, seed: 0 },
    dataset: baseDataset(),
  },
  {
    id: "mislabel",
    label: "Mislabel a pair",
    blurb:
      "The review pair is flipped, so the labeler prefers the rude pan. DPO faithfully chases the wrong answer up.",
    params: { beta: 0.3, learningRate: 1, useAdaptiveWeight: true, seed: 0 },
    dataset: [pair("review", 0, 3, true), pair("advice", 0, 3, false)],
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

// Starting logits for every prompt, optionally jittered by a seeded rng so the
// reference is not perfectly uniform. The policy starts equal to the reference.
function buildLogits(seed: number): Record<string, number[]> {
  const rng = seed === 0 ? null : mulberry32(seed);
  const out: Record<string, number[]> = {};
  for (const prompt of PROMPTS) {
    out[prompt.id] = prompt.completions.map(() =>
      rng ? (rng() - 0.5) * 1.5 : 0,
    );
  }
  return out;
}

export function initialState(presetId = "align"): SimState {
  const preset = getPreset(presetId);
  const refLogits = buildLogits(preset.params.seed);
  const logits: Record<string, number[]> = {};
  for (const prompt of PROMPTS) {
    logits[prompt.id] = [...(refLogits[prompt.id] ?? [])];
  }
  const base: SimState = {
    params: { ...preset.params },
    dataset: preset.dataset.map((p) => ({ ...p })),
    logits,
    refLogits,
    tick: 0,
    loss: 0,
    log: emptyLog(),
  };
  return { ...base, loss: datasetLoss(base) };
}

function snapshot(state: SimState): Snapshot {
  const logits: Record<string, number[]> = {};
  for (const prompt of PROMPTS) {
    logits[prompt.id] = [...stateLogits(state, prompt.id)];
  }
  return {
    params: { ...state.params },
    dataset: state.dataset.map((p) => ({ ...p })),
    logits,
    tick: state.tick,
  };
}

function logged(state: SimState, label: string): SimState {
  return {
    ...state,
    log: record(state.log, state.tick, label, snapshot(state)),
  };
}

function completionName(promptId: string, index: number): string {
  return getPrompt(promptId).completions[index]?.text ?? `#${index}`;
}

function applyIntervention(state: SimState, action: Intervention): SimState {
  switch (action.type) {
    case "setBeta": {
      const next: SimState = {
        ...state,
        params: { ...state.params, beta: action.value },
      };
      return logged(
        { ...next, loss: datasetLoss(next) },
        `beta -> ${action.value.toFixed(2)}`,
      );
    }
    case "setLearningRate": {
      const next: SimState = {
        ...state,
        params: { ...state.params, learningRate: action.value },
      };
      return logged(next, `step size -> ${action.value.toFixed(2)}`);
    }
    case "toggleAdaptiveWeight": {
      const on = !state.params.useAdaptiveWeight;
      const next: SimState = {
        ...state,
        params: { ...state.params, useAdaptiveWeight: on },
      };
      return logged(next, `sigma weight ${on ? "on" : "off"}`);
    }
    case "setSeed": {
      const refLogits = buildLogits(action.value);
      const logits: Record<string, number[]> = {};
      for (const prompt of PROMPTS) {
        logits[prompt.id] = [...(refLogits[prompt.id] ?? [])];
      }
      const next: SimState = {
        ...state,
        params: { ...state.params, seed: action.value },
        logits,
        refLogits,
        tick: 0,
      };
      return logged(
        { ...next, loss: datasetLoss(next) },
        `seed -> ${action.value}`,
      );
    }
    case "flipPair": {
      const dataset = state.dataset.map((p, i) =>
        i === action.index ? { ...p, flipped: !p.flipped } : p,
      );
      const next: SimState = { ...state, dataset };
      const target = dataset[action.index];
      const label = target
        ? `pair ${action.index + 1} ${target.flipped ? "mislabeled" : "restored"}`
        : `pair ${action.index + 1}`;
      return logged({ ...next, loss: datasetLoss(next) }, label);
    }
    case "removePair": {
      const dataset = state.dataset.filter((_, i) => i !== action.index);
      const next: SimState = { ...state, dataset };
      return logged(
        { ...next, loss: datasetLoss(next) },
        `removed pair ${action.index + 1}`,
      );
    }
    case "addPair": {
      const dataset = [
        ...state.dataset,
        pair(action.promptId, action.winner, action.loser),
      ];
      const next: SimState = { ...state, dataset };
      return logged(
        { ...next, loss: datasetLoss(next) },
        `added '${completionName(action.promptId, action.winner)}' > '${completionName(action.promptId, action.loser)}'`,
      );
    }
    case "loadPreset": {
      const fresh = initialState(action.id);
      return logged(fresh, `preset -> ${getPreset(action.id).label}`);
    }
    case "restore": {
      const refLogits = buildLogits(action.snapshot.params.seed);
      const logits: Record<string, number[]> = {};
      for (const prompt of PROMPTS) {
        logits[prompt.id] = [...(action.snapshot.logits[prompt.id] ?? [])];
      }
      const next: SimState = {
        ...state,
        params: { ...action.snapshot.params },
        dataset: action.snapshot.dataset.map((p) => ({ ...p })),
        logits,
        refLogits,
        tick: action.snapshot.tick,
      };
      return logged({ ...next, loss: datasetLoss(next) }, action.label);
    }
  }
}

// The pure reducer. A tick runs one full DPO gradient step over the dataset and
// records the new loss. An intervention changes a parameter or the dataset.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") {
    const { logits, loss } = gradientStep(state);
    const advanced: SimState = {
      ...state,
      logits,
      loss,
      tick: state.tick + 1,
    };
    return logged(advanced, `step ${advanced.tick}: loss ${loss.toFixed(3)}`);
  }
  return applyIntervention(state, input.action);
}
