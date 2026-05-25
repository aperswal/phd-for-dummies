// A pure, deterministic model of zero-shot task transfer from "Language Models
// Are Unsupervised Multitask Learners" (GPT-2). The paper's whole claim is that a
// model trained only to predict the next token, p(x) = prod p(s_n | s_1..s_{n-1}),
// will perform a downstream task with no fine-tuning when you frame the task in
// the prompt, and that it does this better the larger the model. This model
// reproduces that behavior in a small, legible world.
//
// What is real here and what is staged. The autoregressive loop is real: at each
// step the model scores every candidate next token, divides by temperature,
// keeps the top k, softmaxes into a probability distribution, and samples from it
// with a seeded rng. The scores themselves come from hand-built per-task tables
// instead of a trained network, so the patterns a reader recognizes (the gold
// answer winning, the summary tracking the article, the French gloss appearing)
// are legible rather than learned. Capacity is modeled the way the paper measured
// it: a bigger model sharpens the boost toward the correct continuation, which is
// the log-linear improvement of Figure 1. Removing the task hint kills the
// task-conditioned boost, which is the paper's 6.4-point ROUGE drop when the
// "TL;DR:" hint is taken away. No randomness reaches the math except the sampler,
// which draws from the shared seeded rng, so a given seed reproduces the run.

// Indexed read that throws on a genuine out-of-range access instead of returning
// undefined, kept inline so this simulation is self-contained.
function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) {
    throw new RangeError(`index ${index} out of range`);
  }
  return value;
}

// One step of mulberry32 as a pure function: given the generator state, return
// the next value in [0, 1) and the next state. Threading the state through
// immutable sim state keeps every sampled run replayable from a seed.
function nextRandom(state: number): { value: number; state: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, state: a };
}

// The four model sizes from Table 2. The smallest matches the original GPT, the
// largest is GPT-2. Capacity is the single number that drives the log-linear
// effect the paper reports.
export interface ModelSize {
  params: string;
  layers: number;
  dModel: number;
  // Capacity in [0, 1]: how sharply this size concentrates probability on the
  // correct continuation. Set to match the relative ordering of Figure 1.
  capacity: number;
}

export const MODEL_SIZES: ModelSize[] = [
  { params: "117M", layers: 12, dModel: 768, capacity: 0.16 },
  { params: "345M", layers: 24, dModel: 1024, capacity: 0.46 },
  { params: "762M", layers: 36, dModel: 1280, capacity: 0.74 },
  { params: "1542M", layers: 48, dModel: 1600, capacity: 1.0 },
];

export const GPT2_SIZE_INDEX = MODEL_SIZES.length - 1;

// A task is one conditional p(output | input, task). The prompt frames which one
// is active. Each task carries a context the model reads, a continuation it
// should produce when capacity is high and the hint is present, and a set of
// distractor tokens that win when capacity is low or the hint is gone.
export interface Task {
  id: string;
  label: string;
  // The natural-language hint that selects this task, e.g. "TL;DR:" or "A:".
  hint: string;
  // What the reader sees as the conditioning text before generation starts.
  prompt: string;
  // The tokens the model should emit, in order, on the happy path.
  gold: string[];
  // Generic continuations the model falls back to with no task signal. These are
  // what the corpus alone (next-token prediction with no task) produces.
  filler: string[];
  // A short note on what this task is and what the paper found.
  note: string;
}

export const TASKS: Task[] = [
  {
    id: "qa",
    label: "Question answering",
    hint: "A:",
    prompt: "Who wrote the book the origin of species? A:",
    gold: ["Charles", "Darwin"],
    filler: ["the", "author", "of", "many", "books"],
    note: "Natural Questions, zero-shot. GPT-2 answers 4.1% correctly, 5.3x the smallest model. Its top answers are well calibrated.",
  },
  {
    id: "translate",
    label: "Translation (en to fr)",
    hint: "=",
    prompt: "english sentence = french sentence\nI am not a fool =",
    gold: ["Je", "ne", "suis", "pas", "un", "imbecile"],
    filler: ["I", "am", "not", "a", "fool", "again"],
    note: "WMT-14 En-Fr, conditioned on example pairs. GPT-2 reaches 5 BLEU, learned from the small French slice that survived the English filter.",
  },
  {
    id: "summarize",
    label: "Summarization",
    hint: "TL;DR:",
    prompt:
      "The team won the final after a late goal. Fans flooded the streets to celebrate the title. TL;DR:",
    gold: ["The", "team", "won", "the", "title"],
    filler: ["after", "a", "late", "goal", "the", "fans", "flooded"],
    note: "CNN / Daily Mail. The TL;DR: hint induces summary behavior. Remove it and ROUGE drops 6.4 points toward random-3.",
  },
  {
    id: "reading",
    label: "Reading comprehension",
    hint: "A:",
    prompt:
      "The cat sat on the mat because it was warm. Q: why did the cat sit? A:",
    gold: ["it", "was", "warm"],
    filler: ["the", "cat", "sat", "on", "the", "mat"],
    note: "CoQA, zero-shot. GPT-2 reaches 55 F1, matching 3 of 4 baselines that trained on 127k+ examples.",
  },
];

export function getTask(id: string): Task {
  return TASKS.find((task) => task.id === id) ?? at(TASKS, 0);
}

// A candidate next token plus the pieces of its score, exposed so the inspector
// can show why one token wins. logit is the value fed to softmax; prob is its
// share after softmax and top-k masking.
export interface Candidate {
  token: string;
  base: number;
  taskBoost: number;
  logit: number;
  prob: number;
  isGold: boolean;
  masked: boolean;
}

// A deterministic per-token scoring world. The base score is a small generic
// preference shared across tasks (this stands in for the corpus bigram signal).
// The task boost pushes the gold token up, scaled by capacity and by whether the
// hint is present. Distractors get a mild boost so a low-capacity model has
// something wrong to land on.
function baseScore(token: string, position: number): number {
  // A stable pseudo-score from the token and position so the "corpus" is
  // deterministic and the same token does not always win by accident.
  let hash = position * 2654435761;
  for (let i = 0; i < token.length; i++) {
    hash = (hash ^ token.charCodeAt(i)) * 16777619;
  }
  return ((hash >>> 0) % 1000) / 1000;
}

export interface GenParams {
  taskId: string;
  sizeIndex: number;
  temperature: number;
  topK: number;
  seed: number;
  // Whether the natural-language task hint is present in the prompt. The paper's
  // hint-ablation: with it, summary behavior; without it, generic continuation.
  hintPresent: boolean;
}

export interface SimState {
  params: GenParams;
  // Tokens generated so far, in order.
  generated: string[];
  // Cursor into the gold sequence; the next gold token the happy path expects.
  goldCursor: number;
  // Threaded rng state, so the whole run replays from the seed.
  rngState: number;
  tick: number;
  // Whether generation reached the end of the gold sequence or the length cap.
  finished: boolean;
  log: SimEvent[];
  nextEventId: number;
}

export interface SimEvent {
  id: number;
  tick: number;
  label: string;
  snapshot: GenParams;
}

const MAX_TOKENS = 6;

// The candidate set at the current step: the next gold token, the active filler,
// and a couple of fixed distractors, deduped. Keeping the set small makes the
// distribution legible while still being a real softmax over many options.
function candidateTokens(task: Task, goldCursor: number): string[] {
  const set: string[] = [];
  const add = (token: string) => {
    if (!set.includes(token)) set.push(token);
  };
  if (goldCursor < task.gold.length) add(at(task.gold, goldCursor));
  const fillerStart = goldCursor % task.filler.length;
  for (let i = 0; i < 3; i++) {
    add(at(task.filler, (fillerStart + i) % task.filler.length));
  }
  add("the");
  add("and");
  return set;
}

// Score every candidate, divide by temperature, keep the top k by logit, softmax
// the survivors. This is the real next-token distribution the reader samples
// from. The task boost is what the paper's whole argument rests on: the prompt
// makes p(output | input, task) the active conditional, and capacity decides how
// hard the model leans into it.
export function scoreCandidates(
  params: GenParams,
  goldCursor: number,
): Candidate[] {
  const task = getTask(params.taskId);
  const size = at(MODEL_SIZES, clampIndex(params.sizeIndex));
  const tokens = candidateTokens(task, goldCursor);
  const goldToken =
    goldCursor < task.gold.length ? at(task.gold, goldCursor) : null;

  // With the hint present the task signal is full strength; without it the model
  // has no reason to prefer the gold continuation, so the boost collapses.
  const hintFactor = params.hintPresent ? 1 : 0.05;

  const scored = tokens.map((token) => {
    const base = baseScore(token, goldCursor);
    const isGold = token === goldToken;
    // A correct token gets a boost that grows with capacity; this is the
    // log-linear lift of Figure 1 made concrete. A wrong filler token gets a
    // small constant pull so a tiny model has a plausible thing to say instead.
    const taskBoost = isGold ? 3.4 * size.capacity * hintFactor : 0;
    return { token, base, taskBoost, isGold };
  });

  const temp = Math.max(params.temperature, 0.05);
  const withLogits = scored.map((c) => ({
    ...c,
    logit: (c.base + c.taskBoost) / temp,
    prob: 0,
    masked: false,
  }));

  // Top-k masking, exactly the truncation the paper uses for its samples.
  const k = Math.max(1, Math.min(params.topK, withLogits.length));
  const ranked = [...withLogits].sort((a, b) => b.logit - a.logit);
  const kept = new Set(ranked.slice(0, k).map((c) => c.token));
  for (const c of withLogits) {
    c.masked = !kept.has(c.token);
  }

  const live = withLogits.filter((c) => !c.masked);
  const maxLogit = Math.max(...live.map((c) => c.logit));
  const exps = live.map((c) => Math.exp(c.logit - maxLogit));
  const total = exps.reduce((sum, value) => sum + value, 0) || 1;
  live.forEach((c, i) => {
    c.prob = at(exps, i) / total;
  });

  return withLogits;
}

function clampIndex(index: number): number {
  return Math.min(Math.max(index, 0), MODEL_SIZES.length - 1);
}

// Sample one token from the candidate distribution using the threaded rng. The
// rng draw is the only nondeterministic step, and it is seeded, so the run is
// reproducible.
function sampleToken(
  candidates: Candidate[],
  rngState: number,
): { token: string; rngState: number } {
  const live = candidates.filter((c) => !c.masked);
  const draw = nextRandom(rngState);
  let cumulative = 0;
  for (const c of live) {
    cumulative += c.prob;
    if (draw.value <= cumulative) {
      return { token: c.token, rngState: draw.state };
    }
  }
  return { token: at(live, live.length - 1).token, rngState: draw.state };
}

// How well the run so far matches the gold continuation, in [0, 1]. This is the
// emergent quantity worth watching: it rises with capacity and with the hint,
// and collapses when the hint is removed.
export function goldMatch(state: SimState): number {
  const gold = getTask(state.params.taskId).gold;
  if (state.generated.length === 0) return 0;
  let hits = 0;
  for (let i = 0; i < state.generated.length && i < gold.length; i++) {
    if (state.generated[i] === gold[i]) hits += 1;
  }
  return hits / Math.max(state.generated.length, 1);
}

// The probability the model currently assigns to the correct next token, before
// sampling. The paper stresses these probabilities are well calibrated for the
// big model, so this readout climbs as size grows.
export function goldProbability(params: GenParams, goldCursor: number): number {
  const candidates = scoreCandidates(params, goldCursor);
  return candidates.find((c) => c.isGold && !c.masked)?.prob ?? 0;
}

export type Intervention =
  | { type: "setTask"; id: string }
  | { type: "setSize"; index: number }
  | { type: "setTemperature"; value: number }
  | { type: "setTopK"; value: number }
  | { type: "reshuffle" }
  | { type: "toggleHint" }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; params: GenParams; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: GenParams;
}

export const PRESETS: Preset[] = [
  {
    id: "qa-happy",
    label: "Zero-shot Q&A (GPT-2)",
    blurb:
      "The full 1542M model, hint on. Play it and watch 'Charles Darwin' fall out of next-token prediction with no fine-tuning.",
    params: {
      taskId: "qa",
      sizeIndex: GPT2_SIZE_INDEX,
      temperature: 0.7,
      topK: 3,
      seed: 7,
      hintPresent: true,
    },
  },
  {
    id: "summarize-hint",
    label: "Summary with TL;DR:",
    blurb:
      "The TL;DR: hint induces summary behavior. Play it, then toggle the hint off mid-run and watch the output drift back to the article.",
    params: {
      taskId: "summarize",
      sizeIndex: GPT2_SIZE_INDEX,
      temperature: 0.6,
      topK: 3,
      seed: 3,
      hintPresent: true,
    },
  },
  {
    id: "small-model",
    label: "Tiny model fails",
    blurb:
      "Same Q&A prompt on the 117M model. The boost toward the answer is weak, so the wrong token often wins. Bump the size up and the answer returns.",
    params: {
      taskId: "qa",
      sizeIndex: 0,
      temperature: 0.7,
      topK: 4,
      seed: 7,
      hintPresent: true,
    },
  },
  {
    id: "translate",
    label: "Translate en to fr",
    blurb:
      "Conditioned on one example pair, the model glosses 'I am not a fool' into French, the trick from the deleted Reddit posts in WebText.",
    params: {
      taskId: "translate",
      sizeIndex: GPT2_SIZE_INDEX,
      temperature: 0.6,
      topK: 3,
      seed: 5,
      hintPresent: true,
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(presetId = "qa-happy"): SimState {
  const params = { ...getPreset(presetId).params };
  return {
    params,
    generated: [],
    goldCursor: 0,
    rngState: params.seed >>> 0,
    tick: 0,
    finished: false,
    log: [],
    nextEventId: 1,
  };
}

function record(state: SimState, label: string, params: GenParams): SimState {
  const event: SimEvent = {
    id: state.nextEventId,
    tick: state.tick,
    label,
    snapshot: params,
  };
  return {
    ...state,
    params,
    log: [...state.log, event].slice(-60),
    nextEventId: state.nextEventId + 1,
  };
}

// Restart the autoregressive run with new parameters but keep the log, so a
// reader who changes a knob sees a fresh generation under the new reality.
function restart(state: SimState, params: GenParams, label: string): SimState {
  const reset: SimState = {
    ...state,
    params,
    generated: [],
    goldCursor: 0,
    rngState: params.seed >>> 0,
    finished: false,
  };
  return record(reset, label, params);
}

function applyIntervention(state: SimState, action: Intervention): SimState {
  const p = state.params;
  switch (action.type) {
    case "setTask": {
      const next = { ...p, taskId: action.id };
      return restart(state, next, `task -> ${getTask(action.id).label}`);
    }
    case "setSize": {
      const index = clampIndex(action.index);
      const next = { ...p, sizeIndex: index };
      return restart(state, next, `size -> ${at(MODEL_SIZES, index).params}`);
    }
    case "setTemperature":
      return restart(
        state,
        { ...p, temperature: action.value },
        `temperature -> ${action.value.toFixed(2)}`,
      );
    case "setTopK":
      return restart(
        state,
        { ...p, topK: action.value },
        `top-k -> ${action.value}`,
      );
    case "reshuffle": {
      // Advance the seed by a large prime to get a different sample without
      // exposing a meaningless slider value to the reader.
      const nextSeed = (p.seed + 31337) >>> 0;
      return restart(state, { ...p, seed: nextSeed }, "reshuffle");
    }
    case "toggleHint": {
      // Toggling the hint does not restart: it changes the task signal for the
      // rest of the run, which is the point of the mid-run ablation.
      const next = { ...p, hintPresent: !p.hintPresent };
      return record(
        { ...state },
        `hint ${next.hintPresent ? "on" : "removed"}`,
        next,
      );
    }
    case "loadPreset": {
      const preset = getPreset(action.id);
      return restart(state, { ...preset.params }, `preset -> ${preset.label}`);
    }
    case "restore":
      return restart(state, { ...action.params }, action.label);
  }
}

// The pure reducer. A tick samples one next token and appends it; an
// intervention changes a parameter. Every step is deterministic given the seed.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "intervene") {
    return applyIntervention(state, input.action);
  }

  if (state.finished || state.generated.length >= MAX_TOKENS) {
    return { ...state, finished: true };
  }

  const candidates = scoreCandidates(state.params, state.goldCursor);
  const { token, rngState } = sampleToken(candidates, state.rngState);
  const task = getTask(state.params.taskId);
  const matchedGold =
    state.goldCursor < task.gold.length &&
    token === at(task.gold, state.goldCursor);

  const generated = [...state.generated, token];
  const goldCursor = matchedGold ? state.goldCursor + 1 : state.goldCursor;
  const finished =
    generated.length >= MAX_TOKENS || goldCursor >= task.gold.length;

  return {
    ...state,
    generated,
    goldCursor,
    rngState,
    tick: state.tick + 1,
    finished,
  };
}
