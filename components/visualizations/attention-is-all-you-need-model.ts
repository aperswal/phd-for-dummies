// A pure, deterministic model of scaled dot-product attention from "Attention Is
// All You Need". The math is exactly the paper's: every query position scores
// every key by a dot product, divides by sqrt(d_k), softmaxes the scores into
// weights, and returns a weighted sum of the values. Token embeddings and the
// per-head projection matrices are hand-set so the attention patterns mean
// something a reader can recognize (a pronoun finding its noun, neighbors via
// position). A trained model learns those numbers from data; here they are fixed
// so the patterns are legible. The run is fully deterministic — no randomness
// reaches the math.

import { at } from "@/lib/simulation/array";
import {
  emptyLog,
  record,
  type EventLog,
  type SimEvent,
} from "@/lib/simulation/history";

export const SEMANTIC_DIM = 8;
export const POSITIONAL_DIM = 8;
export const D_MODEL = SEMANTIC_DIM + POSITIONAL_DIM;

// Named semantic axes. The first SEMANTIC_DIM dimensions of every token vector
// carry meaning; the rest carry the sinusoidal position stamp.
const SEM = {
  animate: 0,
  object: 1,
  verb: 2,
  negation: 3,
  reason: 4,
  pronoun: 5,
  adjective: 6,
  determiner: 7,
} as const;

type SemKey = keyof typeof SEM;

function semantic(tags: Partial<Record<SemKey, number>>): number[] {
  const vector = new Array<number>(SEMANTIC_DIM).fill(0);
  for (const key of Object.keys(tags) as SemKey[]) {
    vector[SEM[key]] = tags[key] ?? 0;
  }
  return vector;
}

// Sinusoidal positional encoding, PE(pos, 2i) = sin(pos / 10000^(2i/d)) and
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d)), with d = POSITIONAL_DIM. Nearby
// positions share similar stamps, so a head that reads only this block attends
// locally without anyone teaching it to.
export function positionalEncoding(pos: number): number[] {
  const encoding = new Array<number>(POSITIONAL_DIM).fill(0);
  for (let p = 0; p < POSITIONAL_DIM; p++) {
    const pair = Math.floor(p / 2);
    const angle = pos / Math.pow(10000, (2 * pair) / POSITIONAL_DIM);
    encoding[p] = p % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  }
  return encoding;
}

interface TokenSpec {
  text: string;
  tags: Partial<Record<SemKey, number>>;
}

export interface Sentence {
  id: string;
  label: string;
  tokens: string[];
  embeddings: number[][];
  note: string;
}

function buildSentence(
  id: string,
  label: string,
  note: string,
  specs: TokenSpec[],
): Sentence {
  const tokens = specs.map((spec) => spec.text);
  const embeddings = specs.map((spec, position) => [
    ...semantic(spec.tags),
    ...positionalEncoding(position),
  ]);
  return { id, label, tokens, embeddings, note };
}

export const SENTENCES: Sentence[] = [
  buildSentence(
    "animal",
    "The animal didn't cross the street because it was too tired",
    "'it' has two possible antecedents, 'animal' and 'street'. The meaning head resolves it to 'animal'.",
    [
      { text: "The", tags: { determiner: 1 } },
      { text: "animal", tags: { animate: 1 } },
      { text: "didn't", tags: { verb: 0.6, negation: 1 } },
      { text: "cross", tags: { verb: 1 } },
      { text: "the", tags: { determiner: 1 } },
      { text: "street", tags: { object: 1 } },
      { text: "because", tags: { reason: 1 } },
      { text: "it", tags: { pronoun: 1 } },
      { text: "was", tags: { verb: 0.7 } },
      { text: "too", tags: { adjective: 0.5 } },
      { text: "tired", tags: { adjective: 1 } },
    ],
  ),
  buildSentence(
    "cat",
    "The cat sat because it felt safe",
    "A shorter sentence. 'it' resolves to the only animate noun, 'cat'.",
    [
      { text: "The", tags: { determiner: 1 } },
      { text: "cat", tags: { animate: 1 } },
      { text: "sat", tags: { verb: 1 } },
      { text: "because", tags: { reason: 1 } },
      { text: "it", tags: { pronoun: 1 } },
      { text: "felt", tags: { verb: 0.7 } },
      { text: "safe", tags: { adjective: 1 } },
    ],
  ),
];

export function getSentence(id: string): Sentence {
  return SENTENCES.find((sentence) => sentence.id === id) ?? at(SENTENCES, 0);
}

// A head is a set of query/key direction pairs (the rows of W^Q and W^K) plus
// value directions (the rows of W^V). The query projection of token i is the
// vector of its dot products with each query direction; likewise for keys. The
// attention score of i against j is then the dot product of those two
// projections, which is exactly Q_i . K_j.
export interface Head {
  name: string;
  blurb: string;
  qDirs: number[][];
  kDirs: number[][];
  vDirs: number[][];
}

function axis(...active: number[]): number[] {
  const vector = new Array<number>(D_MODEL).fill(0);
  for (const index of active) vector[index] = 1;
  return vector;
}

function scaledAxis(weight: number, ...active: number[]): number[] {
  return axis(...active).map((value) => value * weight);
}

// Value projection shared by every head: read the semantic block straight
// through, so a token's output is a weighted blend of the meanings it attended
// to.
const VALUE_DIRS: number[][] = Array.from({ length: SEMANTIC_DIM }, (_, d) =>
  axis(d),
);

// Positional directions for the neighbor head. It skips the fastest frequency
// pair (dims 0 and 1 of the stamp), because that pair cycles every ~6 positions
// and makes far-apart words score high. With it dropped, the dot product of two
// stamps falls off cleanly as the words get farther apart, which is the local
// attention the head is meant to show.
const NEIGHBOR_DIRS: number[][] = Array.from(
  { length: POSITIONAL_DIM - 2 },
  (_, p) => axis(SEMANTIC_DIM + 2 + p),
);

export const HEADS: Head[] = [
  {
    name: "Meaning",
    blurb:
      "Pronouns reach for an animate noun, verbs reach for their noun arguments, states reach for their subject.",
    qDirs: [
      axis(SEM.pronoun),
      axis(SEM.verb),
      axis(SEM.adjective),
      axis(SEM.reason),
    ],
    kDirs: [
      axis(SEM.animate),
      axis(SEM.animate, SEM.object),
      axis(SEM.animate),
      axis(SEM.verb),
    ],
    vDirs: VALUE_DIRS,
  },
  {
    name: "Neighbors",
    blurb:
      "Reads only the position stamp, so each word leans on the words physically near it. Nothing taught it to; close positions just have similar stamps.",
    qDirs: NEIGHBOR_DIRS,
    kDirs: NEIGHBOR_DIRS,
    vDirs: VALUE_DIRS,
  },
  {
    name: "Broad",
    blurb:
      "A weak, diffuse projection, so the weight spreads almost evenly. This is the averaging a single head falls into, which is why the paper runs several.",
    qDirs: [scaledAxis(0.3, SEM.animate, SEM.object, SEM.verb)],
    kDirs: [scaledAxis(0.3, SEM.animate, SEM.object, SEM.verb)],
    vDirs: VALUE_DIRS,
  },
];

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] ?? 0) * (b[i] ?? 0);
  return sum;
}

function project(vector: number[], dirs: number[][]): number[] {
  return dirs.map((dir) => dot(vector, dir));
}

function softmax(logits: number[]): number[] {
  const finite = logits.filter((value) => Number.isFinite(value));
  const max = finite.length > 0 ? Math.max(...finite) : 0;
  const exps = logits.map((value) =>
    Number.isFinite(value) ? Math.exp(value - max) : 0,
  );
  const total = exps.reduce((sum, value) => sum + value, 0) || 1;
  return exps.map((value) => value / total);
}

export interface AttentionParams {
  sentenceId: string;
  headIndex: number;
  queryIndex: number;
  scaling: boolean;
  causalMask: boolean;
  magnitude: number;
}

export interface AttentionResult {
  tokens: string[];
  queryIndex: number;
  head: Head;
  dk: number;
  rawScores: number[];
  logits: number[];
  weights: number[];
  output: number[];
  maxWeight: number;
  entropy: number;
}

// The whole of equation 1, computed for one query position over every key.
// Embeddings are deterministic, so the same params always produce the same run.
export function computeAttention(params: AttentionParams): AttentionResult {
  const sentence = getSentence(params.sentenceId);
  const embeddings = sentence.embeddings;
  const headIndex =
    params.headIndex >= 0 && params.headIndex < HEADS.length
      ? params.headIndex
      : 0;
  const head = at(HEADS, headIndex);
  const n = sentence.tokens.length;
  const queryIndex = Math.min(Math.max(params.queryIndex, 0), n - 1);

  const dk = head.qDirs.length;
  const query = project(at(embeddings, queryIndex), head.qDirs);

  const rawScores = embeddings.map((vector) =>
    dot(query, project(vector, head.kDirs)),
  );

  // The score magnitude slider is the lever the paper describes for
  // demonstrating softmax saturation (see "Softmax saturation" preset).
  const divisor = params.scaling ? Math.sqrt(dk) : 1;

  const logits = rawScores.map((score, j) => {
    if (params.causalMask && j > queryIndex) return Number.NEGATIVE_INFINITY;
    return (score * params.magnitude) / divisor;
  });

  const weights = softmax(logits);

  const output = head.vDirs.map((_, d) =>
    weights.reduce(
      (sum, weight, j) =>
        sum + weight * at(project(at(embeddings, j), head.vDirs), d),
      0,
    ),
  );

  const maxWeight = weights.reduce((max, value) => Math.max(max, value), 0);
  const entropy = -weights.reduce(
    (sum, value) => sum + (value > 0 ? value * Math.log(value) : 0),
    0,
  );

  return {
    tokens: sentence.tokens,
    queryIndex,
    head,
    dk,
    rawScores,
    logits,
    weights,
    output,
    maxWeight,
    entropy,
  };
}

// A logged moment carries the full parameter set, so the event log can rewind
// the run to exactly that configuration.
export type AttentionEvent = SimEvent<AttentionParams>;

export interface SimState {
  params: AttentionParams;
  tick: number;
  log: EventLog<AttentionParams>;
}

export type Intervention =
  | { type: "setQuery"; index: number }
  | { type: "setHead"; index: number }
  | { type: "setSentence"; id: string }
  | { type: "toggleScaling" }
  | { type: "toggleMask" }
  | { type: "setMagnitude"; value: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; params: AttentionParams; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: AttentionParams;
}

const ANIMAL = "animal";

export const PRESETS: Preset[] = [
  {
    id: "meaning",
    label: "Meaning: it -> animal",
    blurb:
      "The meaning head, scaling on. Step the query to 'it' and watch the weight land on 'animal', not the nearer 'street'.",
    params: {
      sentenceId: ANIMAL,
      headIndex: 0,
      queryIndex: 7,
      scaling: true,
      causalMask: false,
      magnitude: 5,
    },
  },
  {
    id: "saturation",
    label: "Softmax saturation",
    blurb:
      "'because' looks at the verbs, scoring 'cross' highest. With big scores and scaling off the weight collapses onto 'cross' alone. Turn scaling on and 'was' and 'didn't' come back.",
    params: {
      sentenceId: ANIMAL,
      headIndex: 0,
      queryIndex: 6,
      scaling: false,
      causalMask: false,
      magnitude: 16,
    },
  },
  {
    id: "mask",
    label: "Causal mask (decoder)",
    blurb:
      "The decoder mask. Every word past the query is blocked, so the weights form a triangle and no word peeks at its future.",
    params: {
      sentenceId: ANIMAL,
      headIndex: 0,
      queryIndex: 6,
      scaling: true,
      causalMask: true,
      magnitude: 5,
    },
  },
  {
    id: "neighbors",
    label: "Neighbors from position",
    blurb:
      "The neighbor head reads only the position stamp, so each word leans on the words beside it.",
    params: {
      sentenceId: ANIMAL,
      headIndex: 1,
      queryIndex: 3,
      scaling: true,
      causalMask: false,
      magnitude: 8,
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(presetId = "meaning"): SimState {
  return {
    params: { ...getPreset(presetId).params },
    tick: 0,
    log: emptyLog(),
  };
}

function logged(
  state: SimState,
  label: string,
  params: AttentionParams,
): SimState {
  return {
    ...state,
    params,
    log: record(state.log, state.tick, label, params),
  };
}

function tokenName(params: AttentionParams, index: number): string {
  return getSentence(params.sentenceId).tokens[index] ?? `#${index}`;
}

function applyIntervention(
  params: AttentionParams,
  action: Intervention,
): { params: AttentionParams; label: string } {
  switch (action.type) {
    case "setQuery": {
      const next = { ...params, queryIndex: action.index };
      return {
        params: next,
        label: `query -> '${tokenName(next, action.index)}'`,
      };
    }
    case "setHead": {
      const next = { ...params, headIndex: action.index };
      return {
        params: next,
        label: `head -> ${HEADS[action.index]?.name ?? "?"}`,
      };
    }
    case "setSentence": {
      const sentence = getSentence(action.id);
      const queryIndex = Math.min(
        params.queryIndex,
        sentence.tokens.length - 1,
      );
      return {
        params: { ...params, sentenceId: action.id, queryIndex },
        label: `sentence -> ${sentence.id}`,
      };
    }
    case "toggleScaling": {
      const next = { ...params, scaling: !params.scaling };
      return { params: next, label: `scaling ${next.scaling ? "on" : "off"}` };
    }
    case "toggleMask": {
      const next = { ...params, causalMask: !params.causalMask };
      return {
        params: next,
        label: `causal mask ${next.causalMask ? "on" : "off"}`,
      };
    }
    case "setMagnitude":
      return {
        params: { ...params, magnitude: action.value },
        label: `score magnitude -> ${action.value.toFixed(1)}`,
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

// The pure reducer. A tick sweeps the query to the next word; an intervention
// changes one parameter and records what happened. Every step is deterministic.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") {
    const n = getSentence(state.params.sentenceId).tokens.length;
    const queryIndex = (state.params.queryIndex + 1) % n;
    const params = { ...state.params, queryIndex };
    const advanced: SimState = { ...state, tick: state.tick + 1 };
    return logged(
      advanced,
      `query -> '${tokenName(params, queryIndex)}'`,
      params,
    );
  }

  const { params, label } = applyIntervention(state.params, input.action);
  return logged(state, label, params);
}
