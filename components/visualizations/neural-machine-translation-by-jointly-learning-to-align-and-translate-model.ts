// A pure, deterministic model of the soft-alignment decoder from "Neural Machine
// Translation by Jointly Learning to Align and Translate". The math is the
// paper's: for the target word at decoding step i, an additive alignment network
// scores every source annotation h_j with e_ij = v_a . tanh(W_a s_{i-1} + U_a h_j),
// softmaxes those energies over the source into weights alpha_ij that sum to one,
// and reads back the context c_i = sum_j alpha_ij h_j. The decoder state s_{i-1}
// is summarized here by which source content the next target word is reaching for,
// so the alignment patterns are ones a reader can recognize (monotonic copy, and a
// reordered noun phrase). A trained model learns W_a, U_a, v_a and the annotations
// from data; here they are hand-set so the patterns are legible. The run is fully
// deterministic: no randomness is applied to the annotations.

import { at } from "@/lib/simulation/array";
import {
  emptyLog,
  record,
  type EventLog,
  type SimEvent,
} from "@/lib/simulation/history";

// The annotation h_j of a source word carries a few named content channels. The
// BiRNN in the paper builds these from both directions; here each source word
// declares which channels it holds so the alignment network has something to
// match against.
const CHANNELS = [
  "subject",
  "object",
  "action",
  "modifier",
  "function",
  "place",
] as const;

type Channel = (typeof CHANNELS)[number];

export const ANNOTATION_DIM = CHANNELS.length;

function annotation(content: Partial<Record<Channel, number>>): number[] {
  const vector = new Array<number>(ANNOTATION_DIM).fill(0);
  CHANNELS.forEach((channel, index) => {
    vector[index] = content[channel] ?? 0;
  });
  return vector;
}

// One source word and the target word the decoder produces at the same step. The
// `wants` block is what the decoder state s_{i-1} is reaching for when it is about
// to emit this target word; the alignment network matches it against every source
// annotation. `sourceFocus` is the source position the model should land on, used
// only to label the worked example, never to force the math.
interface PairSpec {
  source: string;
  target: string;
  content: Partial<Record<Channel, number>>;
  wants: Partial<Record<Channel, number>>;
}

export interface SentencePair {
  id: string;
  label: string;
  note: string;
  source: string[];
  target: string[];
  annotations: number[][];
  // What the decoder reaches for at each target step, in annotation space.
  queries: number[][];
}

function buildPair(
  id: string,
  label: string,
  note: string,
  specs: PairSpec[],
): SentencePair {
  return {
    id,
    label,
    note,
    source: specs.map((spec) => spec.source),
    target: specs.map((spec) => spec.target),
    annotations: specs.map((spec) => annotation(spec.content)),
    queries: specs.map((spec) => annotation(spec.wants)),
  };
}

// A short monotonic sentence: French and English line up almost one-to-one, so
// the alignment runs straight down the diagonal.
const MONOTONIC = buildPair(
  "monotonic",
  "She read the old letter slowly",
  "Source and target line up almost one to one, so the alignment runs down the diagonal.",
  [
    {
      source: "She",
      target: "Elle",
      content: { subject: 1 },
      wants: { subject: 1 },
    },
    {
      source: "read",
      target: "lisait",
      content: { action: 1 },
      wants: { action: 1 },
    },
    {
      source: "the",
      target: "la",
      content: { function: 1 },
      wants: { function: 1 },
    },
    {
      source: "old",
      target: "vieille",
      content: { modifier: 1 },
      wants: { modifier: 1 },
    },
    {
      source: "letter",
      target: "lettre",
      content: { object: 1 },
      wants: { object: 1 },
    },
    {
      source: "slowly",
      target: "lentement",
      content: { modifier: 0.6, action: 0.4 },
      wants: { modifier: 0.6, action: 0.4 },
    },
  ],
);

// The paper's headline qualitative result: a reordered noun phrase. English
// "European Economic Area" becomes French "zone economique europeenne", so when
// the model writes "zone" it jumps ahead to "Area", then walks back over the two
// adjectives. The wants block for the target words is shifted so the alignment is
// non-monotonic, exactly the jump the paper shows in Figure 3(a).
const REORDER = buildPair(
  "reorder",
  "the European Economic Area",
  "French flips the noun phrase, so writing 'zone' jumps ahead to 'Area' then walks back over the adjectives. This is Figure 3(a).",
  [
    {
      source: "the",
      target: "la",
      content: { function: 1 },
      wants: { function: 1 },
    },
    {
      source: "European",
      target: "zone",
      content: { modifier: 1, place: 0.3 },
      wants: { object: 1, place: 0.5 },
    },
    {
      source: "Economic",
      target: "economique",
      content: { modifier: 1 },
      wants: { modifier: 1 },
    },
    {
      source: "Area",
      target: "europeenne",
      content: { object: 1, place: 1 },
      wants: { modifier: 0.6, place: 1 },
    },
  ],
);

// A long sentence where the fixed-vector bottleneck bites. Translating the tail
// needs content from the head ("doctor" governs "admit"), which a single fixed
// summary built from the last source word cannot hold.
const LONG = buildPair(
  "long",
  "a doctor may admit a patient to a nearby hospital today",
  "A long sentence. The tail needs content from the head, which one fixed summary loses. Turn the bottleneck on and watch the tail fall apart.",
  [
    {
      source: "a",
      target: "un",
      content: { function: 1 },
      wants: { function: 1 },
    },
    {
      source: "doctor",
      target: "medecin",
      content: { subject: 1 },
      wants: { subject: 1 },
    },
    {
      source: "may",
      target: "peut",
      content: { action: 0.5, function: 0.5 },
      wants: { action: 0.5, function: 0.5 },
    },
    {
      source: "admit",
      target: "admettre",
      content: { action: 1 },
      wants: { action: 1 },
    },
    {
      source: "a",
      target: "un",
      content: { function: 1 },
      wants: { function: 1 },
    },
    {
      source: "patient",
      target: "patient",
      content: { object: 1 },
      wants: { object: 1 },
    },
    {
      source: "to",
      target: "dans",
      content: { function: 1, place: 0.4 },
      wants: { function: 1, place: 0.4 },
    },
    {
      source: "a",
      target: "un",
      content: { function: 1 },
      wants: { function: 1 },
    },
    {
      source: "nearby",
      target: "proche",
      content: { modifier: 1, place: 0.5 },
      wants: { modifier: 1, place: 0.5 },
    },
    {
      source: "hospital",
      target: "hopital",
      content: { place: 1, object: 0.5 },
      wants: { place: 1, object: 0.5 },
    },
    {
      source: "today",
      target: "aujourd'hui",
      content: { modifier: 0.5, action: 0.3 },
      wants: { modifier: 0.5, action: 0.3 },
    },
  ],
);

export const PAIRS: SentencePair[] = [MONOTONIC, REORDER, LONG];

export function getPair(id: string): SentencePair {
  return PAIRS.find((pair) => pair.id === id) ?? at(PAIRS, 0);
}

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] ?? 0) * (b[i] ?? 0);
  return sum;
}

function softmax(energies: number[]): number[] {
  const max = energies.length > 0 ? Math.max(...energies) : 0;
  const exps = energies.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0) || 1;
  return exps.map((value) => value / total);
}

export interface AlignParams {
  pairId: string;
  // The decoding step i: which target word the model is about to emit.
  step: number;
  // Off is the paper's RNNsearch (soft attention). On collapses to the basic
  // encoder-decoder, where every step reuses one fixed summary built from the
  // last annotation, blurred by the encoded sentence length.
  bottleneck: boolean;
  // The alignment-network gain v_a, scaling the energies before softmax. Higher
  // sharpens toward a hard alignment; lower blurs toward a uniform read.
  sharpness: number;
}

export interface AlignResult {
  source: string[];
  target: string[];
  step: number;
  // Raw alignment energies e_ij over the source for the current step.
  energies: number[];
  // The attention weights alpha_ij, summing to one over the source.
  weights: number[];
  // The context vector c_i = sum_j alpha_ij h_j.
  context: number[];
  argmax: number;
  maxWeight: number;
  // Shannon entropy of the weight distribution, in nats. High means a soft,
  // spread read; low means a sharp, near-hard alignment.
  entropy: number;
  bottleneck: boolean;
}

// The additive alignment model from equation 6 and the appendix:
// e_ij = v_a . tanh(W_a s_{i-1} + U_a h_j). Here W_a s_{i-1} is the decoder's
// query (what the next target word reaches for) and U_a h_j is the source
// annotation, both already in annotation space, so the tanh of their sum scored
// by the gain v_a gives the energy. Softmax over the source gives the weights,
// and the weighted sum of annotations gives the context.
export function computeAlignment(params: AlignParams): AlignResult {
  const pair = getPair(params.pairId);
  const annotations = pair.annotations;
  const n = pair.source.length;
  const step = Math.min(Math.max(params.step, 0), pair.target.length - 1);
  const query = at(pair.queries, step);

  // When the bottleneck is on, the decoder cannot re-read: every step sees one
  // fixed summary, the last annotation, so the energy is the same constant
  // against every source position and the read collapses to a flat blur. This is
  // the basic encoder-decoder the paper compares against.
  const fixedSummary = at(annotations, n - 1);
  const bottleneckEnergy = params.sharpness * dot(fixedSummary, fixedSummary);
  const energies = annotations.map((h) => {
    if (params.bottleneck) return bottleneckEnergy;
    const blended = query.map((q, d) => Math.tanh(q + (h[d] ?? 0)));
    return params.sharpness * dot(query, blended);
  });

  const weights = softmax(energies);

  const context = new Array<number>(ANNOTATION_DIM).fill(0);
  weights.forEach((weight, j) => {
    const h = at(annotations, j);
    for (let d = 0; d < ANNOTATION_DIM; d++) {
      context[d] = (context[d] ?? 0) + weight * (h[d] ?? 0);
    }
  });

  const argmax = weights.indexOf(Math.max(...weights));
  const maxWeight = weights.reduce((max, value) => Math.max(max, value), 0);
  const entropy = -weights.reduce(
    (sum, value) => sum + (value > 0 ? value * Math.log(value) : 0),
    0,
  );

  return {
    source: pair.source,
    target: pair.target,
    step,
    energies,
    weights,
    context,
    argmax,
    maxWeight,
    entropy,
    bottleneck: params.bottleneck,
  };
}

export type AlignEvent = SimEvent<AlignParams>;

export interface SimState {
  params: AlignParams;
  tick: number;
  log: EventLog<AlignParams>;
}

export type Intervention =
  | { type: "setStep"; step: number }
  | { type: "setPair"; id: string }
  | { type: "toggleBottleneck" }
  | { type: "setSharpness"; value: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; params: AlignParams; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: AlignParams;
}

export const PRESETS: Preset[] = [
  {
    id: "monotonic",
    label: "Monotonic copy",
    blurb:
      "Soft attention on a sentence that lines up word for word. Play it and watch the bright cell walk straight down the diagonal.",
    params: {
      pairId: "monotonic",
      step: 0,
      bottleneck: false,
      sharpness: 4,
    },
  },
  {
    id: "reorder",
    label: "Reordered phrase",
    blurb:
      "The paper's Figure 3(a). Step to the word 'zone' and watch the weight jump ahead to 'Area', then walk back over the adjectives.",
    params: {
      pairId: "reorder",
      step: 1,
      bottleneck: false,
      sharpness: 5,
    },
  },
  {
    id: "bottleneck",
    label: "Fixed-vector bottleneck",
    blurb:
      "The old encoder-decoder. Every step reuses one fixed summary, so the read is the same flat blur no matter which word it writes. Turn the bottleneck off to bring the alignment back.",
    params: {
      pairId: "long",
      step: 6,
      bottleneck: true,
      sharpness: 4,
    },
  },
  {
    id: "long",
    label: "Long sentence, attention on",
    blurb:
      "The same long sentence with soft attention. Step into the tail and watch the read still find the right source word, where the bottleneck lost it.",
    params: {
      pairId: "long",
      step: 6,
      bottleneck: false,
      sharpness: 4,
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(presetId = "monotonic"): SimState {
  return {
    params: { ...getPreset(presetId).params },
    tick: 0,
    log: emptyLog(),
  };
}

function logged(state: SimState, label: string, params: AlignParams): SimState {
  return {
    ...state,
    params,
    log: record(state.log, state.tick, label, params),
  };
}

function targetName(params: AlignParams, step: number): string {
  return getPair(params.pairId).target[step] ?? `#${step}`;
}

function applyIntervention(
  params: AlignParams,
  action: Intervention,
): { params: AlignParams; label: string } {
  switch (action.type) {
    case "setStep": {
      const next = { ...params, step: action.step };
      return {
        params: next,
        label: `write '${targetName(next, action.step)}'`,
      };
    }
    case "setPair": {
      const pair = getPair(action.id);
      const step = Math.min(params.step, pair.target.length - 1);
      return {
        params: { ...params, pairId: action.id, step },
        label: `sentence -> ${pair.id}`,
      };
    }
    case "toggleBottleneck": {
      const next = { ...params, bottleneck: !params.bottleneck };
      return {
        params: next,
        label: `bottleneck ${next.bottleneck ? "on" : "off"}`,
      };
    }
    case "setSharpness":
      return {
        params: { ...params, sharpness: action.value },
        label: `sharpness -> ${action.value.toFixed(1)}`,
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

// The pure reducer. A tick advances the decoder to the next target word and wraps
// at the end of the sentence; an intervention changes one parameter and records
// what happened. Every step is deterministic.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") {
    const n = getPair(state.params.pairId).target.length;
    const nextStep = (state.params.step + 1) % n;
    const params = { ...state.params, step: nextStep };
    const advanced: SimState = { ...state, tick: state.tick + 1 };
    return logged(advanced, `write '${targetName(params, nextStep)}'`, params);
  }

  const { params, label } = applyIntervention(state.params, input.action);
  return logged(state, label, params);
}
