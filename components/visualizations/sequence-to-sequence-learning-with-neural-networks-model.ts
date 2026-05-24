// A pure, deterministic model of the sequence-to-sequence LSTM from "Sequence to
// Sequence Learning with Neural Networks". One LSTM (the encoder) reads the
// source sentence one token at a time and writes each into a single fixed-size
// memory vector; a second LSTM (the decoder) reads that vector and emits the
// target sentence one token at a time until an end-of-sentence token.
//
// The model captures the paper's central, testable claim: a target word can only
// come out right if the signal from its matching source word survives the number
// of recurrent steps between when that source word entered and when its target
// leaves. That count is the "minimal time lag". Reversing the source order (the
// paper's key trick) shrinks the lag for the first correspondences while leaving
// the average distance unchanged, which is why training the reversed model works.
//
// This is a toy that uses an exponential memory-decay law instead of a trained
// LSTM, so the patterns are legible. No randomness reaches the outcome: same
// inputs, same run. It is deterministic and replayable.

// ---------------------------------------------------------------------------
// Self-contained history log (kept local to this file so the sim does not depend
// on shared simulation modules).
// ---------------------------------------------------------------------------

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

const LOG_LIMIT = 60;

function emptyLog<TSnapshot>(): EventLog<TSnapshot> {
  return { events: [], nextEventId: 1 };
}

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
  if (value === undefined) {
    throw new RangeError(`index ${index} out of range`);
  }
  return value;
}

// ---------------------------------------------------------------------------
// Sentences. Each one is a monotonic toy alignment: source word i translates to
// target word i. The animals/garden examples echo the paper's a,b,c -> alpha,
// beta, gamma framing and its long-sentence study.
// ---------------------------------------------------------------------------

export interface Sentence {
  id: string;
  label: string;
  source: string[];
  target: string[];
  note: string;
}

export const SENTENCES: Sentence[] = [
  {
    id: "short",
    label: "a b c",
    source: ["a", "b", "c"],
    target: ["A", "B", "C"],
    note: "The paper's own example: a,b,c maps to A,B,C, where reversing the source puts a next to A.",
  },
  {
    id: "cat",
    label: "the cat sat down",
    source: ["the", "cat", "sat", "down"],
    target: ["le", "chat", "s'est", "assis"],
    note: "A short clause. Each French word lines up with its English source word.",
  },
  {
    id: "long",
    label: "long sentence (9 words)",
    source: [
      "the",
      "small",
      "dog",
      "ran",
      "across",
      "the",
      "wet",
      "green",
      "field",
    ],
    target: [
      "le",
      "petit",
      "chien",
      "courait",
      "sur",
      "le",
      "champ",
      "vert",
      "humide",
    ],
    note: "A long sentence. Forward, the first words sit far from their matches, so the early signal decays past the threshold and the decoder emits UNK.",
  },
];

export function getSentence(id: string): Sentence {
  return SENTENCES.find((sentence) => sentence.id === id) ?? at(SENTENCES, 0);
}

export const UNK = "UNK";
export const EOS = "<EOS>";

// ---------------------------------------------------------------------------
// The forward model: encode, then decode, computing for every target position
// the lag its source signal had to survive and whether it cleared the threshold.
// ---------------------------------------------------------------------------

export interface SeqParams {
  sentenceId: string;
  reversed: boolean;
  memory: number; // tau, how many recurrent steps a signal survives
  capacity: number; // how many words the fixed jar holds before everything fades
  dropEos: boolean; // if true, the decoder never sees its stop signal
  threshold: number; // signal a word needs to come out right
}

export interface TokenTrace {
  targetIndex: number;
  sourceWord: string;
  targetWord: string;
  encoderStep: number; // global step when the matching source word was written
  decoderStep: number; // global step when this target word is emitted
  lag: number; // decoderStep - encoderStep
  signal: number; // retained signal after decay and capacity pressure
  emitted: string; // the word the decoder actually produces
  recovered: boolean;
}

export interface SeqResult {
  source: string[];
  fedOrder: string[]; // the source in the order the encoder actually reads it
  target: string[];
  traces: TokenTrace[];
  emitted: string[]; // the decoder's full output, including EOS
  recoveredCount: number;
  accuracy: number; // fraction of target words recovered
  allRecovered: boolean;
  runaway: boolean; // true when dropEos made the decoder fail to stop
  capacityPressure: number; // 0..1, how much the overfull jar attenuates signal
  encodeLen: number;
}

const RUNAWAY_LIMIT = 24;

function capacityFactor(length: number, capacity: number): number {
  if (length <= capacity) return 1;
  // An overfull jar attenuates every stored signal. The penalty grows smoothly
  // with how far past capacity the sentence runs.
  return capacity / length;
}

export function computeSeq(params: SeqParams): SeqResult {
  const sentence = getSentence(params.sentenceId);
  const source = sentence.source;
  const target = sentence.target;
  const n = source.length;

  // The order the encoder reads the source. Reversing flips it, which is the
  // whole trick. Target order never changes.
  const fedOrder = params.reversed ? [...source].reverse() : [...source];

  // For each source position i, find the global encoder step at which it was
  // written. Forward: position i is written at step i. Reversed: position i is
  // written at step (n - 1 - i), so the first source word is written last.
  const encoderStepOf = (sourcePos: number): number =>
    params.reversed ? n - 1 - sourcePos : sourcePos;

  const pressure = 1 - capacityFactor(n, params.capacity);
  const tau = Math.max(params.memory, 0.05);

  const traces: TokenTrace[] = [];
  for (let i = 0; i < n; i++) {
    const encoderStep = encoderStepOf(i);
    // The decoder emits target word i at global step n + i (it spends n steps
    // reading the source first, then one step per output word).
    const decoderStep = n + i;
    const lag = decoderStep - encoderStep;
    const signal = Math.exp(-lag / tau) * capacityFactor(n, params.capacity);
    const recovered = signal >= params.threshold;
    traces.push({
      targetIndex: i,
      sourceWord: at(source, i),
      targetWord: at(target, i),
      encoderStep,
      decoderStep,
      lag,
      signal,
      emitted: recovered ? at(target, i) : UNK,
      recovered,
    });
  }

  // The decoder's output stream. Normally it appends EOS and stops. With the EOS
  // signal dropped it never stops, so it keeps emitting UNK until a hard limit.
  const emitted = traces.map((trace) => trace.emitted);
  let runaway = false;
  if (params.dropEos) {
    runaway = true;
    const extra = RUNAWAY_LIMIT - emitted.length;
    for (let k = 0; k < extra; k++) emitted.push(UNK);
  } else {
    emitted.push(EOS);
  }

  const recoveredCount = traces.filter((trace) => trace.recovered).length;
  return {
    source,
    fedOrder,
    target,
    traces,
    emitted,
    recoveredCount,
    accuracy: n === 0 ? 0 : recoveredCount / n,
    allRecovered: recoveredCount === n && !runaway,
    runaway,
    capacityPressure: pressure,
    encodeLen: n,
  };
}

// ---------------------------------------------------------------------------
// The stepped simulation. A run walks a cursor through the whole process: one
// step per source word read into the jar (encode phase), then one step per
// target word emitted (decode phase). The cursor is the only thing a tick
// advances; the math above is pure and recomputed from the params for any
// cursor. This is what the view animates.
// ---------------------------------------------------------------------------

export type Phase = "encode" | "decode" | "done";

export interface SeqState {
  params: SeqParams;
  cursor: number; // 0..(encodeLen + emittedLen)
  tick: number;
  log: EventLog<SeqParams>;
}

export interface CursorView {
  phase: Phase;
  encodeProgress: number; // how many source words are in the jar so far
  decodeProgress: number; // how many target words emitted so far
  activeEncodeIndex: number | null; // which fed-order position is being written now
  activeDecodeIndex: number | null; // which target position is being emitted now
  totalSteps: number;
}

export function totalSteps(result: SeqResult): number {
  return result.encodeLen + result.emitted.length;
}

export function viewAtCursor(result: SeqResult, cursor: number): CursorView {
  const encodeLen = result.encodeLen;
  const decodeLen = result.emitted.length;
  const total = encodeLen + decodeLen;
  const clamped = Math.max(0, Math.min(cursor, total));

  if (clamped <= encodeLen) {
    return {
      phase: clamped < encodeLen ? "encode" : decodeLen > 0 ? "decode" : "done",
      encodeProgress: clamped,
      decodeProgress: 0,
      activeEncodeIndex: clamped < encodeLen ? clamped : null,
      activeDecodeIndex: null,
      totalSteps: total,
    };
  }

  const decodeProgress = clamped - encodeLen;
  return {
    phase: decodeProgress >= decodeLen ? "done" : "decode",
    encodeProgress: encodeLen,
    decodeProgress,
    activeEncodeIndex: null,
    activeDecodeIndex: decodeProgress < decodeLen ? decodeProgress : null,
    totalSteps: total,
  };
}

export type Intervention =
  | { type: "toggleReversed" }
  | { type: "setSentence"; id: string }
  | { type: "setMemory"; value: number }
  | { type: "setCapacity"; value: number }
  | { type: "toggleDropEos" }
  | { type: "setThreshold"; value: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; params: SeqParams; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: SeqParams;
}

const BASE_THRESHOLD = 0.12;

export const PRESETS: Preset[] = [
  {
    id: "forward-long",
    label: "Forward (long sentence fails)",
    blurb:
      "A 9-word sentence read in normal order. The first words sit 9 steps from their match, so their signal decays past the threshold and the decoder emits UNK for them. Watch the early outputs break.",
    params: {
      sentenceId: "long",
      reversed: false,
      memory: 4,
      capacity: 12,
      dropEos: false,
      threshold: BASE_THRESHOLD,
    },
  },
  {
    id: "reversed-long",
    label: "Reversed (long sentence works)",
    blurb:
      "The same 9-word sentence read backwards. Now the first source word is written last, so its target comes out one step later and recovers. Reversing fixed the early words without touching the average distance.",
    params: {
      sentenceId: "long",
      reversed: true,
      memory: 4,
      capacity: 12,
      dropEos: false,
      threshold: BASE_THRESHOLD,
    },
  },
  {
    id: "short-clean",
    label: "Short sentence",
    blurb:
      "The paper's a,b,c example. Short enough that forward order already works, so flipping reversal barely changes the output. Switch the sentence to the long one to see reversal earn its keep.",
    params: {
      sentenceId: "short",
      reversed: false,
      memory: 4,
      capacity: 12,
      dropEos: false,
      threshold: BASE_THRESHOLD,
    },
  },
  {
    id: "no-eos",
    label: "No stop token",
    blurb:
      "The decoder never sees its end-of-sentence signal, so it keeps emitting and never stops. This is why every sentence ends with a special <EOS> token.",
    params: {
      sentenceId: "cat",
      reversed: true,
      memory: 4,
      capacity: 12,
      dropEos: true,
      threshold: BASE_THRESHOLD,
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(presetId = "forward-long"): SeqState {
  return {
    params: { ...getPreset(presetId).params },
    cursor: 0,
    tick: 0,
    log: emptyLog(),
  };
}

function logged(state: SeqState, label: string, params: SeqParams): SeqState {
  return {
    ...state,
    params,
    cursor: 0,
    log: record(state.log, state.tick, label, params),
  };
}

function applyIntervention(
  params: SeqParams,
  action: Intervention,
): { params: SeqParams; label: string } {
  switch (action.type) {
    case "toggleReversed": {
      const next = { ...params, reversed: !params.reversed };
      return {
        params: next,
        label: `source ${next.reversed ? "reversed" : "forward"}`,
      };
    }
    case "setSentence": {
      const sentence = getSentence(action.id);
      return {
        params: { ...params, sentenceId: action.id },
        label: `sentence -> ${sentence.label}`,
      };
    }
    case "setMemory":
      return {
        params: { ...params, memory: action.value },
        label: `memory tau -> ${action.value.toFixed(1)}`,
      };
    case "setCapacity":
      return {
        params: { ...params, capacity: action.value },
        label: `capacity -> ${action.value}`,
      };
    case "toggleDropEos": {
      const next = { ...params, dropEos: !params.dropEos };
      return {
        params: next,
        label: `stop token ${next.dropEos ? "dropped" : "restored"}`,
      };
    }
    case "setThreshold":
      return {
        params: { ...params, threshold: action.value },
        label: `threshold -> ${action.value.toFixed(2)}`,
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

// The pure reducer. A tick advances the run cursor by one step (read one source
// word, or emit one target word). An intervention changes a parameter, resets
// the cursor to the start so the new reality plays from the top, and logs it.
export function step(state: SeqState, input: Input): SeqState {
  if (input.kind === "tick") {
    const result = computeSeq(state.params);
    const total = totalSteps(result);
    const cursor = Math.min(state.cursor + 1, total);
    return { ...state, cursor, tick: state.tick + 1 };
  }

  const { params, label } = applyIntervention(state.params, input.action);
  return logged({ ...state, tick: state.tick + 1 }, label, params);
}
