// A pure, deterministic model of in-context learning from "Language Models are
// Few-Shot Learners" (GPT-3). The reader builds a prompt out of an optional task
// description plus a stack of worked examples, then the model runs test queries
// one at a time. Each query succeeds or fails as a seeded Bernoulli trial whose
// probability is the paper's own in-context learning curve: success rises with
// the number of examples in the prompt and rises far faster for a larger model,
// which is the paper's central finding. No weights ever change; the only thing
// that moves the curve is what sits in the prompt and how big the model is.
//
// The success curves are anchored to real numbers from the paper. The arithmetic
// task uses GPT-3 175B's 2-digit addition results from Table 3.9 (76.9% zero-shot,
// 99.6% one-shot, 100% few-shot) and the smaller-model collapse from Figure 3.10.
// The word-unscramble task uses the random-insertion row of Table 3.10 (8.3%,
// 45.4%, 67.2%). The reverse-word task uses the reversed-words row, which stays
// near zero in every setting because the skill cannot be learned from a few
// examples (Table 3.10, RW). A trained model would learn these numbers from data;
// here the curves are fit to the paper so the pattern is legible and honest.

import { emptyLog, record, type EventLog } from "@/lib/simulation/history";

// Eight model sizes from Table 2.1, in billions of parameters. The reader scales
// the model and watches the learning curve's slope change.
export interface ModelSize {
  id: string;
  label: string;
  params: number;
  // log10 of parameter count, used to interpolate the curve's shape across scale.
  scale: number;
}

export const MODEL_SIZES: ModelSize[] = [
  {
    id: "125m",
    label: "125M (Small)",
    params: 0.125,
    scale: Math.log10(0.125),
  },
  { id: "1.3b", label: "1.3B (XL)", params: 1.3, scale: Math.log10(1.3) },
  { id: "13b", label: "13B", params: 13, scale: Math.log10(13) },
  { id: "175b", label: "175B (GPT-3)", params: 175, scale: Math.log10(175) },
];

export function getModelSize(id: string): ModelSize {
  return MODEL_SIZES.find((size) => size.id === id) ?? MODEL_SIZES[3]!;
}

// A task carries a real prompt template and the curve anchors that decide how
// fast the model learns it from examples. `descriptionTokens` and `exampleTokens`
// are rough token costs so the context-window budget bites realistically.
export interface Task {
  id: string;
  label: string;
  description: string;
  // The fraction of the gap to perfect that one example closes at full scale.
  // Higher means the few-shot curve climbs faster.
  learnability: number;
  // The ceiling the largest model reaches with a full prompt (0..1). Reversed
  // words never crosses 0.01 here because the skill can't be learned from a few
  // examples, exactly as the paper reports.
  ceiling: number;
  // Zero-shot success at full scale, before any examples.
  zeroShotBase: number;
  descriptionTokens: number;
  exampleTokens: number;
  examples: WorkedExample[];
  note: string;
}

export interface WorkedExample {
  input: string;
  output: string;
}

const ARITHMETIC: Task = {
  id: "arithmetic",
  label: "2-digit addition",
  description: "Add the two numbers.",
  learnability: 0.62,
  ceiling: 1,
  zeroShotBase: 0.77,
  descriptionTokens: 6,
  exampleTokens: 14,
  examples: [
    { input: "48 + 76", output: "124" },
    { input: "12 + 9", output: "21" },
    { input: "63 + 28", output: "91" },
    { input: "5 + 87", output: "92" },
    { input: "34 + 41", output: "75" },
    { input: "29 + 58", output: "87" },
  ],
  note: "GPT-3 175B scores 76.9% zero-shot and 100% few-shot on 2-digit addition (Table 3.9). Small models stay near zero no matter how many examples they see.",
};

const UNSCRAMBLE: Task = {
  id: "unscramble",
  label: "Unscramble word (random insertion)",
  description: "Remove the inserted symbols and recover the word.",
  learnability: 0.5,
  ceiling: 0.78,
  zeroShotBase: 0.08,
  descriptionTokens: 9,
  exampleTokens: 16,
  examples: [
    { input: "s.u!c/c!e.s s i/o/n", output: "succession" },
    { input: "c#r i$t i&c a*l", output: "critical" },
    { input: "m@e a^n i!n g", output: "meaning" },
    { input: "p)r o(m i*s e", output: "promise" },
    { input: "h!a r%d e#s t", output: "hardest" },
    { input: "j&o u#r n@e y", output: "journey" },
  ],
  note: "Random insertion climbs from 8.3% zero-shot to 67.2% few-shot at 175B (Table 3.10). The task description plus examples both matter.",
};

const REVERSE: Task = {
  id: "reverse",
  label: "Reverse word (the wall)",
  description: "Spell the word backwards.",
  learnability: 0.04,
  ceiling: 0.01,
  zeroShotBase: 0.001,
  descriptionTokens: 5,
  exampleTokens: 12,
  examples: [
    { input: "stcejbo", output: "objects" },
    { input: "yadot", output: "today" },
    { input: "retaw", output: "water" },
    { input: "elbat", output: "table" },
    { input: "thgil", output: "light" },
    { input: "rorrim", output: "mirror" },
  ],
  note: "Reversed words stay near 0% in zero-, one-, and few-shot at every size (Table 3.10). No amount of prompting teaches a skill the model can't do from a few examples.",
};

export const TASKS: Task[] = [ARITHMETIC, UNSCRAMBLE, REVERSE];

export function getTask(id: string): Task {
  return TASKS.find((task) => task.id === id) ?? ARITHMETIC;
}

// The context window is a hard token budget (n_ctx = 2048 in the paper). We use a
// small scaled budget so the cap bites within a handful of demonstrations and the
// reader feels the constraint the paper names.
export const CONTEXT_BUDGET = 96;

// How much of the prompt's token budget the current configuration uses.
export function tokensUsed(
  task: Task,
  exampleCount: number,
  useDescription: boolean,
): number {
  const description = useDescription ? task.descriptionTokens : 0;
  return description + exampleCount * task.exampleTokens + task.exampleTokens;
}

// The fraction of the full-scale learning curve a given model size unlocks. A
// 175B model gets the whole curve; a 125M model gets almost none of the slope,
// which is why its accuracy stays flat as examples pile up. Interpolated on a log
// scale between the smallest and largest models.
function scaleFactor(size: ModelSize): number {
  const min = MODEL_SIZES[0]!.scale;
  const max = MODEL_SIZES[MODEL_SIZES.length - 1]!.scale;
  const t = (size.scale - min) / (max - min);
  // A convex shape: the jump to the largest model is the biggest, matching the
  // sharp step from 13B to 175B in Figures 3.10 and 3.11.
  return t * t;
}

export interface PromptConfig {
  taskId: string;
  modelId: string;
  exampleCount: number;
  useDescription: boolean;
  corruptedExamples: number;
  seed: number;
}

// The paper's in-context learning curve, computed as a probability. Effective
// examples count the clean demonstrations; a corrupted (mislabeled) example
// subtracts learning signal instead of adding it, so wrong demonstrations make
// the model worse. Success approaches the model's reachable ceiling as effective
// examples grow, with the approach rate set by the task's learnability scaled by
// how big the model is.
export function predictSuccess(config: PromptConfig): number {
  const task = getTask(config.taskId);
  const size = getModelSize(config.modelId);
  const factor = scaleFactor(size);

  const clean = Math.max(0, config.exampleCount - config.corruptedExamples);
  const effectiveExamples = clean - config.corruptedExamples * 0.9;

  const reachableCeiling = task.ceiling * factor;
  const descriptionLift = config.useDescription ? 1 : 0.55;

  // Zero-shot floor scaled by how big the model is. A small model barely beats
  // random even with a description.
  const floor = task.zeroShotBase * factor * descriptionLift;

  // Each effective example closes part of the remaining gap to the ceiling. The
  // gap-closing rate is the task's learnability times the model's scale factor,
  // so a small model closes almost no gap per example.
  const closeRate =
    task.learnability * (0.35 + 0.65 * factor) * descriptionLift;
  const gapClosed = 1 - Math.exp(-closeRate * Math.max(0, effectiveExamples));
  const learned = floor + (reachableCeiling - floor) * gapClosed;

  return clamp(learned, 0, 1);
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), high);
}

// A single test query the model attempts. Each is a fresh problem with a known
// answer; the model "answers" it correct or wrong by a seeded Bernoulli draw
// against the predicted success probability for the current prompt.
export interface Trial {
  id: number;
  prompt: string;
  truth: string;
  guess: string;
  correct: boolean;
}

const TEST_PROBLEMS: Record<string, WorkedExample[]> = {
  arithmetic: [
    { input: "57 + 38", output: "95" },
    { input: "84 + 19", output: "103" },
    { input: "26 + 67", output: "93" },
    { input: "45 + 45", output: "90" },
    { input: "73 + 18", output: "91" },
    { input: "9 + 66", output: "75" },
    { input: "38 + 54", output: "92" },
    { input: "61 + 29", output: "90" },
  ],
  unscramble: [
    { input: "f#u t&u r e", output: "future" },
    { input: "s!i g n%a l", output: "signal" },
    { input: "g)a r d(e n", output: "garden" },
    { input: "w@i n d o^w", output: "window" },
    { input: "p*l a!n e t", output: "planet" },
    { input: "b&r i#d g e", output: "bridge" },
  ],
  reverse: [
    { input: "nrettap", output: "pattern" },
    { input: "drowyek", output: "keyword" },
    { input: "ecruos", output: "source" },
    { input: "tnemele", output: "element" },
    { input: "knilpu", output: "uplink" },
    { input: "lebmys", output: "symbol" },
  ],
};

function wrongAnswer(task: Task, truth: string, rng: () => number): string {
  if (task.id === "arithmetic") {
    const value = Number(truth);
    const off = Math.floor(rng() * 9) + 1;
    return String(rng() < 0.5 ? value - off : value + off);
  }
  // For text tasks a wrong guess is the truth with two adjacent letters swapped.
  const chars = truth.split("");
  if (chars.length < 2) return truth + "?";
  const i = Math.floor(rng() * (chars.length - 1));
  [chars[i], chars[i + 1]] = [chars[i + 1]!, chars[i]!];
  return chars.join("");
}

export interface SimState {
  config: PromptConfig;
  trials: Trial[];
  attempted: number;
  correctCount: number;
  rngState: number;
  nextTrialId: number;
  log: EventLog<PromptConfig>;
}

export type Intervention =
  | { type: "setTask"; id: string }
  | { type: "setModel"; id: string }
  | { type: "addExample" }
  | { type: "removeExample" }
  | { type: "toggleDescription" }
  | { type: "corruptExample" }
  | { type: "fixExamples" }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; config: PromptConfig; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  config: PromptConfig;
}

export const PRESETS: Preset[] = [
  {
    id: "few-shot-175b",
    label: "Few-shot, 175B",
    blurb:
      "The full GPT-3 with a task description and a stack of worked examples. Press play and watch the running accuracy climb toward 100% on 2-digit addition.",
    config: {
      taskId: "arithmetic",
      modelId: "175b",
      exampleCount: 5,
      useDescription: true,
      corruptedExamples: 0,
      seed: 7,
    },
  },
  {
    id: "scale-wall",
    label: "Small model, no slope",
    blurb:
      "The same prompt on a 125M model. Add all the examples you like; the curve barely lifts, because the slope of in-context learning comes from scale.",
    config: {
      taskId: "arithmetic",
      modelId: "125m",
      exampleCount: 5,
      useDescription: true,
      corruptedExamples: 0,
      seed: 7,
    },
  },
  {
    id: "zero-shot",
    label: "Zero-shot start",
    blurb:
      "175B, a task description and no examples. Then add examples one at a time and watch each one move the prediction.",
    config: {
      taskId: "unscramble",
      modelId: "175b",
      exampleCount: 0,
      useDescription: true,
      corruptedExamples: 0,
      seed: 3,
    },
  },
  {
    id: "the-wall",
    label: "The reversed-word wall",
    blurb:
      "175B on reversing a word. Pile on examples and the prediction still sits near zero, because some skills can't be learned from a few demonstrations.",
    config: {
      taskId: "reverse",
      modelId: "175b",
      exampleCount: 5,
      useDescription: true,
      corruptedExamples: 0,
      seed: 11,
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? PRESETS[0]!;
}

export function initialState(presetId = "few-shot-175b"): SimState {
  const preset = getPreset(presetId);
  return freshRun({ ...preset.config }, emptyLog());
}

function freshRun(config: PromptConfig, log: EventLog<PromptConfig>): SimState {
  return {
    config,
    trials: [],
    attempted: 0,
    correctCount: 0,
    rngState: config.seed >>> 0,
    nextTrialId: 1,
    log,
  };
}

function maxExamples(task: Task): number {
  return task.examples.length;
}

function clampExamples(task: Task, count: number): number {
  return clamp(count, 0, maxExamples(task));
}

// Drawing a fresh problem, an answer, and the correct/wrong outcome all consume
// the seeded rng stream in a fixed order, so the same seed and the same prompt
// reproduce the same sequence of trials exactly.
function runTrial(state: SimState): SimState {
  const task = getTask(state.config.taskId);
  const probability = predictSuccess(state.config);

  let cursor = state.rngState;
  const draw = (): number => {
    const a = (cursor + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    cursor = a;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const problems = TEST_PROBLEMS[task.id] ?? [];
  const problem = problems[Math.floor(draw() * problems.length)] ?? {
    input: "?",
    output: "?",
  };
  const correct = draw() < probability;
  const guess = correct
    ? problem.output
    : wrongAnswer(task, problem.output, draw);

  const trial: Trial = {
    id: state.nextTrialId,
    prompt: problem.input,
    truth: problem.output,
    guess,
    correct,
  };

  return {
    ...state,
    trials: [...state.trials, trial].slice(-12),
    attempted: state.attempted + 1,
    correctCount: state.correctCount + (correct ? 1 : 0),
    rngState: cursor,
    nextTrialId: state.nextTrialId + 1,
  };
}

function applyIntervention(
  config: PromptConfig,
  action: Intervention,
): { config: PromptConfig; label: string } {
  switch (action.type) {
    case "setTask": {
      const task = getTask(action.id);
      const exampleCount = clampExamples(task, config.exampleCount);
      return {
        config: {
          ...config,
          taskId: action.id,
          exampleCount,
          corruptedExamples: 0,
        },
        label: `task -> ${task.label}`,
      };
    }
    case "setModel":
      return {
        config: { ...config, modelId: action.id },
        label: `model -> ${getModelSize(action.id).label}`,
      };
    case "addExample": {
      const task = getTask(config.taskId);
      const exampleCount = clampExamples(task, config.exampleCount + 1);
      return {
        config: { ...config, exampleCount },
        label: `examples -> ${exampleCount}`,
      };
    }
    case "removeExample": {
      const task = getTask(config.taskId);
      const exampleCount = clampExamples(task, config.exampleCount - 1);
      const corruptedExamples = Math.min(
        config.corruptedExamples,
        exampleCount,
      );
      return {
        config: { ...config, exampleCount, corruptedExamples },
        label: `examples -> ${exampleCount}`,
      };
    }
    case "toggleDescription": {
      const useDescription = !config.useDescription;
      return {
        config: { ...config, useDescription },
        label: `description ${useDescription ? "on" : "off"}`,
      };
    }
    case "corruptExample": {
      const corruptedExamples = Math.min(
        config.corruptedExamples + 1,
        config.exampleCount,
      );
      return {
        config: { ...config, corruptedExamples },
        label: `mislabeled examples -> ${corruptedExamples}`,
      };
    }
    case "fixExamples":
      return {
        config: { ...config, corruptedExamples: 0 },
        label: "examples fixed",
      };
    case "loadPreset": {
      const preset = getPreset(action.id);
      return {
        config: { ...preset.config },
        label: `preset -> ${preset.label}`,
      };
    }
    case "restore":
      return { config: { ...action.config }, label: action.label };
  }
}

// The pure reducer. A tick runs one more test query; an intervention edits the
// prompt or the model and starts a clean run, because the accuracy of the old
// prompt says nothing about the new one. Every change is recorded so the reader
// can rewind to that exact configuration.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") {
    return runTrial(state);
  }

  const { config, label } = applyIntervention(state.config, input.action);
  const log = record(state.log, state.attempted, label, config);
  return freshRun(config, log);
}

export interface RunningStats {
  attempted: number;
  correct: number;
  accuracy: number;
  predicted: number;
  tokensUsed: number;
  overBudget: boolean;
}

export function runningStats(state: SimState): RunningStats {
  const task = getTask(state.config.taskId);
  const used = tokensUsed(
    task,
    state.config.exampleCount,
    state.config.useDescription,
  );
  return {
    attempted: state.attempted,
    correct: state.correctCount,
    accuracy: state.attempted > 0 ? state.correctCount / state.attempted : 0,
    predicted: predictSuccess(state.config),
    tokensUsed: used,
    overBudget: used > CONTEXT_BUDGET,
  };
}
