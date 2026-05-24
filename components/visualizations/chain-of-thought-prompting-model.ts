// A pure, deterministic model of chain-of-thought prompting from "Chain-of-Thought
// Prompting Elicits Reasoning in Large Language Models". The model emits a
// solution one reasoning step at a time, the way an autoregressive language model
// does. Each problem carries its true reasoning chain: a list of steps, where each
// step computes one intermediate quantity from earlier ones, and a final answer
// that depends on the last step. The mechanism the simulation makes pokeable is
// the paper's central claim and its ablations: whether the model writes out the
// reasoning steps at all (the prompt mode), and whether the model is big enough to
// keep each step sound (the scale knob). A step that comes out wrong poisons every
// step downstream, which is exactly the "one reasoning step missing" and "54% had
// major errors" failure the paper reports for small models. Randomness only enters
// through the seeded rng, so a given seed and inputs reproduce the run exactly.

import { at } from "@/lib/simulation/array";
import {
  emptyLog,
  record,
  type EventLog,
  type SimEvent,
} from "@/lib/simulation/history";
import { nextRandom } from "@/lib/simulation/rng";

// One reasoning step. The model computes this step's value from `operands` under
// `op`. `threadInto` names the operand position (0 or 1) that consumes the running
// result of the step before it, which is how the paper's chains carry a number
// forward ("23 - 20 = 3", then "3 + 6 = 9"). A corrupted earlier value flows
// through this slot into later steps. `text` is the natural-language line the model
// emits, with `{v}` standing in for the step's own computed value.
export interface ReasoningStep {
  text: string;
  operands: number[];
  op: "+" | "-" | "*" | "/";
  threadInto?: 0 | 1;
}

export interface Problem {
  id: string;
  label: string;
  question: string;
  steps: ReasoningStep[];
  unit: string;
  note: string;
}

function applyOp(op: ReasoningStep["op"], operands: number[]): number {
  const [a = 0, b = 0] = operands;
  switch (op) {
    case "+":
      return a + b;
    case "-":
      return a - b;
    case "*":
      return a * b;
    case "/":
      return b === 0 ? 0 : a / b;
  }
}

// The famous Figure 1 problem, plus two more multi-step GSM8K-style problems. The
// last step of each is the final answer the standard-prompt model has to produce
// in one shot.
export const PROBLEMS: Problem[] = [
  {
    id: "apples",
    label: "Cafeteria apples",
    question:
      "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?",
    unit: "apples",
    note: "Figure 1's problem. Standard prompting answers 27, the chain answers 9.",
    steps: [
      {
        text: "Start with 23, use 20 for lunch, so 23 - 20 = {v}",
        operands: [23, 20],
        op: "-",
      },
      {
        text: "Then buy 6 more, so 3 + 6 = {v}",
        operands: [3, 6],
        op: "+",
        threadInto: 0,
      },
    ],
  },
  {
    id: "tennis",
    label: "Tennis balls",
    question:
      "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
    unit: "balls",
    note: "The exemplar from Figure 1. Two cans of three is six, plus the five he had.",
    steps: [
      {
        text: "Each can holds 3, and he buys 2 cans, so 2 * 3 = {v}",
        operands: [2, 3],
        op: "*",
      },
      {
        text: "He started with 5, so 5 + 6 = {v}",
        operands: [5, 6],
        op: "+",
        threadInto: 1,
      },
    ],
  },
  {
    id: "servers",
    label: "Server room",
    question:
      "There were 9 computers in the server room. Five more computers were installed each day from Monday to Thursday. How many computers are now in the server room?",
    unit: "computers",
    note: "A three-step chain. The middle step is where small models tend to slip.",
    steps: [
      {
        text: "Monday to Thursday is 4 days, so the days count is 4 * 1 = {v}",
        operands: [4, 1],
        op: "*",
      },
      {
        text: "Five computers each of those 4 days, so 5 * 4 = {v}",
        operands: [5, 4],
        op: "*",
        threadInto: 1,
      },
      {
        text: "Add the 9 that were already there, so 9 + 20 = {v}",
        operands: [9, 20],
        op: "+",
        threadInto: 1,
      },
    ],
  },
];

export function getProblem(id: string): Problem {
  return PROBLEMS.find((problem) => problem.id === id) ?? at(PROBLEMS, 0);
}

// The true value of every step, computed cleanly with no faults. Used as ground
// truth to score the model's emitted chain.
// Replace the threaded operand with the running result of the previous step. A
// step with no `threadInto` uses its constants as written.
function threadOperands(
  step: ReasoningStep,
  prior: number | undefined,
): number[] {
  if (step.threadInto === undefined || prior === undefined)
    return step.operands;
  return step.operands.map((operand, index) =>
    index === step.threadInto ? prior : operand,
  );
}

export function trueValues(problem: Problem): number[] {
  const values: number[] = [];
  for (const step of problem.steps) {
    const prior = values.length > 0 ? at(values, values.length - 1) : undefined;
    values.push(applyOp(step.op, threadOperands(step, prior)));
  }
  return values;
}

export function trueAnswer(problem: Problem): number {
  const values = trueValues(problem);
  return values.length > 0 ? at(values, values.length - 1) : 0;
}

// The five prompt conditions. CoT is the paper's method; the other four are its
// ablations (Section 3.3) plus the standard baseline. Only modes that emit the
// reasoning steps give the model a place to put intermediate computation.
export type PromptMode = "standard" | "cot" | "equation" | "dots" | "after";

export interface ModeSpec {
  id: PromptMode;
  label: string;
  blurb: string;
  // Whether this mode writes natural-language reasoning steps before the answer.
  emitsSteps: boolean;
  // Whether this mode benefits from extra serial compute that helps the model
  // keep a step sound (only natural-language reasoning does, per the ablations).
  reasoningBefore: boolean;
}

export const MODES: ModeSpec[] = [
  {
    id: "standard",
    label: "Standard",
    blurb:
      "The baseline. The model jumps straight to one answer with no steps, so a multi-step problem usually comes out wrong.",
    emitsSteps: false,
    reasoningBefore: false,
  },
  {
    id: "cot",
    label: "Chain of thought",
    blurb:
      "The method. Few-shot exemplars show worked reasoning, so the model writes its own steps before answering.",
    emitsSteps: true,
    reasoningBefore: true,
  },
  {
    id: "equation",
    label: "Equation only",
    blurb:
      "Ablation. The model emits just the equation, no words. It barely helps on GSM8K, since the hard part is turning the words into the equation.",
    emitsSteps: true,
    reasoningBefore: false,
  },
  {
    id: "dots",
    label: "Variable compute (dots)",
    blurb:
      "Ablation. The model emits a row of dots equal to the equation length, spending compute but no reasoning. It scores like the baseline.",
    emitsSteps: false,
    reasoningBefore: false,
  },
  {
    id: "after",
    label: "Reasoning after answer",
    blurb:
      "Ablation. The model gives the answer first, then explains. The explanation can't change the answer, so it scores like the baseline.",
    emitsSteps: true,
    reasoningBefore: false,
  },
];

export function getMode(id: PromptMode): ModeSpec {
  return MODES.find((mode) => mode.id === id) ?? at(MODES, 0);
}

// The scale threshold where chain-of-thought reasoning emerges. Below it the model
// writes fluent but illogical chains; above it the steps come out sound. The paper
// puts this near 100B parameters.
export const EMERGENCE_SCALE_B = 100;

// Probability that a single reasoning step comes out wrong, as a function of model
// scale. A small model fails most steps it has to reason through; a 540B model
// almost never does. This is a smooth stand-in for the sharp emergence curve in
// Figure 4, so the reader can drag scale and watch chains flip from broken to
// sound at the threshold.
export function stepErrorRate(scaleB: number): number {
  const t =
    (Math.log10(Math.max(scaleB, 0.1)) - Math.log10(EMERGENCE_SCALE_B)) * 2.4;
  return 1 / (1 + Math.exp(t * 3));
}

// A token is one emitted unit of the solution: a word in a reasoning step, a dot,
// or the final answer. The view reveals tokens one at a time so the chain reads
// like a model generating it.
export interface EmittedStep {
  index: number;
  text: string;
  value: number;
  trueValue: number;
  faulted: boolean;
  reason: "ok" | "scale" | "injected";
}

export interface Params {
  problemId: string;
  mode: PromptMode;
  scaleB: number;
  seed: number;
  // Steps the reader has manually corrupted, by index. A step here always comes
  // out wrong regardless of scale, which lets the reader watch one bad step poison
  // the answer.
  injected: number[];
}

export interface Solution {
  steps: EmittedStep[];
  finalAnswer: number;
  trueAnswer: number;
  correct: boolean;
  // How many emitted tokens the chain has, so the view can reveal them over time.
  tokenCount: number;
  // The dots line for the variable-compute ablation.
  dots: number;
}

// Deterministic per-step fault decision. We draw one rng value per reasoning step
// from a stream seeded by the seed and the step index, so dragging scale changes
// which steps survive without reshuffling the whole run.
function faultForStep(stepIndex: number, rate: number, seed: number): boolean {
  const draw = nextRandom((seed * 2654435761 + stepIndex * 40503) | 0);
  return draw.value < rate;
}

// Emit the full solution under the current params. Modes that emit reasoning steps
// run each step in order, threading the running result forward. A faulted step
// produces a wrong value (off by a deterministic amount), and every later step
// that depends on it inherits the damage. Modes without reasoning steps collapse
// to a single guessed answer.
export function solve(params: Params): Solution {
  const problem = getProblem(params.problemId);
  const mode = getMode(params.mode);
  const truths = trueValues(problem);
  const answerTruth = trueAnswer(problem);

  if (!mode.emitsSteps) {
    // No place to reason. The model writes one answer; on a multi-step problem it
    // tends to grab the wrong surface number (Figure 1's "27" instead of "9").
    const guess = standardGuess(problem, params.seed);
    const dots = mode.id === "dots" ? equationLength(problem) : 0;
    return {
      steps: [],
      finalAnswer: guess,
      trueAnswer: answerTruth,
      correct: guess === answerTruth,
      tokenCount: 1 + dots,
      dots,
    };
  }

  // Equation-only and reasoning-after gain no serial reasoning benefit, so their
  // steps fault at the baseline (small-model) rate no matter the scale. Only true
  // chain-of-thought lets a bigger model keep steps sound.
  const rate = mode.reasoningBefore
    ? stepErrorRate(params.scaleB)
    : stepErrorRate(EMERGENCE_SCALE_B / 12);

  const emitted: EmittedStep[] = [];
  const runningValues: number[] = [];

  for (let i = 0; i < problem.steps.length; i++) {
    const step = at(problem.steps, i);
    const prior =
      runningValues.length > 0
        ? at(runningValues, runningValues.length - 1)
        : undefined;
    const clean = applyOp(step.op, threadOperands(step, prior));

    const injected = params.injected.includes(i);
    const scaleFault = faultForStep(i, rate, params.seed);
    const faulted = injected || scaleFault;
    const value = faulted ? corruptValue(clean, i, params.seed) : clean;

    runningValues.push(value);
    emitted.push({
      index: i,
      text: step.text.replace("{v}", String(value)),
      value,
      trueValue: at(truths, i),
      faulted,
      reason: injected ? "injected" : scaleFault ? "scale" : "ok",
    });
  }

  const finalAnswer =
    runningValues.length > 0 ? at(runningValues, runningValues.length - 1) : 0;

  // Reasoning-after-answer commits to a guess first; the steps it writes afterward
  // can't move it. So even a sound chain doesn't help.
  const reportedAnswer =
    mode.id === "after" ? standardGuess(problem, params.seed) : finalAnswer;

  const tokenCount = emitted.reduce(
    (sum, step) => sum + step.text.split(" ").length,
    1,
  );

  return {
    steps: emitted,
    finalAnswer: reportedAnswer,
    trueAnswer: answerTruth,
    correct: reportedAnswer === answerTruth,
    tokenCount,
    dots: 0,
  };
}

// The wrong answer a no-reasoning model lands on: it grabs a salient number from
// the question instead of computing. Deterministic so the run reproduces.
function standardGuess(problem: Problem, seed: number): number {
  const numbers = extractNumbers(problem.question);
  if (numbers.length === 0) return 0;
  // A common small-model error is to combine the first two numbers with the last
  // operation, e.g. 23 + 6 - 2 = 27 for the apples problem.
  const draw = nextRandom((seed * 22695477 + problem.id.length) | 0);
  const a = at(numbers, 0);
  const b = numbers.length > 1 ? at(numbers, numbers.length - 1) : 0;
  return draw.value < 0.5 ? a + b : a - b + numbers.length;
}

function corruptValue(clean: number, stepIndex: number, seed: number): number {
  const draw = nextRandom((seed * 374761393 + stepIndex * 668265263) | 0);
  const offsets = [-3, -2, -1, 2, 3, 5, 10];
  const offset = at(offsets, Math.floor(draw.value * offsets.length));
  return clean + offset;
}

function extractNumbers(text: string): number[] {
  const matches = text.match(/\d+/g);
  return matches ? matches.map((value) => Number(value)) : [];
}

function equationLength(problem: Problem): number {
  return problem.steps.reduce((sum, step) => sum + step.operands.length + 2, 0);
}

export type CotEvent = SimEvent<Params>;

export interface SimState {
  params: Params;
  // How many reasoning steps the model has revealed so far. A tick reveals one
  // more, the way an autoregressive model produces them in sequence.
  revealed: number;
  tick: number;
  log: EventLog<Params>;
}

export type Intervention =
  | { type: "setMode"; mode: PromptMode }
  | { type: "setProblem"; id: string }
  | { type: "setScale"; value: number }
  | { type: "setSeed"; value: number }
  | { type: "toggleInjected"; index: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; params: Params; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: Params;
}

export const PRESETS: Preset[] = [
  {
    id: "standard-fails",
    label: "Standard prompting fails",
    blurb:
      "Standard prompting on the apples problem. The model writes one answer and gets it wrong, with no steps to lean on.",
    params: {
      problemId: "apples",
      mode: "standard",
      scaleB: 540,
      seed: 1,
      injected: [],
    },
  },
  {
    id: "cot-solves",
    label: "Chain of thought solves it",
    blurb:
      "Switch to chain of thought at 540B. Step the model and watch it write 23 - 20 = 3, then 3 + 6 = 9, the right answer.",
    params: {
      problemId: "apples",
      mode: "cot",
      scaleB: 540,
      seed: 1,
      injected: [],
    },
  },
  {
    id: "too-small",
    label: "Too small to reason",
    blurb:
      "Chain of thought at 8B on the three-step problem. The model writes fluent steps, but a middle step comes out wrong and poisons the answer.",
    params: {
      problemId: "servers",
      mode: "cot",
      scaleB: 8,
      seed: 3,
      injected: [],
    },
  },
  {
    id: "ablations",
    label: "Ablations don't help",
    blurb:
      "Variable-compute dots on the apples problem. Spending tokens without reasoning scores like the baseline.",
    params: {
      problemId: "apples",
      mode: "dots",
      scaleB: 540,
      seed: 1,
      injected: [],
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(presetId = "standard-fails"): SimState {
  return {
    params: {
      ...getPreset(presetId).params,
      injected: [...getPreset(presetId).params.injected],
    },
    revealed: 0,
    tick: 0,
    log: emptyLog(),
  };
}

function logged(
  state: SimState,
  label: string,
  params: Params,
  revealed: number,
): SimState {
  return {
    ...state,
    params,
    revealed,
    log: record(state.log, state.tick, label, params),
  };
}

function applyIntervention(
  params: Params,
  action: Intervention,
): { params: Params; label: string } {
  switch (action.type) {
    case "setMode":
      return {
        params: { ...params, mode: action.mode },
        label: `mode -> ${getMode(action.mode).label}`,
      };
    case "setProblem": {
      const problem = getProblem(action.id);
      return {
        params: { ...params, problemId: action.id, injected: [] },
        label: `problem -> ${problem.label}`,
      };
    }
    case "setScale":
      return {
        params: { ...params, scaleB: action.value },
        label: `scale -> ${action.value}B`,
      };
    case "setSeed":
      return {
        params: { ...params, seed: action.value },
        label: `seed -> ${action.value}`,
      };
    case "toggleInjected": {
      const has = params.injected.includes(action.index);
      const injected = has
        ? params.injected.filter((index) => index !== action.index)
        : [...params.injected, action.index];
      return {
        params: { ...params, injected },
        label: has
          ? `un-break step ${action.index + 1}`
          : `break step ${action.index + 1}`,
      };
    }
    case "loadPreset": {
      const preset = getPreset(action.id);
      return {
        params: { ...preset.params, injected: [...preset.params.injected] },
        label: `preset -> ${preset.label}`,
      };
    }
    case "restore":
      return {
        params: { ...action.params, injected: [...action.params.injected] },
        label: action.label,
      };
  }
}

// The pure reducer. A tick reveals one more reasoning step (or finishes a
// no-reasoning mode in one step). An intervention changes a parameter, resets the
// reveal, and records the moment. Every step is deterministic given the params.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") {
    const solution = solve(state.params);
    const total = solution.steps.length;
    const next = Math.min(state.revealed + 1, Math.max(total, 1));
    return { ...state, tick: state.tick + 1, revealed: next };
  }

  const { params, label } = applyIntervention(state.params, input.action);
  return logged(state, label, params, 0);
}
