// Headless, deterministic model of the Reflexion loop (Shinn et al., 2023).
// It runs the paper's Algorithm 1 over an abstract multi-step decision task: an
// Actor walks a sequence of decision points, an Evaluator emits a binary
// pass/fail, a Self-Reflection model turns a failure into a verbal lesson, and a
// bounded memory feeds those lessons into the next trial. No React, no DOM, no
// Math.random, no wall-clock, so the same seed and inputs reproduce a run and the
// view can be unit-tested and scrubbed.

// One step of the mulberry32 recurrence, threaded as pure state so the whole
// model stays a deterministic reducer. Same seed, same stream.
function nextRandom(state: number): { value: number; state: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, state: a };
}

export interface SimEvent {
  id: number;
  trial: number;
  label: string;
  snapshot: SimState;
}

// A decision point in the task. Exactly one action is correct. The others are
// wrong, and `trap` marks the wrong action the un-helped Actor is most drawn to,
// which is the paper's recurring failure (the agent "thinks it holds an item it
// does not" and keeps acting on that false belief).
export interface DecisionPoint {
  id: number;
  actionCount: number;
  correctAction: number;
  trapAction: number;
  label: string;
}

// One verbal lesson in memory. A correct reflection names the decision point
// that broke the trajectory and the action to avoid there. A misdiagnosis blames
// the wrong point or steers toward another wrong action, which is how the agent
// falls into a local minimum.
export interface Reflection {
  id: number;
  bornTrial: number;
  pointId: number;
  avoidAction: number;
  suggestAction: number;
  correct: boolean;
  text: string;
}

export type Phase =
  | "ready"
  | "acting"
  | "evaluating"
  | "reflecting"
  | "passed"
  | "exhausted";

export interface TrialOutcome {
  trial: number;
  passed: boolean;
  truthfulPass: boolean;
  wrongPoints: number;
  reflected: boolean;
}

export interface Params {
  reflectionEnabled: boolean;
  reflectionAccuracy: number;
  evaluatorFalsePositiveRate: number;
  memoryCapacity: number;
  maxTrials: number;
}

export interface SimState {
  rng: number;
  seed: number;
  params: Params;
  task: DecisionPoint[];
  trial: number;
  phase: Phase;
  // The choice the Actor has made at each decision point so far this trial.
  trajectory: number[];
  cursor: number;
  // The Evaluator's verdict on the last finished trajectory.
  reward: number | null;
  reportedPass: boolean;
  truthfulPass: boolean;
  memory: Reflection[];
  outcomes: TrialOutcome[];
  log: SimEvent[];
  nextEventId: number;
  nextReflectionId: number;
  // The reflection written at the end of the most recent failed trial, surfaced
  // so the view can show the lesson the agent just learned.
  lastReflection: Reflection | null;
}

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: Params;
  seed: number;
}

const POINT_LABELS = [
  "locate pan",
  "pick up pan",
  "carry to counter",
  "place pan",
];

const ACTION_LABELS: Record<number, string[]> = {
  0: ["check stove", "check counter", "check fridge"],
  1: ["take from stove", "take from sink", "wait"],
  2: ["walk to counter", "walk to sink", "stay put"],
  3: ["put on counter", "put on stove", "drop"],
};

const ACCURATE_TEXTS = [
  "Step {p} failed. I chose {bad} but the pan was not there. Next time {good}.",
  "I incorrectly assumed step {p} was done. I should {good} instead of {bad}.",
  "At step {p} my action {bad} did nothing. The right move is to {good}.",
];

const WRONG_TEXTS = [
  "Step {p} felt wrong, so I will retry {bad} more carefully next time.",
  "The plan failed somewhere. I will be more thorough at step {p} and {bad}.",
  "Step {p} needs another attempt with {bad}; the rest looked fine.",
];

function defaultTask(): DecisionPoint[] {
  return POINT_LABELS.map((label, id) => ({
    id,
    actionCount: 3,
    correctAction: 0,
    trapAction: 1,
    label,
  }));
}

export const PRESETS: Preset[] = [
  {
    id: "reflexion",
    label: "Reflexion",
    blurb:
      "Reflection on, sharp self-reflection, memory of three lessons, an honest evaluator. The agent names each mistake and clears the task in a few trials.",
    params: {
      reflectionEnabled: true,
      reflectionAccuracy: 0.92,
      evaluatorFalsePositiveRate: 0,
      memoryCapacity: 3,
      maxTrials: 12,
    },
    seed: 7,
  },
  {
    id: "react-only",
    label: "ReAct only",
    blurb:
      "No reflection at all. Every trial starts blank, so the agent keeps hitting the same trap and the success curve flatlines, the way ReAct stalls around trial 6 in the paper.",
    params: {
      reflectionEnabled: false,
      reflectionAccuracy: 0.92,
      evaluatorFalsePositiveRate: 0,
      memoryCapacity: 3,
      maxTrials: 12,
    },
    seed: 7,
  },
  {
    id: "local-minimum",
    label: "Local minimum",
    blurb:
      "Reflection on, but the self-reflection keeps misdiagnosing the mistake. The agent rewrites the same vague lesson and never escapes, the failure the paper warns about.",
    params: {
      reflectionEnabled: true,
      reflectionAccuracy: 0.18,
      evaluatorFalsePositiveRate: 0,
      memoryCapacity: 3,
      maxTrials: 12,
    },
    seed: 5,
  },
  {
    id: "flaky-evaluator",
    label: "Flaky evaluator",
    blurb:
      "The evaluator passes wrong trajectories sometimes, like a flaky self-written test. The agent stops early thinking it won while the answer is still wrong, and the safety light flips.",
    params: {
      reflectionEnabled: true,
      reflectionAccuracy: 0.92,
      evaluatorFalsePositiveRate: 0.6,
      memoryCapacity: 3,
      maxTrials: 12,
    },
    seed: 9,
  },
];

function presetById(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? PRESETS[0]!;
}

export function initialState(presetId: string): SimState {
  const preset = presetById(presetId);
  return {
    rng: preset.seed >>> 0,
    seed: preset.seed,
    params: { ...preset.params },
    task: defaultTask(),
    trial: 0,
    phase: "ready",
    trajectory: [],
    cursor: 0,
    reward: null,
    reportedPass: false,
    truthfulPass: false,
    memory: [],
    outcomes: [],
    log: [],
    nextEventId: 1,
    nextReflectionId: 1,
    lastReflection: null,
  };
}

function logEvent(state: SimState, label: string): SimState {
  const event: SimEvent = {
    id: state.nextEventId,
    trial: state.trial,
    label,
    snapshot: { ...state, log: [], lastReflection: state.lastReflection },
  };
  return {
    ...state,
    log: [...state.log, event].slice(-60),
    nextEventId: state.nextEventId + 1,
  };
}

// The Actor's relevant lesson for a decision point: the most recent reflection
// in memory that names this point. Memory is read newest-first, so a fresh
// correction overrides an older one.
function reflectionFor(
  memory: Reflection[],
  pointId: number,
): Reflection | null {
  for (let i = memory.length - 1; i >= 0; i--) {
    const reflection = memory[i]!;
    if (reflection.pointId === pointId) return reflection;
  }
  return null;
}

// The Actor's policy at one decision point. With a correct lesson it takes the
// suggested (right) action. With a misdiagnosis it avoids the named action but,
// because the lesson is wrong, still misses. With no lesson it leans toward the
// trap, which is the un-helped agent's habitual error.
function chooseAction(
  point: DecisionPoint,
  memory: Reflection[],
  rngState: number,
): { action: number; rngState: number } {
  const lesson = reflectionFor(memory, point.id);
  if (lesson && lesson.correct) {
    return { action: lesson.suggestAction, rngState };
  }

  const draw = nextRandom(rngState);
  if (lesson && !lesson.correct) {
    // The misdiagnosis steered the agent toward `suggestAction`, which is wrong
    // by construction, so it keeps failing this point.
    return { action: lesson.suggestAction, rngState: draw.state };
  }

  // No lesson: 65% of the time the un-helped agent walks into the trap, the rest
  // of the time it picks among the remaining actions.
  if (draw.value < 0.65) {
    return { action: point.trapAction, rngState: draw.state };
  }
  const pick = nextRandom(draw.state);
  const alt = Math.floor(pick.value * point.actionCount) % point.actionCount;
  return { action: alt, rngState: pick.state };
}

function actionName(pointId: number, action: number): string {
  const names = ACTION_LABELS[pointId] ?? [];
  return names[action] ?? `action ${action}`;
}

function fillTemplate(
  template: string,
  point: DecisionPoint,
  bad: number,
  good: number,
): string {
  return template
    .replace("{p}", `${point.id + 1} (${point.label})`)
    .replace("{bad}", actionName(point.id, bad))
    .replace("{good}", actionName(point.id, good));
}

function wrongPointsIn(task: DecisionPoint[], trajectory: number[]): number[] {
  const wrong: number[] = [];
  for (let i = 0; i < task.length; i++) {
    if (trajectory[i] !== task[i]!.correctAction) wrong.push(i);
  }
  return wrong;
}

// One Actor decision. Advances the cursor; when the trajectory is complete it
// moves the loop into evaluation.
function actStep(state: SimState): SimState {
  const point = state.task[state.cursor]!;
  const { action, rngState } = chooseAction(point, state.memory, state.rng);
  const trajectory = [...state.trajectory, action];
  const cursor = state.cursor + 1;
  const named = actionName(point.id, action);
  const correct = action === point.correctAction;
  const next: SimState = {
    ...state,
    rng: rngState,
    trajectory,
    cursor,
  };
  const logged = logEvent(
    next,
    `t${state.trial}: step ${point.id + 1} -> ${named}${correct ? "" : " (wrong)"}`,
  );
  if (cursor >= state.task.length) {
    return { ...logged, phase: "evaluating" };
  }
  return logged;
}

// The Evaluator emits its binary verdict. A truly-correct trajectory passes. A
// wrong trajectory normally fails, but with probability evaluatorFalsePositiveRate
// the Evaluator wrongly reports a pass, which the agent then locks in.
function evaluateStep(state: SimState): SimState {
  const wrong = wrongPointsIn(state.task, state.trajectory);
  const truthfulPass = wrong.length === 0;
  let reportedPass = truthfulPass;
  let rngState = state.rng;
  if (!truthfulPass) {
    const draw = nextRandom(rngState);
    rngState = draw.state;
    if (draw.value < state.params.evaluatorFalsePositiveRate) {
      reportedPass = true;
    }
  }

  const outcome: TrialOutcome = {
    trial: state.trial,
    passed: reportedPass,
    truthfulPass,
    wrongPoints: wrong.length,
    reflected: false,
  };

  const base: SimState = {
    ...state,
    rng: rngState,
    reward: reportedPass ? 1 : 0,
    reportedPass,
    truthfulPass,
    outcomes: [...state.outcomes, outcome],
  };

  if (reportedPass) {
    const label = truthfulPass
      ? `t${state.trial}: evaluator PASS (correct)`
      : `t${state.trial}: evaluator PASS (false positive, answer wrong)`;
    return { ...logEvent(base, label), phase: "passed" };
  }

  const logged = logEvent(
    base,
    `t${state.trial}: evaluator FAIL (${wrong.length} wrong step${wrong.length === 1 ? "" : "s"})`,
  );
  if (!state.params.reflectionEnabled) {
    return startNextTrial(logged);
  }
  return { ...logged, phase: "reflecting" };
}

// The Self-Reflection model writes one verbal lesson about the failed trial. With
// probability reflectionAccuracy it correctly names the earliest wrong point and
// the right action; otherwise it misdiagnoses, recommending a still-wrong action.
function reflectStep(state: SimState): SimState {
  const wrong = wrongPointsIn(state.task, state.trajectory);
  const targetIndex = wrong[0] ?? 0;
  const point = state.task[targetIndex]!;
  const chosen = state.trajectory[targetIndex] ?? point.trapAction;

  const draw = nextRandom(state.rng);
  const correct = draw.value < state.params.reflectionAccuracy;

  let suggestAction = point.correctAction;
  let rngState = draw.state;
  if (!correct) {
    // A misdiagnosis points at some action that is still not the correct one.
    const pick = nextRandom(rngState);
    rngState = pick.state;
    let alt = Math.floor(pick.value * point.actionCount) % point.actionCount;
    if (alt === point.correctAction) {
      alt = (alt + 1) % point.actionCount;
    }
    suggestAction = alt;
  }

  const templates = correct ? ACCURATE_TEXTS : WRONG_TEXTS;
  const tIdx = state.nextReflectionId % templates.length;
  const text = fillTemplate(templates[tIdx]!, point, chosen, suggestAction);

  const reflection: Reflection = {
    id: state.nextReflectionId,
    bornTrial: state.trial,
    pointId: point.id,
    avoidAction: chosen,
    suggestAction,
    correct,
    text,
  };

  const memory = [...state.memory, reflection].slice(
    -Math.max(1, state.params.memoryCapacity),
  );

  const outcomes = state.outcomes.map((outcome, i) =>
    i === state.outcomes.length - 1 ? { ...outcome, reflected: true } : outcome,
  );

  const next: SimState = {
    ...state,
    rng: rngState,
    memory,
    outcomes,
    nextReflectionId: state.nextReflectionId + 1,
    lastReflection: reflection,
  };
  const logged = logEvent(
    next,
    `t${state.trial}: reflect on step ${point.id + 1} (${correct ? "on target" : "misdiagnosis"})`,
  );
  return startNextTrial(logged);
}

function startNextTrial(state: SimState): SimState {
  const nextTrial = state.trial + 1;
  if (nextTrial >= state.params.maxTrials) {
    return logEvent(
      { ...state, phase: "exhausted" },
      `t${state.trial}: out of trials, never passed`,
    );
  }
  return {
    ...state,
    trial: nextTrial,
    phase: "acting",
    trajectory: [],
    cursor: 0,
    reward: null,
  };
}

export type Intervention =
  | { type: "loadPreset"; id: string }
  | { type: "setReflectionEnabled"; value: boolean }
  | { type: "setReflectionAccuracy"; value: number }
  | { type: "setFalsePositiveRate"; value: number }
  | { type: "setMemoryCapacity"; value: number }
  | { type: "corruptLastReflection" }
  | { type: "addTrap" }
  | { type: "restore"; snapshot: SimState; label: string };

function applyMemoryCapacity(state: SimState, capacity: number): SimState {
  const bounded = Math.max(1, Math.min(3, Math.round(capacity)));
  const memory = state.memory.slice(-bounded);
  return {
    ...state,
    params: { ...state.params, memoryCapacity: bounded },
    memory,
  };
}

function intervene(state: SimState, action: Intervention): SimState {
  switch (action.type) {
    case "loadPreset":
      return initialState(action.id);
    case "setReflectionEnabled":
      return logEvent(
        {
          ...state,
          params: { ...state.params, reflectionEnabled: action.value },
        },
        `reflection ${action.value ? "on" : "off"}`,
      );
    case "setReflectionAccuracy":
      return {
        ...state,
        params: {
          ...state.params,
          reflectionAccuracy: Math.max(0, Math.min(1, action.value)),
        },
      };
    case "setFalsePositiveRate":
      return {
        ...state,
        params: {
          ...state.params,
          evaluatorFalsePositiveRate: Math.max(0, Math.min(1, action.value)),
        },
      };
    case "setMemoryCapacity":
      return logEvent(
        applyMemoryCapacity(state, action.value),
        `memory window -> ${Math.max(1, Math.min(3, Math.round(action.value)))}`,
      );
    case "corruptLastReflection": {
      if (!state.lastReflection) return state;
      const id = state.lastReflection.id;
      const memory = state.memory.map((reflection) => {
        if (reflection.id !== id || !reflection.correct) return reflection;
        const point = state.task[reflection.pointId]!;
        let alt = (reflection.suggestAction + 1) % point.actionCount;
        if (alt === point.correctAction) alt = (alt + 1) % point.actionCount;
        return {
          ...reflection,
          correct: false,
          suggestAction: alt,
          text: fillTemplate(
            WRONG_TEXTS[0]!,
            point,
            reflection.avoidAction,
            alt,
          ),
        };
      });
      const corrupted = memory.find((r) => r.id === id) ?? state.lastReflection;
      return logEvent(
        { ...state, memory, lastReflection: corrupted },
        `corrupted lesson #${id} into a misdiagnosis`,
      );
    }
    case "addTrap": {
      const id = state.task.length;
      const point: DecisionPoint = {
        id,
        actionCount: 3,
        correctAction: 0,
        trapAction: 1,
        label: `extra step ${id + 1}`,
      };
      return logEvent(
        { ...state, task: [...state.task, point] },
        `added decision point ${id + 1}`,
      );
    }
    case "restore":
      return logEvent({ ...action.snapshot, log: state.log }, action.label);
    default:
      return state;
  }
}

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

// The pure reducer. A tick advances the loop by exactly one micro-step (one Actor
// decision, one evaluation, or one reflection) so the reader can pause between
// any two and aim an intervention at a precise moment.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "intervene") return intervene(state, input.action);

  switch (state.phase) {
    case "ready":
      return { ...state, phase: "acting", trajectory: [], cursor: 0 };
    case "acting":
      return actStep(state);
    case "evaluating":
      return evaluateStep(state);
    case "reflecting":
      return reflectStep(state);
    case "passed":
    case "exhausted":
      return state;
    default:
      return state;
  }
}

export function isTerminal(state: SimState): boolean {
  return state.phase === "passed" || state.phase === "exhausted";
}

// The emergent quantity the reader watches: how many decision points the agent
// currently gets right, out of the whole task. Climbs as good lessons land.
export function masteredPoints(state: SimState): number {
  let count = 0;
  for (const point of state.task) {
    const lesson = reflectionFor(state.memory, point.id);
    if (
      lesson &&
      lesson.correct &&
      lesson.suggestAction === point.correctAction
    ) {
      count += 1;
    }
  }
  return count;
}

// Safety property: every reported pass was a truthful pass. A flaky evaluator
// that reports a pass on a wrong trajectory breaks this.
export function safetyHolds(state: SimState): boolean {
  return state.outcomes.every(
    (outcome) => !outcome.passed || outcome.truthfulPass,
  );
}
