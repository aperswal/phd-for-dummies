// A pure, deterministic model of the OSWorld agent loop from "OSWorld:
// Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer
// Environments". OSWorld frames a computer task as a POMDP: the agent sees an
// observation (a screenshot and/or an accessibility tree), emits a pyautogui
// action (mostly a click at an (x, y) pixel), the real VM executes it, and at
// the end an execution-based script grades the final machine state and returns
// reward 1 (done) or 0 (failed). This model runs that loop one step at a time.
//
// The paper's headline finding is that agents plan well but ground and execute
// badly: over 75% of failures are mouse-click inaccuracies. So here the plan is
// always correct, and each step the agent attempts the next planned click. Two
// things decide whether the click lands: a per-observation grounding rate taken
// from the paper's Table 5 (a11y tree beats raw screenshot, Set-of-Mark helps
// GPT-4V), and the window perturbations from Figure 8 (moving, shrinking, or
// cluttering the window each drag the landing chance down). A miss either burns
// a step on a repeated click or spawns a popup that adds an extra clearing step
// (the "environmental noise dilemma"). The task passes only if every subgoal
// completes before the 15-step budget runs out, which is why long multi-app
// workflows collapse: more required clicks, more chances to miss.
//
// Nothing here calls Math.random. A seed threads through the state so a given
// seed plus the same inputs reproduce the exact run, which is what makes the
// event log and rewind honest.

// One step of the mulberry32 recurrence as a pure function. Threading the
// generator state through immutable sim state keeps the whole reducer
// deterministic without a stateful closure.
function nextRandom(state: number): { value: number; state: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, state: a };
}

function indexed<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) {
    throw new RangeError(`index ${index} out of range`);
  }
  return value;
}

// A logged moment carries the full snapshot so the event log can rewind the run
// to exactly that state. The field is named `tick` to match the shared
// EventLogPanel's event shape; here a tick is one agent action.
export interface SimEvent {
  id: number;
  tick: number;
  label: string;
  snapshot: SimState;
}

export interface EventLog {
  events: SimEvent[];
  nextEventId: number;
}

const LOG_LIMIT = 80;

function emptyLog(): EventLog {
  return { events: [], nextEventId: 1 };
}

// The observation modes the paper evaluates, with the grounding rate each one
// buys. The numbers are the per-step landing chance, set so the simulated
// task-level pass rate lands near the paper's Table 5 overall figures for
// GPT-4 / GPT-4V (a11y tree 12.24%, screenshot 5.26%, Set-of-Mark for GPT-4V
// 11.77%). A trained model's real accuracy comes from the model; here it is a
// fixed rate so the loop is legible.
export interface ObservationMode {
  id: string;
  label: string;
  blurb: string;
  groundRate: number;
}

export const OBSERVATION_MODES: ObservationMode[] = [
  {
    id: "a11y",
    label: "A11y tree",
    blurb:
      "A filtered accessibility tree gives the agent element labels and rough positions. The strongest text-only grounding in the paper, but the tree can run to thousands of tokens.",
    groundRate: 0.74,
  },
  {
    id: "screenshot",
    label: "Screenshot",
    blurb:
      "Raw pixels only, closest to what a human sees. The agent must predict exact coordinates from the image, which is its weakest skill, so clicks miss far more often.",
    groundRate: 0.62,
  },
  {
    id: "som",
    label: "Set-of-Mark",
    blurb:
      "Numbered boxes drawn over the screenshot from the a11y tree, so the agent picks a box index instead of a raw pixel. It helps GPT-4V ground, but the marks add noise on dense screens.",
    groundRate: 0.72,
  },
];

function observationMode(id: string): ObservationMode {
  return (
    OBSERVATION_MODES.find((mode) => mode.id === id) ??
    indexed(OBSERVATION_MODES, 0)
  );
}

// Window perturbations from Figure 8. The agent does relatively well on the
// original layout (50.79% on the probed subset) and each disturbance drops the
// landing chance: moving the window to 36.65%, shrinking it to 15.04%,
// cluttering the screen to 25.39%. We turn those subset rates into a multiplier
// on the per-step grounding rate, so the same shape shows up in the loop.
export type Perturbation = "none" | "move" | "shrink" | "clutter";

const PERTURBATION_FACTOR: Record<Perturbation, number> = {
  none: 1,
  move: 36.65 / 50.79,
  shrink: 15.04 / 50.79,
  clutter: 25.39 / 50.79,
};

export const PERTURBATION_FACTOR_KEYS = Object.keys(
  PERTURBATION_FACTOR,
) as Perturbation[];

export const PERTURBATION_LABEL: Record<Perturbation, string> = {
  none: "Original layout",
  move: "Window moved",
  shrink: "Window shrunk",
  clutter: "Screen cluttered",
};

// A task is a sequence of subgoals the agent must complete in order. Single-app
// tasks are short; multi-app workflows are long, which is the whole reason their
// pass rate collapses. Each subgoal is one grounded click the agent has to land.
export interface TaskSpec {
  id: string;
  label: string;
  app: string;
  subgoals: string[];
  multiApp: boolean;
  note: string;
}

export const TASKS: TaskSpec[] = [
  {
    id: "cookies",
    label: "Clear Amazon cookies",
    app: "Chrome",
    multiApp: false,
    note: "A short single-app task. Open settings, find the site data, delete it. Few clicks, so a decent grounding rate often gets through.",
    subgoals: [
      "open Chrome menu",
      "open Settings",
      "find site cookies",
      "delete amazon.com",
    ],
  },
  {
    id: "rename",
    label: "Rename a spreadsheet sheet",
    app: "LibreOffice Calc",
    multiApp: false,
    note: "A single-app Office task. GUI-heavy clicking on small cells and tabs, which the paper flags as where agents do worst.",
    subgoals: [
      "right-click sheet tab",
      "choose Rename",
      "type new name",
      "save the file",
    ],
  },
  {
    id: "subtitles",
    label: "Strip subtitles from a video",
    app: "VLC + Terminal",
    multiApp: true,
    note: "A multi-app workflow. Split the screen, run ffmpeg in the terminal, save the file. Many grounded steps across two apps, so misses pile up.",
    subgoals: [
      "open Activities",
      "launch Terminal",
      "focus the video file",
      "run ffmpeg -map",
      "run ffmpeg copy",
      "verify output",
    ],
  },
  {
    id: "bookkeeping",
    label: "Update a bookkeeping sheet",
    app: "Chrome + Calc",
    multiApp: true,
    note: "A long multi-app workflow. Download a receipt, read it, transcribe rows into the sheet, recompute totals. The kind of task agents score below 5% on.",
    subgoals: [
      "download the receipt",
      "open the receipt",
      "open the sheet",
      "add the rows",
      "recompute totals",
      "save the workbook",
    ],
  },
];

export function getTask(id: string): TaskSpec {
  return TASKS.find((task) => task.id === id) ?? indexed(TASKS, 0);
}

// What the agent is doing right now, so the view can render the loop honestly.
export type Phase = "ready" | "running" | "done" | "failed";

// One entry on the trajectory tape: the agent attempted a planned click and it
// either landed, missed (a repeated click), or missed into a popup it then has
// to clear.
export type Outcome = "land" | "miss" | "popup";

export interface Attempt {
  step: number;
  subgoalIndex: number;
  subgoal: string;
  outcome: Outcome;
  groundChance: number;
}

export interface SimParams {
  taskId: string;
  observationId: string;
  perturbation: Perturbation;
  stepBudget: number;
  seed: number;
}

export interface SimState {
  params: SimParams;
  rngState: number;
  phase: Phase;
  step: number;
  subgoalIndex: number;
  pendingPopups: number;
  attempts: Attempt[];
  reward: number;
  log: EventLog;
}

export const DEFAULT_BUDGET = 15;

// The effective per-step landing chance for the current config: the
// observation's base grounding rate times the perturbation factor. This is the
// single number the whole outcome turns on.
export function groundChance(params: SimParams): number {
  const base = observationMode(params.observationId).groundRate;
  return base * PERTURBATION_FACTOR[params.perturbation];
}

export type Intervention =
  | { type: "setTask"; id: string }
  | { type: "setObservation"; id: string }
  | { type: "setPerturbation"; value: Perturbation }
  | { type: "setBudget"; value: number }
  | { type: "reshuffle" }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: SimState; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: SimParams;
}

export const PRESETS: Preset[] = [
  {
    id: "single-app",
    label: "Single-app, a11y tree",
    blurb:
      "The agent's best shot. A short Chrome task with the accessibility tree. Run it a few times; some seeds finish, many still miss a click and stall.",
    params: {
      taskId: "cookies",
      observationId: "a11y",
      perturbation: "none",
      stepBudget: DEFAULT_BUDGET,
      seed: 3,
    },
  },
  {
    id: "workflow",
    label: "Multi-app workflow",
    blurb:
      "The hard case. A long two-app workflow. Every extra subgoal is another chance to miss, so the run almost always burns the step budget before finishing.",
    params: {
      taskId: "bookkeeping",
      observationId: "a11y",
      perturbation: "none",
      stepBudget: DEFAULT_BUDGET,
      seed: 1,
    },
  },
  {
    id: "perturbed",
    label: "Shrink the window",
    blurb:
      "Start the rename task on the original layout, then shrink the window mid-run. The grounding chance craters and the same plan starts missing.",
    params: {
      taskId: "rename",
      observationId: "som",
      perturbation: "none",
      stepBudget: DEFAULT_BUDGET,
      seed: 7,
    },
  },
  {
    id: "screenshot",
    label: "Screenshot only",
    blurb:
      "Drop the a11y tree and give the agent raw pixels. It has to guess coordinates, so even the short task misses more often.",
    params: {
      taskId: "rename",
      observationId: "screenshot",
      perturbation: "none",
      stepBudget: DEFAULT_BUDGET,
      seed: 5,
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? indexed(PRESETS, 0);
}

export function initialState(presetId = "single-app"): SimState {
  const params = { ...getPreset(presetId).params };
  return {
    params,
    rngState: params.seed >>> 0,
    phase: "ready",
    step: 0,
    subgoalIndex: 0,
    pendingPopups: 0,
    attempts: [],
    reward: 0,
    log: emptyLog(),
  };
}

function record(state: SimState, label: string): SimState {
  const event: SimEvent = {
    id: state.log.nextEventId,
    tick: state.step,
    label,
    snapshot: { ...state, log: { events: [], nextEventId: 0 } },
  };
  return {
    ...state,
    log: {
      events: [...state.log.events, event].slice(-LOG_LIMIT),
      nextEventId: state.log.nextEventId + 1,
    },
  };
}

// Advance the agent loop by one action. This is the paper's interaction loop:
// observe, attempt the next planned click, let the VM resolve it, and stop when
// the task is done, fails, or the step budget runs out. A miss costs a step; a
// popup costs the missed step plus a future clearing step.
function tick(state: SimState): SimState {
  if (state.phase === "done" || state.phase === "failed") return state;

  if (state.step >= state.params.stepBudget) {
    return record(
      { ...state, phase: "failed" },
      "ran out of steps -> reward 0",
    );
  }

  const task = getTask(state.params.taskId);
  const chance = groundChance(state.params);
  const step = state.step + 1;

  // A pending popup steals this step: the agent has to close the window it
  // accidentally opened before it can get back to the plan.
  if (state.pendingPopups > 0) {
    const attempt: Attempt = {
      step,
      subgoalIndex: state.subgoalIndex,
      subgoal: "close popup",
      outcome: "miss",
      groundChance: chance,
    };
    const cleared: SimState = {
      ...state,
      step,
      pendingPopups: state.pendingPopups - 1,
      attempts: [...state.attempts, attempt],
    };
    return record(cleared, `step ${step}: cleared a popup, no progress`);
  }

  const draw = nextRandom(state.rngState);
  const landed = draw.value < chance;
  const subgoal = indexed(task.subgoals, state.subgoalIndex);

  if (landed) {
    const subgoalIndex = state.subgoalIndex + 1;
    const finished = subgoalIndex >= task.subgoals.length;
    const attempt: Attempt = {
      step,
      subgoalIndex: state.subgoalIndex,
      subgoal,
      outcome: "land",
      groundChance: chance,
    };
    const advanced: SimState = {
      ...state,
      rngState: draw.state,
      step,
      subgoalIndex,
      attempts: [...state.attempts, attempt],
      phase: finished ? "done" : "running",
      reward: finished ? 1 : 0,
    };
    return record(
      advanced,
      finished
        ? `step ${step}: landed '${subgoal}', task done -> reward 1`
        : `step ${step}: landed '${subgoal}'`,
    );
  }

  // A miss. A second draw decides whether the stray click opened a popup,
  // modeling the environmental-noise dilemma the paper describes.
  const popupDraw = nextRandom(draw.state);
  const spawnedPopup = popupDraw.value < 0.35;
  const attempt: Attempt = {
    step,
    subgoalIndex: state.subgoalIndex,
    subgoal,
    outcome: spawnedPopup ? "popup" : "miss",
    groundChance: chance,
  };
  const missed: SimState = {
    ...state,
    rngState: popupDraw.state,
    step,
    pendingPopups: state.pendingPopups + (spawnedPopup ? 1 : 0),
    attempts: [...state.attempts, attempt],
    phase: "running",
  };
  return record(
    missed,
    spawnedPopup
      ? `step ${step}: missed '${subgoal}' and opened a popup`
      : `step ${step}: missed '${subgoal}'`,
  );
}

// Resetting the run keeps the current parameters but clears all progress, so an
// intervention that changes the config also restarts the trajectory cleanly.
function resetWith(params: SimParams, log: EventLog): SimState {
  return {
    params,
    rngState: params.seed >>> 0,
    phase: "ready",
    step: 0,
    subgoalIndex: 0,
    pendingPopups: 0,
    attempts: [],
    reward: 0,
    log,
  };
}

function applyIntervention(
  state: SimState,
  action: Intervention,
): { state: SimState; label: string } {
  switch (action.type) {
    case "setTask": {
      const task = getTask(action.id);
      return {
        state: resetWith({ ...state.params, taskId: action.id }, state.log),
        label: `task -> ${task.label}`,
      };
    }
    case "setObservation": {
      const mode = observationMode(action.id);
      // Changing how the agent sees mid-run is allowed; it shifts the grounding
      // chance for every click from here on, but keeps the progress so far.
      return {
        state: {
          ...state,
          params: { ...state.params, observationId: action.id },
        },
        label: `observation -> ${mode.label}`,
      };
    }
    case "setPerturbation": {
      // The key mid-run intervention from Figure 8: nudge the window while the
      // agent is part-way through and watch the landing chance drop.
      return {
        state: {
          ...state,
          params: { ...state.params, perturbation: action.value },
        },
        label: `window -> ${PERTURBATION_LABEL[action.value]}`,
      };
    }
    case "setBudget":
      return {
        state: {
          ...state,
          params: { ...state.params, stepBudget: action.value },
        },
        label: `step budget -> ${action.value}`,
      };
    case "reshuffle": {
      // Cycle the seed so the reader sees a different trajectory without
      // exposing a numeric dial whose individual values mean nothing.
      const nextSeed = (state.params.seed + 1) % 100;
      return {
        state: resetWith({ ...state.params, seed: nextSeed }, state.log),
        label: `new sample (seed ${nextSeed})`,
      };
    }
    case "loadPreset": {
      const preset = getPreset(action.id);
      return {
        state: resetWith({ ...preset.params }, state.log),
        label: `preset -> ${preset.label}`,
      };
    }
    case "restore":
      return {
        state: { ...action.snapshot, log: state.log },
        label: action.label,
      };
  }
}

// The pure reducer. A tick advances the agent loop one action; an intervention
// changes one parameter and records what happened. Every step is deterministic
// given the seed.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return tick(state);
  const { state: next, label } = applyIntervention(state, input.action);
  return record(next, label);
}
