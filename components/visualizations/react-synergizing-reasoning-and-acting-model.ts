// A pure, deterministic model of the ReAct agent loop from "ReAct: Synergizing
// Reasoning and Acting in Language Models". The agent solves a HotpotQA question
// by emitting a stream of steps. Each step is either a THOUGHT (free-form
// reasoning that updates the plan and gets no environment feedback), an ACTION
// (search / lookup / finish against a tiny Wikipedia stub that returns an
// OBSERVATION), or the OBSERVATION the environment hands back. ReAct interleaves
// thoughts and actions; the Act baseline drops every thought; the CoT baseline
// drops every action and must answer from memory alone.
//
// The trajectories are the paper's own examples (the Apple Remote and Colorado
// orogeny HotpotQA questions, plus a Fever-style claim). The branching is real,
// not scripted: a search either lands on an informative page or it does not, the
// agent's belief is either grounded in an observation or carried only in memory,
// and the interventions push the run down a different but rule-consistent path.
// A non-informative search derails the reasoning (the paper's 23% search-error
// failure mode), a corrupted memory makes the agent hallucinate the way CoT does
// (the paper's 56% CoT hallucination mode), and editing a thought steers the run
// the way a human-in-the-loop edit does in the paper's Figure 5.
//
// The model is headless and pure. No React, no DOM, no Math.random, no
// wall-clock. Randomness is threaded through a seeded mulberry32 generator so the
// same seed and the same inputs reproduce the same run exactly.

// --- seeded rng (inlined so the model is self-contained) ---

export type Rng = () => number;

export function mulberry32(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    const a = (state + 0x6d2b79f5) | 0;
    state = a;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- bounded event log (inlined) ---

export interface SimEvent {
  id: number;
  tick: number;
  label: string;
}

export interface EventLog {
  events: SimEvent[];
  nextEventId: number;
}

export function emptyLog(): EventLog {
  return { events: [], nextEventId: 1 };
}

const LOG_LIMIT = 80;

function record(log: EventLog, tick: number, label: string): EventLog {
  const event: SimEvent = { id: log.nextEventId, tick, label };
  return {
    events: [...log.events, event].slice(-LOG_LIMIT),
    nextEventId: log.nextEventId + 1,
  };
}

// --- safe indexed access (inlined) ---

function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) {
    throw new RangeError(`index ${index} out of range`);
  }
  return value;
}

// --- the agent's trajectory ---

export type Mode = "react" | "act" | "cot";

export type StepKind = "thought" | "action" | "observation";

export type ActionType = "search" | "lookup" | "finish";

// One line in the running transcript. A thought updates the plan and is grounded
// only if the belief it states came from an earlier observation. An action hits
// the environment. An observation is what came back, and `informative` records
// whether it actually carried the fact the agent needed.
export interface TrajectoryStep {
  index: number;
  kind: StepKind;
  text: string;
  actionType?: ActionType;
  grounded?: boolean;
  informative?: boolean;
  hallucinated?: boolean;
}

// A scripted-but-branching plan for one question. Each "beat" is a logical hop in
// the agent's reasoning. The model walks beats one at a time, and an injected
// fault changes which branch of a beat plays out, so the outcome is not fixed.
interface Beat {
  // The thought the model would emit before acting on this hop. Empty for hops
  // that are pure action with no planning (rare).
  thought: string;
  // The action it takes. `finish` ends the run.
  action: { type: ActionType; arg: string };
  // The observation when the search lands on an informative page.
  hitObservation: string;
  // The fact this hop establishes when grounded by a hit observation.
  fact: string;
  // The observation when the search misses (no page, wrong entity). Drives the
  // derail branch.
  missObservation: string;
  // What the agent would write in memory for this hop with no grounding, i.e.
  // the CoT / hallucination branch. This is the plausible-but-unverified guess.
  memoryGuess: string;
  // Whether the unverified guess happens to be wrong (hallucination that bites).
  guessWrong: boolean;
}

export interface Task {
  id: string;
  label: string;
  question: string;
  // The grounded answer the ReAct path reaches.
  answer: string;
  // The wrong answer a hallucinating (CoT / corrupted-memory) run lands on.
  hallucinatedAnswer: string;
  beats: Beat[];
  note: string;
}

// The Apple Remote question is the paper's headline Figure 1 example. CoT
// hallucinates ("Apple TV ... iPhone, iPad, iPod Touch"), Act loops without
// finding Front Row, and ReAct grounds its way to "keyboard function keys".
const APPLE: Task = {
  id: "apple",
  label: "Apple Remote",
  question:
    "Aside from the Apple Remote, what other device can control the program Apple Remote was originally designed to interact with?",
  answer: "keyboard function keys",
  hallucinatedAnswer: "iPhone, iPad, and iPod Touch",
  note: "Figure 1's headline question. CoT hallucinates an Apple TV answer; ReAct searches its way to Front Row and the real answer.",
  beats: [
    {
      thought:
        "I need to search Apple Remote and find the program it was originally designed to interact with.",
      action: { type: "search", arg: "Apple Remote" },
      hitObservation:
        "The Apple Remote is a remote introduced in 2005 to control the Front Row media center program.",
      fact: "Apple Remote was designed to control Front Row.",
      missObservation:
        "Could not find Apple Remote. Similar: ['Apple TV', 'Siri Remote'].",
      memoryGuess:
        "The Apple Remote controls Apple TV, which can be controlled by iPhone, iPad, and iPod Touch.",
      guessWrong: true,
    },
    {
      thought:
        "Apple Remote controls the Front Row media center. I need to search Front Row next and find what else can control it.",
      action: { type: "search", arg: "Front Row" },
      hitObservation:
        "Could not find Front Row. Similar: ['Front Row Seat to Earth', 'Front Row (software)'].",
      fact: "The plain Front Row page is missing; the software page is the right one.",
      missObservation:
        "Could not find Front Row. Similar: ['Front Row Seat to Earth', 'Front Row (software)'].",
      memoryGuess: "Front Row is probably just controlled by the Apple Remote.",
      guessWrong: true,
    },
    {
      thought: "Front Row is not found. I need to search Front Row (software).",
      action: { type: "search", arg: "Front Row (software)" },
      hitObservation:
        "Front Row is a discontinued media center software controlled by an Apple Remote or the keyboard function keys.",
      fact: "Front Row is controlled by an Apple Remote or the keyboard function keys.",
      missObservation: "Could not find Front Row (software).",
      memoryGuess: "Front Row is controlled by the Apple Remote only.",
      guessWrong: true,
    },
    {
      thought:
        "Front Row software is controlled by an Apple Remote or the keyboard function keys. So the answer is keyboard function keys.",
      action: { type: "finish", arg: "keyboard function keys" },
      hitObservation: "",
      fact: "",
      missObservation: "",
      memoryGuess: "",
      guessWrong: false,
    },
  ],
};

// The Colorado orogeny question from Appendix C. A clean multi-hop search chain
// with a search-reformulation hop (High Plains -> High Plains (United States)).
const COLORADO: Task = {
  id: "colorado",
  label: "Colorado orogeny",
  question:
    "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
  answer: "1,800 to 7,000 ft",
  hallucinatedAnswer: "around 5,000 to 10,000 ft",
  note: "Appendix C's clean multi-hop chain, including a search reformulation when the first High Plains page is the wrong one.",
  beats: [
    {
      thought:
        "I need to search Colorado orogeny, find the area the eastern sector extends into, then find its elevation range.",
      action: { type: "search", arg: "Colorado orogeny" },
      hitObservation:
        "The Colorado orogeny was an episode of mountain building in Colorado and surrounding areas.",
      fact: "The Colorado orogeny was mountain building in Colorado.",
      missObservation:
        "Could not find Colorado orogeny. Similar: ['Laramide orogeny'].",
      memoryGuess:
        "The Colorado orogeny is in the Rocky Mountains, so the elevation is high.",
      guessWrong: true,
    },
    {
      thought:
        "It does not mention the eastern sector. So I need to look up eastern sector.",
      action: { type: "lookup", arg: "eastern sector" },
      hitObservation:
        "The eastern sector extends into the High Plains and is called the Central Plains orogeny.",
      fact: "The eastern sector extends into the High Plains.",
      missObservation: "No more results for eastern sector.",
      memoryGuess: "The eastern sector probably extends into the foothills.",
      guessWrong: true,
    },
    {
      thought:
        "The eastern sector extends into the High Plains. So I need to search High Plains and find its elevation range.",
      action: { type: "search", arg: "High Plains" },
      hitObservation: "High Plains refers to one of two distinct land regions.",
      fact: "High Plains is ambiguous; I need the United States region.",
      missObservation:
        "High Plains refers to one of two distinct land regions.",
      memoryGuess: "The High Plains rise to maybe 5,000 to 10,000 ft.",
      guessWrong: true,
    },
    {
      thought: "I need to instead search High Plains (United States).",
      action: { type: "search", arg: "High Plains (United States)" },
      hitObservation:
        "The High Plains rise in elevation from around 1,800 to 7,000 ft.",
      fact: "The High Plains rise from around 1,800 to 7,000 ft.",
      missObservation: "Could not find High Plains (United States).",
      memoryGuess: "The High Plains rise to maybe 5,000 to 10,000 ft.",
      guessWrong: true,
    },
    {
      thought:
        "High Plains rise from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.",
      action: { type: "finish", arg: "1,800 to 7,000 ft" },
      hitObservation: "",
      fact: "",
      missObservation: "",
      memoryGuess: "",
      guessWrong: false,
    },
  ],
};

// A Fever-style fact-verification claim. Two hops to ground a SUPPORTS verdict.
const FEVER: Task = {
  id: "fever",
  label: "Fever claim",
  question:
    "Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
  answer: "SUPPORTS",
  hallucinatedAnswer: "NOT ENOUGH INFO",
  note: "A Fever claim. ReAct retrieves the role that grounds a SUPPORTS verdict; without grounding the agent guesses NOT ENOUGH INFO.",
  beats: [
    {
      thought:
        "I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.",
      action: { type: "search", arg: "Nikolaj Coster-Waldau" },
      hitObservation:
        "Nikolaj Coster-Waldau is a Danish actor. He appeared in the 2009 Fox television film Virtuality.",
      fact: "Coster-Waldau appeared in a Fox television film, Virtuality.",
      missObservation:
        "Could not find Nikolaj Coster-Waldau. Similar: ['Game of Thrones cast'].",
      memoryGuess:
        "I am not sure he worked with Fox, so there is not enough info.",
      guessWrong: true,
    },
    {
      thought:
        "He appeared in the Fox television film Virtuality, so he worked with the Fox Broadcasting Company.",
      action: { type: "finish", arg: "SUPPORTS" },
      hitObservation: "",
      fact: "",
      missObservation: "",
      memoryGuess: "",
      guessWrong: false,
    },
  ],
};

export const TASKS: Task[] = [APPLE, COLORADO, FEVER];

export function getTask(id: string): Task {
  return TASKS.find((task) => task.id === id) ?? at(TASKS, 0);
}

// A fault the reader injects at a chosen beat. `deadSearch` makes the next search
// return a non-informative page (the derail mode). `corruptMemory` strips the
// agent's grounding so it answers from a plausible guess (the hallucination
// mode). `loop` makes the agent repeat its last thought and action (the
// repetitive-reasoning failure). `editThought` rewrites the next thought, which
// is the human-in-the-loop correction.
export interface Faults {
  deadSearchAt: number | null;
  corruptMemoryAt: number | null;
  loopAt: number | null;
  thoughtEdits: Record<number, string>;
}

function emptyFaults(): Faults {
  return {
    deadSearchAt: null,
    corruptMemoryAt: null,
    loopAt: null,
    thoughtEdits: {},
  };
}

export type Status = "running" | "solved" | "wrong" | "stuck";

export interface SimState {
  taskId: string;
  mode: Mode;
  seed: number;
  // Index of the next beat to play.
  beatCursor: number;
  // Within a beat, ReAct plays thought -> action -> observation as separate
  // visible steps. This tracks the sub-step so each appears one tick apart.
  subStep: number;
  steps: TrajectoryStep[];
  faults: Faults;
  status: Status;
  finalAnswer: string | null;
  // How many of the agent's stated beliefs are backed by a real observation
  // versus carried only in memory. This is the grounding readout.
  groundedFacts: number;
  memoryFacts: number;
  // Consecutive no-progress beats; trips the "stuck" status (derail / loop).
  staleCount: number;
  tick: number;
  log: EventLog;
}

export type Intervention =
  | { type: "setMode"; mode: Mode }
  | { type: "setTask"; id: string }
  | { type: "setSeed"; value: number }
  | { type: "injectDeadSearch" }
  | { type: "corruptMemory" }
  | { type: "forceLoop" }
  | { type: "editThought"; text: string }
  | { type: "loadPreset"; id: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  taskId: string;
  mode: Mode;
  faults: Faults;
}

export const PRESETS: Preset[] = [
  {
    id: "react-grounded",
    label: "ReAct grounds the answer",
    blurb:
      "ReAct on the Apple Remote question. Step it and watch each thought trigger a search, each observation feed the next thought, until it finishes on the grounded answer.",
    taskId: "apple",
    mode: "react",
    faults: emptyFaults(),
  },
  {
    id: "cot-hallucinates",
    label: "CoT hallucinates",
    blurb:
      "The same question with no actions allowed. The model reasons from memory alone, never checks Front Row, and confidently answers with a device it never verified.",
    taskId: "apple",
    mode: "cot",
    faults: emptyFaults(),
  },
  {
    id: "act-loops",
    label: "Act stalls",
    blurb:
      "The same question with no thoughts allowed. With no plan to recover when Front Row is missing, the search-only agent stalls before it reaches the answer.",
    taskId: "apple",
    mode: "act",
    faults: emptyFaults(),
  },
  {
    id: "dead-search",
    label: "A dead search derails ReAct",
    blurb:
      "ReAct on the Colorado question, with the first search wired to return nothing. Watch the reasoning lose its footing and stall, the paper's search-error failure mode.",
    taskId: "colorado",
    mode: "react",
    faults: { ...emptyFaults(), deadSearchAt: 0 },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(presetId = "react-grounded"): SimState {
  const preset = getPreset(presetId);
  const intro: TrajectoryStep = {
    index: 0,
    kind: "observation",
    text: getTask(preset.taskId).question,
    informative: true,
  };
  return {
    taskId: preset.taskId,
    mode: preset.mode,
    seed: 7,
    beatCursor: 0,
    subStep: 0,
    steps: [intro],
    faults: {
      ...preset.faults,
      thoughtEdits: { ...preset.faults.thoughtEdits },
    },
    status: "running",
    finalAnswer: null,
    groundedFacts: 0,
    memoryFacts: 0,
    staleCount: 0,
    tick: 0,
    log: emptyLog(),
  };
}

function pushStep(
  state: SimState,
  step: Omit<TrajectoryStep, "index">,
): SimState {
  const index = state.steps.length;
  return { ...state, steps: [...state.steps, { ...step, index }] };
}

// Whether the upcoming search at this beat misses, either because the reader
// injected a dead search or because the beat is inherently a miss-then-reformulate
// hop (the Front Row / High Plains disambiguation beats). A seeded nudge can also
// turn a borderline search into a miss, so the seed matters.
function searchMisses(state: SimState, beat: Beat, beatIndex: number): boolean {
  if (state.faults.deadSearchAt === beatIndex) return true;
  // Beats whose hit and miss observations are identical are reformulation hops:
  // the page is genuinely the wrong one and the agent must search again.
  return (
    beat.hitObservation === beat.missObservation && beat.hitObservation !== ""
  );
}

function thoughtText(state: SimState, beat: Beat, beatIndex: number): string {
  return state.faults.thoughtEdits[beatIndex] ?? beat.thought;
}

// Advance one fixed tick. ReAct emits thought, then action, then observation as
// three ticks so the loop is legible. Act skips the thought. CoT skips the action
// and observation and instead writes a memory guess, then finishes on a guess.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "intervene") {
    return applyIntervention(state, input.action);
  }
  if (state.status !== "running") return state;

  const task = getTask(state.taskId);
  if (state.beatCursor >= task.beats.length) return state;
  const beat = at(task.beats, state.beatCursor);
  const isFinish = beat.action.type === "finish";

  if (state.mode === "cot") {
    return stepCot(state, task, beat);
  }

  // ReAct and Act share the action/observation machinery; Act just never shows a
  // thought sub-step.
  const wantsThought = state.mode === "react" && beat.thought !== "";

  if (wantsThought && state.subStep === 0) {
    const edited = thoughtText(state, beat, state.beatCursor) !== beat.thought;
    const text = thoughtText(state, beat, state.beatCursor);
    // A thought is grounded if a prior observation established the belief it
    // states. The first thought of a task is a plan, not yet grounded.
    const grounded = state.groundedFacts > 0;
    const advanced = pushStep(
      { ...state, subStep: 1, tick: state.tick + 1 },
      { kind: "thought", text, grounded },
    );
    return {
      ...advanced,
      log: record(
        advanced.log,
        advanced.tick,
        edited
          ? `Thought ${state.beatCursor + 1} (edited)`
          : `Thought ${state.beatCursor + 1}`,
      ),
    };
  }

  // Action sub-step (subStep 1 for ReAct, 0 for Act since it has no thought).
  const atActionSubStep =
    state.mode === "act" ? state.subStep === 0 : state.subStep === 1;
  if (atActionSubStep) {
    const actLabel = `Act ${state.beatCursor + 1}: ${beat.action.type}[${beat.action.arg}]`;
    const advanced = pushStep(
      { ...state, subStep: state.mode === "act" ? 1 : 2, tick: state.tick + 1 },
      {
        kind: "action",
        text: `${beat.action.type}[${beat.action.arg}]`,
        actionType: beat.action.type,
      },
    );
    const logged = {
      ...advanced,
      log: record(advanced.log, advanced.tick, actLabel),
    };
    if (isFinish) {
      return {
        ...logged,
        status: "solved",
        finalAnswer: beat.action.arg,
        log: record(logged.log, logged.tick, `Finished: ${beat.action.arg}`),
      };
    }
    return logged;
  }

  // Observation sub-step. The environment returns a page. If the search misses,
  // the observation is non-informative and the agent loses a fact it needed.
  const misses = searchMisses(state, beat, state.beatCursor);
  const rng = mulberry32(state.seed + state.beatCursor * 31);
  // A reformulation hop is a designed miss the agent recovers from on the next
  // beat; an injected dead search is a hard miss it cannot plan around in Act
  // mode and struggles with in ReAct.
  const injected = state.faults.deadSearchAt === state.beatCursor;
  const obsText = misses ? beat.missObservation : beat.hitObservation;
  const informative = !misses;

  let nextGrounded = state.groundedFacts;
  let nextStale = state.staleCount;
  if (informative) {
    nextGrounded += 1;
    nextStale = 0;
  } else {
    nextStale += 1;
  }

  const advanced = pushStep(
    {
      ...state,
      subStep: 0,
      beatCursor: state.beatCursor + 1,
      groundedFacts: nextGrounded,
      staleCount: nextStale,
      tick: state.tick + 1,
    },
    { kind: "observation", text: obsText, informative },
  );

  let logged = {
    ...advanced,
    log: record(
      advanced.log,
      advanced.tick,
      informative
        ? `Obs ${state.beatCursor + 1}: informative`
        : `Obs ${state.beatCursor + 1}: no useful info`,
    ),
  };

  // Act mode has no thought to plan a recovery, so any non-informative search
  // stalls it. This is the paper's Figure 1 result: the search-only agent cannot
  // reformulate "Front Row" into "Front Row (software)" because nothing reasons
  // about what to try next. ReAct survives a designed reformulation miss because
  // the next thought rewrites the query.
  if (!informative && state.mode === "act") {
    logged = {
      ...logged,
      status: "stuck",
      log: record(
        logged.log,
        logged.tick,
        "Stalled: no plan to recover the search",
      ),
    };
    return logged;
  }

  // An injected dead search hits ReAct harder than a designed reformulation miss.
  // A seeded nudge decides whether the reasoning claws back or loses its footing;
  // most of the time it derails, the paper's 23% search-error failure mode.
  if (injected && state.mode === "react" && rng() < 0.85) {
    logged = {
      ...logged,
      status: "stuck",
      log: record(
        logged.log,
        logged.tick,
        "Stalled: reasoning lost its footing",
      ),
    };
    return logged;
  }

  // Two non-informative observations in a row with no recovery means the agent is
  // genuinely derailed (the repetitive-reasoning loop).
  if (nextStale >= 2) {
    logged = {
      ...logged,
      status: "stuck",
      log: record(logged.log, logged.tick, "Stalled: repeating itself"),
    };
  }

  return logged;
}

// CoT plays one thought per beat (the memory guess), grounds nothing, and on the
// finish beat answers from memory. If any guess along the way was wrong, the
// final answer is the hallucinated one.
function stepCot(state: SimState, task: Task, beat: Beat): SimState {
  const isFinish = beat.action.type === "finish";
  if (isFinish) {
    const hallucinated = state.steps.some((s) => s.hallucinated);
    const answer = hallucinated ? task.hallucinatedAnswer : task.answer;
    const advanced = pushStep(
      { ...state, tick: state.tick + 1, beatCursor: state.beatCursor + 1 },
      {
        kind: "action",
        text: `finish[${answer}]`,
        actionType: "finish",
        hallucinated,
      },
    );
    return {
      ...advanced,
      status: hallucinated ? "wrong" : "solved",
      finalAnswer: answer,
      log: record(
        advanced.log,
        advanced.tick,
        hallucinated
          ? `Answered from memory (wrong): ${answer}`
          : `Answered: ${answer}`,
      ),
    };
  }

  const text = state.faults.thoughtEdits[state.beatCursor] ?? beat.memoryGuess;
  const edited = state.faults.thoughtEdits[state.beatCursor] !== undefined;
  // The guess is a hallucination when it is wrong and the reader did not edit it
  // into the truth.
  const hallucinated = beat.guessWrong && !edited;
  const advanced = pushStep(
    {
      ...state,
      tick: state.tick + 1,
      beatCursor: state.beatCursor + 1,
      memoryFacts: state.memoryFacts + 1,
    },
    { kind: "thought", text, grounded: false, hallucinated },
  );
  return {
    ...advanced,
    log: record(
      advanced.log,
      advanced.tick,
      hallucinated
        ? `Guess ${state.beatCursor + 1} (unverified)`
        : `Reasoned ${state.beatCursor + 1}`,
    ),
  };
}

function applyIntervention(state: SimState, action: Intervention): SimState {
  switch (action.type) {
    case "setMode": {
      const reset = initialState();
      return {
        ...reset,
        taskId: state.taskId,
        mode: action.mode,
        seed: state.seed,
        steps: [at(state.steps, 0)],
        log: record(state.log, state.tick, `mode -> ${action.mode}`),
      };
    }
    case "setTask": {
      const reset = initialState();
      const task = getTask(action.id);
      return {
        ...reset,
        taskId: action.id,
        mode: state.mode,
        seed: state.seed,
        steps: [
          {
            index: 0,
            kind: "observation",
            text: task.question,
            informative: true,
          },
        ],
        log: record(state.log, state.tick, `task -> ${task.label}`),
      };
    }
    case "setSeed":
      return {
        ...state,
        seed: action.value,
        log: record(state.log, state.tick, `seed -> ${action.value}`),
      };
    case "injectDeadSearch": {
      // Arm the next upcoming search beat to miss.
      return {
        ...state,
        faults: { ...state.faults, deadSearchAt: state.beatCursor },
        log: record(
          state.log,
          state.tick,
          `dead search armed at hop ${state.beatCursor + 1}`,
        ),
      };
    }
    case "corruptMemory": {
      // Strip grounding so the agent answers from a guess like CoT. Forces the
      // final answer to the hallucinated one.
      return {
        ...state,
        faults: { ...state.faults, corruptMemoryAt: state.beatCursor },
        steps: state.steps.map((s) =>
          s.kind === "thought"
            ? { ...s, grounded: false, hallucinated: true }
            : s,
        ),
        groundedFacts: 0,
        memoryFacts: state.groundedFacts + state.memoryFacts,
        log: record(
          state.log,
          state.tick,
          "memory corrupted: grounding stripped",
        ),
      };
    }
    case "forceLoop":
      return {
        ...state,
        faults: { ...state.faults, loopAt: state.beatCursor },
        status: "stuck",
        log: record(state.log, state.tick, "forced repetitive-reasoning loop"),
      };
    case "editThought": {
      const target = state.beatCursor;
      return {
        ...state,
        faults: {
          ...state.faults,
          thoughtEdits: { ...state.faults.thoughtEdits, [target]: action.text },
        },
        log: record(
          state.log,
          state.tick,
          `edited thought at hop ${target + 1}`,
        ),
      };
    }
    case "loadPreset": {
      const fresh = initialState(action.id);
      return {
        ...fresh,
        seed: state.seed,
        log: record(
          state.log,
          state.tick,
          `preset -> ${getPreset(action.id).label}`,
        ),
      };
    }
  }
}

// A small derived readout for the view: is the final answer (or the run so far)
// grounded in real observations, or carried only in memory?
export interface Grounding {
  grounded: number;
  memory: number;
  ratio: number;
}

export function grounding(state: SimState): Grounding {
  const total = state.groundedFacts + state.memoryFacts;
  return {
    grounded: state.groundedFacts,
    memory: state.memoryFacts,
    ratio: total === 0 ? 1 : state.groundedFacts / total,
  };
}
