// A pure, deterministic model of an LLM agent solving a CRMArena task, from
// Huang et al. 2024, "CRMArena: Understanding the Capacity of LLM Agents to
// Perform Professional CRM Tasks in Realistic Environments". The paper frames
// every task as a POMDP: an agent issues tool calls into a synthesized
// Salesforce org, reads back observations, and submits one final answer that an
// exact-match grader scores 0 or 1.
//
// A task here is a chain of dependent steps. Each step needs one correct tool
// call: resolve an object ID, look up its history, aggregate the numbers, then
// submit. The output of an earlier step feeds the arguments of a later one, so a
// single wrong call derails everything after it. That dependency chain is the
// thing the paper says trips agents up, and it is what the reader pokes here.
//
// Each tick is one agent turn. The agent picks the right tool for the current
// step with a probability that depends on its reasoning strength, the agentic
// framework (Act/ReAct/Function Calling), and whether the toolset gives it a
// clean task-specific function or only raw SOQL/SOSL queries. A wrong call
// returns an error observation; with execution feedback the agent can retry,
// and stronger models recover more often. The reader can crash that loop
// mid-run: drop a tool result, corrupt an argument, pull a needed function out
// of the toolset, or make the task unanswerable so the only correct answer is
// "None". All randomness comes from a seeded stream threaded through the state,
// so a seed plus a list of inputs reproduces a run exactly.

import { at } from "@/lib/simulation/array";
import { emptyLog, record, type EventLog } from "@/lib/simulation/history";
import { mulberry32 } from "@/lib/simulation/rng";

export const MAX_TURNS = 12;

// The four agent profiles map onto the paper's headline models. The numbers are
// hand-set so the relative ordering matches Table 2: a strong reasoner clears
// most chains, gpt-4o is solid, claude-3.5 is mid, and a weak open model stalls.
export interface AgentProfile {
  id: string;
  name: string;
  blurb: string;
  // Base chance of picking the correct tool for a step, before framework and
  // tool bonuses.
  skill: number;
  // Chance of fixing a wrong call after seeing the error observation. The
  // paper's deepseek-r1 note (poor at adjusting to feedback) lives here.
  recovery: number;
  // How well the model exploits a clean task-specific function. Weak models gain
  // little or are even hurt, which is the Table 4 finding.
  toolAffinity: number;
}

export const AGENTS: AgentProfile[] = [
  {
    id: "o1",
    name: "o1 (reasoner)",
    blurb:
      "A strong reasoning model. It clears long dependency chains and recovers from most errors, the best overall in the paper.",
    skill: 0.86,
    recovery: 0.8,
    toolAffinity: 1,
  },
  {
    id: "gpt4o",
    name: "gpt-4o",
    blurb:
      "The most cost-effective workhorse. Solid tool choice and good recovery, and it gains the most from task-specific functions.",
    skill: 0.74,
    recovery: 0.62,
    toolAffinity: 1,
  },
  {
    id: "claude35",
    name: "claude-3.5-sonnet",
    blurb:
      "Capable but mid-pack on these chains. It improves with function calling, like the other strong proprietary models.",
    skill: 0.6,
    recovery: 0.5,
    toolAffinity: 0.85,
  },
  {
    id: "llama8b",
    name: "llama-3.1-8b (weak)",
    blurb:
      "A weak open model. It rarely chains calls correctly and barely recovers, and human-crafted functions can hurt it more than help.",
    skill: 0.24,
    recovery: 0.18,
    toolAffinity: 0.35,
  },
];

export function getAgent(id: string): AgentProfile {
  return AGENTS.find((agent) => agent.id === id) ?? at(AGENTS, 0);
}

// The three agentic frameworks from the paper. ReAct adds a thought before each
// action and helps most models. Function Calling helps strong models but can
// drag weak ones down. Act (no thought) is the weakest scaffold.
export interface Framework {
  id: string;
  name: string;
  blurb: string;
  // Multiplier on the correct-tool probability.
  skillMul: number;
  // Multiplier on recovery: ReAct's thought step buys the most error recovery.
  recoveryMul: number;
}

export const FRAMEWORKS: Framework[] = [
  {
    id: "act",
    name: "Act",
    blurb: "Action only, no thought step. The weakest scaffold.",
    skillMul: 0.78,
    recoveryMul: 0.7,
  },
  {
    id: "react",
    name: "ReAct",
    blurb:
      "A thought before every action. The reasoning step lifts both tool choice and error recovery.",
    skillMul: 1,
    recoveryMul: 1,
  },
  {
    id: "fc",
    name: "Function Calling",
    blurb:
      "The model calls typed functions directly. Great for strong models, but weak function-callers stumble on the schema.",
    skillMul: 1.05,
    recoveryMul: 0.85,
  },
];

export function getFramework(id: string): Framework {
  return (
    FRAMEWORKS.find((framework) => framework.id === id) ?? at(FRAMEWORKS, 1)
  );
}

// The toolset the agent gets. General-purpose is raw SOQL/SOSL the agent must
// compose itself. Task-specific gives clean Python wrappers, but only a model
// with enough function-calling skill turns that into a win.
export type Toolset = "general" | "specific";

// One step in a task's dependency chain. The label is what the agent must get
// right at this turn (the right object ID, the right query, the right
// aggregation). `needsFunction` marks a step that a removed task-specific
// function would have served, so pulling that function out of the toolset on a
// task-specific run breaks exactly this step.
export interface TaskStep {
  label: string;
  needsFunction: boolean;
}

export interface CrmTask {
  id: string;
  name: string;
  persona: string;
  query: string;
  steps: TaskStep[];
  // True when the only correct answer is "None" (a false-presupposition query).
  // The agent must resist inventing an answer.
  answerable: boolean;
}

export const TASKS: CrmTask[] = [
  {
    id: "ned",
    name: "Named Entity Disambiguation",
    persona: "Service Agent",
    query: "Show me the running shoes I purchased last year.",
    answerable: true,
    steps: [
      { label: "find the customer's contact ID", needsFunction: false },
      { label: "pull the customer's order history", needsFunction: true },
      { label: "match 'running shoes' to one product", needsFunction: true },
      { label: "return the matching order ID", needsFunction: false },
    ],
  },
  {
    id: "tii",
    name: "Top Issue Identification",
    persona: "Service Analyst",
    query: "What was the most reported issue for product X?",
    answerable: true,
    steps: [
      { label: "resolve product X to its ID", needsFunction: false },
      { label: "query every case for that product", needsFunction: true },
      { label: "count issues and take the top one", needsFunction: true },
      { label: "return the top issue", needsFunction: false },
    ],
  },
  {
    id: "ncr",
    name: "New Case Routing",
    persona: "Service Manager",
    query: "Which agent should this new case go to?",
    answerable: true,
    steps: [
      { label: "read the case subject and skill needed", needsFunction: false },
      { label: "list agents with that skill", needsFunction: true },
      { label: "rank them by availability", needsFunction: true },
      { label: "return the best agent ID", needsFunction: false },
    ],
  },
  {
    id: "htu-none",
    name: "Handle Time (unanswerable)",
    persona: "Service Manager",
    query: "Which agent had the shortest handle time last month?",
    answerable: false,
    steps: [
      { label: "query handle times for last month", needsFunction: true },
      { label: "find no cases were handled then", needsFunction: false },
      { label: "answer 'None', resist inventing one", needsFunction: false },
    ],
  },
];

export function getTask(id: string): CrmTask {
  return TASKS.find((task) => task.id === id) ?? at(TASKS, 0);
}

// A turn's outcome, kept so the view can render the trace and color each turn.
export type TurnKind =
  | "correct"
  | "error"
  | "recovered"
  | "submit-correct"
  | "submit-wrong"
  | "dropped"
  | "corrupted";

export interface Turn {
  index: number;
  stepIndex: number;
  kind: TurnKind;
  detail: string;
}

export type RunStatus = "running" | "solved" | "failed";

export interface CrmParams {
  agentId: string;
  frameworkId: string;
  toolset: Toolset;
  taskId: string;
  seed: number;
}

export interface CrmState {
  params: CrmParams;
  // Cursor into the task's step chain.
  step: number;
  turn: number;
  status: RunStatus;
  // The internal rng cursor, threaded so the reducer stays pure.
  rngState: number;
  turns: Turn[];
  // Steps whose task-specific function the reader pulled this run. A general
  // toolset has none of them removed (they were never there); on a specific
  // toolset, removing one breaks that step's clean path.
  removedFunctionSteps: number[];
  // The reader can answer for the agent on the current step. -1 means no forced
  // answer pending.
  forcedSubmit: "none" | "guess" | null;
  log: EventLog<CrmParams>;
}

// Pulls one value from the seeded stream and returns the advanced cursor, so the
// reducer never holds a stateful closure and a seed reproduces the exact run.
function draw(rngState: number): { value: number; next: number } {
  // mulberry32 is built to be seeded fresh, so we re-seed from the cursor and
  // fold the single draw back into a new cursor. This keeps determinism while
  // staying inside the shared rng helper.
  const rng = mulberry32(rngState);
  const value = rng();
  const next = (rngState + 0x9e3779b9) | 0;
  return { value, next };
}

// The probability the agent gets the current step's tool call right, given its
// profile, the framework, and the toolset. A clean task-specific function lifts
// a strong model and barely moves a weak one; a removed function on that step
// drops the agent back to composing raw queries.
function stepSuccessChance(
  state: CrmState,
  agent: AgentProfile,
  framework: Framework,
): number {
  const task = getTask(state.params.taskId);
  const stepSpec = at(task.steps, state.step);
  let chance = agent.skill * framework.skillMul;

  const functionRemoved = state.removedFunctionSteps.includes(state.step);
  if (state.params.toolset === "specific" && stepSpec.needsFunction) {
    if (functionRemoved) {
      // The clean wrapper is gone; fall back to composing raw queries, which is
      // harder, especially for this step.
      chance *= 0.55;
    } else {
      // Task-specific functions help, scaled by the model's ability to use them.
      chance *= 0.6 + 0.55 * agent.toolAffinity;
    }
  }

  // The unanswerable step rewards restraint, which weak models lack: they tend
  // to invent an answer instead of returning "None".
  if (!task.answerable && stepSpec.label.includes("resist")) {
    chance = 0.25 + 0.7 * agent.skill;
  }

  return Math.max(0.02, Math.min(0.98, chance));
}

export interface Intervention {
  type:
    | "loadPreset"
    | "setAgent"
    | "setFramework"
    | "setToolset"
    | "setTask"
    | "setSeed"
    | "dropResult"
    | "corruptArg"
    | "removeFunction"
    | "forceGuess"
    | "restore";
  id?: string;
  value?: number;
  params?: CrmParams;
  label?: string;
}

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

function freshRunFields(
  params: CrmParams,
): Pick<
  CrmState,
  | "step"
  | "turn"
  | "status"
  | "rngState"
  | "turns"
  | "removedFunctionSteps"
  | "forcedSubmit"
> {
  return {
    step: 0,
    turn: 0,
    status: "running",
    rngState: (params.seed * 0x100000 + 0x1234567) | 0,
    turns: [],
    removedFunctionSteps: [],
    forcedSubmit: null,
  };
}

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: CrmParams;
}

export const PRESETS: Preset[] = [
  {
    id: "reasoner-wins",
    label: "Reasoner clears it",
    blurb:
      "o1 under ReAct on a clean chain. Step it and watch the agent thread the dependent calls and submit the right answer.",
    params: {
      agentId: "o1",
      frameworkId: "react",
      toolset: "general",
      taskId: "ned",
      seed: 3,
    },
  },
  {
    id: "weak-stalls",
    label: "Weak model stalls",
    blurb:
      "A weak open model under Act. It botches an early call and rarely recovers, so the chain falls apart before the submit.",
    params: {
      agentId: "llama8b",
      frameworkId: "act",
      toolset: "general",
      taskId: "tii",
      seed: 5,
    },
  },
  {
    id: "tools-backfire",
    label: "Tools backfire (weak)",
    blurb:
      "The same weak model, now handed task-specific functions. Run it and compare: the clean wrappers do not save a model that cannot call them.",
    params: {
      agentId: "llama8b",
      frameworkId: "fc",
      toolset: "specific",
      taskId: "tii",
      seed: 5,
    },
  },
  {
    id: "false-premise",
    label: "False-presupposition",
    blurb:
      "An unanswerable query. The correct answer is 'None'. Strong models hold back; weak ones invent an agent that never existed.",
    params: {
      agentId: "claude35",
      frameworkId: "react",
      toolset: "general",
      taskId: "htu-none",
      seed: 2,
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(presetId = "reasoner-wins"): CrmState {
  const params = { ...getPreset(presetId).params };
  return { params, ...freshRunFields(params), log: emptyLog() };
}

function logged(state: CrmState, label: string): CrmState {
  return { ...state, log: record(state.log, state.turn, label, state.params) };
}

// Advance the agent by one turn. The agent works the current step: it either
// gets the tool call right and moves on, hits an error and may recover, or burns
// a turn. When the last step lands, it submits, and the grader scores it.
function advance(state: CrmState): CrmState {
  if (state.status !== "running") return state;

  const task = getTask(state.params.taskId);
  const agent = getAgent(state.params.agentId);
  const framework = getFramework(state.params.frameworkId);

  // A forced guess short-circuits the loop: the reader answered for the agent.
  if (state.forcedSubmit === "guess") {
    const turn: Turn = {
      index: state.turn + 1,
      stepIndex: state.step,
      kind: "submit-wrong",
      detail: task.answerable
        ? "forced early guess, graded wrong"
        : "guessed an answer to an unanswerable query",
    };
    return logged(
      {
        ...state,
        turn: state.turn + 1,
        status: "failed",
        forcedSubmit: null,
        turns: [...state.turns, turn],
      },
      "forced guess -> failed",
    );
  }

  if (state.turn >= MAX_TURNS) {
    const turn: Turn = {
      index: state.turn + 1,
      stepIndex: state.step,
      kind: "submit-wrong",
      detail: "ran out of turns before reaching submit",
    };
    return logged(
      {
        ...state,
        turn: state.turn + 1,
        status: "failed",
        turns: [...state.turns, turn],
      },
      "out of turns -> failed",
    );
  }

  const stepSpec = at(task.steps, state.step);
  const isLastStep = state.step === task.steps.length - 1;

  const roll = draw(state.rngState);
  const chance = stepSuccessChance(state, agent, framework);
  const success = roll.value < chance;

  if (success) {
    if (isLastStep) {
      // The agent submits. On an answerable task a cleared chain grades correct;
      // on an unanswerable one, clearing the chain means it returned "None".
      const turn: Turn = {
        index: state.turn + 1,
        stepIndex: state.step,
        kind: "submit-correct",
        detail: task.answerable
          ? "submitted the right answer"
          : "returned 'None', the right call",
      };
      return logged(
        {
          ...state,
          turn: state.turn + 1,
          rngState: roll.next,
          status: "solved",
          turns: [...state.turns, turn],
        },
        "submit -> solved",
      );
    }

    const turn: Turn = {
      index: state.turn + 1,
      stepIndex: state.step,
      kind: "correct",
      detail: `${stepSpec.label}`,
    };
    return logged(
      {
        ...state,
        turn: state.turn + 1,
        step: state.step + 1,
        rngState: roll.next,
        turns: [...state.turns, turn],
      },
      `step ${state.step + 1} ok`,
    );
  }

  // Wrong call. The agent sees the error observation and may recover next turn.
  const recoveryRoll = draw(roll.next);
  const recovered = recoveryRoll.value < agent.recovery * framework.recoveryMul;

  const turn: Turn = {
    index: state.turn + 1,
    stepIndex: state.step,
    kind: recovered ? "recovered" : "error",
    detail: recovered
      ? `error, then fixed: ${stepSpec.label}`
      : `wrong call on: ${stepSpec.label}`,
  };

  if (recovered) {
    if (isLastStep) {
      const submitTurn: Turn = {
        index: state.turn + 1,
        stepIndex: state.step,
        kind: "submit-correct",
        detail: "recovered, then submitted the right answer",
      };
      return logged(
        {
          ...state,
          turn: state.turn + 1,
          rngState: recoveryRoll.next,
          status: "solved",
          turns: [...state.turns, submitTurn],
        },
        "recovered submit -> solved",
      );
    }
    return logged(
      {
        ...state,
        turn: state.turn + 1,
        step: state.step + 1,
        rngState: recoveryRoll.next,
        turns: [...state.turns, turn],
      },
      `step ${state.step + 1} recovered`,
    );
  }

  return logged(
    {
      ...state,
      turn: state.turn + 1,
      rngState: recoveryRoll.next,
      turns: [...state.turns, turn],
    },
    `error on step ${state.step + 1}`,
  );
}

function applyIntervention(
  state: CrmState,
  action: Intervention,
): { state: CrmState; label: string } {
  switch (action.type) {
    case "loadPreset": {
      const preset = getPreset(action.id ?? "");
      const params = { ...preset.params };
      return {
        state: { ...state, params, ...freshRunFields(params) },
        label: `preset -> ${preset.label}`,
      };
    }
    case "setAgent": {
      const params = {
        ...state.params,
        agentId: action.id ?? state.params.agentId,
      };
      return {
        state: { ...state, params, ...freshRunFields(params) },
        label: `agent -> ${getAgent(params.agentId).name}`,
      };
    }
    case "setFramework": {
      const params = {
        ...state.params,
        frameworkId: action.id ?? state.params.frameworkId,
      };
      return {
        state: { ...state, params, ...freshRunFields(params) },
        label: `framework -> ${getFramework(params.frameworkId).name}`,
      };
    }
    case "setToolset": {
      const toolset: Toolset =
        action.id === "specific" ? "specific" : "general";
      const params = { ...state.params, toolset };
      return {
        state: { ...state, params, ...freshRunFields(params) },
        label: `toolset -> ${toolset}`,
      };
    }
    case "setTask": {
      const params = {
        ...state.params,
        taskId: action.id ?? state.params.taskId,
      };
      return {
        state: { ...state, params, ...freshRunFields(params) },
        label: `task -> ${getTask(params.taskId).name}`,
      };
    }
    case "setSeed": {
      const params = {
        ...state.params,
        seed: action.value ?? state.params.seed,
      };
      return {
        state: { ...state, params, ...freshRunFields(params) },
        label: `seed -> ${params.seed}`,
      };
    }
    case "dropResult": {
      // The next correct call's result vanishes, so the agent burns a turn even
      // when it called the right tool. We model it as a forced error this turn.
      if (state.status !== "running") return { state, label: "drop (no-op)" };
      const turn: Turn = {
        index: state.turn + 1,
        stepIndex: state.step,
        kind: "dropped",
        detail: "tool result dropped before the agent could read it",
      };
      const next = advanceAfterForcedError(state, turn);
      return { state: next, label: "dropped a tool result" };
    }
    case "corruptArg": {
      if (state.status !== "running")
        return { state, label: "corrupt (no-op)" };
      const turn: Turn = {
        index: state.turn + 1,
        stepIndex: state.step,
        kind: "corrupted",
        detail: "an argument from the previous step was corrupted",
      };
      const next = advanceAfterForcedError(state, turn);
      return { state: next, label: "corrupted an argument" };
    }
    case "removeFunction": {
      // Pull the task-specific function for the current step. Only bites on a
      // specific toolset and a step that needed that function.
      const stepIndex = state.step;
      if (state.removedFunctionSteps.includes(stepIndex)) {
        return { state, label: "function already removed" };
      }
      return {
        state: {
          ...state,
          removedFunctionSteps: [...state.removedFunctionSteps, stepIndex],
        },
        label: `removed the function for step ${stepIndex + 1}`,
      };
    }
    case "forceGuess": {
      return {
        state: { ...state, forcedSubmit: "guess" },
        label: "forced the agent to guess now",
      };
    }
    case "restore": {
      const params = action.params ?? state.params;
      return {
        state: { ...state, params, ...freshRunFields(params) },
        label: action.label ?? "restored",
      };
    }
    default:
      return { state, label: "no-op" };
  }
}

// A forced error (dropped result or corrupted argument) costs the agent this
// turn and then runs the normal recovery roll, so even a strong model can be
// knocked back by an ill-timed strike.
function advanceAfterForcedError(state: CrmState, turn: Turn): CrmState {
  const agent = getAgent(state.params.agentId);
  const framework = getFramework(state.params.frameworkId);
  const task = getTask(state.params.taskId);
  const isLastStep = state.step === task.steps.length - 1;

  const recoveryRoll = draw(state.rngState);
  const recovered =
    recoveryRoll.value < agent.recovery * framework.recoveryMul * 0.85;

  if (state.turn + 1 >= MAX_TURNS && !recovered) {
    return logged(
      {
        ...state,
        turn: state.turn + 1,
        rngState: recoveryRoll.next,
        status: "failed",
        turns: [...state.turns, turn],
      },
      turn.detail,
    );
  }

  if (recovered && isLastStep) {
    const submitTurn: Turn = {
      ...turn,
      kind: "submit-correct",
      detail: `${turn.detail}, recovered and submitted`,
    };
    return logged(
      {
        ...state,
        turn: state.turn + 1,
        rngState: recoveryRoll.next,
        status: "solved",
        turns: [...state.turns, submitTurn],
      },
      `${turn.kind} then solved`,
    );
  }

  if (recovered) {
    return logged(
      {
        ...state,
        turn: state.turn + 1,
        step: state.step + 1,
        rngState: recoveryRoll.next,
        turns: [
          ...state.turns,
          { ...turn, detail: `${turn.detail}, recovered` },
        ],
      },
      `${turn.kind} then recovered`,
    );
  }

  return logged(
    {
      ...state,
      turn: state.turn + 1,
      rngState: recoveryRoll.next,
      turns: [...state.turns, turn],
    },
    turn.detail,
  );
}

// The pure reducer. A tick advances the agent one turn; an intervention reshapes
// the run and records what happened. Every step is deterministic given the seed.
export function step(state: CrmState, input: Input): CrmState {
  if (input.kind === "tick") return advance(state);
  const { state: next, label } = applyIntervention(state, input.action);
  return logged(next, label);
}

// Read-only views the component renders.
export function totalSteps(params: CrmParams): number {
  return getTask(params.taskId).steps.length;
}

export function progressLabel(state: CrmState): string {
  const task = getTask(state.params.taskId);
  if (state.status === "solved") return "solved";
  if (state.status === "failed") return "failed";
  return `step ${state.step + 1} of ${task.steps.length}`;
}
