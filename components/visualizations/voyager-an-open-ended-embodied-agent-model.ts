// A pure, deterministic model of VOYAGER's lifelong-learning loop from "VOYAGER:
// An Open-Ended Embodied Agent with Large Language Models". The real agent runs
// in Minecraft via Mineflayer and calls GPT-4. Here the world is a small tech
// tree of craftable items, GPT-4's coding is modelled as a per-round success
// probability that depends on task difficulty, the coder strength, and how many
// relevant skills the library can hand back, and the curriculum, skill library,
// and iterative-prompt refiner are the paper's real control flow. No randomness
// reaches the loop unless it draws from the seeded rng threaded through state, so
// a given seed reproduces the exact run.
//
// The mechanism kept faithful: a task is proposed by the curriculum from the
// agent's frontier, relevant skills are retrieved, the coder writes a program and
// gets up to MAX_ROUNDS rounds of refinement from environment feedback and
// execution errors, a self-verification critic decides pass or fail, and a passed
// program is committed to the skill library keyed by its description so harder
// tasks can compose it. Turn the library off and progress plateaus; turn
// self-verification off and unverified skills get committed; swap the curriculum
// for a random one and the frontier stalls. Those are the paper's own ablations.

import {
  record,
  emptyLog,
  type EventLog,
  type SimEvent,
} from "@/lib/simulation/history";
import { nextRandom } from "@/lib/simulation/rng";
import { at } from "@/lib/simulation/array";

// One node of the tech tree. `needs` are the item ids a task to make this item
// can compose from the skill library; an item with all its needs already in the
// library is far easier to code. `difficulty` is the base hardness GPT-4 faces
// writing the program from scratch.
export interface TechItem {
  id: string;
  label: string;
  tier: number;
  needs: string[];
  difficulty: number;
}

// A small but real Minecraft tech tree: wood up through the diamond tools the
// paper tracks. Tiers gate the curriculum so it proposes reachable tasks.
export const TECH_TREE: TechItem[] = [
  { id: "log", label: "Mine Wood Log", tier: 0, needs: [], difficulty: 0.15 },
  {
    id: "planks",
    label: "Craft Planks",
    tier: 0,
    needs: ["log"],
    difficulty: 0.18,
  },
  {
    id: "stick",
    label: "Craft Stick",
    tier: 0,
    needs: ["planks"],
    difficulty: 0.2,
  },
  {
    id: "table",
    label: "Make Crafting Table",
    tier: 0,
    needs: ["planks"],
    difficulty: 0.22,
  },
  {
    id: "wood_pickaxe",
    label: "Craft Wooden Pickaxe",
    tier: 1,
    needs: ["planks", "stick", "table"],
    difficulty: 0.32,
  },
  {
    id: "cobblestone",
    label: "Mine Cobblestone",
    tier: 1,
    needs: ["wood_pickaxe"],
    difficulty: 0.3,
  },
  {
    id: "stone_pickaxe",
    label: "Craft Stone Pickaxe",
    tier: 2,
    needs: ["cobblestone", "stick", "table"],
    difficulty: 0.42,
  },
  {
    id: "furnace",
    label: "Make Furnace",
    tier: 2,
    needs: ["cobblestone", "table"],
    difficulty: 0.44,
  },
  {
    id: "raw_iron",
    label: "Mine Raw Iron",
    tier: 3,
    needs: ["stone_pickaxe"],
    difficulty: 0.55,
  },
  {
    id: "iron_ingot",
    label: "Smelt Iron Ingot",
    tier: 3,
    needs: ["raw_iron", "furnace"],
    difficulty: 0.58,
  },
  {
    id: "iron_pickaxe",
    label: "Craft Iron Pickaxe",
    tier: 4,
    needs: ["iron_ingot", "stick", "table"],
    difficulty: 0.68,
  },
  {
    id: "iron_sword",
    label: "Craft Iron Sword",
    tier: 4,
    needs: ["iron_ingot", "stick", "table"],
    difficulty: 0.66,
  },
  {
    id: "shield",
    label: "Craft Shield",
    tier: 4,
    needs: ["planks", "iron_ingot", "table"],
    difficulty: 0.64,
  },
  {
    id: "combat_zombie",
    label: "Combat Zombie",
    tier: 5,
    needs: ["iron_sword", "shield"],
    difficulty: 0.78,
  },
  {
    id: "diamond",
    label: "Mine Diamond",
    tier: 5,
    needs: ["iron_pickaxe"],
    difficulty: 0.85,
  },
];

export function getItem(id: string): TechItem {
  return TECH_TREE.find((item) => item.id === id) ?? at(TECH_TREE, 0);
}

// A skill stored in the library: the item it makes plus how many simpler skills
// it composed when it was written. The composition count is the paper's point,
// since a skill built from many smaller skills was cheap to write.
export interface Skill {
  itemId: string;
  label: string;
  composedFrom: number;
}

export type Curriculum = "automatic" | "manual" | "random";
export type Coder = "gpt-4" | "gpt-3.5";

export interface Params {
  curriculum: Curriculum;
  coder: Coder;
  skillLibrary: boolean;
  selfVerification: boolean;
  hallucinationRate: number;
}

// One slot of the iterative prompting mechanism: the current task, which round of
// refinement it is on, and whether the round just executed cleanly. A task lives
// here while it is being coded and refined.
export interface ActiveTask {
  itemId: string;
  label: string;
  round: number;
  difficulty: number;
  retrievedSkills: number;
  hallucinated: boolean;
  lastRoundExecuted: boolean;
}

export type Phase =
  | "propose"
  | "retrieve"
  | "code"
  | "execute"
  | "verify"
  | "commit"
  | "idle";

export interface StepRecord {
  itemId: string;
  label: string;
  phase: Phase;
  detail: string;
}

export interface SimState {
  params: Params;
  rng: number;
  tick: number;
  phase: Phase;
  active: ActiveTask | null;
  library: Skill[];
  unlocked: string[];
  attemptedFailures: string[];
  committedUnverified: number;
  iterations: number;
  failedAttempts: number;
  history: number[];
  last: StepRecord | null;
  log: EventLog<SimState>;
}

export const MAX_ROUNDS = 4;

function rand(state: SimState): { value: number; next: SimState } {
  const drawn = nextRandom(state.rng);
  return { value: drawn.value, next: { ...state, rng: drawn.state } };
}

// The set of items the agent could make next: every item whose tech prerequisites
// (the previous tier's needs) are satisfied and that is not already unlocked.
function frontier(state: SimState): TechItem[] {
  const have = new Set(state.unlocked);
  return TECH_TREE.filter((item) => {
    if (have.has(item.id)) return false;
    return item.needs.every((need) => have.has(need) || isPrimitive(need));
  });
}

// Wood and planks are reachable from bare hands, so the very first tasks have no
// satisfied prerequisite to gate on.
function isPrimitive(id: string): boolean {
  return id === "log";
}

function librarySkillsFor(state: SimState, item: TechItem): number {
  if (!state.params.skillLibrary) return 0;
  const have = new Set(state.library.map((skill) => skill.itemId));
  return item.needs.filter((need) => have.has(need)).length;
}

// The chance GPT-4 writes a working program this round. Harder tasks start lower;
// every relevant skill the library retrieves pulls it up (composition is cheap);
// GPT-3.5 codes far worse; later rounds improve because environment feedback and
// the error trace narrow the bug. This stands in for the LLM the real loop calls.
function codeSuccessChance(
  params: Params,
  difficulty: number,
  retrievedSkills: number,
  round: number,
  hallucinated: boolean,
): number {
  if (hallucinated) return 0;
  const coderBase = params.coder === "gpt-4" ? 0.62 : 0.28;
  const skillBoost = params.skillLibrary ? retrievedSkills * 0.13 : 0;
  const roundBoost = (round - 1) * 0.12;
  const raw = coderBase - difficulty * 0.55 + skillBoost + roundBoost;
  return Math.max(0.04, Math.min(0.96, raw));
}

// The curriculum's pick of the next task. The automatic curriculum prefers the
// lowest-tier unsolved item so it climbs in order; the random curriculum picks
// any reachable item with no regard to readiness, which is why it stalls; the
// manual curriculum follows a fixed hand-set order and ignores the agent's state.
const MANUAL_ORDER = [
  "log",
  "planks",
  "stick",
  "table",
  "wood_pickaxe",
  "cobblestone",
  "stone_pickaxe",
  "furnace",
  "raw_iron",
  "iron_ingot",
  "iron_pickaxe",
  "iron_sword",
  "shield",
  "combat_zombie",
  "diamond",
];

function proposeTask(state: SimState): {
  item: TechItem | null;
  next: SimState;
} {
  const reachable = frontier(state);
  if (reachable.length === 0) return { item: null, next: state };

  if (state.params.curriculum === "manual") {
    const have = new Set(state.unlocked);
    const nextId = MANUAL_ORDER.find((id) => !have.has(id));
    const item = nextId ? getItem(nextId) : null;
    // The manual order can name a task whose prerequisites are not built yet,
    // which the agent then cannot complete, so it falls through to nothing.
    if (item && reachable.some((candidate) => candidate.id === item.id)) {
      return { item, next: state };
    }
    return { item: null, next: state };
  }

  if (state.params.curriculum === "random") {
    const drawn = rand(state);
    const index = Math.floor(drawn.value * reachable.length);
    return {
      item: at(reachable, Math.min(index, reachable.length - 1)),
      next: drawn.next,
    };
  }

  // Automatic: lowest tier first, ties broken by lowest difficulty, which keeps
  // the proposed task hard enough to learn from but not impossible.
  const sorted = [...reachable].sort(
    (a, b) => a.tier - b.tier || a.difficulty - b.difficulty,
  );
  return { item: at(sorted, 0), next: state };
}

export type Intervention =
  | { type: "setCurriculum"; value: Curriculum }
  | { type: "setCoder"; value: Coder }
  | { type: "toggleSkillLibrary" }
  | { type: "toggleSelfVerification" }
  | { type: "setHallucination"; value: number }
  | { type: "forceTask"; itemId: string }
  | { type: "setSeed"; value: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: SimState; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

function logged(state: SimState, label: string): SimState {
  return { ...state, log: record(state.log, state.tick, label, state) };
}

function startTask(state: SimState, item: TechItem, next: SimState): SimState {
  const drawn = rand(next);
  const hallucinated = drawn.value < state.params.hallucinationRate;
  const task: ActiveTask = {
    itemId: item.id,
    label: item.label,
    round: 1,
    difficulty: item.difficulty,
    retrievedSkills: librarySkillsFor(state, item),
    hallucinated,
    lastRoundExecuted: false,
  };
  const detail = hallucinated
    ? "proposed a task for an item that doesn't exist"
    : task.retrievedSkills > 0
      ? `retrieved ${task.retrievedSkills} skill${task.retrievedSkills === 1 ? "" : "s"} to compose`
      : "no relevant skills to retrieve";
  return {
    ...drawn.next,
    phase: "retrieve",
    active: task,
    last: { itemId: item.id, label: item.label, phase: "propose", detail },
  };
}

// One advance of the loop. With no active task, the curriculum proposes one. With
// an active task, the coder takes a round: it either executes cleanly and reaches
// self-verification, or it fails and refines, until MAX_ROUNDS runs out and the
// task is shelved for the curriculum to reattempt later.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "intervene") {
    return applyIntervention(state, input.action);
  }

  const ticked: SimState = { ...state, tick: state.tick + 1 };

  if (ticked.active === null) {
    const { item, next } = proposeTask(ticked);
    if (item === null) {
      return {
        ...next,
        phase: "idle",
        last: {
          itemId: "",
          label: "",
          phase: "idle",
          detail:
            ticked.params.curriculum === "random"
              ? "random curriculum proposed nothing reachable"
              : "no further task; the frontier is exhausted",
        },
      };
    }
    return startTask(ticked, item, next);
  }

  return runRound(ticked, ticked.active);
}

function runRound(state: SimState, task: ActiveTask): SimState {
  const item = getItem(task.itemId);
  const chance = codeSuccessChance(
    state.params,
    task.difficulty,
    task.retrievedSkills,
    task.round,
    task.hallucinated,
  );
  const drawn = rand(state);
  const executed = drawn.value < chance;
  const afterExecute: SimState = {
    ...drawn.next,
    iterations: drawn.next.iterations + 1,
  };

  if (!executed) {
    if (task.round >= MAX_ROUNDS) {
      return shelveTask(afterExecute, task, item);
    }
    const detail = task.hallucinated
      ? "execution error: no such item in the game"
      : "environment feedback: missing materials, refining the program";
    return {
      ...afterExecute,
      phase: "code",
      active: { ...task, round: task.round + 1, lastRoundExecuted: false },
      last: { itemId: item.id, label: item.label, phase: "execute", detail },
    };
  }

  return verifyTask(afterExecute, task, item);
}

// Self-verification: a critic decides whether the program truly completed the
// task. With verification on, the critic is usually right, so only real successes
// commit. With it off, every executed program commits, even partial ones, which
// is how unverified skills enter the library and progress quietly stops being
// real (the paper's -73% ablation).
function verifyTask(
  state: SimState,
  task: ActiveTask,
  item: TechItem,
): SimState {
  if (!state.params.selfVerification) {
    return commitTask(state, task, item, true);
  }
  const drawn = rand(state);
  const criticPasses = drawn.value < 0.9;
  const after = { ...drawn.next };
  if (criticPasses) {
    return commitTask(after, task, item, false);
  }
  if (task.round >= MAX_ROUNDS) {
    return shelveTask(after, task, item);
  }
  return {
    ...after,
    phase: "code",
    active: { ...task, round: task.round + 1, lastRoundExecuted: true },
    last: {
      itemId: item.id,
      label: item.label,
      phase: "verify",
      detail: "self-verification: task not complete, trying again",
    },
  };
}

function commitTask(
  state: SimState,
  task: ActiveTask,
  item: TechItem,
  unverified: boolean,
): SimState {
  const have = new Set(state.unlocked);
  const newlyUnlocked = have.has(item.id)
    ? state.unlocked
    : [...state.unlocked, item.id];
  const skill: Skill = {
    itemId: item.id,
    label: item.label,
    composedFrom: task.retrievedSkills,
  };
  const library = state.params.skillLibrary
    ? state.library.some((existing) => existing.itemId === item.id)
      ? state.library
      : [...state.library, skill]
    : state.library;
  const committed: SimState = {
    ...state,
    phase: "commit",
    active: null,
    unlocked: newlyUnlocked,
    library,
    committedUnverified: state.committedUnverified + (unverified ? 1 : 0),
    history: [...state.history, newlyUnlocked.length],
    last: {
      itemId: item.id,
      label: item.label,
      phase: "commit",
      detail: unverified
        ? "committed to the skill library without verification"
        : `verified and committed${task.retrievedSkills > 0 ? `, composed from ${task.retrievedSkills}` : ""}`,
    },
  };
  return logged(committed, `unlocked ${item.label}`);
}

function shelveTask(
  state: SimState,
  task: ActiveTask,
  item: TechItem,
): SimState {
  const detail = task.hallucinated
    ? "gave up after 4 rounds: the item does not exist"
    : "gave up after 4 rounds, curriculum will reattempt later";
  const shelved: SimState = {
    ...state,
    phase: "idle",
    active: null,
    failedAttempts: state.failedAttempts + 1,
    attemptedFailures: [...state.attemptedFailures, item.id].slice(-12),
    history: [...state.history, state.unlocked.length],
    last: { itemId: item.id, label: item.label, phase: "verify", detail },
  };
  return logged(shelved, `shelved ${item.label}`);
}

function applyIntervention(state: SimState, action: Intervention): SimState {
  switch (action.type) {
    case "setCurriculum":
      return logged(
        {
          ...state,
          params: { ...state.params, curriculum: action.value },
          active: null,
          phase: "idle",
        },
        `curriculum -> ${action.value}`,
      );
    case "setCoder":
      return logged(
        { ...state, params: { ...state.params, coder: action.value } },
        `coder -> ${action.value}`,
      );
    case "toggleSkillLibrary":
      return logged(
        {
          ...state,
          params: { ...state.params, skillLibrary: !state.params.skillLibrary },
        },
        `skill library ${!state.params.skillLibrary ? "on" : "off"}`,
      );
    case "toggleSelfVerification":
      return logged(
        {
          ...state,
          params: {
            ...state.params,
            selfVerification: !state.params.selfVerification,
          },
        },
        `self-verification ${!state.params.selfVerification ? "on" : "off"}`,
      );
    case "setHallucination":
      return logged(
        {
          ...state,
          params: { ...state.params, hallucinationRate: action.value },
        },
        `hallucination rate -> ${Math.round(action.value * 100)}%`,
      );
    case "forceTask": {
      const item = getItem(action.itemId);
      const started = startTask(state, item, state);
      return logged(started, `forced task -> ${item.label}`);
    }
    case "setSeed":
      return logged(
        { ...state, rng: action.value >>> 0 },
        `seed -> ${action.value}`,
      );
    case "loadPreset": {
      const preset = getPreset(action.id);
      return initialState(preset.id);
    }
    case "restore":
      return logged({ ...action.snapshot }, action.label);
  }
}

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: Params;
  seed: number;
}

export const PRESETS: Preset[] = [
  {
    id: "full",
    label: "Full Voyager",
    blurb:
      "Every part on. The automatic curriculum climbs the tech tree, GPT-4 writes the code, the skill library composes, and self-verification gates every commit.",
    params: {
      curriculum: "automatic",
      coder: "gpt-4",
      skillLibrary: true,
      selfVerification: true,
      hallucinationRate: 0,
    },
    seed: 7,
  },
  {
    id: "no-library",
    label: "No skill library",
    blurb:
      "Same agent, library off. Nothing composes, so each task is coded from scratch and the climb plateaus in the later, harder tiers.",
    params: {
      curriculum: "automatic",
      coder: "gpt-4",
      skillLibrary: false,
      selfVerification: true,
      hallucinationRate: 0,
    },
    seed: 7,
  },
  {
    id: "random-curriculum",
    label: "Random curriculum",
    blurb:
      "The curriculum proposes any reachable task at random instead of climbing in order, so it wastes rounds on tasks the agent isn't ready for.",
    params: {
      curriculum: "random",
      coder: "gpt-4",
      skillLibrary: true,
      selfVerification: true,
      hallucinationRate: 0,
    },
    seed: 7,
  },
  {
    id: "no-verification",
    label: "No self-verification",
    blurb:
      "The critic is off, so any program that runs gets committed, even partial ones. Watch the unverified-skill count climb while real progress slows.",
    params: {
      curriculum: "automatic",
      coder: "gpt-4",
      skillLibrary: true,
      selfVerification: false,
      hallucinationRate: 0,
    },
    seed: 7,
  },
  {
    id: "hallucinating",
    label: "Hallucinating curriculum",
    blurb:
      "The curriculum sometimes proposes items that don't exist, like a copper sword. The coder burns four rounds on them and the agent shelves them.",
    params: {
      curriculum: "automatic",
      coder: "gpt-4",
      skillLibrary: true,
      selfVerification: true,
      hallucinationRate: 0.35,
    },
    seed: 7,
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

export function initialState(presetId = "full"): SimState {
  const preset = getPreset(presetId);
  return {
    params: { ...preset.params },
    rng: preset.seed >>> 0,
    tick: 0,
    phase: "idle",
    active: null,
    library: [],
    unlocked: [],
    attemptedFailures: [],
    committedUnverified: 0,
    iterations: 0,
    failedAttempts: 0,
    history: [0],
    last: null,
    log: emptyLog(),
  };
}

export type VoyagerEvent = SimEvent<SimState>;

// How far up the tree the agent has climbed, named for the readout. The diamond
// tier is the milestone only the full agent reaches in the paper.
export function highestTier(state: SimState): number {
  const have = new Set(state.unlocked);
  return TECH_TREE.reduce(
    (max, item) => (have.has(item.id) ? Math.max(max, item.tier) : max),
    0,
  );
}
