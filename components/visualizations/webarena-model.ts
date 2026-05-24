// A pure, deterministic model of how WebArena scores a web agent. The paper's
// core move is to score by *functional correctness*: run the agent's actions in
// a real, stateful website and check whether the world ended up in the intended
// state, instead of comparing the agent's action string against one reference
// script (the surface-form metric that earlier benchmarks used). This model
// runs a small web world as a state machine. Each action deterministically
// transitions the world. Two reward functions then read the world: a functional
// reward that checks the end state (and intermediate states, the paper's r_prog)
// and a surface-form reward that compares the action sequence to a reference.
// The two disagree exactly where the paper says they should, on a correct
// alternative path. No Math.random and no wall clock reach the model, so the
// same plan reproduces the same run.

// Mulberry32, inlined so this simulation is self-contained. A given seed yields
// the same stream, which is what lets a run be replayed and scrubbed.
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Indexed read that throws on a real out-of-range access instead of quietly
// returning undefined, so the strict compiler keeps reads typed as T.
function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) {
    throw new RangeError(`index ${index} out of range`);
  }
  return value;
}

// The slice of the world an action can touch. A web task in WebArena writes to
// real backing state: a shopping cart, the current page URL, a repository's
// files, posted forum content. We track a small representative set so the reward
// functions have something concrete to read.
export interface WorldState {
  url: string;
  cart: string[];
  repoFiles: Record<string, string>;
  posts: { subreddit: string; body: string }[];
  answer: string | null;
  gaveUp: boolean;
}

function freshWorld(): WorldState {
  return {
    url: "home",
    cart: [],
    repoFiles: {},
    posts: [],
    answer: null,
    gaveUp: false,
  };
}

// One concrete action the agent can issue. This mirrors the paper's compound
// action space (element ops, tab ops, URL navigation) reduced to the verbs each
// task needs. `kind` is the surface form the reference script is written in;
// `apply` is how the action changes the world.
export interface AgentAction {
  kind: string;
  label: string;
  apply: (world: WorldState) => WorldState;
  // True when this action moves the world toward the goal. An off-target action
  // leaves `progress` false, which is how the functional reward ends up failing
  // without any scripted outcome.
  progress: boolean;
}

function step(world: WorldState, mutate: (w: WorldState) => void): WorldState {
  const next: WorldState = {
    ...world,
    cart: [...world.cart],
    repoFiles: { ...world.repoFiles },
    posts: world.posts.map((post) => ({ ...post })),
  };
  mutate(next);
  return next;
}

// A reward function reads a trajectory (the sequence of world states the agent
// produced) and returns 1 or 0. The functional reward checks the end state and,
// where the task demands it, an intermediate state, which is the paper's r_prog
// over s_1..s_T. The surface reward never reads the world; it only compares the
// action kinds to the reference script.
export interface RewardCheck {
  label: string;
  pass: (trajectory: WorldState[]) => boolean;
}

export interface TaskDef {
  id: string;
  title: string;
  intent: string;
  category: "information" | "navigation" | "content";
  achievable: boolean;
  // The reference action script. Surface-form scoring compares the agent's
  // action kinds against this list, in order.
  reference: string[];
  // The outcome checks the functional reward runs against the world the agent
  // produced. Every check must pass for the task to count as solved.
  checks: RewardCheck[];
  // Named plans the agent can run. Each is a list of actions. "reference"
  // reproduces the script exactly; "alternative" reaches the same goal a
  // different way; "offTarget" ends in the wrong state; "giveUp" stops early.
  plans: Record<PlanName, AgentAction[]>;
}

export type PlanName = "reference" | "alternative" | "offTarget" | "giveUp";

const lastUrlIncludes = (needle: string) => (trajectory: WorldState[]) =>
  at(trajectory, trajectory.length - 1).url.includes(needle);

// ----- Task 1: a content task with two valid paths to the same end state. -----
// This is where functional and surface-form scoring split. Both plans put the
// itinerary in the README; only the reference plan matches the reference
// action string.
const itineraryTask: TaskDef = {
  id: "itinerary",
  title: "Log a museum itinerary",
  intent:
    "Find Pittsburgh's art museums on the map and log the visiting order in my travel repo README.",
  category: "content",
  achievable: true,
  reference: ["goto", "goto", "type", "click"],
  checks: [
    {
      label: "README lists all three museums",
      pass: (trajectory) => {
        const readme = at(trajectory, trajectory.length - 1).repoFiles[
          "README"
        ];
        if (readme === undefined) return false;
        return (
          readme.includes("Carnegie") &&
          readme.includes("Mattress") &&
          readme.includes("Warhol")
        );
      },
    },
  ],
  plans: {
    reference: [
      {
        kind: "goto",
        label: "goto(map): look up museums",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "map?q=museums")),
      },
      {
        kind: "goto",
        label: "goto(gitlab): open the repo",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "gitlab/travel")),
      },
      {
        kind: "type",
        label: "type README: Carnegie, Mattress, Warhol",
        progress: true,
        apply: (w) =>
          step(
            w,
            (n) => (n.repoFiles["README"] = "Carnegie -> Mattress -> Warhol"),
          ),
      },
      {
        kind: "click",
        label: "click(Commit)",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "gitlab/travel?committed")),
      },
    ],
    // Same end state, different surface form: the agent uses the map's search
    // box and the GitLab web IDE instead of a direct goto plus a type. The
    // README ends up identical, so the functional reward passes while the action
    // kinds no longer line up with the reference.
    alternative: [
      {
        kind: "click",
        label: "click(search) then type museums",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "map?q=museums")),
      },
      {
        kind: "click",
        label: "click(repo link): open the repo",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "gitlab/travel")),
      },
      {
        kind: "click",
        label: "click(Web IDE)",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "gitlab/travel/ide")),
      },
      {
        kind: "type",
        label: "type README: Carnegie, Mattress, Warhol",
        progress: true,
        apply: (w) =>
          step(
            w,
            (n) => (n.repoFiles["README"] = "Carnegie -> Mattress -> Warhol"),
          ),
      },
    ],
    offTarget: [
      {
        kind: "goto",
        label: "goto(map): look up museums",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "map?q=museums")),
      },
      {
        kind: "goto",
        label: "goto(gitlab): open the repo",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "gitlab/travel")),
      },
      {
        kind: "type",
        label: "type README: only Carnegie (missed two)",
        progress: false,
        apply: (w) => step(w, (n) => (n.repoFiles["README"] = "Carnegie")),
      },
      {
        kind: "click",
        label: "click(Commit)",
        progress: false,
        apply: (w) => step(w, (n) => (n.url = "gitlab/travel?committed")),
      },
    ],
    giveUp: [
      {
        kind: "goto",
        label: "goto(map): look up museums",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "map?q=museums")),
      },
      {
        kind: "stop",
        label: "stop: 'this looks impossible'",
        progress: false,
        apply: (w) => step(w, (n) => (n.gaveUp = true)),
      },
    ],
  },
};

// ----- Task 2: a content/transaction task (place an order). -----
const orderTask: TaskDef = {
  id: "order",
  title: "Order an ergonomic chair",
  intent: "Buy the ergonomic chair with the best rating.",
  category: "content",
  achievable: true,
  reference: ["goto", "click", "click"],
  checks: [
    {
      label: "cart holds the ergonomic chair",
      pass: (trajectory) =>
        at(trajectory, trajectory.length - 1).cart.includes("ergo-chair"),
    },
    {
      label: "checkout page reached",
      pass: lastUrlIncludes("checkout"),
    },
  ],
  plans: {
    reference: [
      {
        kind: "goto",
        label: "goto(shop): chairs",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "shop/chairs")),
      },
      {
        kind: "click",
        label: "click(Add to Cart) on top-rated chair",
        progress: true,
        apply: (w) =>
          step(w, (n) => {
            n.cart.push("ergo-chair");
            n.url = "shop/cart";
          }),
      },
      {
        kind: "click",
        label: "click(Checkout)",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "shop/checkout")),
      },
    ],
    // A valid second path: sort by rating first, then add, then check out. The
    // cart and checkout end identical; the surface form gains a sort click.
    alternative: [
      {
        kind: "click",
        label: "click(Sort by rating)",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "shop/chairs?sort=rating")),
      },
      {
        kind: "click",
        label: "click(Add to Cart) on top result",
        progress: true,
        apply: (w) =>
          step(w, (n) => {
            n.cart.push("ergo-chair");
            n.url = "shop/cart";
          }),
      },
      {
        kind: "click",
        label: "click(Checkout)",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "shop/checkout")),
      },
    ],
    offTarget: [
      {
        kind: "goto",
        label: "goto(shop): chairs",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "shop/chairs")),
      },
      {
        kind: "click",
        label: "click(Add to Cart) on the cheap stool",
        progress: false,
        apply: (w) =>
          step(w, (n) => {
            n.cart.push("stool");
            n.url = "shop/cart";
          }),
      },
      {
        kind: "click",
        label: "click(Checkout)",
        progress: false,
        apply: (w) => step(w, (n) => (n.url = "shop/checkout")),
      },
    ],
    giveUp: [
      {
        kind: "stop",
        label: "stop: 'no chair found'",
        progress: false,
        apply: (w) => step(w, (n) => (n.gaveUp = true)),
      },
    ],
  },
};

// ----- Task 3: an unachievable task (the contact-number example). -----
// WebArena labels some intents N/A: the site genuinely lacks the data. The right
// behavior is to stop and say so, which only happens when the UA hint is on.
const unachievableTask: TaskDef = {
  id: "unachievable",
  title: "Find OneStopShop's phone number",
  intent: "Tell me the contact phone number of OneStopShop.",
  category: "information",
  achievable: false,
  reference: ["stop"],
  checks: [
    {
      label: "agent answered N/A (no phone exists on the site)",
      pass: (trajectory) =>
        at(trajectory, trajectory.length - 1).answer === "N/A",
    },
  ],
  plans: {
    // With the UA hint, the agent recognizes there is no contact page and stops
    // with N/A, which is the only correct outcome.
    reference: [
      {
        kind: "goto",
        label: "goto(shop): look for a contact page",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "shop")),
      },
      {
        kind: "stop",
        label: "answer: 'N/A' (no contact info on the site)",
        progress: true,
        apply: (w) => step(w, (n) => (n.answer = "N/A")),
      },
    ],
    alternative: [
      {
        kind: "click",
        label: "click(Help) looking for contact",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "shop/help")),
      },
      {
        kind: "stop",
        label: "answer: 'N/A' (no contact info on the site)",
        progress: true,
        apply: (w) => step(w, (n) => (n.answer = "N/A")),
      },
    ],
    // Without recognizing impossibility the agent hallucinates a number.
    offTarget: [
      {
        kind: "goto",
        label: "goto(shop): look for a contact page",
        progress: true,
        apply: (w) => step(w, (n) => (n.url = "shop")),
      },
      {
        kind: "stop",
        label: "answer: '1-800-555-0199' (made up)",
        progress: false,
        apply: (w) => step(w, (n) => (n.answer = "1-800-555-0199")),
      },
    ],
    giveUp: [
      {
        kind: "stop",
        label: "answer: 'N/A'",
        progress: true,
        apply: (w) => step(w, (n) => (n.answer = "N/A")),
      },
    ],
  },
};

export const TASKS: TaskDef[] = [itineraryTask, orderTask, unachievableTask];

export function getTask(id: string): TaskDef {
  return TASKS.find((task) => task.id === id) ?? at(TASKS, 0);
}

// The agent's behavior on a feasible task when the UA hint is on. The paper's
// error analysis found GPT-4 calls 54.9% of feasible tasks impossible once the
// hint is present, so on a feasible task the hint can flip a good plan into an
// early give-up. We model that as a probability the agent bails, drawn from the
// seeded rng so a seed reproduces the run.
const FALSE_IMPOSSIBLE_RATE = 0.549;

// ----- Reward evaluation. -----
export interface RewardResult {
  functional: number;
  surface: number;
  checks: { label: string; pass: boolean }[];
  surfaceDiff: { ref: string; got: string; match: boolean }[];
}

function evaluateFunctional(
  task: TaskDef,
  trajectory: WorldState[],
): { score: number; checks: { label: string; pass: boolean }[] } {
  if (trajectory.length === 0) {
    return {
      score: 0,
      checks: task.checks.map((check) => ({ label: check.label, pass: false })),
    };
  }
  const checks = task.checks.map((check) => ({
    label: check.label,
    pass: check.pass(trajectory),
  }));
  const score = checks.every((check) => check.pass) ? 1 : 0;
  return { score, checks };
}

// Surface-form scoring: an exact, ordered comparison of action kinds against the
// reference script. This is the metric the paper argues against. It penalizes a
// correct alternative path because the strings differ.
function evaluateSurface(
  task: TaskDef,
  actions: AgentAction[],
): { score: number; diff: { ref: string; got: string; match: boolean }[] } {
  const length = Math.max(task.reference.length, actions.length);
  const diff: { ref: string; got: string; match: boolean }[] = [];
  let allMatch = actions.length === task.reference.length;
  for (let i = 0; i < length; i++) {
    const ref = task.reference[i] ?? "-";
    const got = actions[i]?.kind ?? "-";
    const match = ref === got;
    if (!match) allMatch = false;
    diff.push({ ref, got, match });
  }
  return { score: allMatch ? 1 : 0, diff };
}

export function evaluateRun(
  task: TaskDef,
  actions: AgentAction[],
  trajectory: WorldState[],
): RewardResult {
  const functional = evaluateFunctional(task, trajectory);
  const surface = evaluateSurface(task, actions);
  return {
    functional: functional.score,
    surface: surface.score,
    checks: functional.checks,
    surfaceDiff: surface.diff,
  };
}

// ----- The stepped simulation. -----
export type Metric = "functional" | "surface";

export interface SimParams {
  taskId: string;
  plan: PlanName;
  uaHint: boolean;
  metric: Metric;
  seed: number;
}

export interface LogEntry {
  id: number;
  step: number;
  label: string;
  snapshot: SimParams;
}

export interface SimState {
  params: SimParams;
  cursor: number; // how many actions of the current plan have executed
  trajectory: WorldState[]; // world after each executed action, plus the start
  actions: AgentAction[]; // the actions executed so far
  finished: boolean;
  log: LogEntry[];
  nextLogId: number;
}

export type Intervention =
  | { type: "setTask"; id: string }
  | { type: "setPlan"; plan: PlanName }
  | { type: "toggleUaHint" }
  | { type: "setMetric"; metric: Metric }
  | { type: "setSeed"; value: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; params: SimParams; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

// Resolve which plan the agent actually runs. The UA hint is the paper's lever.
// On an unachievable task the hint is needed to stop correctly; with it off the
// agent hallucinates (offTarget). On a feasible task the hint can backfire: a
// fraction of the time the agent wrongly decides the task is impossible and
// bails early, which is the 54.9% false-impossible behavior.
function resolvePlan(params: SimParams): {
  plan: PlanName;
  note: string;
} {
  const task = getTask(params.taskId);
  if (!task.achievable) {
    if (params.uaHint) {
      return {
        plan: "reference",
        note: "UA hint on: agent can stop with N/A.",
      };
    }
    return {
      plan: "offTarget",
      note: "UA hint off: agent has no way to declare impossible, so it hallucinates.",
    };
  }
  // Feasible task. The reader's chosen plan stands, unless the UA hint trips the
  // agent into a false give-up.
  if (params.uaHint && params.plan !== "giveUp") {
    const rng = mulberry32(params.seed * 1000 + 7);
    if (rng() < FALSE_IMPOSSIBLE_RATE) {
      return {
        plan: "giveUp",
        note: "UA hint on a feasible task: agent wrongly called it impossible and stopped.",
      };
    }
    return { plan: params.plan, note: "UA hint on, but the agent kept going." };
  }
  return { plan: params.plan, note: "" };
}

export function getResolvedPlan(params: SimParams): {
  plan: PlanName;
  note: string;
  actions: AgentAction[];
} {
  const { plan, note } = resolvePlan(params);
  const task = getTask(params.taskId);
  return { plan, note, actions: task.plans[plan] };
}

export function initialState(params: SimParams): SimState {
  return {
    params,
    cursor: 0,
    trajectory: [freshWorld()],
    actions: [],
    finished: false,
    log: [],
    nextLogId: 1,
  };
}

function reset(params: SimParams): SimState {
  return initialState(params);
}

function logged(state: SimState, label: string): SimState {
  const entry: LogEntry = {
    id: state.nextLogId,
    step: state.cursor,
    label,
    snapshot: state.params,
  };
  return {
    ...state,
    log: [...state.log, entry].slice(-60),
    nextLogId: state.nextLogId + 1,
  };
}

// Advance the agent by one action. The world transitions deterministically; the
// trajectory grows. When the plan runs out, the run finishes and the rewards can
// be read off the final trajectory.
function tick(state: SimState): SimState {
  if (state.finished) return state;
  const { actions } = getResolvedPlan(state.params);
  if (state.cursor >= actions.length) {
    return logged(
      { ...state, finished: true },
      "run finished: rewards evaluated",
    );
  }
  const action = at(actions, state.cursor);
  const world = at(state.trajectory, state.trajectory.length - 1);
  const nextWorld = action.apply(world);
  const advanced: SimState = {
    ...state,
    cursor: state.cursor + 1,
    trajectory: [...state.trajectory, nextWorld],
    actions: [...state.actions, action],
  };
  const finishedNow = advanced.cursor >= actions.length;
  return logged(
    { ...advanced, finished: finishedNow },
    finishedNow ? `${action.label}  (run finished)` : action.label,
  );
}

function applyIntervention(
  params: SimParams,
  action: Intervention,
): { params: SimParams; label: string } {
  switch (action.type) {
    case "setTask": {
      const task = getTask(action.id);
      return {
        params: { ...params, taskId: action.id },
        label: `task -> ${task.title}`,
      };
    }
    case "setPlan":
      return {
        params: { ...params, plan: action.plan },
        label: `plan -> ${action.plan}`,
      };
    case "toggleUaHint": {
      const next = !params.uaHint;
      return {
        params: { ...params, uaHint: next },
        label: `UA hint ${next ? "on" : "off"}`,
      };
    }
    case "setMetric":
      return {
        params: { ...params, metric: action.metric },
        label: `metric -> ${action.metric}`,
      };
    case "setSeed":
      return {
        params: { ...params, seed: action.value },
        label: `seed -> ${action.value}`,
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

// The pure reducer. A tick advances the agent one action. Any intervention
// changes a parameter and resets the run from a fresh world, because the metric,
// plan, task, or hint determines the whole trajectory from the start. Every step
// is deterministic given the params and the seed.
export function reduce(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return tick(state);
  const { params, label } = applyIntervention(state.params, input.action);
  // A view-only metric switch shouldn't throw away the run.
  if (input.action.type === "setMetric") {
    return logged({ ...state, params }, label);
  }
  const fresh = reset(params);
  return logged(fresh, label);
}

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: SimParams;
}

export const PRESETS: Preset[] = [
  {
    id: "alt-path",
    label: "Valid alternative path",
    blurb:
      "A feasible content task run a different but correct way. Watch the functional reward pass while the surface-form metric fails, because the action string no longer matches the reference.",
    params: {
      taskId: "itinerary",
      plan: "alternative",
      uaHint: false,
      metric: "functional",
      seed: 1,
    },
  },
  {
    id: "off-target",
    label: "Off-target failure",
    blurb:
      "The agent commits the README but misses two museums. The end state is wrong, so the functional reward fails even though the agent finished without error.",
    params: {
      taskId: "itinerary",
      plan: "offTarget",
      uaHint: false,
      metric: "functional",
      seed: 1,
    },
  },
  {
    id: "unachievable",
    label: "Unachievable, UA hint on",
    blurb:
      "The site has no contact number. With the UA hint the agent stops and answers N/A, the only correct outcome. Turn the hint off and it hallucinates a number.",
    params: {
      taskId: "unachievable",
      plan: "reference",
      uaHint: true,
      metric: "functional",
      seed: 1,
    },
  },
  {
    id: "false-impossible",
    label: "UA hint backfires",
    blurb:
      "A feasible order task with the UA hint on. About 55% of the time the agent wrongly calls it impossible and gives up. Nudge the seed and watch the run flip between finishing and bailing.",
    params: {
      taskId: "order",
      plan: "reference",
      uaHint: true,
      metric: "functional",
      seed: 2,
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}
