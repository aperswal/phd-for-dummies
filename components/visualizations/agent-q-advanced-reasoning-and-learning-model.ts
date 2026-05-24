// A pure, deterministic model of the Agent Q search loop from "Agent Q: Advanced
// Reasoning and Learning for Autonomous AI Agents". The mechanism is the paper's
// own: a single base LLM plays both the actor (it proposes K candidate actions at
// each web page) and the critic (it ranks those actions, giving AI process
// feedback). Monte Carlo Tree Search runs over web pages, selecting nodes by the
// UCB1 rule of Eq. 7, rolling a trajectory out to a terminal page, and
// backpropagating the binary outcome reward by Eq. 8. The value that drives
// selection is the blend of Eq. 10, Q = alpha * Qtilde + (1 - alpha) * Qhat,
// where Qtilde is the empirical MCTS value and Qhat is the critic's rank score.
// Sibling actions whose blended values differ by more than a threshold become
// the preferred/rejected pairs the policy is later fine-tuned on with DPO.
//
// The web task is a fixed booking tree (book a restaurant reservation), modelled
// after the paper's OpenTable figures. The actor's proposal scores, the critic's
// rank scores, and each action's true reach-the-goal probability are hand-set so
// a reader can recognise the search finding the correct path, a greedy run
// stalling in a wrong branch, and a misleading critic dragging the search off
// course. A trained agent would learn those numbers; here they are fixed so the
// behaviour is legible. All randomness flows through a seeded rng threaded in the
// state, so a given seed reproduces the run exactly.

import { at } from "@/lib/simulation/array";
import { emptyLog, record, type EventLog } from "@/lib/simulation/history";
import { nextRandom } from "@/lib/simulation/rng";

// One candidate action the actor proposed at a page. `proposal` is how strongly
// the actor wants it (its sampling weight); `criticRank` is the critic's process
// feedback score Qhat in [0, 1]; `reach` is the true probability that committing
// to this action eventually reaches the goal, used by the rollout. `child` names
// the page this action leads to.
export interface ActionSpec {
  id: string;
  label: string;
  command: string;
  proposal: number;
  criticRank: number;
  reach: number;
  child: string;
}

// A page in the booking task. `terminal` marks a leaf the rollout stops at;
// `success` marks the one page that completes the booking.
export interface PageSpec {
  id: string;
  label: string;
  note: string;
  terminal: boolean;
  success: boolean;
  actions: ActionSpec[];
}

// The fixed booking task, modelled on the paper's OpenTable run: the agent starts
// on a search page, can land on the wrong restaurant or the right one, then must
// open the date selector, fix the date, and complete the reservation. The two
// FAILURE pages mirror Figure 1's red nodes; the SUCCESS page is the green one.
const TASK_PAGES: PageSpec[] = [
  {
    id: "search",
    label: "OpenTable search",
    note: "The agent opens OpenTable and must reach the right restaurant page.",
    terminal: false,
    success: false,
    actions: [
      {
        id: "to-wrong",
        label: "Open the wrong restaurant",
        command: "CLICK first-result",
        proposal: 0.6,
        criticRank: 0.35,
        reach: 0.1,
        child: "wrong-restaurant",
      },
      {
        id: "to-right",
        label: "Open the correct restaurant",
        command: "CLICK matched-result",
        proposal: 0.4,
        criticRank: 0.8,
        reach: 0.85,
        child: "right-restaurant",
      },
    ],
  },
  {
    id: "wrong-restaurant",
    label: "Wrong restaurant",
    note: "A plausible but incorrect restaurant. The booking can never complete here.",
    terminal: false,
    success: false,
    actions: [
      {
        id: "wrong-book",
        label: "Try to book here anyway",
        command: "CLICK find-a-table",
        proposal: 0.7,
        criticRank: 0.3,
        reach: 0.0,
        child: "wrong-fail",
      },
      {
        id: "wrong-home",
        label: "Go back to the homepage",
        command: "GOTO opentable.com",
        proposal: 0.3,
        criticRank: 0.45,
        reach: 0.0,
        child: "homepage-fail",
      },
    ],
  },
  {
    id: "right-restaurant",
    label: "Correct restaurant",
    note: "The right restaurant. The date is wrong by default and must be fixed.",
    terminal: false,
    success: false,
    actions: [
      {
        id: "pick-wrong-date",
        label: "Submit with the default date",
        command: "CLICK find-a-table",
        proposal: 0.55,
        criticRank: 0.3,
        reach: 0.05,
        child: "date-fail",
      },
      {
        id: "open-date",
        label: "Open the date selector",
        command: "CLICK date-selector",
        proposal: 0.45,
        criticRank: 0.78,
        reach: 0.92,
        child: "date-selector",
      },
    ],
  },
  {
    id: "date-selector",
    label: "Date selector",
    note: "The calendar is open. Pick the date the user asked for, then complete.",
    terminal: false,
    success: false,
    actions: [
      {
        id: "set-date",
        label: "Select the requested date",
        command: "CLICK requested-date",
        proposal: 0.5,
        criticRank: 0.82,
        reach: 0.97,
        child: "complete",
      },
      {
        id: "close-date",
        label: "Close without changing the date",
        command: "CLICK close",
        proposal: 0.5,
        criticRank: 0.25,
        reach: 0.04,
        child: "date-fail",
      },
    ],
  },
  {
    id: "complete",
    label: "Reservation booked",
    note: "Correct restaurant, correct date, reservation confirmed. Reward 1.",
    terminal: true,
    success: true,
    actions: [],
  },
  {
    id: "wrong-fail",
    label: "Booked wrong restaurant",
    note: "A reservation at the wrong restaurant. Reward 0.",
    terminal: true,
    success: false,
    actions: [],
  },
  {
    id: "homepage-fail",
    label: "Lost on homepage",
    note: "Wandered back to the homepage out of steps. Reward 0.",
    terminal: true,
    success: false,
    actions: [],
  },
  {
    id: "date-fail",
    label: "Wrong date booked",
    note: "Right restaurant, wrong date, so the task fails. Reward 0.",
    terminal: true,
    success: false,
    actions: [],
  },
];

function getPage(id: string): PageSpec {
  const page = TASK_PAGES.find((p) => p.id === id);
  if (page === undefined) throw new RangeError(`unknown page ${id}`);
  return page;
}

// A node in the search tree. It wraps one page reached by a specific path, and
// tracks the MCTS bookkeeping: how many of the K proposed actions are expanded,
// the empirical value Qtilde and visit count N per action (Eq. 8), and whether
// the critic at this node has been poisoned by the reader.
export interface TreeNode {
  id: number;
  parentId: number | null;
  pageId: string;
  depth: number;
  visits: number;
  // Per proposed action, the running MCTS value (Qtilde) and the child node id
  // once the action has been expanded.
  actionVisits: number[];
  actionValue: number[];
  childIds: (number | null)[];
  poisoned: boolean;
  dead: boolean;
}

// The search is either waiting to start, mid-run, or finished. The view reads
// "done" to stop offering more steps once the budget is spent.
export type Phase = "idle" | "running" | "done";

export interface PrefPair {
  nodeId: number;
  pageLabel: string;
  preferred: string;
  rejected: string;
  gap: number;
}

export interface Params {
  cExp: number; // UCB1 exploration constant
  alpha: number; // Eq. 10 blend: weight on the empirical MCTS value
  threshold: number; // DPO pair gap threshold (theta in Algorithm 1)
  budget: number; // number of MCTS rollouts to run
  rolloutNoise: number; // how stochastic the rollout policy is (0 = greedy)
}

export interface SimState {
  rng: number;
  params: Params;
  nodes: TreeNode[];
  rootId: number;
  phase: Phase;
  // The path of node ids selected this iteration, so the view can light the
  // branch the last rollout took.
  selectedPath: number[];
  iteration: number;
  successes: number;
  prefPairs: PrefPair[];
  log: EventLog<string>;
}

export type Intervention =
  | { type: "setCExp"; value: number }
  | { type: "setAlpha"; value: number }
  | { type: "setThreshold"; value: number }
  | { type: "setBudget"; value: number }
  | { type: "setRolloutNoise"; value: number }
  | { type: "poisonNode"; nodeId: number }
  | { type: "killNode"; nodeId: number }
  | { type: "loadPreset"; id: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

// The blended value of Eq. 10. Qtilde is the empirical MCTS estimate for the
// action; Qhat is the critic rank. A poisoned node flips the critic so its best
// action looks worst, which is how the reader injects a misleading critic.
export function blendedValue(
  node: TreeNode,
  actionIndex: number,
  alpha: number,
): number {
  const page = getPage(node.pageId);
  const action = at(page.actions, actionIndex);
  const empirical = at(node.actionValue, actionIndex);
  const criticRank = node.poisoned ? 1 - action.criticRank : action.criticRank;
  return alpha * empirical + (1 - alpha) * criticRank;
}

function rand(state: SimState): { value: number; state: SimState } {
  const next = nextRandom(state.rng);
  return { value: next.value, state: { ...state, rng: next.state } };
}

function findNode(state: SimState, id: number): TreeNode {
  const node = state.nodes.find((n) => n.id === id);
  if (node === undefined) throw new RangeError(`unknown node ${id}`);
  return node;
}

function makeNode(
  id: number,
  parentId: number | null,
  pageId: string,
  depth: number,
): TreeNode {
  const page = getPage(pageId);
  const k = page.actions.length;
  return {
    id,
    parentId,
    pageId,
    depth,
    visits: 0,
    actionVisits: new Array<number>(k).fill(0),
    actionValue: new Array<number>(k).fill(0),
    childIds: new Array<number | null>(k).fill(null),
    poisoned: false,
    dead: false,
  };
}

const ROOT_PAGE = "search";

export function initialState(params: Params, seed = 7): SimState {
  const root = makeNode(0, null, ROOT_PAGE, 0);
  return {
    rng: seed >>> 0,
    params,
    nodes: [root],
    rootId: 0,
    phase: "idle",
    selectedPath: [],
    iteration: 0,
    successes: 0,
    prefPairs: [],
    log: emptyLog(),
  };
}

function expandedCount(node: TreeNode): number {
  let count = 0;
  for (const id of node.childIds) if (id !== null) count += 1;
  return count;
}

// UCB1 of Eq. 7 with the blended exploit term wired to the live alpha. An
// expanded action scores its blended value plus an exploration bonus that grows
// with parent visits and shrinks with this action's visits. An unexpanded action
// is a frontier the agent can open. Opening the very first action at a node is
// free, ordered by the actor's raw proposal (the only signal before any rollout
// evidence). Opening any further unexpanded sibling costs exploration: its bonus
// scales with c_exp, so with c_exp set to zero the agent commits to whatever it
// opened first and never tries the alternative, which is the paper's greedy
// failure. A dead child (one the reader killed) drops out so it is never picked.
function ucb1(node: TreeNode, actionIndex: number, params: Params): number {
  const page = getPage(node.pageId);
  const action = at(page.actions, actionIndex);
  const childId = at(node.childIds, actionIndex);
  if (childId !== null) {
    const exploit = blendedValue(node, actionIndex, params.alpha);
    const childVisits = at(node.actionVisits, actionIndex);
    const explore =
      params.cExp * Math.sqrt(Math.log(node.visits + 1) / (1 + childVisits));
    return exploit + explore;
  }
  if (expandedCount(node) === 0) {
    // First action ever opened here: take the actor's most-proposed action.
    return 5 + action.proposal;
  }
  // A further unexpanded sibling. Only worth opening if exploration is on.
  return params.cExp * (1 + action.proposal);
}

// Pick the UCB1-best action at a node, skipping any action whose expanded child
// the reader killed. Ties break toward the lower index, which keeps the run
// deterministic.
function bestActionByUcb(state: SimState, node: TreeNode): number {
  const page = getPage(node.pageId);
  let bestIndex = -1;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < page.actions.length; i++) {
    const childId = at(node.childIds, i);
    if (childId !== null && findNode(state, childId).dead) continue;
    const score = ucb1(node, i, state.params);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }
  return bestIndex;
}

interface SelectResult {
  state: SimState;
  path: number[];
  leafId: number;
  actionIndex: number;
}

// Selection phase: descend from the root following UCB1 until we reach an action
// that has not been expanded yet (or a terminal page, or a node with no live
// action left because the reader killed its only branch). Returns the path of
// node ids and the action index to expand at the leaf. A guard caps the descent
// so a fully-explored subtree can never loop forever.
function select(state: SimState): SelectResult {
  const path: number[] = [state.rootId];
  let currentId = state.rootId;
  let guard = 0;
  while (guard < 32) {
    guard += 1;
    const node = findNode(state, currentId);
    const page = getPage(node.pageId);
    if (page.terminal) {
      return { state, path, leafId: currentId, actionIndex: -1 };
    }
    const actionIndex = bestActionByUcb(state, node);
    if (actionIndex === -1) {
      // Every action here is dead. Re-run the rollout from this node anyway so
      // the iteration still produces a (failing) outcome rather than stalling.
      return { state, path, leafId: currentId, actionIndex: -1 };
    }
    const childId = at(node.childIds, actionIndex);
    if (childId === null) {
      return { state, path, leafId: currentId, actionIndex };
    }
    currentId = childId;
    path.push(currentId);
  }
  return { state, path, leafId: currentId, actionIndex: -1 };
}

// Expansion: create the child node for the chosen action at the leaf.
function expand(
  state: SimState,
  leafId: number,
  actionIndex: number,
): { state: SimState; childId: number } {
  const leaf = findNode(state, leafId);
  const page = getPage(leaf.pageId);
  const action = at(page.actions, actionIndex);
  const newId = state.nodes.length;
  const child = makeNode(newId, leafId, action.child, leaf.depth + 1);
  const nodes = state.nodes.map((n) => {
    if (n.id !== leafId) return n;
    const childIds = [...n.childIds];
    childIds[actionIndex] = newId;
    return { ...n, childIds };
  });
  return { state: { ...state, nodes: [...nodes, child] }, childId: newId };
}

// Rollout: from the expanded child, follow the policy to a terminal page and
// return the binary reward. The policy samples actions weighted by their
// proposal score, biased by `reach` (a good action is more likely to be taken),
// with rolloutNoise controlling how greedy it is. Each action's `reach` is the
// true chance it leads to the goal, so a low-reach branch usually ends in
// failure. Dead actions are skipped.
function rollout(
  startState: SimState,
  childId: number,
): { state: SimState; reward: number; pages: string[] } {
  let state = startState;
  let node = findNode(state, childId);
  const pages: string[] = [getPage(node.pageId).label];
  let guard = 0;
  while (guard < 12) {
    guard += 1;
    const page = getPage(node.pageId);
    if (page.terminal) {
      return { state, reward: page.success ? 1 : 0, pages };
    }
    const weights = page.actions.map((action, i) => {
      const childNodeId = node.childIds[i] ?? null;
      if (childNodeId !== null && findNode(state, childNodeId).dead) return 0;
      const greedyPull = 1 - state.params.rolloutNoise;
      return action.proposal * (1 - greedyPull) + action.reach * greedyPull;
    });
    const total = weights.reduce((s, w) => s + w, 0);
    const draw = rand(state);
    state = draw.state;
    let threshold = draw.value * (total || 1);
    let chosen = 0;
    for (let i = 0; i < weights.length; i++) {
      threshold -= at(weights, i);
      if (threshold <= 0) {
        chosen = i;
        break;
      }
    }
    const action = at(page.actions, chosen);
    // Whether the step actually advances toward the goal is gated by `reach`:
    // even a good action can slip, and a bad action almost always lands in a
    // failure page.
    const slip = rand(state);
    state = slip.state;
    const nextPageId =
      slip.value < action.reach ? action.child : failureChild(page, action);
    pages.push(getPage(nextPageId).label);
    node = makeNode(-1, null, nextPageId, node.depth + 1);
  }
  return { state, reward: 0, pages };
}

// When a rollout step slips, it lands on the nearest failure page reachable from
// the current page, so the rollout still terminates honestly.
function failureChild(page: PageSpec, action: ActionSpec): string {
  if (getPage(action.child).terminal) return action.child;
  const fail = TASK_PAGES.find(
    (p) => p.terminal && !p.success && reachableFailureFor(page.id) === p.id,
  );
  return fail?.id ?? action.child;
}

function reachableFailureFor(pageId: string): string {
  switch (pageId) {
    case "search":
      return "homepage-fail";
    case "wrong-restaurant":
      return "wrong-fail";
    case "right-restaurant":
      return "date-fail";
    case "date-selector":
      return "date-fail";
    default:
      return "homepage-fail";
  }
}

// Backpropagation of Eq. 8: walk the selected path bottom-up, incrementing visit
// counts and folding the rollout reward into each action's running mean Qtilde.
function backprop(
  state: SimState,
  path: number[],
  actionAtLeaf: number,
  reward: number,
): SimState {
  const nodes = state.nodes.map((n) => ({ ...n }));
  // Walk the selected path from leaf to root, folding the reward into each
  // node's visit count and the running mean of the action it took next.
  for (let depth = path.length - 1; depth >= 0; depth--) {
    const nodeId = at(path, depth);
    const node = nodes.find((n) => n.id === nodeId);
    if (node === undefined) continue;
    node.visits += 1;
    // The action taken from this node toward the next node in the path.
    const actionIndex =
      depth === path.length - 1
        ? actionAtLeaf
        : node.childIds.findIndex((cid) => cid === at(path, depth + 1));
    if (actionIndex >= 0 && actionIndex < node.actionVisits.length) {
      const n = at(node.actionVisits, actionIndex);
      const q = at(node.actionValue, actionIndex);
      node.actionValue[actionIndex] = (q * n + reward) / (n + 1);
      node.actionVisits[actionIndex] = n + 1;
    }
  }
  return { ...state, nodes };
}

// After a round of search, form DPO preference pairs from sibling actions whose
// blended values differ by more than the threshold (Algorithm 1, Eq. 10). The
// higher-valued action is preferred, the lower rejected.
function buildPrefPairs(state: SimState): PrefPair[] {
  const pairs: PrefPair[] = [];
  for (const node of state.nodes) {
    const page = getPage(node.pageId);
    if (page.actions.length < 2) continue;
    if (node.visits === 0) continue;
    const values = page.actions.map((_, i) =>
      blendedValue(node, i, state.params.alpha),
    );
    for (let i = 0; i < page.actions.length; i++) {
      for (let j = i + 1; j < page.actions.length; j++) {
        const gap = Math.abs(at(values, i) - at(values, j));
        if (gap < state.params.threshold) continue;
        const iWins = at(values, i) >= at(values, j);
        const win = iWins ? i : j;
        const lose = iWins ? j : i;
        pairs.push({
          nodeId: node.id,
          pageLabel: page.label,
          preferred: at(page.actions, win).label,
          rejected: at(page.actions, lose).label,
          gap,
        });
      }
    }
  }
  return pairs;
}

function logged(state: SimState, label: string): SimState {
  return {
    ...state,
    log: record(state.log, state.iteration, label, label),
  };
}

// One full MCTS iteration runs as one tick: select, expand, roll out, backprop,
// then rebuild the preference pairs. A tick always completes a whole rollout, so
// the tree grows by at most one node per step and the reader can step through the
// search one rollout at a time.
function runIteration(state: SimState): SimState {
  if (state.iteration >= state.params.budget) {
    return { ...state, phase: "done" };
  }
  const selection = select(state);
  let working = selection.state;
  const leaf = findNode(working, selection.leafId);
  const leafPage = getPage(leaf.pageId);

  let childId = selection.leafId;
  const path = [...selection.path];

  if (!leafPage.terminal && selection.actionIndex >= 0) {
    const expansion = expand(working, selection.leafId, selection.actionIndex);
    working = expansion.state;
    childId = expansion.childId;
    path.push(childId);
  }

  const roll = rollout(working, childId);
  working = roll.state;

  // Backprop along the selected path (the nodes that existed before this
  // expansion). The action taken at the original leaf is the one we just
  // expanded; deeper nodes infer their action from the path. A terminal leaf
  // reached with no expansion contributes no action update, only its visit.
  working = backprop(
    working,
    selection.path,
    selection.actionIndex,
    roll.reward,
  );

  const successes = working.successes + (roll.reward >= 1 ? 1 : 0);
  const iteration = working.iteration + 1;
  working = {
    ...working,
    iteration,
    successes,
    selectedPath: path,
    phase: iteration >= working.params.budget ? "done" : "running",
  };
  working = { ...working, prefPairs: buildPrefPairs(working) };
  const outcome = roll.reward >= 1 ? "goal reached" : "failed";
  return logged(
    working,
    `iter ${iteration}: ${roll.pages.join(" -> ")} (${outcome})`,
  );
}

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: Params;
  seed: number;
  poison?: string[]; // page ids whose critic starts poisoned
}

export const PRESETS: Preset[] = [
  {
    id: "guided",
    label: "Guided search",
    blurb:
      "A trustworthy critic and balanced exploration. The search threads the correct path (right restaurant, open date, set date) and the Q-values concentrate there.",
    params: {
      cExp: 1.2,
      alpha: 0.5,
      threshold: 0.2,
      budget: 24,
      rolloutNoise: 0.3,
    },
    seed: 7,
  },
  {
    id: "greedy",
    label: "Greedy DPO trap",
    blurb:
      "Exploration off, so the agent commits to whatever looks best first and never tries the alternative. This is the paper's greedy-search failure: it stalls in the wrong branch and rarely books.",
    params: {
      cExp: 0,
      alpha: 0.95,
      threshold: 0.2,
      budget: 24,
      rolloutNoise: 0.05,
    },
    seed: 7,
  },
  {
    id: "misleading",
    label: "Misleading critic",
    blurb:
      "The critic at the search page is poisoned, so it ranks the wrong restaurant highest, and alpha leans on the critic. Watch the search waste its budget down the dead branch.",
    params: {
      cExp: 1.0,
      alpha: 0.2,
      threshold: 0.2,
      budget: 24,
      rolloutNoise: 0.3,
    },
    seed: 7,
    poison: ["search"],
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((p) => p.id === id) ?? at(PRESETS, 0);
}

export function presetState(id: string): SimState {
  const preset = getPreset(id);
  let state = initialState(preset.params, preset.seed);
  if (preset.poison) {
    const poison = new Set(preset.poison);
    state = {
      ...state,
      nodes: state.nodes.map((n) =>
        poison.has(n.pageId) ? { ...n, poisoned: true } : n,
      ),
    };
  }
  return logged(state, `preset -> ${preset.label}`);
}

function applyIntervention(state: SimState, action: Intervention): SimState {
  switch (action.type) {
    case "setCExp":
      return logged(
        { ...state, params: { ...state.params, cExp: action.value } },
        `c_exp -> ${action.value.toFixed(2)}`,
      );
    case "setAlpha":
      return logged(
        { ...state, params: { ...state.params, alpha: action.value } },
        `alpha -> ${action.value.toFixed(2)}`,
      );
    case "setThreshold": {
      const next = {
        ...state,
        params: { ...state.params, threshold: action.value },
      };
      return logged(
        { ...next, prefPairs: buildPrefPairs(next) },
        `threshold -> ${action.value.toFixed(2)}`,
      );
    }
    case "setBudget":
      return logged(
        { ...state, params: { ...state.params, budget: action.value } },
        `budget -> ${action.value}`,
      );
    case "setRolloutNoise":
      return logged(
        { ...state, params: { ...state.params, rolloutNoise: action.value } },
        `rollout noise -> ${action.value.toFixed(2)}`,
      );
    case "poisonNode": {
      const target = findNode(state, action.nodeId);
      const next = {
        ...state,
        nodes: state.nodes.map((n) =>
          n.id === action.nodeId ? { ...n, poisoned: !n.poisoned } : n,
        ),
        highlightNodeId: action.nodeId,
      };
      const verb = target.poisoned ? "un-poisoned" : "poisoned";
      return logged(
        { ...next, prefPairs: buildPrefPairs(next) },
        `critic ${verb} at ${getPage(target.pageId).label}`,
      );
    }
    case "killNode": {
      const target = findNode(state, action.nodeId);
      if (action.nodeId === state.rootId) return state; // never kill the root
      const next = {
        ...state,
        nodes: state.nodes.map((n) =>
          n.id === action.nodeId ? { ...n, dead: !n.dead } : n,
        ),
        highlightNodeId: action.nodeId,
      };
      const verb = target.dead ? "revived" : "killed";
      return logged(next, `${verb} ${getPage(target.pageId).label}`);
    }
    case "loadPreset":
      return presetState(action.id);
  }
}

// The pure reducer. A tick runs one MCTS iteration; an intervention changes a
// parameter or pokes a node and records what happened. Every step is
// deterministic given the seed carried in state.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return runIteration(state);
  return applyIntervention(state, input.action);
}

// The view needs to read a page's static spec (its label, note, and actions)
// from a node's pageId, so expose the lookup.
export function pageOf(pageId: string): PageSpec {
  return getPage(pageId);
}
