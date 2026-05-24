// A pure, deterministic model of AlphaZero's Monte-Carlo Tree Search from
// "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning
// Algorithm". The search itself is exactly the paper's: every simulation
// descends the tree by the PUCT rule, picking the child that maximizes
//
//   U(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a)),
//
// expands the first unvisited leaf, reads a value from the network, and backs
// that value up the path it took, flipping sign at each ply so both players play
// to win. The move finally played is the child of the root with the most visits.
//
// To keep the search real but legible, it runs over a small fixed game tree
// whose leaves carry true outcomes (a self-contained stand-in for chess, where a
// full board would need millions of nodes). The "network" is a lookup table of a
// prior P and a value v per node, hand-set so a reader can build the case the
// paper cares about: a prior that points at the wrong move, which enough guided
// simulations correct on their own. The leaf values the search reads are the
// network's v estimates, not the true leaf outcomes, so the search can be fooled
// exactly when the net is wrong and under-explored. Randomness only enters
// through optional Dirichlet-style root noise, drawn from a seeded generator, so
// a given seed reproduces the exact same run.

// One step of the mulberry32 recurrence as a pure function: given the
// generator's internal state, return the next value in [0, 1) and the next
// state, so a reducer can thread the state through immutable sim state and stay
// deterministic without a stateful closure.
function nextRandom(state: number): { value: number; next: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, next: a };
}

// Indexed access that throws on a genuine out-of-range read instead of quietly
// returning undefined, so the hot loops stay typed as the element type while
// still failing loudly on a real bug.
function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) {
    throw new RangeError(`index ${index} out of range`);
  }
  return value;
}

// A node in the fixed game tree. `prior` is the network's P(s, a) for the move
// that leads here from its parent; `netValue` is the network's value estimate
// v(s) of this node from the perspective of the player to move at it; `trueValue`
// is the game's real outcome at a leaf, used only to score the move the search
// finally picks, never read by the search itself.
export interface GameNode {
  id: number;
  parent: number | null;
  depth: number;
  label: string;
  prior: number;
  netValue: number;
  trueValue: number;
  children: number[];
}

export interface GameTree {
  nodes: GameNode[];
  root: number;
}

// Per-node search statistics, the part MCTS actually grows. `visits` is N(s, a),
// `valueSum` is the running total of backed-up values from this node's subtree,
// and Q = valueSum / visits is the mean value. `expanded` marks that the node's
// children have been added to the live tree.
export interface NodeStats {
  visits: number;
  valueSum: number;
  expanded: boolean;
}

export interface ScenarioSpec {
  id: string;
  label: string;
  blurb: string;
  tree: GameTree;
  bestMove: number;
}

// Helper to build a tree from a flat spec. Each entry names its parent, its move
// label, the network prior on the edge into it, the network's value estimate, and
// (for leaves) the true outcome.
interface NodeSpec {
  parent: number | null;
  label: string;
  prior: number;
  netValue: number;
  trueValue: number;
}

function buildTree(specs: NodeSpec[]): GameTree {
  const nodes: GameNode[] = specs.map((spec, id) => ({
    id,
    parent: spec.parent,
    depth: 0,
    label: spec.label,
    prior: spec.prior,
    netValue: spec.netValue,
    trueValue: spec.trueValue,
    children: [],
  }));
  for (const node of nodes) {
    if (node.parent !== null) {
      at(nodes, node.parent).children.push(node.id);
      node.depth = at(nodes, node.parent).depth + 1;
    }
  }
  return { nodes, root: 0 };
}

export function getNode(tree: GameTree, id: number): GameNode {
  return at(tree.nodes, id);
}

export function isLeaf(tree: GameTree, id: number): boolean {
  return getNode(tree, id).children.length === 0;
}

// The "net is wrong" tree. Three root moves. The network's prior loves move A
// (0.7), which the true game scores as a loss, and it underrates move C (0.1),
// which is the real win. The leaf values the net reports are deliberately off:
// it thinks A is good and C is mediocre. A search that explores enough still
// drives visits onto C, because C's children carry winning values the net
// reluctantly admits once it looks. Under-explore and the search just echoes the
// prior and plays the losing move A.
const WRONG_NET: ScenarioSpec = {
  id: "wrong-net",
  label: "Net is wrong",
  blurb:
    "The prior loves move A, a loss. Move C is the real win but the prior barely backs it. Run enough simulations with healthy exploration and visits drift onto C. Starve the search and it plays the prior's mistake.",
  tree: buildTree([
    { parent: null, label: "root", prior: 1, netValue: 0, trueValue: 0 },
    { parent: 0, label: "A", prior: 0.7, netValue: 0.35, trueValue: -1 },
    { parent: 0, label: "B", prior: 0.2, netValue: -0.1, trueValue: 0 },
    { parent: 0, label: "C", prior: 0.1, netValue: 0.05, trueValue: 1 },
    // A's replies: the opponent has a strong refutation, so A's leaves are bad.
    { parent: 1, label: "A1", prior: 0.6, netValue: -0.8, trueValue: -1 },
    { parent: 1, label: "A2", prior: 0.4, netValue: -0.6, trueValue: -1 },
    // B's replies: a quiet draw.
    { parent: 2, label: "B1", prior: 0.5, netValue: 0.0, trueValue: 0 },
    { parent: 2, label: "B2", prior: 0.5, netValue: 0.05, trueValue: 0 },
    // C's replies: the opponent has no good answer, so C's leaves are winning.
    { parent: 3, label: "C1", prior: 0.5, netValue: 0.7, trueValue: 1 },
    { parent: 3, label: "C2", prior: 0.5, netValue: 0.6, trueValue: 1 },
  ]),
  bestMove: 3,
};

// The aligned tree. The network's prior already points at the best move and its
// values are honest, so even a handful of simulations confirm the prior and play
// the winning move. This is the common case where search just sharpens an
// already-good net.
const ALIGNED_NET: ScenarioSpec = {
  id: "aligned-net",
  label: "Net is right",
  blurb:
    "The prior already favors the winning move A, and its values are honest. A few simulations confirm the prior and play A. This is search sharpening a net that is already good.",
  tree: buildTree([
    { parent: null, label: "root", prior: 1, netValue: 0, trueValue: 0 },
    { parent: 0, label: "A", prior: 0.6, netValue: 0.6, trueValue: 1 },
    { parent: 0, label: "B", prior: 0.25, netValue: 0.0, trueValue: 0 },
    { parent: 0, label: "C", prior: 0.15, netValue: -0.4, trueValue: -1 },
    { parent: 1, label: "A1", prior: 0.5, netValue: 0.7, trueValue: 1 },
    { parent: 1, label: "A2", prior: 0.5, netValue: 0.65, trueValue: 1 },
    { parent: 2, label: "B1", prior: 0.5, netValue: 0.0, trueValue: 0 },
    { parent: 2, label: "B2", prior: 0.5, netValue: -0.05, trueValue: 0 },
    { parent: 3, label: "C1", prior: 0.5, netValue: -0.5, trueValue: -1 },
    { parent: 3, label: "C2", prior: 0.5, netValue: -0.4, trueValue: -1 },
  ]),
  bestMove: 1,
};

// A trap tree. Move A looks great to a shallow look (its own value is high) but
// hides a refutation one ply deeper. Only a search that descends past A's first
// reply discovers the trap and backs the loss up to A. This shows search
// catching a tactic the net's single-position value misses.
const TRAP_NET: ScenarioSpec = {
  id: "trap",
  label: "Hidden trap",
  blurb:
    "Move A looks strong on the surface but hides a refutation a ply deeper. Only a search that pushes past A's first reply uncovers the trap and shifts visits to the safe move B.",
  tree: buildTree([
    { parent: null, label: "root", prior: 1, netValue: 0, trueValue: 0 },
    { parent: 0, label: "A", prior: 0.55, netValue: 0.5, trueValue: -1 },
    { parent: 0, label: "B", prior: 0.3, netValue: 0.1, trueValue: 1 },
    { parent: 0, label: "C", prior: 0.15, netValue: -0.2, trueValue: 0 },
    // A's reply hides the refutation: the opponent has a crushing answer.
    { parent: 1, label: "A1", prior: 0.5, netValue: -0.9, trueValue: -1 },
    { parent: 1, label: "A2", prior: 0.5, netValue: -0.85, trueValue: -1 },
    // B's replies are genuinely good for us.
    { parent: 2, label: "B1", prior: 0.5, netValue: 0.55, trueValue: 1 },
    { parent: 2, label: "B2", prior: 0.5, netValue: 0.5, trueValue: 1 },
    { parent: 3, label: "C1", prior: 0.5, netValue: 0.0, trueValue: 0 },
    { parent: 3, label: "C2", prior: 0.5, netValue: -0.05, trueValue: 0 },
  ]),
  bestMove: 2,
};

export const SCENARIOS: ScenarioSpec[] = [WRONG_NET, ALIGNED_NET, TRAP_NET];

export function getScenario(id: string): ScenarioSpec {
  return SCENARIOS.find((scenario) => scenario.id === id) ?? at(SCENARIOS, 0);
}

export type Phase = "select" | "expand" | "evaluate" | "backup" | "idle";

export interface SimEvent {
  id: number;
  sim: number;
  label: string;
  snapshot: Snapshot;
}

// A snapshot fully describes a moment the reader can rewind to: the scenario, the
// search statistics, the simulation counter, the rng cursor, and the parameters.
// Rewinding restores every one of these, so the replayed run is bit-identical.
export interface Snapshot {
  scenarioId: string;
  stats: Record<number, NodeStats>;
  completed: number;
  rngState: number;
  params: SearchParams;
}

export interface SearchParams {
  cPuct: number;
  rootNoise: number;
  simBudget: number;
  seed: number;
}

export interface SimState {
  scenarioId: string;
  stats: Record<number, NodeStats>;
  completed: number;
  phase: Phase;
  path: number[];
  activeLeaf: number | null;
  lastValue: number | null;
  rngState: number;
  params: SearchParams;
  events: SimEvent[];
  nextEventId: number;
}

export type Intervention =
  | { type: "setCPuct"; value: number }
  | { type: "setRootNoise"; value: number }
  | { type: "setSimBudget"; value: number }
  | { type: "setSeed"; value: number }
  | { type: "loadScenario"; id: string }
  | { type: "restore"; snapshot: Snapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

const EVENT_LIMIT = 80;

// The root is expanded before the first simulation, the way AlphaZero expands
// the root node once at the start of a search so every simulation descends into a
// child. Without this the first simulation would stop at the root and visit no
// move, leaving the visit count off by one.
function emptyStats(tree: GameTree): Record<number, NodeStats> {
  const stats: Record<number, NodeStats> = {};
  for (const node of tree.nodes) {
    stats[node.id] = {
      visits: 0,
      valueSum: 0,
      expanded: node.id === tree.root,
    };
  }
  return stats;
}

export function defaultParams(): SearchParams {
  return { cPuct: 1.5, rootNoise: 0, simBudget: 40, seed: 7 };
}

export function initialState(
  scenarioId = "wrong-net",
  params: SearchParams = defaultParams(),
): SimState {
  const tree = getScenario(scenarioId).tree;
  return {
    scenarioId,
    stats: emptyStats(tree),
    completed: 0,
    phase: "idle",
    path: [],
    activeLeaf: null,
    lastValue: null,
    rngState: params.seed >>> 0,
    params: { ...params },
    events: [],
    nextEventId: 1,
  };
}

function statsOf(state: SimState, id: number): NodeStats {
  return state.stats[id] ?? { visits: 0, valueSum: 0, expanded: false };
}

export function meanValue(stat: NodeStats): number {
  return stat.visits === 0 ? 0 : stat.valueSum / stat.visits;
}

// The exploration prior on the root's edges, optionally mixed with seeded
// Dirichlet-style noise the way the paper adds Dir(alpha) at the root to force
// exploration. The noise here is a simple seeded reproduction: draw one positive
// number per child, normalize, and blend by `rootNoise`. With rootNoise 0 the
// raw prior is used unchanged.
function rootPriors(
  state: SimState,
  tree: GameTree,
): { priors: Record<number, number>; rngState: number } {
  const children = getNode(tree, tree.root).children;
  const priors: Record<number, number> = {};
  if (state.params.rootNoise <= 0) {
    for (const child of children) priors[child] = getNode(tree, child).prior;
    return { priors, rngState: state.rngState };
  }
  let rngState = state.rngState;
  const noise: number[] = [];
  let total = 0;
  for (let i = 0; i < children.length; i++) {
    const draw = nextRandom(rngState);
    rngState = draw.next;
    const sample = -Math.log(Math.max(draw.value, 1e-9));
    noise.push(sample);
    total += sample;
  }
  children.forEach((child, i) => {
    const n = total > 0 ? at(noise, i) / total : 1 / children.length;
    priors[child] =
      (1 - state.params.rootNoise) * getNode(tree, child).prior +
      state.params.rootNoise * n;
  });
  return { priors, rngState };
}

// The PUCT score for one edge, exactly the paper's selection rule. `parentVisits`
// is the sum of visits over the parent's children.
export function puctScore(
  q: number,
  prior: number,
  parentVisits: number,
  childVisits: number,
  cPuct: number,
): number {
  const u = cPuct * prior * (Math.sqrt(parentVisits) / (1 + childVisits));
  return q + u;
}

// Descend from the root by PUCT until reaching a node that has not been expanded
// (a leaf of the live tree). The sign on Q flips with depth because a value good
// for the player to move at a node is bad for the player one ply up, so each
// player picks the move that looks best from its own side.
function selectPath(
  state: SimState,
  tree: GameTree,
  priors: Record<number, number>,
): number[] {
  const path = [tree.root];
  let current = tree.root;
  while (statsOf(state, current).expanded && !isLeaf(tree, current)) {
    const children = getNode(tree, current).children;
    const parentVisits = children.reduce(
      (sum, child) => sum + statsOf(state, child).visits,
      0,
    );
    let best = at(children, 0);
    let bestScore = Number.NEGATIVE_INFINITY;
    for (const child of children) {
      const stat = statsOf(state, child);
      const prior =
        current === tree.root
          ? (priors[child] ?? getNode(tree, child).prior)
          : getNode(tree, child).prior;
      // Q is stored from the perspective of the player to move at the child.
      // Negate so the parent reads it from its own perspective.
      const q = -meanValue(stat);
      const score = puctScore(
        q,
        prior,
        parentVisits,
        stat.visits,
        state.params.cPuct,
      );
      if (score > bestScore) {
        bestScore = score;
        best = child;
      }
    }
    path.push(best);
    current = best;
  }
  return path;
}

// Back the leaf value up the selected path, flipping sign at each ply so the
// value is always stored from the perspective of the player to move at that node.
function backup(
  stats: Record<number, NodeStats>,
  path: number[],
  leafDepth: number,
  value: number,
): Record<number, NodeStats> {
  const next = { ...stats };
  for (const id of path) {
    const prev = next[id] ?? { visits: 0, valueSum: 0, expanded: false };
    const depth = path.indexOf(id);
    const signed = (leafDepth - depth) % 2 === 0 ? value : -value;
    next[id] = {
      visits: prev.visits + 1,
      valueSum: prev.valueSum + signed,
      expanded: prev.expanded,
    };
  }
  return next;
}

function pushEvent(
  state: SimState,
  label: string,
  override?: Partial<Snapshot>,
): SimState {
  const snapshot: Snapshot = {
    scenarioId: state.scenarioId,
    stats: state.stats,
    completed: state.completed,
    rngState: state.rngState,
    params: state.params,
    ...override,
  };
  const event: SimEvent = {
    id: state.nextEventId,
    sim: state.completed,
    label,
    snapshot,
  };
  return {
    ...state,
    events: [...state.events, event].slice(-EVENT_LIMIT),
    nextEventId: state.nextEventId + 1,
  };
}

// Run one full simulation: select a path, expand the leaf, read the network's
// value at it, and back that value up. Recorded as a single event so the reader
// can rewind to just before it. This advances the search by one of the budget's
// simulations.
function runSimulation(state: SimState): SimState {
  if (state.completed >= state.params.simBudget) return state;
  const tree = getScenario(state.scenarioId).tree;
  const { priors, rngState } = rootPriors(state, tree);
  const path = selectPath({ ...state, rngState }, tree, priors);
  const leaf = at(path, path.length - 1);
  const leafNode = getNode(tree, leaf);

  // Expand: mark this leaf's children live so the next simulation can descend
  // past it. Leaves of the game tree (true terminals) never expand.
  const expandedStats = { ...state.stats };
  const leafStat = statsOf(state, leaf);
  if (!isLeaf(tree, leaf)) {
    expandedStats[leaf] = { ...leafStat, expanded: true };
  }

  // Evaluate: the value the search reads is the network's estimate at the leaf,
  // not the true outcome. This is why a wrong, under-explored net gets fooled.
  const value = leafNode.netValue;
  const backedUp = backup(expandedStats, path, leafNode.depth, value);

  const advanced: SimState = {
    ...state,
    stats: backedUp,
    completed: state.completed + 1,
    phase: "backup",
    path,
    activeLeaf: leaf,
    lastValue: value,
    rngState,
  };
  const pathLabels = path.map((id) => getNode(tree, id).label).join(" > ");
  return pushEvent(
    advanced,
    `sim ${advanced.completed}: ${pathLabels} (v=${value.toFixed(2)})`,
  );
}

function applyIntervention(state: SimState, action: Intervention): SimState {
  switch (action.type) {
    case "setCPuct": {
      const params = { ...state.params, cPuct: action.value };
      return pushEvent(
        { ...state, params },
        `c_puct -> ${action.value.toFixed(2)}`,
      );
    }
    case "setRootNoise": {
      const params = { ...state.params, rootNoise: action.value };
      return pushEvent(
        { ...state, params },
        `root noise -> ${action.value.toFixed(2)}`,
      );
    }
    case "setSimBudget": {
      const params = { ...state.params, simBudget: action.value };
      return pushEvent({ ...state, params }, `sim budget -> ${action.value}`);
    }
    case "setSeed": {
      const params = { ...state.params, seed: action.value };
      // Only reset the rng cursor if the search has not started, so changing the
      // seed mid-run does not silently rewrite history already recorded.
      const rngState =
        state.completed === 0 ? action.value >>> 0 : state.rngState;
      return pushEvent(
        { ...state, params, rngState },
        `seed -> ${action.value}`,
      );
    }
    case "loadScenario":
      return initialState(action.id, state.params);
    case "restore": {
      const snap = action.snapshot;
      return pushEvent(
        {
          ...state,
          scenarioId: snap.scenarioId,
          stats: snap.stats,
          completed: snap.completed,
          rngState: snap.rngState,
          params: snap.params,
          phase: "idle",
          path: [],
          activeLeaf: null,
          lastValue: null,
        },
        action.label,
        snap,
      );
    }
  }
}

// The pure reducer. A tick runs one full simulation; an intervention changes a
// parameter, loads a scenario, or rewinds. Every step is deterministic given the
// state, which carries the rng cursor.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return runSimulation(state);
  return applyIntervention(state, input.action);
}

export interface MoveStat {
  id: number;
  label: string;
  visits: number;
  meanValue: number;
  prior: number;
  trueValue: number;
}

// The root move statistics, the part the reader watches. Visits is what AlphaZero
// turns into the policy target and what the played move is chosen by.
export function rootMoveStats(state: SimState): MoveStat[] {
  const tree = getScenario(state.scenarioId).tree;
  return getNode(tree, tree.root).children.map((child) => {
    const node = getNode(tree, child);
    const stat = statsOf(state, child);
    return {
      id: child,
      label: node.label,
      visits: stat.visits,
      // Stored from the child's perspective; negate for the root's perspective.
      meanValue: -meanValue(stat),
      prior: node.prior,
      trueValue: node.trueValue,
    };
  });
}

// The move AlphaZero would play: the most-visited root child. Ties break toward
// the lower id for determinism.
export function selectedMove(state: SimState): MoveStat | null {
  const moves = rootMoveStats(state);
  if (moves.every((move) => move.visits === 0)) return null;
  return moves.reduce((best, move) =>
    move.visits > best.visits ? move : best,
  );
}

// Whether the search currently plays the scenario's true best move. This is the
// invariant indicator: with enough simulations and healthy exploration it flips
// to true even when the prior was wrong.
export function playsBestMove(state: SimState): boolean {
  const chosen = selectedMove(state);
  if (!chosen) return false;
  return chosen.id === getScenario(state.scenarioId).bestMove;
}
