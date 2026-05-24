// A pure, deterministic model of Monte-Carlo Tree Search from Remi Coulom's 2006
// paper "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search",
// the work that introduced MCTS and the Crazy Stone Go program.
//
// One simulation does four things, exactly as Section 2 describes them:
//   1. descend from the root, at each visited node picking the next move by a
//      selectivity rule until it reaches a node that has not been expanded yet;
//   2. expand that node (the paper creates a child the second time a node is
//      visited);
//   3. run a random rollout from the new leaf, which returns a value;
//   4. back the value up the visited path, updating each node's mean and variance
//      with the chosen backup operator.
//
// The underlying "game" is a small fixed tree. Every leaf has a hidden true
// value; a rollout from a leaf samples a noisy value around that true mean. One
// move is a trap: its true value is high, but its rollout noise is wide, so its
// first few samples can look bad. A pruning rule that trusts early samples kills
// it; Coulom's selectivity keeps probing it and finds it. All randomness comes
// from a seeded stream threaded through the state, so a seed plus a list of
// inputs reproduces a run exactly. No Math.random, no wall clock.
//
// Selectivity (Section 3.2). With moves ordered mu_0 > mu_1 > ... the urgency of
// move i is
//   u_i = exp( -2.4 * (mu_0 - mu_i) / sqrt(2 * (sigma0^2 + sigma_i^2)) ) + eps_i
// and a move is chosen with probability proportional to u_i. eps_i is a small
// floor so urgency never reaches zero, which is why no move is ever fully pruned.
//
// Backup (Section 4, Figure 1). For an external (rarely visited) node the value
// is the plain mean Sigma/S. For an internal node the operators are
//   mean        : Sigma/S, always.
//   max         : the largest child value (negamax), which overestimates.
//   robust-max  : the value of the most-simulated child.
//   mix         : a weighted blend that starts near the mean and slides toward
//                 the robust max as the visit count grows, the paper's best
//                 operator.

// ---------------------------------------------------------------------------
// Seeded RNG. mulberry32 threaded as pure (value, state) so the reducer stays
// deterministic. Kept inline so this simulation is self-contained.
// ---------------------------------------------------------------------------

function nextRandom(state: number): { value: number; state: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, state: a };
}

// Box-Muller from two uniforms, returning one standard normal and the advanced
// rng state.
function nextGaussian(state: number): { value: number; state: number } {
  const first = nextRandom(state);
  const second = nextRandom(first.state);
  const u1 = Math.max(first.value, 1e-9);
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * second.value);
  return { value: z, state: second.state };
}

function at<T>(items: readonly T[], index: number): T {
  const value = items[index];
  if (value === undefined) {
    throw new RangeError(`index ${index} out of range`);
  }
  return value;
}

// ---------------------------------------------------------------------------
// The underlying game tree the search explores.
// ---------------------------------------------------------------------------

export interface MoveSpec {
  label: string;
  // The hidden true value of this move (in [-1, 1], higher is better for the
  // player to move at the root).
  trueValue: number;
  // The width of the rollout noise from this move. A wide value is what makes a
  // good move look bad early.
  noise: number;
  // Whether this move is the one worth finding: high true value behind wide
  // noise.
  trap: boolean;
}

export interface GameSpec {
  id: string;
  label: string;
  blurb: string;
  moves: MoveSpec[];
}

export const GAMES: GameSpec[] = [
  {
    id: "trap",
    label: "Hidden gem",
    blurb:
      "Move C is the best move, but its rollouts are noisy so it looks bad at first. Greedy and pruning starve it; Coulom's selectivity keeps probing and finds it.",
    moves: [
      { label: "A", trueValue: 0.25, noise: 0.18, trap: false },
      { label: "B", trueValue: 0.1, noise: 0.18, trap: false },
      { label: "C", trueValue: 0.55, noise: 0.85, trap: true },
      { label: "D", trueValue: -0.3, noise: 0.2, trap: false },
    ],
  },
  {
    id: "clear",
    label: "Clear best",
    blurb:
      "One move is plainly best with low noise. Every selectivity rule converges fast, so this is the baseline.",
    moves: [
      { label: "A", trueValue: 0.6, noise: 0.2, trap: false },
      { label: "B", trueValue: 0.05, noise: 0.2, trap: false },
      { label: "C", trueValue: -0.1, noise: 0.2, trap: false },
      { label: "D", trueValue: -0.4, noise: 0.2, trap: false },
    ],
  },
  {
    id: "close",
    label: "Two close rivals",
    blurb:
      "A and C are nearly tied, both with moderate noise. The search has to spend simulations to separate them.",
    moves: [
      { label: "A", trueValue: 0.42, noise: 0.4, trap: false },
      { label: "B", trueValue: -0.15, noise: 0.25, trap: false },
      { label: "C", trueValue: 0.38, noise: 0.4, trap: true },
      { label: "D", trueValue: -0.35, noise: 0.2, trap: false },
    ],
  },
];

export function getGame(id: string): GameSpec {
  return GAMES.find((game) => game.id === id) ?? at(GAMES, 0);
}

// ---------------------------------------------------------------------------
// Tree nodes. The root's children are the candidate moves; each move node has a
// short fixed chain of follow-up leaves so a rollout has a real path to walk.
// ---------------------------------------------------------------------------

export interface TreeNode {
  id: number;
  parentId: number | null;
  // Index of the move at the root this node sits under, for color and grouping.
  moveIndex: number;
  label: string;
  depth: number;
  childIds: number[];
  // Whether the search has created this node's children yet.
  expanded: boolean;
  // Monte-Carlo statistics: count S, sum Sigma, sum of squares Sigma2.
  visits: number;
  sumValue: number;
  sumSquares: number;
  // The backed-up value and variance the operator produced.
  value: number;
  variance: number;
  // The hidden true value used to sample rollouts that reach this node.
  trueValue: number;
  noise: number;
}

export type BackupOperator = "mean" | "max" | "robust-max" | "mix";
export type SelectivityRule = "coulom" | "greedy" | "uniform" | "pruning";

export interface SimParams {
  gameId: string;
  backup: BackupOperator;
  selectivity: SelectivityRule;
  // The threshold S above which a node is treated as internal (Section 2). Below
  // it, the node uses the plain mean.
  expandThreshold: number;
  seed: number;
}

export interface SimState {
  params: SimParams;
  nodes: TreeNode[];
  rootId: number;
  rngState: number;
  // The simulation count, which is also the virtual clock.
  simulations: number;
  // Which moves a pruning rule has cut off (indices into the root's children).
  pruned: number[];
  // The path of node ids the most recent simulation walked, for the view to
  // highlight, plus the value that rollout returned.
  lastPath: number[];
  lastValue: number | null;
  log: SimEvent[];
  nextEventId: number;
}

export interface SimEvent {
  id: number;
  simulations: number;
  label: string;
  snapshot: SimSnapshot;
}

// A snapshot small enough to store and rewind to: the full node array plus the
// counters. Because the model is pure, restoring a snapshot reproduces the rest
// of the run under any new inputs.
export interface SimSnapshot {
  nodes: TreeNode[];
  rngState: number;
  simulations: number;
  pruned: number[];
}

// ---------------------------------------------------------------------------
// Tree construction.
// ---------------------------------------------------------------------------

const FOLLOW_UP_DEPTH = 2;

function makeNode(
  id: number,
  parentId: number | null,
  moveIndex: number,
  label: string,
  depth: number,
  trueValue: number,
  noise: number,
): TreeNode {
  return {
    id,
    parentId,
    moveIndex,
    label,
    depth,
    childIds: [],
    expanded: false,
    visits: 0,
    sumValue: 0,
    sumSquares: 0,
    value: 0,
    variance: 1,
    trueValue,
    noise,
  };
}

function buildInitialTree(game: GameSpec): {
  nodes: TreeNode[];
  rootId: number;
} {
  const nodes: TreeNode[] = [];
  const root = makeNode(0, null, -1, "root", 0, 0, 0);
  // The root's candidate moves exist from the start, so the very first
  // simulation descends into a move rather than stopping at the root.
  root.expanded = true;
  nodes.push(root);
  game.moves.forEach((move, index) => {
    const node = makeNode(
      nodes.length,
      root.id,
      index,
      move.label,
      1,
      move.trueValue,
      move.noise,
    );
    root.childIds.push(node.id);
    nodes.push(node);
  });
  return { nodes, rootId: root.id };
}

function node(state: SimState, id: number): TreeNode {
  return at(
    state.nodes,
    state.nodes.findIndex((entry) => entry.id === id),
  );
}

// ---------------------------------------------------------------------------
// Backup. Mean for external nodes; the chosen operator for internal nodes.
// ---------------------------------------------------------------------------

function externalValue(n: TreeNode): number {
  return n.visits > 0 ? n.sumValue / n.visits : 0;
}

// Variance of an external node, the paper's formula with a high-variance virtual
// prior. P (the board size term) is fixed to a small constant here since this is
// an abstract tree, not a 9x9 goban.
function externalVariance(n: TreeNode): number {
  const P = 0.5;
  const S = n.visits;
  if (S <= 0) return 1;
  const mu = externalValue(n);
  return (n.sumSquares - S * mu * mu + 4 * P * P) / (S + 1);
}

function childValuesSorted(
  state: SimState,
  parent: TreeNode,
): { value: number; visits: number }[] {
  return parent.childIds
    .map((id) => {
      const child = node(state, id);
      return { value: child.value, visits: child.visits };
    })
    .sort((a, b) => b.value - a.value);
}

// The mix operator from Figure 1, slid to this abstract setting. MeanWeight grows
// with the simulation count, so the value starts near the mean and moves toward
// the robust max as the node gathers visits.
function mixValue(state: SimState, parent: TreeNode): number {
  const sorted = childValuesSorted(state, parent);
  const mean = externalValue(parent);
  if (sorted.length === 0) return mean;
  const robustMax = at(sorted, 0).value;
  const meanWeight = 4;
  const grown =
    parent.visits > 16 ? meanWeight * (parent.visits / 16) : meanWeight;
  const weightToMax = parent.visits / (parent.visits + grown);
  return mean * (1 - weightToMax) + robustMax * weightToMax;
}

function robustMaxValue(state: SimState, parent: TreeNode): number {
  let best = parent.childIds[0];
  let bestVisits = -1;
  for (const id of parent.childIds) {
    const child = node(state, id);
    if (child.visits > bestVisits) {
      bestVisits = child.visits;
      best = id;
    }
  }
  return best === undefined ? externalValue(parent) : node(state, best).value;
}

function maxValue(state: SimState, parent: TreeNode): number {
  const sorted = childValuesSorted(state, parent);
  return sorted.length > 0 ? at(sorted, 0).value : externalValue(parent);
}

function backedUpValue(
  state: SimState,
  n: TreeNode,
  operator: BackupOperator,
): number {
  const internal = n.expanded && n.visits > state.params.expandThreshold;
  if (!internal || n.childIds.length === 0) return externalValue(n);
  switch (operator) {
    case "mean":
      return externalValue(n);
    case "max":
      return maxValue(state, n);
    case "robust-max":
      return robustMaxValue(state, n);
    case "mix":
      return mixValue(state, n);
  }
}

// ---------------------------------------------------------------------------
// Selectivity. Returns the chosen child index, drawing from the rng when the
// rule is stochastic.
// ---------------------------------------------------------------------------

interface MoveStat {
  index: number; // index into childIds
  childId: number;
  mu: number;
  sigma2: number;
  pruned: boolean;
}

function statsFor(state: SimState, parent: TreeNode): MoveStat[] {
  return parent.childIds.map((childId, index) => {
    const child = node(state, childId);
    const isRootChild = parent.id === state.rootId;
    return {
      index,
      childId,
      mu: child.value,
      sigma2: Math.max(child.variance, 1e-4),
      pruned: isRootChild && state.pruned.includes(child.moveIndex),
    };
  });
}

// Coulom's urgency, Section 3.2. mu_0 is the best current value among the
// candidates; eps is the floor that keeps every move alive.
function urgencies(stats: MoveStat[]): number[] {
  const best = stats.reduce((max, s) => Math.max(max, s.mu), -Infinity);
  const sigma0 =
    stats.reduce((acc, s) => (s.mu === best ? s.sigma2 : acc), 1) || 1;
  const N = stats.length;
  return stats.map((s, rank) => {
    const denom = Math.sqrt(2 * (sigma0 + s.sigma2)) || 1;
    const main = Math.exp((-2.4 * (best - s.mu)) / denom);
    const eps = (0.1 + Math.pow(2, -(rank + 1))) / N;
    return main + eps;
  });
}

function chooseWeighted(
  weights: number[],
  rngState: number,
): { index: number; rngState: number } {
  const total = weights.reduce((sum, w) => sum + w, 0) || 1;
  const draw = nextRandom(rngState);
  let target = draw.value * total;
  for (let i = 0; i < weights.length; i++) {
    target -= weights[i] ?? 0;
    if (target <= 0) return { index: i, rngState: draw.state };
  }
  return { index: weights.length - 1, rngState: draw.state };
}

function selectChild(
  state: SimState,
  parent: TreeNode,
  rngState: number,
): { childIndex: number; rngState: number } {
  const stats = statsFor(state, parent);
  const live = stats.filter((s) => !s.pruned);
  const pool = live.length > 0 ? live : stats;

  switch (state.params.selectivity) {
    case "greedy": {
      // Exploit the best value, breaking ties toward the least-visited so a fresh
      // run still spreads its first samples.
      let best = at(pool, 0);
      for (const s of pool) {
        if (
          s.mu > best.mu ||
          (s.mu === best.mu &&
            node(state, s.childId).visits < node(state, best.childId).visits)
        ) {
          best = s;
        }
      }
      return { childIndex: best.index, rngState };
    }
    case "uniform": {
      const draw = nextRandom(rngState);
      const picked = at(pool, Math.floor(draw.value * pool.length));
      return { childIndex: picked.index, rngState: draw.state };
    }
    case "pruning": {
      // Progressive pruning gives each surviving move a handful of samples, then
      // exploits the best. The cutting itself happens in the step. A move that
      // draws bad early samples gets pruned before its mean can recover, which is
      // the danger Section 3.1 names.
      const WARMUP = 6;
      const underSampled = pool.filter(
        (s) => node(state, s.childId).visits < WARMUP,
      );
      if (underSampled.length > 0) {
        let fewest = at(underSampled, 0);
        for (const s of underSampled) {
          if (
            node(state, s.childId).visits < node(state, fewest.childId).visits
          ) {
            fewest = s;
          }
        }
        return { childIndex: fewest.index, rngState };
      }
      let best = at(pool, 0);
      for (const s of pool) if (s.mu > best.mu) best = s;
      return { childIndex: best.index, rngState };
    }
    case "coulom":
    default: {
      const weights = urgencies(pool);
      const chosen = chooseWeighted(weights, rngState);
      return {
        childIndex: at(pool, chosen.index).index,
        rngState: chosen.rngState,
      };
    }
  }
}

// ---------------------------------------------------------------------------
// One simulation: descend, expand, rollout, back up.
// ---------------------------------------------------------------------------

// A rollout from a move node samples a noisy value around the deepest reached
// node's true value. The sign flips each ply so the value is always from the
// perspective of the player to move at the root (negamax convention).
function rollout(
  trueValue: number,
  noise: number,
  rngState: number,
): { value: number; rngState: number } {
  const g = nextGaussian(rngState);
  const raw = trueValue + g.value * noise;
  return { value: Math.max(-1, Math.min(1, raw)), rngState: g.state };
}

function cloneNodes(nodes: TreeNode[]): TreeNode[] {
  return nodes.map((n) => ({ ...n, childIds: [...n.childIds] }));
}

function recordEvent(state: SimState, label: string): SimState {
  const snapshot: SimSnapshot = {
    nodes: cloneNodes(state.nodes),
    rngState: state.rngState,
    simulations: state.simulations,
    pruned: [...state.pruned],
  };
  const event: SimEvent = {
    id: state.nextEventId,
    simulations: state.simulations,
    label,
    snapshot,
  };
  return {
    ...state,
    log: [...state.log, event].slice(-60),
    nextEventId: state.nextEventId + 1,
  };
}

// Progressive pruning, Section 3.1: cut a move whose value is far enough below
// the best that, given the spread, it is very unlikely to be better. This is the
// dangerous rule the paper warns about, kept here so the reader can watch it kill
// the trap move.
function applyPruning(state: SimState): number[] {
  if (state.params.selectivity !== "pruning") return state.pruned;
  const root = node(state, state.rootId);
  const stats = statsFor(state, root).filter((s) => !s.pruned);
  if (stats.length <= 1) return state.pruned;
  const best = stats.reduce(
    (max, s) => (s.mu > max.mu ? s : max),
    at(stats, 0),
  );
  const pruned = [...state.pruned];
  for (const s of stats) {
    const child = node(state, s.childId);
    if (child.visits < 5 || s.childId === best.childId) continue;
    // The central-limit standard error of this move's mean. Progressive pruning
    // trusts this even though, as Section 3.1 warns, the distribution is not
    // stationary inside a search tree, so a move with bad early luck gets cut.
    const standardError = Math.sqrt(s.sigma2 / child.visits);
    if (best.mu - s.mu > 1.3 * standardError) pruned.push(child.moveIndex);
  }
  return pruned;
}

function runOneSimulation(state: SimState): SimState {
  let rngState = state.rngState;
  const path: number[] = [state.rootId];
  let current = node(state, state.rootId);

  // Descend until we reach a node that is not expanded yet.
  while (current.expanded && current.childIds.length > 0) {
    const choice = selectChild(state, current, rngState);
    rngState = choice.rngState;
    const childId = at(current.childIds, choice.childIndex);
    path.push(childId);
    current = node(state, childId);
  }

  // Expand: the paper creates children the second time a node is visited. The
  // first visit just marks it; on a later visit (visits >= 1) we add a fixed
  // chain of follow-up leaves so the rollout walks a real path.
  let nodes = state.nodes;
  if (
    !current.expanded &&
    current.visits >= 1 &&
    current.depth < 1 + FOLLOW_UP_DEPTH
  ) {
    nodes = cloneNodes(nodes);
    const idx = nodes.findIndex((n) => n.id === current.id);
    const parent = at(nodes, idx);
    let nextId = nodes.reduce((max, n) => Math.max(max, n.id), 0) + 1;
    // One follow-up child carrying the same move's true value, slightly decayed,
    // so deeper search refines the estimate rather than changing the answer.
    const child = makeNode(
      nextId,
      parent.id,
      parent.moveIndex,
      `${parent.label}.${parent.childIds.length + 1}`,
      parent.depth + 1,
      parent.trueValue * 0.95,
      parent.noise * 0.8,
    );
    parent.childIds = [...parent.childIds, child.id];
    parent.expanded = true;
    nodes.push(child);
    nextId += 1;
    current = child;
    path.push(child.id);
  }

  // Rollout from the reached node.
  const result = rollout(current.trueValue, current.noise, rngState);
  rngState = result.rngState;
  const value = result.value;

  // Back up: update raw statistics along the path, then recompute each node's
  // operator value from leaf to root so children are current before parents read
  // them.
  const updated = cloneNodes(nodes);
  for (const id of path) {
    const n = at(
      updated,
      updated.findIndex((entry) => entry.id === id),
    );
    n.visits += 1;
    n.sumValue += value;
    n.sumSquares += value * value;
  }

  let working: SimState = {
    ...state,
    nodes: updated,
    rngState,
    simulations: state.simulations + 1,
    lastPath: path,
    lastValue: value,
  };

  const ordered = [...path].reverse();
  for (const id of ordered) {
    const idx = working.nodes.findIndex((entry) => entry.id === id);
    const n = at(working.nodes, idx);
    n.variance = externalVariance(n);
    n.value = backedUpValue(working, n, working.params.backup);
  }

  working = { ...working, pruned: applyPruning(working) };
  return working;
}

// ---------------------------------------------------------------------------
// Presets.
// ---------------------------------------------------------------------------

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: SimParams;
}

export const PRESETS: Preset[] = [
  {
    id: "coulom-mix",
    label: "Coulom: find the gem",
    blurb:
      "Coulom selectivity with the mix backup on the hidden-gem tree. Run it and watch move C overtake A as its noisy rollouts settle.",
    params: {
      gameId: "trap",
      backup: "mix",
      selectivity: "coulom",
      expandThreshold: 4,
      seed: 7,
    },
  },
  {
    id: "pruning-trap",
    label: "Pruning kills the gem",
    blurb:
      "Progressive pruning on the same tree. Run it and watch C get pruned after a few unlucky rollouts, so the search settles on the worse move A.",
    params: {
      gameId: "trap",
      backup: "mean",
      selectivity: "pruning",
      expandThreshold: 4,
      seed: 10,
    },
  },
  {
    id: "max-overestimate",
    label: "Max overestimates",
    blurb:
      "Coulom selectivity with the max backup. The root value sits above the true best because the max of noisy children is biased high.",
    params: {
      gameId: "trap",
      backup: "max",
      selectivity: "coulom",
      expandThreshold: 4,
      seed: 7,
    },
  },
  {
    id: "greedy-stuck",
    label: "Greedy gets stuck",
    blurb:
      "Greedy selectivity on the close-rivals tree. It pours samples into whichever rival looks best first and is slow to correct.",
    params: {
      gameId: "close",
      backup: "mix",
      selectivity: "greedy",
      expandThreshold: 4,
      seed: 3,
    },
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

// ---------------------------------------------------------------------------
// Public reducer surface.
// ---------------------------------------------------------------------------

export type Intervention =
  | { type: "setBackup"; value: BackupOperator }
  | { type: "setSelectivity"; value: SelectivityRule }
  | { type: "setGame"; id: string }
  | { type: "setThreshold"; value: number }
  | { type: "setSeed"; value: number }
  | { type: "burst"; count: number }
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: SimSnapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

export function initialState(presetId = "coulom-mix"): SimState {
  const params = { ...getPreset(presetId).params };
  const game = getGame(params.gameId);
  const { nodes, rootId } = buildInitialTree(game);
  return {
    params,
    nodes,
    rootId,
    rngState: params.seed >>> 0,
    simulations: 0,
    pruned: [],
    lastPath: [],
    lastValue: null,
    log: [],
    nextEventId: 1,
  };
}

function resetWith(params: SimParams): SimState {
  const game = getGame(params.gameId);
  const { nodes, rootId } = buildInitialTree(game);
  return {
    params,
    nodes,
    rootId,
    rngState: params.seed >>> 0,
    simulations: 0,
    pruned: [],
    lastPath: [],
    lastValue: null,
    log: [],
    nextEventId: 1,
  };
}

// Public read helpers for the view.

export function rootChildren(state: SimState): TreeNode[] {
  return node(state, state.rootId).childIds.map((id) => node(state, id));
}

export function rootValue(state: SimState): number {
  return node(state, state.rootId).value;
}

// The move the search would play now: the most-visited root child (robust
// choice). Returns the move index, or null before any simulation.
export function recommendedMove(state: SimState): number | null {
  const children = rootChildren(state).filter(
    (c) => !state.pruned.includes(c.moveIndex),
  );
  if (children.length === 0 || state.simulations === 0) return null;
  let best = at(children, 0);
  for (const c of children) if (c.visits > best.visits) best = c;
  return best.moveIndex;
}

export function trueBestMove(state: SimState): number {
  const game = getGame(state.params.gameId);
  let bestIndex = 0;
  let bestValue = -Infinity;
  game.moves.forEach((move, index) => {
    if (move.trueValue > bestValue) {
      bestValue = move.trueValue;
      bestIndex = index;
    }
  });
  return bestIndex;
}

// The safety property the sim watches: the move the search recommends is the
// move with the highest true value. An intervention that prunes the gem breaks
// this and the indicator flips.
export function recommendationCorrect(state: SimState): boolean | null {
  const rec = recommendedMove(state);
  if (rec === null) return null;
  return rec === trueBestMove(state);
}

function backupLabel(operator: BackupOperator): string {
  return operator;
}

function applyIntervention(
  state: SimState,
  action: Intervention,
): { state: SimState; label: string } {
  switch (action.type) {
    case "setBackup": {
      const params = { ...state.params, backup: action.value };
      // Recompute every node's value under the new operator without losing the
      // gathered statistics.
      const next: SimState = { ...state, params };
      const recomputed = cloneNodes(next.nodes);
      const recomputedState: SimState = { ...next, nodes: recomputed };
      // Leaves first: a simple depth-descending order is enough since deeper
      // nodes have larger depth.
      const order = [...recomputed].sort((a, b) => b.depth - a.depth);
      for (const n of order) {
        n.variance = externalVariance(n);
        n.value = backedUpValue(recomputedState, n, params.backup);
      }
      return {
        state: recomputedState,
        label: `backup -> ${backupLabel(action.value)}`,
      };
    }
    case "setSelectivity":
      return {
        state: {
          ...state,
          params: { ...state.params, selectivity: action.value },
        },
        label: `selectivity -> ${action.value}`,
      };
    case "setGame":
      return {
        state: resetWith({ ...state.params, gameId: action.id }),
        label: `tree -> ${action.id}`,
      };
    case "setThreshold":
      return {
        state: {
          ...state,
          params: { ...state.params, expandThreshold: action.value },
        },
        label: `expand threshold -> ${action.value}`,
      };
    case "setSeed":
      return {
        state: resetWith({ ...state.params, seed: action.value }),
        label: `seed -> ${action.value}`,
      };
    case "burst": {
      let working = state;
      for (let i = 0; i < action.count; i++)
        working = runOneSimulation(working);
      return { state: working, label: `ran ${action.count} simulations` };
    }
    case "loadPreset": {
      const preset = getPreset(action.id);
      return {
        state: resetWith({ ...preset.params }),
        label: `preset -> ${preset.label}`,
      };
    }
    case "restore": {
      const { snapshot } = action;
      return {
        state: {
          ...state,
          nodes: cloneNodes(snapshot.nodes),
          rngState: snapshot.rngState,
          simulations: snapshot.simulations,
          pruned: [...snapshot.pruned],
          lastPath: [],
          lastValue: null,
        },
        label: action.label,
      };
    }
  }
}

// The pure reducer. A tick runs one simulation; an intervention changes a
// parameter or replays a snapshot. Deterministic given the seed and the input
// list.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") return runOneSimulation(state);
  const { state: next, label } = applyIntervention(state, input.action);
  return recordEvent(next, label);
}
