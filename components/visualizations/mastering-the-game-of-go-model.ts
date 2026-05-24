import { nextRandom } from "@/lib/simulation/rng";
import { emptyLog, record, type EventLog } from "@/lib/simulation/history";

// A headless, deterministic model of AlphaGo's APV-MCTS over a small synthetic
// game tree. The paper's real position has b ~ 250 and d ~ 150; here a fixed
// shallow tree with hidden true leaf values stands in for one Go position, so
// the search dynamics (PUCT selection, value/rollout leaf evaluation, visit-
// count backup, picking the most-visited move) are exactly the paper's rules
// while staying legible. No Math.random, no Date.now: the rng seed/cursor lives
// in state, so the same seed and inputs reproduce the run for scrub and replay.

export interface TreeNode {
  id: number;
  parent: number | null;
  // Index of this node among its parent's children, used to label moves.
  childSlot: number;
  depth: number;
  // The player to move at this node. 0 = the searcher (root player), 1 = opponent.
  toMove: 0 | 1;
  children: number[];
  expanded: boolean;
  // The hidden true minimax value of this node from the root player's view, in
  // [-1, 1]. The search never reads this directly; it only sees it filtered
  // through the (perturbable) value network and rollouts.
  trueValue: number;
  // Per-edge statistics from the paper: prior P, visit count N, accumulated
  // value W, mean action value Q. Stored on the child via its own id; the
  // root's own edge stats are unused.
  prior: number;
  visits: number;
  totalValue: number;
  // Reader perturbations applied to this node.
  priorBias: number; // multiplies the policy prior on the edge into this node
  valueLie: number | null; // overrides the value-net estimate of this node
  pruned: boolean; // selection skips this subtree
}

export interface SimParams {
  // Mixing parameter from V(s_L) = (1 - lambda) v_theta + lambda z. The paper's
  // best result is lambda = 0.5.
  lambda: number;
  // Exploration constant c_puct in u(s,a) = c_puct P sqrt(sum N) / (1 + N).
  cPuct: number;
  // How noisy the fast rollout is. 0 = rollout returns the leaf's true value;
  // higher spreads the single-rollout result around it.
  rolloutNoise: number;
  // How accurate the value network is. 0 = perfect; higher adds a fixed-by-seed
  // estimation error to every value-net read.
  valueNoise: number;
}

export interface LastSim {
  // The root-to-leaf path of node ids the last simulation selected.
  path: number[];
  leaf: number;
  vTheta: number; // value-net estimate at the leaf
  rollout: number; // rollout outcome z at the leaf
  evaluation: number; // combined V(s_L)
  expanded: boolean;
}

export interface SimState {
  nodes: TreeNode[];
  root: number;
  params: SimParams;
  rng: number; // mulberry32 cursor
  baseSeed: number;
  simulations: number;
  last: LastSim | null;
  log: EventLog<Snapshot>;
}

export interface Snapshot {
  // A full snapshot is the whole node array plus the cursor, which is enough to
  // replay or rewind exactly.
  nodes: TreeNode[];
  rng: number;
  simulations: number;
  params: SimParams;
}

export type Intervention =
  | { type: "loadPreset"; id: string }
  | { type: "setLambda"; value: number }
  | { type: "setCPuct"; value: number }
  | { type: "setRolloutNoise"; value: number }
  | { type: "setValueNoise"; value: number }
  | { type: "setSeed"; value: number }
  | { type: "boostPrior"; node: number }
  | { type: "resetPrior"; node: number }
  | { type: "lieValue"; node: number; value: number }
  | { type: "clearLie"; node: number }
  | { type: "prune"; node: number }
  | { type: "unprune"; node: number }
  | { type: "restore"; snapshot: Snapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

// One value in [0, 1) plus the advanced cursor, threaded so the model stays pure.
function draw(cursor: number): { value: number; cursor: number } {
  const next = nextRandom(cursor);
  return { value: next.value, cursor: next.state };
}

function gaussian(cursor: number): { value: number; cursor: number } {
  // Box-Muller from two uniforms, deterministic given the cursor.
  const a = draw(cursor);
  const b = draw(a.cursor);
  const r = Math.sqrt(-2 * Math.log(a.value + 1e-12));
  return { value: r * Math.cos(2 * Math.PI * b.value), cursor: b.cursor };
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

// ---- Tree shapes -----------------------------------------------------------

interface TreeShape {
  // branching[d] = number of children a node at depth d gets when expanded.
  branching: number[];
  // leafValues maps a leaf's slot path (joined with ".") to its true value.
  // A leaf is any node at depth === branching.length.
  leafValues: Record<string, number>;
  basePriors: number[][]; // basePriors[d] = the policy prior over the d-level children
}

// A two-ply tree: the root has three candidate moves; each leads to an opponent
// reply with three options; opponent minimizes, so the root value of a move is
// the worst reply the opponent can choose. The hidden leaf values are set so one
// root move is clearly best under correct minimax, which is the truth the search
// should recover.
function buildShape(presetId: string): TreeShape {
  const branching = [3, 3];
  // Root move 0 ("center"): opponent's best replies leave it strong.
  // Root move 1 ("contact"): looks tempting (a high-prior trap) but the opponent
  //   has a refutation that makes it bad.
  // Root move 2 ("corner"): solid but a touch worse than center.
  const leafValues: Record<string, number> = {
    "0.0": 0.85,
    "0.1": 0.55,
    "0.2": 0.7, // move 0 minimax value = 0.55
    "1.0": 0.95,
    "1.1": -0.6,
    "1.2": 0.4, // move 1 minimax value = -0.6 (the trap)
    "2.0": 0.5,
    "2.1": 0.3,
    "2.2": 0.45, // move 2 minimax value = 0.3
  };

  // Honest policy: priors roughly track quality. The "trap" preset hands the bad
  // move 1 a high prior so the reader can watch PUCT fight a misleading network.
  let rootPriors = [0.45, 0.3, 0.25];
  if (presetId === "trap" || presetId === "value-lies") {
    rootPriors = [0.2, 0.62, 0.18];
  }
  const replyPriors = [0.4, 0.3, 0.3];
  return {
    branching,
    leafValues,
    basePriors: [rootPriors, replyPriors],
  };
}

function makeNode(
  id: number,
  parent: number | null,
  childSlot: number,
  depth: number,
  toMove: 0 | 1,
  prior: number,
  trueValue: number,
): TreeNode {
  return {
    id,
    parent,
    childSlot,
    depth,
    toMove,
    children: [],
    expanded: false,
    trueValue,
    prior,
    visits: 0,
    totalValue: 0,
    priorBias: 1,
    valueLie: null,
    pruned: false,
  };
}

// ---- Presets ---------------------------------------------------------------

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: SimParams;
}

export const PRESETS: Preset[] = [
  {
    id: "honest",
    label: "Honest search",
    blurb:
      "Good priors and a fair value network. The visit count piles onto the move that is actually best.",
    params: { lambda: 0.5, cPuct: 1.6, rolloutNoise: 0.35, valueNoise: 0.1 },
  },
  {
    id: "value-lies",
    label: "Value net lies",
    blurb:
      "The value network is fooled into loving a losing move, with rollouts switched off (lambda 0). Search follows the lie. Drag lambda up to 0.5 and the rollouts pull it back.",
    params: { lambda: 0, cPuct: 1.6, rolloutNoise: 0.35, valueNoise: 0.1 },
  },
  {
    id: "trap",
    label: "Greedy trap",
    blurb:
      "A bad move carries a high prior and exploration is near zero. Search tunnels into it and never checks the move that refutes it. Raise the exploration constant to escape.",
    params: { lambda: 0.5, cPuct: 0.15, rolloutNoise: 0.35, valueNoise: 0.1 },
  },
];

function paramsFor(presetId: string): SimParams {
  const preset = PRESETS.find((p) => p.id === presetId) ?? PRESETS[0]!;
  return { ...preset.params };
}

// ---- Initial state ---------------------------------------------------------

export function initialState(
  presetId: string,
  seedOverride?: number,
): SimState {
  const shape = buildShape(presetId);
  const nodes: TreeNode[] = [];
  const root = makeNode(0, null, 0, 0, 0, 1, 0);
  nodes.push(root);

  // The trap preset starts with the bad move's prior already boosted via the
  // shape's basePriors; value-lies additionally pins a lie once children exist.
  const seed = seedOverride ?? 7;
  let state: SimState = {
    nodes,
    root: 0,
    params: paramsFor(presetId),
    rng: seed >>> 0,
    baseSeed: seed >>> 0,
    simulations: 0,
    last: null,
    log: emptyLog<Snapshot>(),
  };

  // Eagerly expand the root so the three candidate moves are visible before the
  // first simulation, matching the paper's figure 3 where the root already
  // shows its move priors.
  state = expand(state, 0, shape);

  if (presetId === "value-lies" || presetId === "trap") {
    // Pin a lie on the trap move so the value network over-rates that whole
    // subtree. In value-lies the rollouts are off (lambda 0) so nothing checks
    // the lie; in trap the rollouts are on but exploration is too low to gather
    // enough of them on the refuting reply to crash the inflated mean.
    const trapMove = state.nodes.find(
      (n) => n.parent === 0 && n.childSlot === 1,
    );
    if (trapMove) {
      state = applyIntervention(state, {
        type: "lieValue",
        node: trapMove.id,
        value: 0.9,
      });
    }
  }
  return state;
}

// The shape is rebuilt per call from the current preset's branching/priors. We
// store the active preset implicitly through the prior layout already baked into
// the nodes, so expansion only needs branching and leaf values, which are fixed.
const FIXED_SHAPE = buildShape("honest");

function expand(state: SimState, id: number, shape: TreeShape): SimState {
  const nodes = state.nodes.map((n) => ({ ...n }));
  const node = nodes[id];
  if (!node || node.expanded) return { ...state, nodes };
  const count = shape.branching[node.depth];
  if (count === undefined) return { ...state, nodes }; // leaf, nothing to expand

  const priorRow = shape.basePriors[node.depth] ?? [];
  const childToMove: 0 | 1 = node.toMove === 0 ? 1 : 0;
  for (let slot = 0; slot < count; slot++) {
    const childId = nodes.length;
    const childDepth = node.depth + 1;
    let trueValue = 0;
    if (shape.branching[childDepth] === undefined) {
      const path = [...buildSlotPath(nodes, id), slot].join(".");
      trueValue = shape.leafValues[path] ?? 0;
    }
    const prior = priorRow[slot] ?? 1 / count;
    const child = makeNode(
      childId,
      id,
      slot,
      childDepth,
      childToMove,
      prior,
      trueValue,
    );
    nodes.push(child);
    node.children.push(childId);
  }
  node.expanded = true;
  return { ...state, nodes };
}

function buildSlotPath(nodes: TreeNode[], id: number): number[] {
  const slots: number[] = [];
  let current: number | null = id;
  while (current !== null) {
    const node: TreeNode | undefined = nodes[current];
    if (!node || node.parent === null) break;
    slots.unshift(node.childSlot);
    current = node.parent;
  }
  return slots;
}

// ---- The search rules (the paper) ------------------------------------------

// u(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a)).
function explorationBonus(
  cPuct: number,
  prior: number,
  childVisits: number,
  parentTotalVisits: number,
): number {
  return (
    (cPuct * prior * Math.sqrt(Math.max(1, parentTotalVisits))) /
    (1 + childVisits)
  );
}

function meanValue(node: TreeNode): number {
  return node.visits === 0 ? 0 : node.totalValue / node.visits;
}

// Select the child of `parent` that maximizes Q + u, sign-flipping Q for the
// opponent so the searcher always picks the move best for itself. Pruned
// children are skipped.
function selectChild(state: SimState, parentId: number): number | null {
  const parent = state.nodes[parentId];
  if (!parent || parent.children.length === 0) return null;
  const total = parent.children.reduce(
    (sum, cid) => sum + (state.nodes[cid]?.visits ?? 0),
    0,
  );
  let best: number | null = null;
  let bestScore = -Infinity;
  for (const cid of parent.children) {
    const child = state.nodes[cid];
    if (!child || child.pruned) continue;
    // Q is stored from the root player's view; the opponent at an odd ply wants
    // to minimize it, so the searcher (choosing the opponent's move during
    // descent) evaluates -Q for the opponent's turn.
    const q = parent.toMove === 0 ? meanValue(child) : -meanValue(child);
    const bias = child.priorBias;
    const u = explorationBonus(
      state.params.cPuct,
      child.prior * bias,
      child.visits,
      total,
    );
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      best = cid;
    }
  }
  return best;
}

// A value-net lie pinned on a node misleads the network about that whole part
// of the board, so it carries down to every descendant leaf. This walks up from
// the leaf to find the nearest ancestor (or the leaf itself) that carries a lie.
function inheritedLie(state: SimState, id: number): number | null {
  let current: number | null = id;
  while (current !== null) {
    const node: TreeNode | undefined = state.nodes[current];
    if (!node) break;
    if (node.valueLie !== null) return node.valueLie;
    current = node.parent;
  }
  return null;
}

// The value network's read of a leaf: its true value plus a seeded estimation
// error, unless a lie sits on it or any ancestor (the net mis-reads that region).
function valueNetwork(
  state: SimState,
  node: TreeNode,
): { value: number; cursor: number } {
  const lie = inheritedLie(state, node.id);
  if (lie !== null) {
    return { value: clamp(lie, -1, 1), cursor: state.rng };
  }
  if (state.params.valueNoise <= 0) {
    return { value: node.trueValue, cursor: state.rng };
  }
  const g = gaussian(state.rng);
  return {
    value: clamp(node.trueValue + g.value * state.params.valueNoise, -1, 1),
    cursor: g.cursor,
  };
}

// One fast rollout: returns the leaf's true value perturbed by rollout noise,
// the single-sample z in the paper. With noise 0 it is exact.
function rollout(
  state: SimState,
  node: TreeNode,
  cursor: number,
): { value: number; cursor: number } {
  if (state.params.rolloutNoise <= 0) {
    return { value: node.trueValue, cursor };
  }
  const g = gaussian(cursor);
  return {
    value: clamp(node.trueValue + g.value * state.params.rolloutNoise, -1, 1),
    cursor: g.cursor,
  };
}

// One APV-MCTS simulation: select to a leaf, expand it, evaluate by the value-
// rollout blend, back the value up the path. Pure: returns a fresh state.
function simulate(state: SimState): SimState {
  let nodes = state.nodes.map((n) => ({ ...n }));
  let working: SimState = { ...state, nodes };

  // Selection.
  const path: number[] = [working.root];
  let currentId = working.root;
  while (true) {
    const current = nodes[currentId];
    if (!current || !current.expanded || current.children.length === 0) break;
    const nextId = selectChild(working, currentId);
    if (nextId === null) break;
    path.push(nextId);
    currentId = nextId;
  }

  const leaf = nodes[currentId]!;
  let expandedNow = false;

  // Expansion: a leaf that is not terminal and has been visited at least once
  // gets its children. The paper expands once the visit count crosses a
  // threshold; here that threshold is 1 so the tree grows visibly.
  const isTerminal = FIXED_SHAPE.branching[leaf.depth] === undefined;
  if (!isTerminal && !leaf.expanded && leaf.visits >= 1) {
    working = expand(working, currentId, FIXED_SHAPE);
    nodes = working.nodes.map((n) => ({ ...n }));
    expandedNow = true;
  }

  // Evaluation: value network plus a single rollout, mixed by lambda.
  const vRead = valueNetwork(working, nodes[currentId]!);
  let cursor = vRead.cursor;
  const vTheta = vRead.value;
  const roll = rollout(working, nodes[currentId]!, cursor);
  cursor = roll.cursor;
  const z = roll.value;
  const lambda = working.params.lambda;
  const evaluation = (1 - lambda) * vTheta + lambda * z;

  // Backup: every edge on the path gains a visit and the evaluation, and its
  // mean action value is implied by W / N. Stored from the root player's view.
  for (const nid of path) {
    const node = nodes[nid];
    if (!node) continue;
    node.visits += 1;
    node.totalValue += evaluation;
  }

  const last: LastSim = {
    path,
    leaf: currentId,
    vTheta,
    rollout: z,
    evaluation,
    expanded: expandedNow,
  };

  return {
    ...working,
    nodes,
    rng: cursor,
    simulations: working.simulations + 1,
    last,
  };
}

// ---- Readouts ---------------------------------------------------------------

// The move the search would play: the most-visited child of the root, exactly
// the paper's selection rule.
export function chosenMove(state: SimState): number | null {
  const root = state.nodes[state.root];
  if (!root || root.children.length === 0) return null;
  let best: number | null = null;
  let bestVisits = -1;
  for (const cid of root.children) {
    const child = state.nodes[cid];
    if (!child) continue;
    if (child.visits > bestVisits) {
      bestVisits = child.visits;
      best = cid;
    }
  }
  return best;
}

// The true-best root move under minimax on the hidden leaf values. The searcher
// is the maximizer at the root; each move's value is the opponent's best
// (minimizing) reply.
export function trueBestMove(state: SimState): number | null {
  const root = state.nodes[state.root];
  if (!root || root.children.length === 0) return null;
  let best: number | null = null;
  let bestValue = -Infinity;
  for (const cid of root.children) {
    const child = state.nodes[cid];
    if (!child) continue;
    const value = minimaxValue(state, cid);
    if (value > bestValue) {
      bestValue = value;
      best = cid;
    }
  }
  return best;
}

export function minimaxValue(state: SimState, id: number): number {
  const node = state.nodes[id];
  if (!node) return 0;
  const path = buildSlotPath(state.nodes, id);
  const isLeafDepth = FIXED_SHAPE.branching[node.depth] === undefined;
  if (isLeafDepth) {
    return FIXED_SHAPE.leafValues[path.join(".")] ?? node.trueValue;
  }
  // Build the child true-values from the fixed shape regardless of whether the
  // search has expanded this node yet, so the truth is always defined.
  const count = FIXED_SHAPE.branching[node.depth]!;
  const values: number[] = [];
  for (let slot = 0; slot < count; slot++) {
    values.push(childMinimax(state, [...path, slot], node.depth + 1));
  }
  // toMove 0 maximizes, toMove 1 minimizes.
  return node.toMove === 0 ? Math.max(...values) : Math.min(...values);
}

function childMinimax(state: SimState, path: number[], depth: number): number {
  const isLeafDepth = FIXED_SHAPE.branching[depth] === undefined;
  if (isLeafDepth) {
    return FIXED_SHAPE.leafValues[path.join(".")] ?? 0;
  }
  const count = FIXED_SHAPE.branching[depth]!;
  const values: number[] = [];
  for (let slot = 0; slot < count; slot++) {
    values.push(childMinimax(state, [...path, slot], depth + 1));
  }
  // Even depth (root player) maximizes; odd depth (opponent) minimizes.
  return depth % 2 === 0 ? Math.max(...values) : Math.min(...values);
}

export function rootValueError(state: SimState): number {
  const chosen = chosenMove(state);
  const truth = trueBestMove(state);
  if (chosen === null || truth === null) return 1;
  const chosenTrue = minimaxValue(state, chosen);
  const truthTrue = minimaxValue(state, truth);
  return Math.abs(truthTrue - chosenTrue);
}

export function isCorrect(state: SimState): boolean {
  const chosen = chosenMove(state);
  const truth = trueBestMove(state);
  if (chosen === null || truth === null) return false;
  // Correct when the chosen move's true value matches the best move's true
  // value (ties on truth count as correct).
  return (
    Math.abs(minimaxValue(state, chosen) - minimaxValue(state, truth)) < 1e-9
  );
}

export function totalRootVisits(state: SimState): number {
  const root = state.nodes[state.root];
  if (!root) return 0;
  return root.children.reduce(
    (sum, cid) => sum + (state.nodes[cid]?.visits ?? 0),
    0,
  );
}

export const MOVE_NAMES = ["center", "contact", "corner"] as const;

export function snapshotOf(state: SimState): Snapshot {
  return {
    nodes: state.nodes.map((n) => ({ ...n, children: [...n.children] })),
    rng: state.rng,
    simulations: state.simulations,
    params: { ...state.params },
  };
}

// ---- Interventions ----------------------------------------------------------

function logEvent(state: SimState, label: string): EventLog<Snapshot> {
  return record(state.log, state.simulations, label, snapshotOf(state));
}

function applyIntervention(state: SimState, action: Intervention): SimState {
  switch (action.type) {
    case "loadPreset":
      return initialState(action.id, state.baseSeed);
    case "setLambda":
      return {
        ...state,
        params: { ...state.params, lambda: clamp(action.value, 0, 1) },
        log: logEvent(state, `lambda = ${action.value.toFixed(2)}`),
      };
    case "setCPuct":
      return {
        ...state,
        params: { ...state.params, cPuct: clamp(action.value, 0, 4) },
        log: logEvent(state, `c_puct = ${action.value.toFixed(2)}`),
      };
    case "setRolloutNoise":
      return {
        ...state,
        params: { ...state.params, rolloutNoise: clamp(action.value, 0, 1) },
      };
    case "setValueNoise":
      return {
        ...state,
        params: { ...state.params, valueNoise: clamp(action.value, 0, 1) },
      };
    case "setSeed": {
      const next = initialState(presetGuess(state), action.value >>> 0);
      return { ...next, log: logEvent(next, `seed = ${action.value}`) };
    }
    case "boostPrior": {
      const nodes = state.nodes.map((n) =>
        n.id === action.node ? { ...n, priorBias: 3 } : { ...n },
      );
      const node = nodes[action.node];
      const name = node ? moveLabel(state, node) : `node ${action.node}`;
      return {
        ...state,
        nodes,
        log: logEvent({ ...state, nodes }, `boost prior on ${name}`),
      };
    }
    case "resetPrior": {
      const nodes = state.nodes.map((n) =>
        n.id === action.node ? { ...n, priorBias: 1 } : { ...n },
      );
      return { ...state, nodes };
    }
    case "lieValue": {
      const nodes = state.nodes.map((n) =>
        n.id === action.node
          ? { ...n, valueLie: clamp(action.value, -1, 1) }
          : { ...n },
      );
      const node = nodes[action.node];
      const name = node ? moveLabel(state, node) : `node ${action.node}`;
      return {
        ...state,
        nodes,
        log: logEvent(
          { ...state, nodes },
          `value net lies on ${name}: ${action.value.toFixed(2)}`,
        ),
      };
    }
    case "clearLie": {
      const nodes = state.nodes.map((n) =>
        n.id === action.node ? { ...n, valueLie: null } : { ...n },
      );
      return { ...state, nodes };
    }
    case "prune": {
      const nodes = state.nodes.map((n) =>
        n.id === action.node ? { ...n, pruned: true } : { ...n },
      );
      const node = nodes[action.node];
      const name = node ? moveLabel(state, node) : `node ${action.node}`;
      return {
        ...state,
        nodes,
        log: logEvent({ ...state, nodes }, `prune ${name}`),
      };
    }
    case "unprune": {
      const nodes = state.nodes.map((n) =>
        n.id === action.node ? { ...n, pruned: false } : { ...n },
      );
      return { ...state, nodes };
    }
    case "restore": {
      const snap = action.snapshot;
      return {
        ...state,
        nodes: snap.nodes.map((n) => ({ ...n, children: [...n.children] })),
        rng: snap.rng,
        simulations: snap.simulations,
        params: { ...snap.params },
        last: null,
        log: logEvent(state, action.label),
      };
    }
  }
}

// A root-child node's name uses the move list; deeper nodes report their move
// plus a reply index.
function moveLabel(state: SimState, node: TreeNode): string {
  if (node.parent === state.root) {
    return MOVE_NAMES[node.childSlot] ?? `move ${node.childSlot}`;
  }
  const parent = node.parent !== null ? state.nodes[node.parent] : undefined;
  const parentName =
    parent && parent.parent === state.root
      ? (MOVE_NAMES[parent.childSlot] ?? `move ${parent.childSlot}`)
      : "node";
  return `${parentName} reply ${node.childSlot + 1}`;
}

// The seed/preset interventions need to rebuild the tree. We infer the preset
// from the prior layout (the trap presets boost move 1's prior), which is the
// only thing that distinguishes their starting trees.
function presetGuess(state: SimState): string {
  const move1 = state.nodes.find(
    (n) => n.parent === state.root && n.childSlot === 1,
  );
  if (move1 && move1.prior > 0.5) {
    // value-lies runs at lambda 0 (rollouts off); trap keeps lambda 0.5.
    return state.params.lambda <= 0.01 ? "value-lies" : "trap";
  }
  return "honest";
}

// ---- The reducer ------------------------------------------------------------

export function step(state: SimState, input: Input): SimState {
  if (input.kind === "tick") {
    return simulate(state);
  }
  return applyIntervention(state, input.action);
}
