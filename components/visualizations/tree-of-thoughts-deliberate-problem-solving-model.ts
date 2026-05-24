// A pure, deterministic model of Tree of Thoughts (Yao et al., 2023) run on the
// paper's Game of 24 task. The LLM's two jobs in the paper become two functions
// here: a thought generator that proposes the next intermediate equations from a
// state, and a state evaluator that scores each child sure / maybe / impossible.
// On top of those, the search algorithm (breadth-first with beam width b, or
// depth-first with a prune threshold and backtracking) explores the tree, keeps
// the promising nodes, prunes the rest, and backtracks when a path dies. This is
// exactly the ToT-BFS and ToT-DFS structure from Algorithms 1 and 2.
//
// Game of 24 is finite and exactly decidable, so the "truth" of whether 24 is
// still reachable from a state is computed by real arithmetic, not faked. That
// lets the evaluator be honest (read the truth) or noisy (a seeded chance to call
// a reachable state impossible), which is the paper's real failure mode: a good
// evaluator that wrongly prunes the only path and the search fails. All
// randomness is drawn from the seeded stream threaded through the state, so a
// seed plus a list of inputs reproduces a run exactly.

import { at } from "@/lib/simulation/array";
import {
  emptyLog,
  record,
  type EventLog,
  type SimEvent,
} from "@/lib/simulation/history";
import { nextRandom } from "@/lib/simulation/rng";

export type Verdict = "sure" | "maybe" | "impossible";

// What the search has decided about a node so far. frontier nodes are waiting to
// be expanded; expanded nodes have produced children; kept nodes survived a beam
// or threshold cut; pruned nodes were cut by the search; solution is a state that
// reaches 24; dead is a leaf with no children that is not a solution.
export type NodeStatus =
  | "frontier"
  | "expanded"
  | "kept"
  | "pruned"
  | "solution"
  | "dead";

export interface ThoughtNode {
  id: number;
  parentId: number | null;
  depth: number;
  // The numbers still available at this state. A solution state is [24].
  numbers: number[];
  // The equation that produced this state from its parent, e.g. "4 + 9 = 13".
  thought: string;
  status: NodeStatus;
  verdict: Verdict;
  // The numeric value the evaluator maps the verdict to (1 / 0.5 / 0.001).
  value: number;
  // Ground truth: can 24 still be reached from this state by real arithmetic.
  reachable: boolean;
  // True when the evaluator's verdict disagrees with the truth (a mislabel).
  misjudged: boolean;
}

const VALUE_OF: Record<Verdict, number> = {
  sure: 1,
  maybe: 0.5,
  impossible: 0.001,
};

// Pretty-print a number the way the puzzle reads, keeping small fractions exact
// enough to recognize (e.g. 8/3) without a wall of decimals.
function fmt(value: number): string {
  if (Math.abs(value - Math.round(value)) < 1e-9)
    return String(Math.round(value));
  return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

// Two numbers and an operation produce one result. Game of 24 allows rationals as
// intermediates (8 / (3 - 8 / 3) = 24), so division is kept whenever the divisor
// is non-zero, and subtraction is kept both ways. The larger-first ordering on +
// and * drops mirror duplicates so the tree stays legible.
function combine(a: number, b: number): { value: number; op: string }[] {
  const out: { value: number; op: string }[] = [];
  out.push({ value: a + b, op: "+" });
  out.push({ value: a * b, op: "*" });
  if (a - b >= 0) out.push({ value: a - b, op: "-" });
  if (b - a > 0) out.push({ value: b - a, op: "-r" });
  if (b !== 0) out.push({ value: a / b, op: "/" });
  if (a !== 0 && a !== b) out.push({ value: b / a, op: "/r" });
  return out;
}

// Every distinct next state from picking an unordered pair of the remaining
// numbers and applying one operation. Returns the child's remaining numbers and
// the equation text. Deduplicated so the tree stays legible.
function nextStates(
  numbers: number[],
): { numbers: number[]; thought: string }[] {
  const results: { numbers: number[]; thought: string }[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < numbers.length; i++) {
    for (let j = i + 1; j < numbers.length; j++) {
      const a = at(numbers, i);
      const b = at(numbers, j);
      const rest = numbers.filter((_, idx) => idx !== i && idx !== j);
      for (const combo of combine(a, b)) {
        const left = combo.op.endsWith("r") ? b : a;
        const right = combo.op.endsWith("r") ? a : b;
        const symbol = combo.op[0];
        const thought = `${fmt(left)} ${symbol} ${fmt(right)} = ${fmt(combo.value)}`;
        const childNumbers = [...rest, combo.value].sort((x, y) => x - y);
        const key = `${childNumbers.map(fmt).join(",")}|${thought}`;
        if (seen.has(key)) continue;
        seen.add(key);
        results.push({ numbers: childNumbers, thought });
      }
    }
  }
  return results;
}

// Ground truth, computed by exhaustive arithmetic: starting from these numbers,
// can repeated pair-combining land on exactly 24. Memo-free but tiny (at most 4
// numbers), so it stays fast.
export function canReach24(numbers: number[]): boolean {
  if (numbers.length === 1) return Math.abs(at(numbers, 0) - 24) < 1e-9;
  for (let i = 0; i < numbers.length; i++) {
    for (let j = i + 1; j < numbers.length; j++) {
      const a = at(numbers, i);
      const b = at(numbers, j);
      const rest = numbers.filter((_, idx) => idx !== i && idx !== j);
      const candidates = [a + b, a * b, a - b, b - a];
      if (b !== 0) candidates.push(a / b);
      if (a !== 0) candidates.push(b / a);
      for (const value of candidates) {
        if (canReach24([...rest, value])) return true;
      }
    }
  }
  return false;
}

// The evaluator. Honest mode reads the ground truth: reachable -> sure, else
// impossible, with a near-24 single number called sure. Noisy mode keeps that as
// a base but, with probability evalNoise drawn from the seeded stream, downgrades
// a truly reachable state to impossible (the paper's pruning failure) or upgrades
// a dead state to maybe. The verdict is what the search acts on; the truth is
// only ever used to color the honesty light, never to pick branches.
function evaluate(
  numbers: number[],
  reachable: boolean,
  honest: boolean,
  noise: number,
  rngState: number,
): { verdict: Verdict; rngState: number } {
  if (numbers.length === 1) {
    const won = Math.abs(at(numbers, 0) - 24) < 1e-9;
    return { verdict: won ? "sure" : "impossible", rngState };
  }
  const truthful: Verdict = reachable ? "sure" : "impossible";
  if (honest) return { verdict: truthful, rngState };

  const draw = nextRandom(rngState);
  if (draw.value < noise) {
    // A confident mistake, the failure the paper warns about.
    const wrong: Verdict = reachable ? "impossible" : "maybe";
    return { verdict: wrong, rngState: draw.state };
  }
  return { verdict: truthful, rngState: draw.state };
}

export type Algorithm = "bfs" | "dfs";

export interface Params {
  algorithm: Algorithm;
  beam: number; // b, BFS states kept per depth
  threshold: number; // v_th, DFS prune cutoff on value
  honest: boolean; // evaluator reads ground truth when true
  evalNoise: number; // chance a noisy evaluator mislabels a reachable state
  proposals: number; // k, candidates generated per expansion (caps children)
  puzzleId: string;
}

export interface Puzzle {
  id: string;
  label: string;
  numbers: number[];
  solvable: boolean;
  note: string;
}

export const PUZZLES: Puzzle[] = [
  {
    id: "4-9-10-13",
    label: "4 9 10 13",
    numbers: [4, 9, 10, 13],
    solvable: true,
    note: "The paper's worked example. (10 - 4) * (13 - 9) = 24.",
  },
  {
    id: "2-4-6-8",
    label: "2 4 6 8",
    numbers: [2, 4, 6, 8],
    solvable: true,
    note: "Many routes reach 24, e.g. (2 + 4) * (8 - 4)... wide, easy tree.",
  },
  {
    id: "1-1-1-1",
    label: "1 1 1 1",
    numbers: [1, 1, 1, 1],
    solvable: false,
    note: "Too small to reach 24. The search should fail honestly, never invent a solution.",
  },
  {
    id: "3-3-8-8",
    label: "3 3 8 8",
    numbers: [3, 3, 8, 8],
    solvable: true,
    note: "Only 8 / (3 - 8 / 3) = 24 works, a single narrow path a tight beam can miss.",
  },
];

export function getPuzzle(id: string): Puzzle {
  return PUZZLES.find((puzzle) => puzzle.id === id) ?? at(PUZZLES, 0);
}

export interface SearchSnapshot {
  params: Params;
}

export type ToTEvent = SimEvent<SearchSnapshot>;

// The whole live model. nodes is the tree built so far (index by id is positions
// in the array since ids are assigned in order). frontier holds the ids the
// search still has to consider. dfsStack is the explicit DFS path for backtrack.
// done is set when a solution is found or the search exhausts the frontier.
export interface SimState {
  params: Params;
  nodes: ThoughtNode[];
  frontier: number[];
  dfsStack: number[];
  nextId: number;
  tick: number;
  done: boolean;
  outcome: "searching" | "solved" | "failed";
  solutionId: number | null;
  visited: number; // nodes the search has expanded (the Figure 3a x-axis)
  rngState: number;
  log: EventLog<SearchSnapshot>;
}

const ROOT_ID = 0;

function makeRoot(numbers: number[]): ThoughtNode {
  return {
    id: ROOT_ID,
    parentId: null,
    depth: 0,
    numbers,
    thought: numbers.join(" "),
    status: "frontier",
    verdict: "maybe",
    value: VALUE_OF.maybe,
    reachable: canReach24(numbers),
    misjudged: false,
  };
}

export function initialState(params: Params, seed = 7): SimState {
  const puzzle = getPuzzle(params.puzzleId);
  const numbers = [...puzzle.numbers].sort((a, b) => a - b);
  return {
    params,
    nodes: [makeRoot(numbers)],
    frontier: [ROOT_ID],
    dfsStack: [ROOT_ID],
    nextId: 1,
    tick: 0,
    done: false,
    outcome: "searching",
    solutionId: null,
    visited: 0,
    rngState: seed >>> 0,
    log: emptyLog(),
  };
}

function node(state: SimState, id: number): ThoughtNode {
  const found = state.nodes.find((n) => n.id === id);
  if (found === undefined) throw new RangeError(`no node with id ${id}`);
  return found;
}

// Expand one parent into its evaluated children, capped at k proposals, biased so
// the higher-value (sure first) candidates survive the cap, matching how the
// generator's k samples then feed the evaluator.
function expand(
  state: SimState,
  parentId: number,
): { children: ThoughtNode[]; rngState: number } {
  const parent = node(state, parentId);
  const { honest, evalNoise, proposals } = state.params;
  let rngState = state.rngState;
  // The generator proposes k candidates, then the evaluator scores all of them.
  // Capping on generation (not on the verdict) keeps a wrongly-pruned reachable
  // node visible, which is the whole point of letting the reader see misjudges.
  // Solution states are always kept so the search can never silently drop a win.
  // The generator is a decent proposer: it tends to suggest moves that keep 24
  // in reach, so order the candidates reachable-first before capping to k. That
  // is the generator's job. The evaluator that scores them next is a separate
  // role and can still lie, which is where misjudges and failures come from.
  const proposed = nextStates(parent.numbers);
  const solutions = proposed.filter(
    (c) => c.numbers.length === 1 && Math.abs(at(c.numbers, 0) - 24) < 1e-9,
  );
  const others = proposed
    .filter((c) => !solutions.includes(c))
    .sort(
      (a, b) => Number(canReach24(b.numbers)) - Number(canReach24(a.numbers)),
    );
  const capped = [...solutions, ...others].slice(
    0,
    Math.max(proposals, solutions.length),
  );

  const children = capped.map((candidate, index) => {
    const reachable = canReach24(candidate.numbers);
    const result = evaluate(
      candidate.numbers,
      reachable,
      honest,
      evalNoise,
      rngState,
    );
    rngState = result.rngState;
    const verdict = result.verdict;
    const isSolution =
      candidate.numbers.length === 1 &&
      Math.abs(at(candidate.numbers, 0) - 24) < 1e-9;
    return {
      id: state.nextId + index,
      parentId,
      depth: parent.depth + 1,
      numbers: candidate.numbers,
      thought: candidate.thought,
      status: isSolution
        ? ("solution" as NodeStatus)
        : ("frontier" as NodeStatus),
      verdict,
      value: VALUE_OF[verdict],
      reachable,
      misjudged: verdict === "impossible" ? reachable : false,
    };
  });
  return { children, rngState };
}

function logged(
  state: SimState,
  label: string,
  next: Partial<SimState>,
): SimState {
  const merged = { ...state, ...next };
  return {
    ...merged,
    log: record(state.log, state.tick, label, { params: state.params }),
  };
}

// One BFS layer: expand every frontier node at the current shallowest depth,
// evaluate the children, then keep the best b across that whole layer and prune
// the rest. This is S_t <- argmax over size-b subsets of sum of values, Algorithm
// 1, with ties broken by id for determinism.
function stepBfs(state: SimState): SimState {
  if (state.frontier.length === 0) {
    return logged(state, "frontier empty -> failed", {
      done: true,
      outcome: "failed",
    });
  }
  const minDepth = Math.min(
    ...state.frontier.map((id) => node(state, id).depth),
  );
  const layer = state.frontier.filter(
    (id) => node(state, id).depth === minDepth,
  );
  const rest = state.frontier.filter(
    (id) => node(state, id).depth !== minDepth,
  );

  let nodes = state.nodes.map((n) =>
    layer.includes(n.id) ? { ...n, status: "expanded" as NodeStatus } : n,
  );
  let nextId = state.nextId;
  let rngState = state.rngState;
  let allChildren: ThoughtNode[] = [];

  for (const parentId of layer) {
    const result = expand({ ...state, nodes, nextId, rngState }, parentId);
    rngState = result.rngState;
    nodes = [...nodes, ...result.children];
    nextId += result.children.length;
    allChildren = [...allChildren, ...result.children];
  }

  const solution = allChildren.find((c) => c.status === "solution");
  if (solution) {
    return logged(
      { ...state, nodes, nextId, rngState },
      `solved: ${solution.thought}`,
      {
        nodes,
        nextId,
        rngState,
        frontier: [],
        done: true,
        outcome: "solved",
        solutionId: solution.id,
        visited: state.visited + layer.length,
      },
    );
  }

  // Pruning happens on the verdict, so an "impossible" verdict drops a child even
  // if it is truly reachable. The beam then keeps the best b of the survivors.
  const alive = allChildren.filter((c) => c.verdict !== "impossible");
  const beamKept = alive
    .slice()
    .sort((a, b) => b.value - a.value || a.id - b.id)
    .slice(0, state.params.beam);
  const keptIds = new Set(beamKept.map((c) => c.id));

  nodes = nodes.map((n) => {
    if (!allChildren.some((c) => c.id === n.id)) return n;
    if (keptIds.has(n.id)) return { ...n, status: "kept" as NodeStatus };
    return { ...n, status: "pruned" as NodeStatus };
  });

  const newFrontier = [...rest, ...beamKept.map((c) => c.id)];
  if (newFrontier.length === 0) {
    return logged(
      { ...state, nodes, nextId, rngState },
      "all pruned -> failed",
      {
        nodes,
        nextId,
        rngState,
        frontier: [],
        done: true,
        outcome: "failed",
        visited: state.visited + layer.length,
      },
    );
  }

  return logged(
    { ...state, nodes, nextId, rngState },
    `depth ${minDepth + 1}: kept ${beamKept.length} of ${allChildren.length}`,
    {
      nodes,
      nextId,
      rngState,
      frontier: newFrontier,
      visited: state.visited + layer.length,
    },
  );
}

// One DFS action: take the top of the stack, expand it, and dive into its most
// promising child whose value clears the threshold. If none clears it, prune the
// subtree and backtrack to the parent, exactly Algorithm 2.
function stepDfs(state: SimState): SimState {
  if (state.dfsStack.length === 0) {
    return logged(state, "stack empty -> failed", {
      done: true,
      outcome: "failed",
    });
  }
  const currentId = at(state.dfsStack, state.dfsStack.length - 1);
  const current = node(state, currentId);

  const result = expand(state, currentId);
  const nextId = state.nextId + result.children.length;
  let nodes = [
    ...state.nodes.map((n) =>
      n.id === currentId ? { ...n, status: "expanded" as NodeStatus } : n,
    ),
    ...result.children,
  ];

  const solution = result.children.find((c) => c.status === "solution");
  if (solution) {
    return logged(
      { ...state, nodes, nextId, rngState: result.rngState },
      `solved: ${solution.thought}`,
      {
        nodes,
        nextId,
        rngState: result.rngState,
        done: true,
        outcome: "solved",
        solutionId: solution.id,
        visited: state.visited + 1,
      },
    );
  }

  const promising = result.children
    .filter((c) => c.value > state.params.threshold)
    .sort((a, b) => b.value - a.value || a.id - b.id);

  if (promising.length === 0) {
    // Dead end. Prune this subtree's children and backtrack.
    const childIds = new Set(result.children.map((c) => c.id));
    nodes = nodes.map((n) => {
      if (childIds.has(n.id)) return { ...n, status: "pruned" as NodeStatus };
      if (n.id === currentId) return { ...n, status: "dead" as NodeStatus };
      return n;
    });
    const stack = state.dfsStack.slice(0, -1);
    if (stack.length === 0) {
      return logged(
        { ...state, nodes, nextId, rngState: result.rngState },
        "backtracked past root -> failed",
        {
          nodes,
          nextId,
          rngState: result.rngState,
          dfsStack: [],
          done: true,
          outcome: "failed",
          visited: state.visited + 1,
        },
      );
    }
    return logged(
      { ...state, nodes, nextId, rngState: result.rngState },
      `dead end at depth ${current.depth}, backtrack`,
      {
        nodes,
        nextId,
        rngState: result.rngState,
        dfsStack: stack,
        visited: state.visited + 1,
      },
    );
  }

  const dive = at(promising, 0);
  const pruned = result.children.filter((c) => c.id !== dive.id);
  const prunedIds = new Set(pruned.map((c) => c.id));
  nodes = nodes.map((n) => {
    if (n.id === dive.id) return { ...n, status: "kept" as NodeStatus };
    if (prunedIds.has(n.id))
      return n.value > state.params.threshold
        ? n
        : { ...n, status: "pruned" as NodeStatus };
    return n;
  });

  return logged(
    { ...state, nodes, nextId, rngState: result.rngState },
    `dive into ${dive.thought}`,
    {
      nodes,
      nextId,
      rngState: result.rngState,
      dfsStack: [...state.dfsStack, dive.id],
      visited: state.visited + 1,
    },
  );
}

export type Intervention =
  | { type: "setAlgorithm"; algorithm: Algorithm }
  | { type: "setBeam"; value: number }
  | { type: "setThreshold"; value: number }
  | { type: "toggleHonest" }
  | { type: "setNoise"; value: number }
  | { type: "setProposals"; value: number }
  | { type: "setPuzzle"; id: string }
  | { type: "flipVerdict"; nodeId: number } // make the evaluator lie / correct it
  | { type: "forcePrune"; nodeId: number } // cut a node the search kept
  | { type: "forceKeep"; nodeId: number } // rescue a node the search pruned
  | { type: "loadPreset"; id: string }
  | { type: "restore"; snapshot: SearchSnapshot; label: string };

export type Input =
  | { kind: "tick" }
  | { kind: "intervene"; action: Intervention };

const VERDICT_CYCLE: Verdict[] = ["sure", "maybe", "impossible"];

function cycleVerdict(verdict: Verdict): Verdict {
  const index = VERDICT_CYCLE.indexOf(verdict);
  return at(VERDICT_CYCLE, (index + 1) % VERDICT_CYCLE.length);
}

function withNode(
  state: SimState,
  nodeId: number,
  change: (n: ThoughtNode) => ThoughtNode,
): ThoughtNode[] {
  return state.nodes.map((n) => (n.id === nodeId ? change(n) : n));
}

function applyIntervention(state: SimState, action: Intervention): SimState {
  switch (action.type) {
    case "setAlgorithm":
      return logged(state, `algorithm -> ${action.algorithm.toUpperCase()}`, {
        params: { ...state.params, algorithm: action.algorithm },
      });
    case "setBeam":
      return logged(state, `beam b -> ${action.value}`, {
        params: { ...state.params, beam: action.value },
      });
    case "setThreshold":
      return logged(state, `threshold -> ${action.value.toFixed(2)}`, {
        params: { ...state.params, threshold: action.value },
      });
    case "toggleHonest": {
      const honest = !state.params.honest;
      return logged(state, `evaluator -> ${honest ? "honest" : "noisy"}`, {
        params: { ...state.params, honest },
      });
    }
    case "setNoise":
      return logged(state, `eval noise -> ${action.value.toFixed(2)}`, {
        params: { ...state.params, evalNoise: action.value },
      });
    case "setProposals":
      return logged(state, `proposals k -> ${action.value}`, {
        params: { ...state.params, proposals: action.value },
      });
    case "setPuzzle":
      return logged(
        initialState({ ...state.params, puzzleId: action.id }, state.rngState),
        `puzzle -> ${getPuzzle(action.id).label}`,
        {},
      );
    case "flipVerdict": {
      const target = state.nodes.find((n) => n.id === action.nodeId);
      if (!target || target.status === "solution") return state;
      const verdict = cycleVerdict(target.verdict);
      const nodes = withNode(state, action.nodeId, (n) => ({
        ...n,
        verdict,
        value: VALUE_OF[verdict],
        misjudged: verdict === "impossible" ? n.reachable : false,
      }));
      return logged(state, `verdict on #${action.nodeId} -> ${verdict}`, {
        nodes,
      });
    }
    case "forcePrune": {
      const nodes = withNode(state, action.nodeId, (n) => ({
        ...n,
        status: "pruned",
      }));
      const frontier = state.frontier.filter((id) => id !== action.nodeId);
      const dfsStack = state.dfsStack.filter((id) => id !== action.nodeId);
      return logged(state, `force-prune #${action.nodeId}`, {
        nodes,
        frontier,
        dfsStack,
      });
    }
    case "forceKeep": {
      const target = state.nodes.find((n) => n.id === action.nodeId);
      if (!target) return state;
      const nodes = withNode(state, action.nodeId, (n) => ({
        ...n,
        status: "kept",
      }));
      const frontier = state.frontier.includes(action.nodeId)
        ? state.frontier
        : [...state.frontier, action.nodeId];
      return logged(state, `force-keep #${action.nodeId}`, {
        nodes,
        frontier,
        done: false,
        outcome: state.outcome === "failed" ? "searching" : state.outcome,
      });
    }
    case "loadPreset": {
      const preset = getPreset(action.id);
      return logged(
        initialState(preset.params, preset.seed),
        `preset -> ${preset.label}`,
        {},
      );
    }
    case "restore":
      return logged(
        initialState(action.snapshot.params, state.rngState),
        action.label,
        {},
      );
  }
}

// The pure reducer. A tick advances the search one action (one BFS layer or one
// DFS dive/backtrack). An intervention changes a parameter or pokes a node and
// records what happened. Every step is deterministic given the seed.
export function step(state: SimState, input: Input): SimState {
  if (input.kind === "intervene") {
    return applyIntervention(state, input.action);
  }
  if (state.done) return state;
  const advanced: SimState = { ...state, tick: state.tick + 1 };
  return state.params.algorithm === "bfs"
    ? stepBfs(advanced)
    : stepDfs(advanced);
}

// Runs the search to completion from a fresh state, used by tests and to read off
// an outcome without a render loop. Bounded so a pathological run can't spin.
export function runToEnd(params: Params, seed = 7, maxTicks = 200): SimState {
  let state = initialState(params, seed);
  let ticks = 0;
  while (!state.done && ticks < maxTicks) {
    state = step(state, { kind: "tick" });
    ticks += 1;
  }
  return state;
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
    id: "bfs-wide",
    label: "BFS, wide beam (b=5)",
    blurb:
      "The paper's 74% setting. A wide beam keeps the winning branch alive at every depth, so the search reaches 24.",
    params: {
      algorithm: "bfs",
      beam: 5,
      threshold: 0.3,
      honest: true,
      evalNoise: 0.3,
      proposals: 6,
      puzzleId: "4-9-10-13",
    },
    seed: 7,
  },
  {
    id: "bfs-narrow",
    label: "BFS, narrow beam (b=1)",
    blurb:
      "A noisy evaluator plus a beam of 1 keeps only the single best-rated node per depth, so one bad score drops the winning branch and the search fails. Widen the beam to b=5 and the answer comes back.",
    params: {
      algorithm: "bfs",
      beam: 1,
      threshold: 0.3,
      honest: false,
      evalNoise: 0.5,
      proposals: 6,
      puzzleId: "4-9-10-13",
    },
    seed: 7,
  },
  {
    id: "noisy-eval",
    label: "Evaluator lies",
    blurb:
      "A noisy evaluator calls reachable states impossible and prunes them, the red misjudge flags. The only path gets cut and the search fails. Turn the evaluator honest and rerun to watch the same beam succeed.",
    params: {
      algorithm: "bfs",
      beam: 2,
      threshold: 0.3,
      honest: false,
      evalNoise: 0.55,
      proposals: 6,
      puzzleId: "4-9-10-13",
    },
    seed: 3,
  },
  {
    id: "dfs",
    label: "DFS with backtracking",
    blurb:
      "Depth-first dives into the best child, prunes below the threshold, and backtracks on a dead end. Raise the threshold to make it prune harder.",
    params: {
      algorithm: "dfs",
      beam: 5,
      threshold: 0.3,
      honest: true,
      evalNoise: 0.3,
      proposals: 6,
      puzzleId: "4-9-10-13",
    },
    seed: 7,
  },
];

export function getPreset(id: string): Preset {
  return PRESETS.find((preset) => preset.id === id) ?? at(PRESETS, 0);
}

// Children of a node, in id order, for the view's tree layout.
export function childrenOf(state: SimState, parentId: number): ThoughtNode[] {
  return state.nodes
    .filter((n) => n.parentId === parentId)
    .sort((a, b) => a.id - b.id);
}
